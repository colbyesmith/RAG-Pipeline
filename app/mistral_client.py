from __future__ import annotations

import asyncio
import json
from typing import Any

import httpx

from app.config import Settings


class MistralError(RuntimeError):
    pass


async def _post_json_with_retries(
    client: httpx.AsyncClient,
    settings: Settings,
    url: str,
    headers: dict[str, str],
    body: dict[str, Any],
    *,
    timeout: float = 120.0,
) -> httpx.Response:
    """POST with exponential backoff on 429 / 503 (rate limit / overload)."""
    last: httpx.Response | None = None
    max_retries = max(0, settings.mistral_api_max_retries)
    base = max(0.1, settings.mistral_api_retry_base_seconds)

    for attempt in range(max_retries + 1):
        last = await client.post(url, headers=headers, json=body, timeout=timeout)
        if last.status_code in (429, 503) and attempt < max_retries:
            ra = last.headers.get("Retry-After")
            if ra is not None:
                try:
                    wait = float(ra)
                except ValueError:
                    wait = min(base * (2**attempt), 60.0)
            else:
                wait = min(base * (2**attempt), 60.0)
            await asyncio.sleep(wait)
            continue
        return last
    assert last is not None
    return last


async def mistral_embed(client: httpx.AsyncClient, settings: Settings, texts: list[str]) -> list[list[float]]:
    if not settings.mistral_api_key:
        raise MistralError("MISTRAL_API_KEY is not set")
    url = f"{settings.mistral_base_url.rstrip('/')}/embeddings"
    headers = {"Authorization": f"Bearer {settings.mistral_api_key}", "Content-Type": "application/json"}
    payload: dict[str, Any] = {"model": settings.embed_model, "input": texts}
    r = await _post_json_with_retries(client, settings, url, headers, payload, timeout=120.0)
    if r.status_code >= 400:
        raise MistralError(f"Embeddings error {r.status_code}: {r.text[:500]}")
    data = r.json()
    out = []
    for item in sorted(data.get("data", []), key=lambda x: x.get("index", 0)):
        out.append(item["embedding"])
    return out


async def mistral_chat(
    client: httpx.AsyncClient,
    settings: Settings,
    messages: list[dict[str, str]],
    *,
    temperature: float = 0.2,
    max_tokens: int = 1024,
    response_format: dict[str, Any] | None = None,
) -> str:
    if not settings.mistral_api_key:
        raise MistralError("MISTRAL_API_KEY is not set")
    url = f"{settings.mistral_base_url.rstrip('/')}/chat/completions"
    headers = {"Authorization": f"Bearer {settings.mistral_api_key}", "Content-Type": "application/json"}
    body: dict[str, Any] = {
        "model": settings.chat_model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if response_format is not None:
        body["response_format"] = response_format
    r = await _post_json_with_retries(client, settings, url, headers, body, timeout=120.0)
    if r.status_code >= 400:
        raise MistralError(f"Chat error {r.status_code}: {r.text[:500]}")
    data = r.json()
    try:
        return data["choices"][0]["message"]["content"]
    except (KeyError, IndexError) as e:
        raise MistralError(f"Unexpected chat response: {data}") from e


def parse_json_object(text: str) -> dict[str, Any]:
    text = text.strip()
    if text.startswith("```"):
        text = text.removeprefix("```json").removeprefix("```").strip()
        if text.endswith("```"):
            text = text[: text.rfind("```")].strip()
    return json.loads(text)
