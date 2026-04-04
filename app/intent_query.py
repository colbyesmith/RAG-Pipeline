"""
Intent detection and query rewriting for retrieval.

Fast path avoids LLM for obvious chitchat. Otherwise one structured Mistral call returns
whether to retrieve plus keyword/semantic query variants.
"""

from __future__ import annotations

import re
from typing import Any

import httpx

from app.config import Settings
from app.mistral_client import mistral_chat, parse_json_object


_CHITCHAT = re.compile(
    r"^\s*(hi+|hello+|hey+|good\s+(morning|afternoon|evening)|howdy|yo)\b[!.?\s]*$",
    re.I,
)
_THANKS = re.compile(r"^\s*(thanks?|thank you|thx|cheers)\b[!.?\s]*$", re.I)
_BYE = re.compile(r"^\s*(bye|goodbye|see you|cya)\b[!.?\s]*$", re.I)


def fast_intent(query: str) -> dict[str, Any] | None:
    q = query.strip()
    if not q:
        return {
            "needs_retrieval": False,
            "intent": "empty",
            "retrieval_query_semantic": "",
            "retrieval_query_keywords": "",
            "direct_reply": "Ask a question about your uploaded PDFs.",
        }
    if _CHITCHAT.match(q):
        return {
            "needs_retrieval": False,
            "intent": "greeting",
            "retrieval_query_semantic": "",
            "retrieval_query_keywords": "",
            "direct_reply": "Hello! Upload PDFs and ask questions about their contents.",
        }
    if _THANKS.match(q):
        return {
            "needs_retrieval": False,
            "intent": "thanks",
            "retrieval_query_semantic": "",
            "retrieval_query_keywords": "",
            "direct_reply": "You're welcome.",
        }
    if _BYE.match(q):
        return {
            "needs_retrieval": False,
            "intent": "farewell",
            "retrieval_query_semantic": "",
            "retrieval_query_keywords": "",
            "direct_reply": "Goodbye!",
        }
    return None


INTENT_SCHEMA_PROMPT = """You classify user messages for a document Q&A assistant.

Return a single JSON object with keys:
- needs_retrieval (boolean): true if answering well requires searching uploaded PDFs.
- intent: one of "factual", "list", "compare", "summary", "other".
- retrieval_query_semantic: a concise English sentence capturing meaning for semantic search (empty if needs_retrieval false).
- retrieval_query_keywords: space-separated keywords for lexical search (empty if needs_retrieval false).
- direct_reply: short friendly reply if needs_retrieval is false, else empty string.

Rules:
- Greetings, thanks, meta questions about the bot, or vague chat => needs_retrieval false.
- Questions about document content, policies, numbers, definitions => needs_retrieval true.
- For intent "list", user wants enumerated items. "compare" wants differences/similarities. "summary" wants overview.

User message:
"""


async def classify_and_rewrite(
    client: httpx.AsyncClient,
    settings: Settings,
    query: str,
) -> dict[str, Any]:
    hit = fast_intent(query)
    if hit is not None:
        return hit

    messages = [
        {"role": "system", "content": "You output only valid JSON objects."},
        {"role": "user", "content": INTENT_SCHEMA_PROMPT + query},
    ]
    raw = await mistral_chat(
        client,
        settings,
        messages,
        temperature=0.1,
        max_tokens=256,
        response_format={"type": "json_object"},
    )
    data = parse_json_object(raw)
    data.setdefault("needs_retrieval", True)
    data.setdefault("intent", "factual")
    data.setdefault("retrieval_query_semantic", query)
    data.setdefault("retrieval_query_keywords", query)
    data.setdefault("direct_reply", "")
    return data


