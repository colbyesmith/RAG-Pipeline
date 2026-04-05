"""Answer generation with intent-specific templates and citation grounding."""

from __future__ import annotations

import re

import httpx

from app.config import Settings
from app.evidence import evidence_check, merge_context
from app.hybrid_search import RankedChunk
from app.mistral_client import mistral_chat


_CITATION_RULE = (
    "CITATION RULE: Every sentence or bullet that states a fact from CONTEXT must end with an inline "
    "citation in parentheses, using the EXACT filename and page from that chunk's header. "
    "Format: (Source: <filename exactly as given>, p.N) or (Source: <filename>, p.N–M) for a page range. "
    "If one sentence combines two chunks, use two citations in order. Do not cite a file or page not in CONTEXT.\n"
    "If CONTEXT does not address the question (wrong topic or missing facts), give a short statement only. "
    "Do NOT add (Source: …) lines or list filenames/pages—retrieved chunks are not evidence for that reply.\n\n"
)

# Heuristic: model said the KB doesn't cover the question (strip API citations / footers).
_CONTEXT_DENIAL_SNIPPETS = (
    "provided context does not contain",
    "context does not contain",
    "does not contain any information",
    "does not contain information about",
    "no information in the provided",
    "no information about",
    "cannot answer this question from",
    "cannot answer from the provided",
    "not mentioned in the provided context",
    "not discussed in the provided",
    "outside the scope of the provided",
    "not found in the provided context",
    "context is insufficient to",
    "insufficient information in the",
    "none of the provided passages",
)


def looks_like_context_denial(answer: str) -> bool:
    """True when the reply is mainly 'your PDFs don't cover this' (not a cited factual claim)."""
    t = answer.lstrip()
    t = re.sub(r"^(unfortunately|sorry),?\s+", "", t, flags=re.IGNORECASE)
    start = t[:240].lower()
    opening_ok = start.startswith(
        (
            "the provided context",
            "the context",
            "based on the provided",
            "based on the context",
            "none of the provided",
            "none of the passages",
            "none of these",
            "i cannot",
            "i'm unable",
            "i am unable",
            "there is no information",
            "there isn't information",
            "there is insufficient",
            "the documents ",
            "the passages ",
            "according to the provided context,",
            "no information is available",
            "no relevant information",
        )
    )
    if not opening_ok:
        return False
    head = t[:1400].lower()
    return any(s in head for s in _CONTEXT_DENIAL_SNIPPETS)


def strip_trailing_inline_sources(answer: str) -> str:
    """Remove a trailing '(Source: …)' block the model may still emit."""
    m = re.search(r"\n\s*\(Source:", answer, re.IGNORECASE)
    if m:
        return answer[: m.start()].strip()
    return answer.strip()


def build_system_prompt(intent: str) -> str:
    base = (
        "You are a careful assistant that answers ONLY using the provided CONTEXT from the user's PDFs. "
        "If context is insufficient or off-topic for the question, say so briefly in one or two sentences. "
        "Quote or paraphrase faithfully when you do answer; do not invent facts. "
        "When CONTEXT combines passages that answer different parts of the question, link them explicitly.\n\n"
        + _CITATION_RULE
    )
    if intent == "list":
        return (
            base
            + "Format: markdown bullet list. Each bullet must end with its (Source: …) citation.\n"
        )
    if intent == "compare":
        return (
            base
            + "Format: markdown table; include a final column **Source** with (filename, page) for each row, "
            "or put the citation at the end of the Details cell.\n"
        )
    if intent == "summary":
        return base + "Format: 3-6 bullet points; each bullet ends with (Source: …) as above.\n"
    return base + "Format: concise paragraphs; each factual sentence ends with (Source: …).\n"


def _page_label(page_start: int, page_end: int) -> str:
    if page_start == page_end:
        return f"p.{page_start}"
    return f"p.{page_start}–{page_end}"


def format_context_chunk(r: RankedChunk) -> str:
    ch = r.stored.chunk
    return (
        f"### CHUNK {ch.id}\n"
        f"Source document: {ch.source_file}\n"
        f"Pages: {_page_label(ch.page_start, ch.page_end)}\n"
        f"{ch.text}"
    )


def build_user_prompt(question: str, ranked: list[RankedChunk]) -> str:
    parts = [format_context_chunk(r) for r in ranked]
    ctx = "\n\n".join(parts)
    return (
        f"QUESTION:\n{question}\n\nCONTEXT:\n{ctx}\n\n"
        f"Answer using only CONTEXT. Use the exact `Source document` filenames in every (Source: …) citation."
    )


async def generate_answer(
    client: httpx.AsyncClient,
    settings: Settings,
    *,
    question: str,
    intent: str,
    ranked: list[RankedChunk],
) -> tuple[str, list[str]]:
    messages = [
        {"role": "system", "content": build_system_prompt(intent)},
        {"role": "user", "content": build_user_prompt(question, ranked)},
    ]
    answer = await mistral_chat(client, settings, messages, temperature=0.2, max_tokens=900)
    ctx = merge_context([r.stored.chunk.text for r in ranked])
    flags = evidence_check(answer, ctx, settings.evidence_overlap_min)
    return answer.strip(), flags
