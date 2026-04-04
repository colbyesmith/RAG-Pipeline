"""Answer generation with intent-specific templates and citation grounding."""

from __future__ import annotations

import httpx

from app.config import Settings
from app.evidence import evidence_check, merge_context
from app.hybrid_search import RankedChunk
from app.mistral_client import mistral_chat


_CITATION_RULE = (
    "CITATION RULE: Every sentence or bullet that states a fact from CONTEXT must end with an inline "
    "citation in parentheses, using the EXACT filename and page from that chunk's header. "
    "Format: (Source: <filename exactly as given>, p.N) or (Source: <filename>, p.N–M) for a page range. "
    "If one sentence combines two chunks, use two citations in order. Do not cite a file or page not in CONTEXT.\n\n"
)


def build_system_prompt(intent: str) -> str:
    base = (
        "You are a careful assistant that answers ONLY using the provided CONTEXT from the user's PDFs. "
        "If context is insufficient, say so briefly. Quote or paraphrase faithfully; do not invent facts. "
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
