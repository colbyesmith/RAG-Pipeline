"""
Optional second retrieval hop: propose a follow-up query from the LLM, search again, merge results.

This is lightweight multi-hop *retrieval* (not full agentic planning): good for questions that bridge
two facets (e.g. "How does X relate to Y?" when X and Y surface in different chunks).
"""

from __future__ import annotations

import json
import logging
from typing import Any

import httpx

from app.bm25 import BM25Index
from app.chunk_store import StoredChunk
from app.config import Settings
from app.hybrid_search import RankedChunk
from app.mistral_client import mistral_chat, parse_json_object
from app.semantic_rank import embed_query_vector

logger = logging.getLogger(__name__)


def _extract_followup_query(data: dict[str, Any]) -> str:
    for key in ("followup_query", "follow_up_query", "followUpQuery", "second_query"):
        raw = data.get(key)
        if raw is not None and str(raw).strip():
            return str(raw).strip()
    return ""


def fallback_second_query(user_question: str, retrieval_semantic: str, retrieval_keywords: str) -> str | None:
    """
    When the LLM returns no follow-up, use a formulation distinct from the primary semantic query
    so the second hybrid pass can surface different chunks (keyword-heavy vs paraphrase).
    """
    sem = retrieval_semantic.strip()
    sem_l = sem.lower()
    for cand in (retrieval_keywords.strip(), user_question.strip()):
        if len(cand) >= 4 and cand.lower() != sem_l:
            return cand
    return None


def _previews_for_prompt(ranked: list[RankedChunk], max_chunks: int = 5, max_chars: int = 280) -> str:
    lines = []
    for i, r in enumerate(ranked[:max_chunks], start=1):
        t = r.stored.chunk.text.replace("\n", " ")[:max_chars]
        lines.append(f"{i}. ({r.stored.chunk.source_file} p.{r.stored.chunk.page_start}) {t}")
    return "\n".join(lines) if lines else "(no previews)"


async def propose_followup_query(
    client: httpx.AsyncClient,
    settings: Settings,
    *,
    user_question: str,
    retrieval_semantic: str,
    retrieval_keywords: str,
    first_hop_ranked: list[RankedChunk],
) -> str | None:
    """
    Returns a follow-up search string for hop-2, or None only if no usable variant exists.
    Multi-hop is always attempted: we prefer the LLM's query, then a keyword/user-question fallback.
    """
    previews = _previews_for_prompt(first_hop_ranked)
    user_msg = (
        f"Original user question:\n{user_question}\n\n"
        f"Primary retrieval query already used:\n{retrieval_semantic}\n\n"
        f"Top retrieved passage previews (may be incomplete):\n{previews}\n\n"
        'Return JSON only: {"needs_second_search": boolean, "followup_query": string}. '
        "Set needs_second_search true whenever additional passages may help (comparisons, multiple entities, "
        "or facts not clearly in previews). If true, followup_query must be a short English search string "
        "that is NOT a verbatim copy of the primary retrieval query—use different keywords or focus. "
        "If a second angle is unnecessary, set needs_second_search false and followup_query to empty string."
    )
    raw = await mistral_chat(
        client,
        settings,
        [
            {"role": "system", "content": "You output only valid JSON objects."},
            {"role": "user", "content": user_msg},
        ],
        temperature=0.1,
        max_tokens=200,
        response_format={"type": "json_object"},
    )
    try:
        data = parse_json_object(raw)
    except (json.JSONDecodeError, TypeError, ValueError) as e:
        logger.warning("multi_hop JSON parse failed: %s", e)
        return fallback_second_query(user_question, retrieval_semantic, retrieval_keywords)

    fq = _extract_followup_query(data)
    if fq and len(fq) >= 4 and fq.lower() != retrieval_semantic.strip().lower():
        return fq

    return fallback_second_query(user_question, retrieval_semantic, retrieval_keywords)


def merge_two_hop_candidates(
    chunks: list[StoredChunk],
    bm25: BM25Index,
    hop1: list[RankedChunk],
    hop2: list[RankedChunk],
    q_emb_primary: list[float],
    q_emb_follow: list[float],
    kw_primary: str,
    kw_follow: str,
    merge_top: int,
) -> list[RankedChunk]:
    """
    Union chunks from both hops; re-score with max(dense primary, dense follow) and max(BM25).
    RRF score kept as max of hop-specific RRF when present.
    """
    if not hop2:
        return hop1[:merge_top]

    qv_p = embed_query_vector(q_emb_primary)
    qv_f = embed_query_vector(q_emb_follow)
    sp_p = bm25.scores(kw_primary)
    sp_f = bm25.scores(kw_follow)

    rrf_by_id: dict[str, float] = {}
    for r in hop1:
        rrf_by_id[r.stored.chunk.id] = max(rrf_by_id.get(r.stored.chunk.id, 0.0), r.rrf_score)
    for r in hop2:
        rrf_by_id[r.stored.chunk.id] = max(rrf_by_id.get(r.stored.chunk.id, 0.0), r.rrf_score)

    id_to_idx = {chunks[i].chunk.id: i for i in range(len(chunks))}
    union_ids = {r.stored.chunk.id for r in hop1} | {r.stored.chunk.id for r in hop2}

    def _dot(qv, i: int) -> float:
        emb = chunks[i].embedding
        if emb is None:
            return 0.0
        return float((qv * emb).sum())

    rows: list[tuple[int, float, float, float]] = []
    for cid in union_ids:
        i = id_to_idx[cid]
        sm = max(_dot(qv_p, i), _dot(qv_f, i))
        bm = max(sp_p[i], sp_f[i])
        rrf = rrf_by_id.get(cid, 0.0)
        rows.append((i, sm, bm, rrf))

    rows.sort(key=lambda t: (t[1], t[2], t[3]), reverse=True)
    out: list[RankedChunk] = []
    for i, sm, bm, rrf in rows[:merge_top]:
        out.append(RankedChunk(stored=chunks[i], rrf_score=rrf, semantic_score=sm, bm25_score=bm))
    return out
