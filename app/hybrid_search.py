"""
Hybrid retrieval: combine dense (semantic) and sparse (BM25) rankings.

We use Reciprocal Rank Fusion (RRF) — a standard, library-free merge that is robust when
one modality dominates. A light re-rank boosts chunks that appear strongly in both lists.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from app.bm25 import BM25Index
from app.chunk_store import StoredChunk
from app.semantic_rank import cosine_scores, embed_query_vector


@dataclass
class RankedChunk:
    stored: StoredChunk
    rrf_score: float
    semantic_score: float
    bm25_score: float
    cross_encoder_score: float | None = None


def _rank_order(scores: list[float]) -> list[int]:
    return sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)


def reciprocal_rank_fusion(
    sem_order: list[int],
    bm25_order: list[int],
    rrf_k: int,
    n: int,
) -> np.ndarray:
    rrf = np.zeros(n, dtype=np.float64)
    for r, idx in enumerate(sem_order):
        rrf[idx] += 1.0 / (rrf_k + r + 1)
    for r, idx in enumerate(bm25_order):
        rrf[idx] += 1.0 / (rrf_k + r + 1)
    return rrf


def _minmax(x: list[float]) -> list[float]:
    if not x:
        return []
    lo, hi = min(x), max(x)
    if hi - lo < 1e-9:
        return [1.0 for _ in x]
    return [(v - lo) / (hi - lo) for v in x]


def hybrid_search(
    chunks: list[StoredChunk],
    bm25: BM25Index,
    query_semantic: str,
    query_keywords: str,
    query_embedding: list[float],
    top_k: int,
    rrf_k: int,
) -> list[RankedChunk]:
    if not chunks:
        return []

    qv = embed_query_vector(query_embedding)
    sem = cosine_scores(qv, chunks)
    sparse = bm25.scores(query_keywords)

    sem_order = _rank_order(sem)
    bm_order = _rank_order(sparse)
    rrf = reciprocal_rank_fusion(sem_order, bm_order, rrf_k, len(chunks))

    sem_n = _minmax(sem)
    sp_n = _minmax(sparse)
    combined = []
    for i in range(len(chunks)):
        # Re-rank: RRF + small bonus when both signals are above median
        bonus = 0.0
        if sem_n[i] > 0.5 and sp_n[i] > 0.5:
            bonus = 0.05
        combined.append((i, float(rrf[i]) + bonus, sem[i], sparse[i]))

    combined.sort(key=lambda t: t[1], reverse=True)
    out: list[RankedChunk] = []
    for i, rrf_s, s_s, b_s in combined[:top_k]:
        out.append(RankedChunk(stored=chunks[i], rrf_score=rrf_s, semantic_score=s_s, bm25_score=b_s))
    return out
