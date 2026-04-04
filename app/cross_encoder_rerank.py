"""
Second-stage retrieval: cross-encoder scores (query, passage) pairs jointly.

Uses sentence-transformers CrossEncoder (runs locally; first download pulls model weights).
Disabled by default via settings — enable when `sentence-transformers` is installed.
"""

from __future__ import annotations

import logging

from app.hybrid_search import RankedChunk

logger = logging.getLogger(__name__)

_ce_model = None
_ce_model_name: str | None = None

_MAX_PASSAGE_CHARS = 2000


def _get_cross_encoder(model_name: str):
    global _ce_model, _ce_model_name
    try:
        from sentence_transformers import CrossEncoder
    except ImportError as e:
        raise RuntimeError(
            "sentence-transformers is not installed. Run: pip install sentence-transformers"
        ) from e
    if _ce_model is None or _ce_model_name != model_name:
        logger.info("Loading cross-encoder model: %s", model_name)
        _ce_model = CrossEncoder(model_name)
        _ce_model_name = model_name
    return _ce_model


def rerank_with_cross_encoder(
    query: str,
    candidates: list[RankedChunk],
    top_k: int,
    model_name: str,
) -> list[RankedChunk]:
    """
    Re-order `candidates` by cross-encoder relevance; return top `top_k`.
    On failure, returns candidates[:top_k] unchanged (first-stage order).
    """
    if not candidates:
        return []
    if top_k <= 0:
        return []

    try:
        model = _get_cross_encoder(model_name)
    except RuntimeError as e:
        logger.warning("%s", e)
        return candidates[:top_k]

    pairs = [(query, r.stored.chunk.text[:_MAX_PASSAGE_CHARS]) for r in candidates]
    scores = model.predict(pairs, show_progress_bar=False, batch_size=16)

    scored = list(zip(candidates, scores, strict=True))
    scored.sort(key=lambda x: x[1], reverse=True)

    out: list[RankedChunk] = []
    for r, s in scored[:top_k]:
        out.append(
            RankedChunk(
                stored=r.stored,
                rrf_score=r.rrf_score,
                semantic_score=r.semantic_score,
                bm25_score=r.bm25_score,
                cross_encoder_score=float(s),
            )
        )
    return out
