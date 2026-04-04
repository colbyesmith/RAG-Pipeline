"""Dense retrieval via cosine similarity (embeddings from Mistral API)."""

from __future__ import annotations

import numpy as np

from app.chunk_store import StoredChunk


def embed_query_vector(vec: list[float]) -> np.ndarray:
    v = np.asarray(vec, dtype=np.float64)
    n = np.linalg.norm(v)
    if n > 0:
        v = v / n
    return v


def cosine_scores(query_vec: np.ndarray, chunks: list[StoredChunk]) -> list[float]:
    scores: list[float] = []
    for sc in chunks:
        if sc.embedding is None:
            scores.append(0.0)
            continue
        scores.append(float(np.dot(query_vec, sc.embedding)))
    return scores
