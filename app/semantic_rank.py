"""Dense retrieval via cosine similarity (embeddings from Mistral API)."""

from __future__ import annotations

import numpy as np

from app.chunk_store import StoredChunk
from app.config import Settings
from app.vector_ann import get_chunk_ann


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


def dense_scores_for_hybrid(
    query_vec: np.ndarray,
    chunks: list[StoredChunk],
    settings: Settings,
    *,
    retrieve_pool_k: int,
) -> list[float]:
    """
    Full-length dense scores for RRF. Uses FAISS HNSW when enabled and the corpus is large
    enough; otherwise exact cosine. Chunks not returned by ANN get score 0.0 (rank at tail).
    """
    n = len(chunks)
    if n == 0:
        return []
    if not settings.rag_ann_enabled or n < settings.rag_ann_min_chunks:
        return cosine_scores(query_vec, chunks)

    ann = get_chunk_ann()
    if ann.ntotal != n:
        return cosine_scores(query_vec, chunks)

    k = min(n, max(settings.rag_ann_neighbors, retrieve_pool_k * 2, 64))
    scores, indices = ann.search(query_vec, k)
    sem = [0.0] * n
    for j in range(indices.shape[1]):
        idx = int(indices[0, j])
        if idx < 0:
            continue
        if idx < n:
            sem[idx] = float(scores[0, j])
    return sem
