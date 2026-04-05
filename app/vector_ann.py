"""
Approximate nearest neighbors for dense retrieval (FAISS HNSW + inner product).

Embeddings are L2-normalized in ChunkStore; inner product equals cosine similarity.
"""

from __future__ import annotations

import threading

import faiss
import numpy as np

from app.config import Settings

_singleton_lock = threading.Lock()
_chunk_ann: ChunkVectorANN | None = None


class ChunkVectorANN:
    """Row i in the index matches chunks[i] in the store."""

    def __init__(
        self,
        hnsw_m: int = 32,
        ef_construction: int = 40,
        ef_search: int = 128,
    ) -> None:
        self._hnsw_m = hnsw_m
        self._ef_construction = ef_construction
        self._ef_search = ef_search
        self._dim: int | None = None
        self._index: faiss.IndexHNSWFlat | None = None
        self._lock = threading.RLock()

    def clear(self) -> None:
        with self._lock:
            self._dim = None
            self._index = None

    def extend(self, vectors: np.ndarray) -> None:
        vs = np.asarray(vectors, dtype=np.float32)
        if vs.ndim == 1:
            vs = vs.reshape(1, -1)
        if vs.size == 0:
            return
        _, d = vs.shape
        faiss.normalize_L2(vs)
        with self._lock:
            if self._index is None:
                self._dim = d
                self._index = faiss.IndexHNSWFlat(d, self._hnsw_m, faiss.METRIC_INNER_PRODUCT)
                self._index.hnsw.efConstruction = self._ef_construction
            elif d != self._dim:
                raise ValueError(f"embedding dim {d} does not match index dim {self._dim}")
            self._index.add(vs)

    @property
    def ntotal(self) -> int:
        with self._lock:
            if self._index is None:
                return 0
            return int(self._index.ntotal)

    def search(self, query_vec: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        """Returns (scores, indices) with shapes (1, k'); IP similarity, higher is better."""
        q = np.asarray(query_vec, dtype=np.float32).reshape(1, -1)
        if q.size == 0:
            return np.zeros((1, 0), dtype=np.float32), np.zeros((1, 0), dtype=np.int64)
        faiss.normalize_L2(q)
        with self._lock:
            if self._index is None or self._index.ntotal == 0:
                return np.zeros((1, 0), dtype=np.float32), np.zeros((1, 0), dtype=np.int64)
            if self._dim is None or q.shape[1] != self._dim:
                return np.zeros((1, 0), dtype=np.float32), np.zeros((1, 0), dtype=np.int64)
            kk = min(int(k), int(self._index.ntotal))
            self._index.hnsw.efSearch = self._ef_search
            return self._index.search(q, kk)


def get_chunk_ann() -> ChunkVectorANN:
    global _chunk_ann
    with _singleton_lock:
        if _chunk_ann is None:
            _chunk_ann = ChunkVectorANN()
        return _chunk_ann


def reset_chunk_ann(
    hnsw_m: int = 32,
    ef_construction: int = 40,
    ef_search: int = 128,
) -> ChunkVectorANN:
    global _chunk_ann
    with _singleton_lock:
        _chunk_ann = ChunkVectorANN(
            hnsw_m=hnsw_m,
            ef_construction=ef_construction,
            ef_search=ef_search,
        )
        return _chunk_ann


def configure_vector_ann_from_settings(settings: Settings) -> None:
    reset_chunk_ann(
        hnsw_m=settings.rag_ann_hnsw_m,
        ef_construction=settings.rag_ann_ef_construction,
        ef_search=settings.rag_ann_ef_search,
    )
