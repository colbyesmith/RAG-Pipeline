"""
In-memory document store with optional embedding vectors.
No external vector database: vectors live alongside chunk records in RAM.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass

import numpy as np

from app.pdf_ingest import TextChunk
from app.vector_ann import get_chunk_ann


@dataclass
class StoredChunk:
    chunk: TextChunk
    embedding: np.ndarray | None = None


class ChunkStore:
    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._items: list[StoredChunk] = []

    def clear(self) -> None:
        with self._lock:
            self._items.clear()
        get_chunk_ann().clear()

    def add_chunks(self, chunks: list[TextChunk], embeddings: list[list[float]] | None) -> int:
        raw_vecs: list[np.ndarray | None] = []
        with self._lock:
            for i, ch in enumerate(chunks):
                emb = None
                if embeddings is not None and i < len(embeddings):
                    v = np.asarray(embeddings[i], dtype=np.float64)
                    n = np.linalg.norm(v)
                    if n > 0:
                        v = v / n
                    emb = v
                self._items.append(StoredChunk(chunk=ch, embedding=emb))
                raw_vecs.append(emb.astype(np.float32) if emb is not None else None)

        dim = next((int(v.shape[0]) for v in raw_vecs if v is not None), None)
        if dim is not None:
            batch = np.stack(
                [v if v is not None else np.zeros(dim, dtype=np.float32) for v in raw_vecs],
                axis=0,
            )
            get_chunk_ann().extend(batch)
        return len(chunks)

    def all_with_embeddings(self) -> list[StoredChunk]:
        with self._lock:
            return list(self._items)

    def count(self) -> int:
        with self._lock:
            return len(self._items)


store = ChunkStore()
