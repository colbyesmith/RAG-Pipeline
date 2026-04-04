"""
In-memory document store with optional embedding vectors.
No external vector database: vectors live alongside chunk records in RAM.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field

import numpy as np

from app.pdf_ingest import TextChunk


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

    def add_chunks(self, chunks: list[TextChunk], embeddings: list[list[float]] | None) -> int:
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
            return len(chunks)

    def all_with_embeddings(self) -> list[StoredChunk]:
        with self._lock:
            return list(self._items)

    def count(self) -> int:
        with self._lock:
            return len(self._items)


store = ChunkStore()
