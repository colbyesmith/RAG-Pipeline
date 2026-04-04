"""
Okapi BM25 — implemented directly from the standard formula (no search engine library).
"""

from __future__ import annotations

import math
from collections import defaultdict

from app.chunk_store import StoredChunk
from app.tokenize import tokenize


def _avg_doc_len(lengths: list[int]) -> float:
    if not lengths:
        return 1.0
    return sum(lengths) / len(lengths)


class BM25Index:
    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self._doc_tokens: list[list[str]] = []
        self._df: defaultdict[str, int] = defaultdict(int)
        self._N = 0

    def build(self, chunks: list[StoredChunk]) -> None:
        self._doc_tokens = [tokenize(sc.chunk.text) for sc in chunks]
        self._df = defaultdict(int)
        for toks in self._doc_tokens:
            seen = set(toks)
            for t in seen:
                self._df[t] += 1
        self._N = len(self._doc_tokens)

    def scores(self, query: str) -> list[float]:
        q = tokenize(query)
        if not q or self._N == 0:
            return [0.0] * self._N

        lengths = [len(d) for d in self._doc_tokens]
        avgdl = _avg_doc_len(lengths)
        scores = [0.0] * self._N

        for qi in q:
            df = self._df.get(qi, 0)
            idf = math.log(1 + (self._N - df + 0.5) / (df + 0.5))
            for i, doc in enumerate(self._doc_tokens):
                f = doc.count(qi)
                if f == 0:
                    continue
                denom = f + self.k1 * (1 - self.b + self.b * lengths[i] / avgdl)
                scores[i] += idf * (f * (self.k1 + 1)) / denom
        return scores
