"""Post-generation checks: overlap between answer sentences and retrieved context."""

from __future__ import annotations

import re

from app.tokenize import tokenize


_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")


def jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0


def sentence_token_sets(text: str) -> list[tuple[str, set[str]]]:
    parts = _SENT_SPLIT.split(text.strip())
    out: list[tuple[str, set[str]]] = []
    for p in parts:
        p = p.strip()
        if len(p) < 8:
            continue
        toks = set(tokenize(p))
        if len(toks) < 3:
            continue
        out.append((p, toks))
    return out


def evidence_check(answer: str, context: str, min_overlap: float) -> list[str]:
    ctx = set(tokenize(context))
    flags: list[str] = []
    for sent, st in sentence_token_sets(answer):
        score = jaccard(st, ctx)
        if score < min_overlap:
            flags.append(f"low_support: {sent[:120]}{'...' if len(sent) > 120 else ''}")
    return flags


def merge_context(chunks_text: list[str]) -> str:
    return "\n\n".join(chunks_text)
