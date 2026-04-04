"""Minimal tokenizer for keyword/BM25 paths (no dedicated search library)."""

from __future__ import annotations

import re

_TOKEN = re.compile(r"[a-z0-9]+", re.I)

# Small English stopword set to reduce noise in BM25 (hand-rolled, not from a library).
_STOP = frozenset(
    "a an the and or but if in on at to for of as is was are were be been being "
    "it this that these those with from by not no yes do does did has have had "
    "you your we our they their he she his her i me my what which who whom when where "
    "why how all each every both few more most other some such than too very can will "
    "just about into through during before after above below between again further then "
    "once here there any same so than too very s t don should now d ll re ve m".split()
)


def tokenize(text: str) -> list[str]:
    return [t.lower() for t in _TOKEN.findall(text) if t.lower() not in _STOP and len(t) > 1]
