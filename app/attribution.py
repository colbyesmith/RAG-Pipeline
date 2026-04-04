"""Roll up retrieval scores by source PDF for API / UI transparency."""

from __future__ import annotations

import statistics
from collections import defaultdict

from app.hybrid_search import RankedChunk
from app.schemas import DocumentScoreSummary


def summarize_scores_by_document(ranked: list[RankedChunk]) -> list[DocumentScoreSummary]:
    by_file: dict[str, list[RankedChunk]] = defaultdict(list)
    for r in ranked:
        by_file[r.stored.chunk.source_file].append(r)

    out: list[DocumentScoreSummary] = []
    for fname, rows in by_file.items():
        sims = [x.semantic_score for x in rows]
        bm = [x.bm25_score for x in rows]
        rr = [x.rrf_score for x in rows]
        ce_vals = [x.cross_encoder_score for x in rows if x.cross_encoder_score is not None]
        out.append(
            DocumentScoreSummary(
                source_file=fname,
                chunks_used=len(rows),
                semantic_similarity_max=max(sims),
                semantic_similarity_mean=float(statistics.mean(sims)),
                bm25_score_max=max(bm),
                rrf_score_max=max(rr),
                cross_encoder_score_max=max(ce_vals) if ce_vals else None,
            )
        )
    out.sort(key=lambda s: (-s.semantic_similarity_max, s.source_file))
    return out
