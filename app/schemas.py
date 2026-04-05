from typing import Any

from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=8000)


class IngestTimings(BaseModel):
    """Wall time per ingest phase (helps spot PDF CPU vs Mistral API)."""

    pdf_extract_and_chunk_s: float
    embedding_s: float


class IngestResponse(BaseModel):
    ingested_files: list[str]
    chunks_added: int
    message: str
    timings: IngestTimings | None = None


class Citation(BaseModel):
    chunk_id: str
    source_file: str
    page_start: int
    page_end: int
    similarity: float | None = None
    bm25_score: float | None = None
    rrf_score: float | None = None
    cross_encoder_score: float | None = None


class DocumentScoreSummary(BaseModel):
    """Aggregated retrieval scores for one PDF among chunks sent to the model."""

    source_file: str
    chunks_used: int
    semantic_similarity_max: float
    semantic_similarity_mean: float
    bm25_score_max: float
    rrf_score_max: float
    cross_encoder_score_max: float | None = None


class QueryResponse(BaseModel):
    answer: str
    needs_retrieval: bool
    intent: str
    retrieval_skipped_reason: str | None = None
    insufficient_evidence: bool = False
    document_scores: list[DocumentScoreSummary] = Field(default_factory=list)
    citations: list[Citation] = Field(default_factory=list)
    policy_flags: list[str] = Field(default_factory=list)
    hallucination_flags: list[str] = Field(default_factory=list)
    debug: dict[str, Any] | None = None
