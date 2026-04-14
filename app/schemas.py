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


class QueryResponse(BaseModel):
    answer: str
    needs_retrieval: bool
    intent: str
    retrieval_skipped_reason: str | None = None
    insufficient_evidence: bool = False
    policy_flags: list[str] = Field(default_factory=list)
    hallucination_flags: list[str] = Field(default_factory=list)
    debug: dict[str, Any] | None = None
