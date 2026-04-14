from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    mistral_api_key: str = ""
    mistral_base_url: str = "https://api.mistral.ai/v1"
    chat_model: str = "mistral-small-latest"
    embed_model: str = "mistral-embed"
    # Reduce 429s: retries on 429/503; optional pause between embedding batches if you hit limits.
    mistral_api_max_retries: int = Field(default=5, ge=0, le=20)
    mistral_api_retry_base_seconds: float = Field(default=1.0, ge=0.1, le=30.0)
    # Mistral caps total tokens per embeddings request (error 3210 if exceeded)—lower batch size or chunk length if you get 400s.
    mistral_embed_batch_size: int = Field(default=128, ge=1, le=512)
    mistral_embed_batch_delay_ms: int = Field(default=0, ge=0, le=120_000)
    # PDF: skip slow pypdf paths when PyMuPDF already extracted enough text; pypdf fallback uses plain only.
    pdf_extract_fast: bool = True
    pdf_fitz_min_chars_skip_pypdf: int = Field(default=48, ge=0, le=5000)

    rag_top_k: int = 8
    rag_retrieve_k: int = 24
    rag_cross_encoder_enabled: bool = True
    cross_encoder_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    # Multi-hop merge pool size (second hop is always attempted when the model returns a follow-up query).
    rag_multi_hop_pool_k: int = Field(default=20, ge=8, le=100)
    rag_similarity_threshold: float = 0.32
    # FAISS HNSW approximate dense retrieval (inner product = cosine on L2-normalized vectors).
    rag_ann_enabled: bool = True
    rag_ann_min_chunks: int = Field(default=384, ge=32, le=1_000_000)
    rag_ann_neighbors: int = Field(default=512, ge=32, le=8192)
    rag_ann_hnsw_m: int = Field(default=32, ge=8, le=64)
    rag_ann_ef_construction: int = Field(default=200, ge=40, le=800)
    rag_ann_ef_search: int = Field(default=128, ge=16, le=512)
    rrf_k: int = 60
    upload_max_mb: int = 25
    # Larger chunk_size_chars → fewer chunks, faster/cheaper ingest (fewer embed calls), coarser citations.
    # Roughly: tokens ≈ chars / 4 for English. Overlap ~12–15% of chunk size is a common default.
    chunk_size_chars: int = Field(default=800, ge=32, le=8000)
    chunk_overlap_chars: int = Field(default=120, ge=0, le=2000)
    # Sentence–context Jaccard must be >= this to skip a "low_support" flag. Lower = fewer flags (more lenient).
    evidence_overlap_min: float = Field(default=0.10, ge=0.0, le=1.0)


@lru_cache
def get_settings() -> Settings:
    return Settings()
