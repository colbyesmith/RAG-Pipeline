from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    mistral_api_key: str = ""
    mistral_base_url: str = "https://api.mistral.ai/v1"
    chat_model: str = "mistral-small-latest"
    embed_model: str = "mistral-embed"
    # Reduce 429s: retries on 429/503, smaller batches, pause between embedding batches (ingest).
    mistral_api_max_retries: int = Field(default=5, ge=0, le=20)
    mistral_api_retry_base_seconds: float = Field(default=1.0, ge=0.1, le=30.0)
    mistral_embed_batch_size: int = Field(default=8, ge=1, le=64)
    mistral_embed_batch_delay_ms: int = Field(default=250, ge=0, le=120_000)

    rag_top_k: int = 8
    rag_retrieve_k: int = 24
    rag_cross_encoder_enabled: bool = False
    cross_encoder_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    rag_multi_hop_enabled: bool = False
    rag_multi_hop_pool_k: int = Field(default=20, ge=8, le=100)
    rag_similarity_threshold: float = 0.32
    rrf_k: int = 60
    upload_max_mb: int = 25
    # Smaller values → more chunks, finer retrieval (roughly tokens ≈ chars / 4 for English).
    # ~80 chars ≈ ~20 tokens; ~40 ≈ ~10 tokens (very fine, more embed calls, noisier singles).
    chunk_size_chars: int = Field(default=400, ge=32, le=8000)
    chunk_overlap_chars: int = Field(default=80, ge=0, le=2000)
    evidence_overlap_min: float = 0.22


@lru_cache
def get_settings() -> Settings:
    return Settings()
