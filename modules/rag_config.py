"""
rag_config.py — typed configuration for knowledge_rag.

Reads from environment variables (and optionally a .env file via
pydantic-settings), validates types/ranges at startup instead of
failing deep inside a request.
"""

from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field, field_validator


class RagConfig(BaseSettings):
    # SQLite
    memory_db: str = Field(default="/home/kidus/eliteomni_memory.db", alias="MEMORY_DB")

    # Mistral
    mistral_api_key: str = Field(default="", alias="MISTRAL_API_KEY")
    embed_model: str = Field(default="mistral-embed", alias="RAG_EMBED_MODEL")
    embed_batch_size: int = Field(default=32, alias="RAG_EMBED_BATCH_SIZE")
    embed_max_retries: int = Field(default=3, alias="RAG_EMBED_MAX_RETRIES")
    embed_dim_fallback: int = Field(default=1024, alias="RAG_EMBED_DIM_FALLBACK")

    # Qdrant
    qdrant_url: str = Field(default="http://localhost:6333", alias="QDRANT_URL")
    qdrant_api_key: str = Field(default="", alias="QDRANT_API_KEY")
    qdrant_collection: str = Field(default="eliteomni_chunks", alias="RAG_COLLECTION")
    qdrant_timeout_s: float = Field(default=10.0, alias="QDRANT_TIMEOUT_S")

    # Retrieval
    default_top_k: int = Field(default=5, alias="RAG_DEFAULT_TOP_K")
    default_fetch_k: int = Field(default=20, alias="RAG_DEFAULT_FETCH_K")
    vector_weight: float = Field(default=0.7, alias="RAG_VECTOR_WEIGHT")
    bm25_weight: float = Field(default=0.3, alias="RAG_BM25_WEIGHT")

    # Concurrency
    max_concurrent_embed_requests: int = Field(default=5, alias="RAG_MAX_CONCURRENT_EMBED")

    @field_validator("default_top_k", "default_fetch_k", "embed_batch_size", "embed_max_retries")
    @classmethod
    def _positive(cls, v):
        if v <= 0:
            raise ValueError("must be > 0")
        return v

    @field_validator("vector_weight", "bm25_weight")
    @classmethod
    def _weight_range(cls, v):
        if not (0.0 <= v <= 1.0):
            raise ValueError("weight must be between 0 and 1")
        return v

    model_config = SettingsConfigDict(env_file=".env", populate_by_name=True)


config = RagConfig()
