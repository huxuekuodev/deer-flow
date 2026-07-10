"""Configuration for the semantic search cache."""

from __future__ import annotations

from pydantic import BaseModel, Field


class SearchCacheConfig(BaseModel):
    """Semantic-cache configuration for web-search tools.

    When enabled, search results are stored in Redis Stack with their
    embedding vectors.  Subsequent queries whose embedding cosine-similarity
    exceeds *similarity_threshold* are served from the cache instead of
    hitting the upstream search API.
    """

    enabled: bool = Field(
        default=False,
        description="Enable the semantic search cache powered by Redis Stack.",
    )
    redis_url: str = Field(
        default="redis://localhost:6379/0",
        description="Redis Stack connection URL (requires RediSearch module).",
    )
    ttl_seconds: int = Field(
        default=1800,
        ge=0,
        description="Cache entry time-to-live in seconds. 0 disables expiry (not recommended).",
    )
    similarity_threshold: float = Field(
        default=0.85,
        ge=0.0,
        le=1.0,
        description="Minimum cosine similarity for a cache hit. Higher values mean stricter matches.",
    )
    embedding_model: str = Field(
        default="text-embedding-3-small",
        description="OpenAI-compatible embedding model name for query vectorisation.",
    )
    embedding_base_url: str | None = Field(
        default=None,
        description="Optional base URL override for the embedding endpoint (openai-compatible gateway).",
    )
    embedding_dimensions: int = Field(
        default=1536,
        description="Dimensionality of the embedding vectors produced by embedding_model.",
    )
    embedding_api_key: str | None = Field(
        default=None,
        description="Optional API key for the embedding endpoint (openai-compatible gateway).",
    )
