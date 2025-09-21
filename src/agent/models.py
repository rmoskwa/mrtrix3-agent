"""Pydantic data models for the MRtrix3 agent module."""

import os
from typing import Optional, TYPE_CHECKING, Any
from pydantic import BaseModel, Field, field_validator

# Use TYPE_CHECKING to avoid circular imports
if TYPE_CHECKING:
    from supabase import AsyncClient
    from .embedding_service import EmbeddingService
else:
    AsyncClient = Any
    EmbeddingService = Any


class DocumentResult(BaseModel):
    """Clean document output for agent consumption."""

    title: str = Field(..., description="Document title")
    content: str = Field(..., description="Document content")


class SearchToolParameters(BaseModel):
    """Parameters for the search_knowledgebase tool."""

    query: str = Field(
        ..., description="Natural language search query", min_length=1, max_length=500
    )

    @field_validator("query")
    @classmethod
    def validate_query(cls, v: str) -> str:
        """Validate and clean search query."""
        if not v or not v.strip():
            raise ValueError("Query cannot be empty")
        return v.strip()


class BaseDependencies(BaseModel):
    """Base dependencies that tools can extend."""

    model_config = {"arbitrary_types_allowed": True}


class SearchKnowledgebaseDependencies(BaseDependencies):
    """Dependencies specifically for the search_knowledgebase tool."""

    supabase_client: AsyncClient = Field(
        ..., description="Initialized async Supabase client"
    )
    embedding_model: str = Field(
        default_factory=lambda: os.getenv("EMBEDDING_MODEL"),
        description="Google generative AI embedding model name",
    )
    embedding_service: Optional[EmbeddingService] = Field(
        default=None, description="Cached embedding service instance"
    )
    rate_limiter: Optional[object] = Field(
        default=None, description="Agent-specific rate limiting"
    )
    config: Optional["AgentConfiguration"] = Field(
        default=None, description="Agent configuration settings"
    )


class AgentConfiguration(BaseModel):
    """Configuration model for the MRtrix3 assistant agent."""

    model_name: str = Field(
        default="gemini-2.5-flash", description="Correct Gemini model identifier"
    )
    embedding_model: str = Field(
        default="gemini-embedding-001", description="Embedding model name"
    )
    embedding_dimensions: int = Field(
        default=768, description="Vector dimensions", ge=1, le=2048
    )
    max_search_results: int = Field(
        default=3, description="Top-k for similarity search", ge=1, le=10
    )
    return_top_n: int = Field(
        default=2, description="Documents to return to agent", ge=1, le=10
    )
    similarity_threshold: float = Field(
        default=0.7, description="Cosine similarity threshold", ge=0.0, le=1.0
    )
    system_prompt: str = Field(default="", description="Agent's base instructions")
