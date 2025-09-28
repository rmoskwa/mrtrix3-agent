"""Pydantic data models for the MRtrix3 agent module."""

import os
from typing import Optional, TYPE_CHECKING, Any, List, Union
from pydantic import BaseModel, Field, field_validator

# Use TYPE_CHECKING to avoid circular imports
if TYPE_CHECKING:
    from supabase import AsyncClient
    from chromadb.api import ClientAPI
    from chromadb import Collection
    from .embedding_service import EmbeddingService
else:
    AsyncClient = Any
    ClientAPI = Any
    Collection = Any
    EmbeddingService = Any


class DocumentResult(BaseModel):
    """Clean document output for agent consumption."""

    title: str = Field(..., description="Document title")
    content: str = Field(..., description="Document content")


class SearchToolParameters(BaseModel):
    """Parameters for the search_knowledgebase tool."""

    queries: Union[str, List[str]] = Field(
        ...,
        description="Single query string or list of natural language search queries",
    )

    @field_validator("queries")
    @classmethod
    def validate_queries(cls, v: Union[str, List[str]]) -> List[str]:
        """Validate and clean search queries, ensure it's always a list."""
        # Convert single string to list
        if isinstance(v, str):
            queries = [v]
        else:
            queries = v

        # Validate each query
        validated = []
        for query in queries:
            if not query or not query.strip():
                continue  # Skip empty queries
            cleaned = query.strip()[:500]  # Limit each query length
            validated.append(cleaned)

        if not validated:
            raise ValueError("At least one non-empty query is required")

        return validated


class BaseDependencies(BaseModel):
    """Base dependencies that tools can extend."""

    model_config = {"arbitrary_types_allowed": True}


class SearchKnowledgebaseDependencies(BaseDependencies):
    """Dependencies specifically for the search_knowledgebase tool with dual database support."""

    supabase_client: AsyncClient = Field(
        ..., description="Initialized async Supabase client for sync operations"
    )
    chromadb_client: Optional[ClientAPI] = Field(
        default=None, description="ChromaDB client for local operations"
    )
    chromadb_collection: Optional[Collection] = Field(
        default=None, description="ChromaDB collection for document storage"
    )
    chromadb_path: Optional[str] = Field(
        default=None, description="Path to ChromaDB storage directory"
    )
    embedding_model: str = Field(
        default_factory=lambda: os.getenv("EMBEDDING_MODEL"),
        description="Google generative AI embedding model name",
    )
    embedding_api_key: Optional[str] = Field(
        default=None, description="Google API key for embeddings"
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
        default=2, description="Top-k for similarity search", ge=1, le=10
    )
    return_top_n: int = Field(
        default=2, description="Documents to return to agent", ge=1, le=10
    )
    similarity_threshold: float = Field(
        default=0.7, description="Cosine similarity threshold", ge=0.0, le=1.0
    )
    system_prompt: str = Field(default="", description="Agent's base instructions")
