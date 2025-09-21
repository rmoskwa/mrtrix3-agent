"""Pydantic data models for the MRtrix3 agent module."""

from typing import Any, Optional

from pydantic import BaseModel, Field


class DocumentResult(BaseModel):
    """Clean document output for agent consumption."""

    title: str = Field(..., description="Document title")
    content: str = Field(..., description="Document content")


class SearchToolParameters(BaseModel):
    """Parameters for the search_knowledgebase tool."""

    query: str = Field(..., description="Natural language search query")


class BaseDependencies(BaseModel):
    """Base dependencies that tools can extend."""

    model_config = {"arbitrary_types_allowed": True}


class SearchKnowledgebaseDependencies(BaseDependencies):
    """Dependencies specifically for the search_knowledgebase tool."""

    supabase_client: Any = Field(..., description="Initialized Supabase client")
    embedding_model: Any = Field(
        ..., description="Google generative AI embedding model"
    )
    rate_limiter: Optional[Any] = Field(
        default=None, description="Agent-specific rate limiting"
    )


class AgentConfiguration(BaseModel):
    """Configuration model for the MRtrix3 assistant agent."""

    model_name: str = Field(
        default="gemini-2.5-flash", description="Correct Gemini model identifier"
    )
    embedding_model: str = Field(
        default="gemini-embedding-001", description="Embedding model name"
    )
    embedding_dimensions: int = Field(default=768, description="Vector dimensions")
    max_search_results: int = Field(
        default=3, description="Top-k for similarity search"
    )
    return_top_n: int = Field(default=2, description="Documents to return to agent")
    system_prompt: str = Field(default="", description="Agent's base instructions")
