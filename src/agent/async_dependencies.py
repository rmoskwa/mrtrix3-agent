"""Async dependencies setup for the MRtrix3 agent."""

import os
from typing import Optional
from supabase import acreate_client, AsyncClient
from pydantic import BaseModel, Field
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class AsyncSearchKnowledgebaseDependencies(BaseModel):
    """Dependencies specifically for the async search_knowledgebase tool."""

    model_config = {"arbitrary_types_allowed": True}

    supabase_client: AsyncClient = Field(..., description="Async Supabase client")
    embedding_model: str = Field(
        default_factory=lambda: os.getenv("EMBEDDING_MODEL"),
        description="Embedding model name",
    )
    rate_limiter: Optional[any] = Field(
        default=None, description="Agent-specific rate limiting"
    )


async def create_async_dependencies() -> AsyncSearchKnowledgebaseDependencies:
    """
    Create async dependencies for the agent.

    Returns:
        AsyncSearchKnowledgebaseDependencies with initialized async Supabase client
    """
    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_KEY")

    if not url:
        raise ValueError("SUPABASE_URL not found in environment")
    if not key:
        raise ValueError("SUPABASE_KEY not found in environment")

    # Create async Supabase client
    supabase = await acreate_client(url, key)

    embedding_model = os.getenv("EMBEDDING_MODEL")
    if not embedding_model:
        raise ValueError("EMBEDDING_MODEL not found in environment")

    return AsyncSearchKnowledgebaseDependencies(
        supabase_client=supabase,
        embedding_model=embedding_model,
        rate_limiter=None,
    )
