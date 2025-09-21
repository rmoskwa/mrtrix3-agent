"""Async dependencies setup for the MRtrix3 agent."""

import os
from typing import Optional, Any
from supabase import acreate_client, AsyncClient
from pydantic import BaseModel, Field, SkipValidation
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)
from dotenv import load_dotenv
import logging

logger = logging.getLogger(__name__)

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
    rate_limiter: Optional[SkipValidation[Any]] = Field(
        default=None, description="Agent-specific rate limiting"
    )


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry=retry_if_exception_type((ConnectionError, TimeoutError)),
    reraise=True,
)
async def create_async_dependencies() -> AsyncSearchKnowledgebaseDependencies:
    """
    Create async dependencies for the agent with retry logic.

    Returns:
        AsyncSearchKnowledgebaseDependencies with initialized async Supabase client

    Raises:
        ConnectionError: After 3 failed attempts to connect to Supabase
    """
    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_KEY")

    if not url:
        raise ValueError("SUPABASE_URL not found in environment")
    if not key:
        raise ValueError("SUPABASE_KEY not found in environment")

    try:
        # Create async Supabase client with retry
        logger.info("Creating async Supabase client...")
        supabase = await acreate_client(url, key)
        logger.info("Supabase client created successfully")
    except Exception as e:
        logger.error(f"Failed to create Supabase client: {e}")
        raise ConnectionError(f"Failed to connect to Supabase: {e}") from e

    embedding_model = os.getenv("EMBEDDING_MODEL")
    if not embedding_model:
        raise ValueError("EMBEDDING_MODEL not found in environment")

    return AsyncSearchKnowledgebaseDependencies(
        supabase_client=supabase,
        embedding_model=embedding_model,
        rate_limiter=None,
    )
