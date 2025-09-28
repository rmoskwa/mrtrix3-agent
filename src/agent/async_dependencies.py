"""Async dependencies setup for the MRtrix3 agent."""

import os
from typing import Optional, Any
from pathlib import Path
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
import chromadb

logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()


class AsyncSearchKnowledgebaseDependencies(BaseModel):
    """Dependencies specifically for the async search_knowledgebase tool with dual database support."""

    model_config = {"arbitrary_types_allowed": True}

    supabase_client: AsyncClient = Field(..., description="Async Supabase client")
    chromadb_client: Optional[SkipValidation[Any]] = Field(
        default=None, description="ChromaDB client for local operations"
    )
    chromadb_collection: Optional[SkipValidation[Any]] = Field(
        default=None, description="ChromaDB collection for document storage"
    )
    chromadb_path: Optional[str] = Field(
        default=None, description="Path to ChromaDB storage directory"
    )
    embedding_model: str = Field(
        default_factory=lambda: os.getenv("EMBEDDING_MODEL"),
        description="Embedding model name",
    )
    embedding_api_key: Optional[str] = Field(
        default=None, description="Google API key for embeddings"
    )
    embedding_service: Optional[SkipValidation[Any]] = Field(
        default=None, description="Cached embedding service instance"
    )
    rate_limiter: Optional[SkipValidation[Any]] = Field(
        default=None, description="Agent-specific rate limiting"
    )
    config: Optional[SkipValidation[Any]] = Field(
        default=None, description="Agent configuration settings"
    )


def setup_chromadb_client(storage_path: str):
    """
    Set up ChromaDB persistent client.

    Args:
        storage_path: Path to ChromaDB storage directory.

    Returns:
        Initialized ChromaDB client.
    """
    # Ensure the storage directory exists
    path = Path(storage_path)
    path.mkdir(parents=True, exist_ok=True)

    # Create persistent client with telemetry disabled
    client = chromadb.PersistentClient(
        path=str(path), settings=chromadb.Settings(anonymized_telemetry=False)
    )
    return client


def initialize_chromadb_collection(client):
    """
    Initialize or get the ChromaDB collection for MRtrix3 documents.

    Args:
        client: ChromaDB client instance.

    Returns:
        ChromaDB collection for storing documents.
    """
    # Get or create the collection with metadata for tracking
    collection = client.get_or_create_collection(
        name="mrtrix3_documents",
        metadata={
            "description": "MRtrix3 documentation for semantic search",
            "embedding_dimensions": "768",
            "source": "Supabase sync",
        },
    )
    return collection


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry=retry_if_exception_type((ConnectionError, TimeoutError)),
    reraise=True,
)
async def create_async_dependencies() -> AsyncSearchKnowledgebaseDependencies:
    """
    Create async dependencies for the agent with retry logic and dual database support.

    Returns:
        AsyncSearchKnowledgebaseDependencies with initialized async Supabase client and ChromaDB

    Raises:
        ConnectionError: After 3 failed attempts to connect to Supabase
    """
    # First validate environment to set up API keys
    from .dependencies import validate_environment

    env_vars = validate_environment()

    from .config import SUPABASE_URL, SUPABASE_ANON_KEY

    url = SUPABASE_URL
    key = SUPABASE_ANON_KEY

    try:
        # Create async Supabase client with retry
        supabase = await acreate_client(url, key)
    except Exception as e:
        logger.error(f"Failed to create Supabase client: {e}")
        raise ConnectionError(f"Failed to connect to Supabase: {e}") from e

    from .config import EMBEDDING_MODEL, APP_NAME
    from platformdirs import user_data_dir

    embedding_model = EMBEDDING_MODEL

    # Set up ChromaDB for local operations
    default_chromadb_path = Path(user_data_dir(APP_NAME)) / "chromadb"
    chromadb_path = os.getenv("CHROMADB_PATH", str(default_chromadb_path))
    chromadb_path = os.path.expanduser(chromadb_path)

    try:
        chromadb_client = setup_chromadb_client(chromadb_path)
        chromadb_collection = initialize_chromadb_collection(chromadb_client)
    except Exception as e:
        logger.error(f"Failed to initialize ChromaDB: {e}")
        # ChromaDB is optional, so we don't raise an error
        chromadb_client = None
        chromadb_collection = None

    return AsyncSearchKnowledgebaseDependencies(
        supabase_client=supabase,
        chromadb_client=chromadb_client,
        chromadb_collection=chromadb_collection,
        chromadb_path=chromadb_path if chromadb_client else None,
        embedding_model=embedding_model,
        embedding_api_key=env_vars.get("GOOGLE_API_KEY_EMBEDDING"),
        embedding_service=None,
        rate_limiter=None,
        config=None,
    )
