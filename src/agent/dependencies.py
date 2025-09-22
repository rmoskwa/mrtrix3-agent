"""Dependency injection setup for the MRtrix3 agent with ChromaDB and Supabase support."""

import os
from pathlib import Path
from typing import Optional

import chromadb
from chromadb.api import ClientAPI
from dotenv import load_dotenv
from supabase import Client, create_client

from .models import SearchKnowledgebaseDependencies


def validate_environment() -> dict:
    """
    Validate required environment variables.

    Returns:
        Dictionary of validated environment variables.

    Raises:
        ValueError: If any required environment variable is missing or invalid.
    """
    load_dotenv()

    required_vars = [
        "SUPABASE_URL",
        "SUPABASE_KEY",
        "GEMINI_API_KEY",
        "GOOGLE_API_KEY_EMBEDDING",
    ]

    env_vars = {}

    for var in required_vars:
        value = os.getenv(var)
        if not value:
            raise ValueError(f"Required environment variable {var} is not set")
        env_vars[var] = value

    embedding_model = os.getenv("EMBEDDING_MODEL")
    if embedding_model != "gemini-embedding-001":
        raise ValueError(
            f"EMBEDDING_MODEL must be 'gemini-embedding-001', got '{embedding_model}'"
        )
    env_vars["EMBEDDING_MODEL"] = embedding_model

    embedding_dimensions = os.getenv("EMBEDDING_DIMENSIONS")
    if embedding_dimensions != "768":
        raise ValueError(
            f"EMBEDDING_DIMENSIONS must be '768', got '{embedding_dimensions}'"
        )
    env_vars["EMBEDDING_DIMENSIONS"] = int(embedding_dimensions)

    # Add ChromaDB storage path configuration
    chromadb_path = os.getenv("CHROMADB_PATH", "~/.mrtrix3-agent/chromadb")
    env_vars["CHROMADB_PATH"] = os.path.expanduser(chromadb_path)

    return env_vars


def setup_chromadb_client(storage_path: str) -> ClientAPI:
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

    # Create persistent client
    client = chromadb.PersistentClient(path=str(path))

    return client


def initialize_chromadb_collection(client: ClientAPI) -> chromadb.Collection:
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


def setup_search_knowledgebase_dependencies() -> SearchKnowledgebaseDependencies:
    """Set up dependencies for the search_knowledgebase tool with dual database support.

    Returns:
        Configured SearchKnowledgebaseDependencies instance.
    """
    env_vars = validate_environment()

    # Set up Supabase client for sync operations
    supabase_client: Client = create_client(
        env_vars["SUPABASE_URL"], env_vars["SUPABASE_KEY"]
    )

    # Set up ChromaDB client for local operations
    chromadb_client = setup_chromadb_client(env_vars["CHROMADB_PATH"])
    chromadb_collection = initialize_chromadb_collection(chromadb_client)

    # Set up embedding model configuration
    import google.generativeai as genai

    genai.configure(api_key=env_vars["GOOGLE_API_KEY_EMBEDDING"])

    # Initialize rate limiter placeholder
    class AgentRateLimiter:
        """Placeholder rate limiter for agent-specific rate limiting."""

        pass

    rate_limiter = AgentRateLimiter()

    # Create dependencies with both database clients
    deps = SearchKnowledgebaseDependencies(
        supabase_client=supabase_client,
        embedding_model=env_vars["EMBEDDING_MODEL"],  # Pass as string
        rate_limiter=rate_limiter,
    )

    # Add ChromaDB references as additional attributes
    # Note: update the model to include these
    deps.chromadb_client = chromadb_client
    deps.chromadb_collection = chromadb_collection
    deps.chromadb_path = env_vars["CHROMADB_PATH"]

    return deps


def setup_dependencies() -> SearchKnowledgebaseDependencies:
    """
    Set up and return SearchKnowledgebaseDependencies instance.

    Returns:
        Configured SearchKnowledgebaseDependencies instance with dual database support.
    """
    return setup_search_knowledgebase_dependencies()


def check_chromadb_initialized(storage_path: Optional[str] = None) -> bool:
    """
    Check if ChromaDB has been initialized with data.

    Args:
        storage_path: Optional path to ChromaDB storage. If not provided, uses environment default.

    Returns:
        True if ChromaDB has data, False otherwise.
    """
    if storage_path is None:
        env_vars = validate_environment()
        storage_path = env_vars["CHROMADB_PATH"]

    try:
        client = chromadb.PersistentClient(path=str(storage_path))
        collection = client.get_collection("mrtrix3_documents")
        # Check if collection has any documents
        count = collection.count()
        return count > 0
    except Exception:
        # Collection doesn't exist or other error
        return False
