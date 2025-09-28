"""Dependency injection setup for the MRtrix3 agent with ChromaDB and Supabase support."""

import os
from pathlib import Path
from typing import Optional

import chromadb
from chromadb.api import ClientAPI
from dotenv import load_dotenv
from platformdirs import user_data_dir, user_config_dir
from supabase import Client, create_client

from .config import (
    SUPABASE_URL,
    SUPABASE_ANON_KEY,
    EMBEDDING_MODEL,
    EMBEDDING_DIMENSIONS,
    APP_NAME,
)
from .models import SearchKnowledgebaseDependencies


def validate_environment() -> dict:
    """
    Validate required environment variables and configuration.

    Loads configuration from (in order of priority, highest to lowest):
    1. Environment variables already set (highest priority)
    2. User config directory (~/.config/mrtrix3-agent/config or platform equivalent)
    3. .env file in current directory (for development)

    Returns:
        Dictionary containing:
        - SUPABASE_URL, SUPABASE_KEY: Hardcoded read-only credentials
        - GOOGLE_API_KEY: User-provided Gemini API key
        - GOOGLE_API_KEY_EMBEDDING: Same as GOOGLE_API_KEY unless overridden
        - EMBEDDING_MODEL, EMBEDDING_DIMENSIONS: Hardcoded values
        - CHROMADB_PATH: Local storage path for ChromaDB

    Raises:
        ValueError: If GOOGLE_API_KEY is not found in any configuration source.
    """
    # Get platform-specific config directory
    config_dir = Path(user_config_dir(APP_NAME))
    config_file = config_dir / "config"

    # Load from user config if it exists
    if config_file.exists():
        load_dotenv(config_file, override=False)

    # Load from .env for development (lower priority)
    # Try multiple locations for .env file
    possible_env_paths = [
        Path.cwd() / ".env",  # Current directory
        Path(__file__).parent.parent.parent / ".env",  # Project root
    ]

    for env_path in possible_env_paths:
        if env_path.exists():
            load_dotenv(env_path, override=False)
            break
    else:
        # Fallback to default load_dotenv behavior
        load_dotenv(override=False)

    env_vars = {}

    # Use hardcoded Supabase credentials (read-only access)
    env_vars["SUPABASE_URL"] = SUPABASE_URL
    env_vars["SUPABASE_KEY"] = SUPABASE_ANON_KEY

    # Only Google API keys need to be provided by the user
    required_user_vars = [
        "GOOGLE_API_KEY",
    ]

    for var in required_user_vars:
        value = os.getenv(var)
        if not value:
            error_msg = f"Required API key {var} is not set.\n\n"
            error_msg += "Please run 'setup-mrtrixbot' to configure your API keys."
            raise ValueError(error_msg)
        env_vars[var] = value

        # Set it in the environment for PydanticAI to use
        os.environ[var] = value

    # Use same key for embeddings if not separately provided
    env_vars["GOOGLE_API_KEY_EMBEDDING"] = os.getenv(
        "GOOGLE_API_KEY_EMBEDDING", env_vars["GOOGLE_API_KEY"]
    )

    # Use hardcoded embedding configuration
    env_vars["EMBEDDING_MODEL"] = EMBEDDING_MODEL
    env_vars["EMBEDDING_DIMENSIONS"] = EMBEDDING_DIMENSIONS

    # Add ChromaDB storage path configuration
    # Use platformdirs for data storage
    default_chromadb_path = Path(user_data_dir(APP_NAME)) / "chromadb"
    chromadb_path = os.getenv("CHROMADB_PATH", str(default_chromadb_path))
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

    # Create persistent client with telemetry disabled
    client = chromadb.PersistentClient(
        path=str(path), settings=chromadb.Settings(anonymized_telemetry=False)
    )

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
        embedding_api_key=env_vars["GOOGLE_API_KEY_EMBEDDING"],  # Pass API key
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
        client = chromadb.PersistentClient(
            path=str(storage_path),
            settings=chromadb.Settings(anonymized_telemetry=False),
        )
        collection = client.get_collection("mrtrix3_documents")
        # Check if collection has any documents
        count = collection.count()
        return count > 0
    except Exception:
        # Collection doesn't exist or other error
        return False
