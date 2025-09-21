"""Dependency injection setup for the MRtrix3 agent."""

import os

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

    return env_vars


def setup_search_knowledgebase_dependencies() -> SearchKnowledgebaseDependencies:
    """Set up dependencies for the search_knowledgebase tool.

    Returns:
        Configured SearchKnowledgebaseDependencies instance.
    """
    env_vars = validate_environment()

    supabase_client: Client = create_client(
        env_vars["SUPABASE_URL"], env_vars["SUPABASE_KEY"]
    )

    import google.generativeai as genai

    genai.configure(api_key=env_vars["GOOGLE_API_KEY_EMBEDDING"])
    embedding_model = genai.GenerativeModel("gemini-embedding-001")

    # Placeholder rate limiter
    class AgentRateLimiter:
        """Placeholder rate limiter for agent-specific rate limiting."""

        pass

    rate_limiter = AgentRateLimiter()

    return SearchKnowledgebaseDependencies(
        supabase_client=supabase_client,
        embedding_model=embedding_model,
        rate_limiter=rate_limiter,
    )


def setup_dependencies() -> SearchKnowledgebaseDependencies:
    """
    Set up and return SearchKnowledgebaseDependencies instance.

    Returns:
        Configured SearchKnowledgebaseDependencies instance.
    """
    return setup_search_knowledgebase_dependencies()
