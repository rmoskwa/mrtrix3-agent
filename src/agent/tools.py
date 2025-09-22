"""
MRtrix3 documentation search tool for PydanticAI agent using ChromaDB.

This module implements the search_knowledgebase tool that allows the AI agent
to autonomously retrieve MRtrix3 documentation context from the local ChromaDB
database when needed to answer questions.
"""

import logging
import re
import json
from typing import List, Optional, Dict, Any
from pathlib import Path
from pydantic_ai import RunContext, ModelRetry
from .models import DocumentResult, SearchKnowledgebaseDependencies
from .embedding_service import EmbeddingService

logger = logging.getLogger("agent.tools")


async def search_knowledgebase(
    ctx: RunContext[SearchKnowledgebaseDependencies], query: str
) -> List[DocumentResult]:
    """
    Tool to search MRtrix3 documentation in local ChromaDB.

    Call this when you needs additional context to answer questions
    about MRtrix3 commands, tutorials, guides, or references.

    Args:
        ctx: PydanticAI context with dependencies including ChromaDB client,
             embedding service, and rate limiter
        query: Natural language search query from the agent

    Returns:
        List of matching documents formatted with XML blocks
    """
    # Input validation and sanitization
    if not query or not query.strip():
        logger.warning("Empty query provided")
        return []

    # Sanitize query - limit length and remove special characters that could be malicious
    query = query.strip()[:500]  # Limit query length
    sanitized_query = re.sub(
        r"[^\w\s\-.,!?]", " ", query
    )  # Keep alphanumeric and basic punctuation

    logger.info(f"Agent searching for: {sanitized_query}")

    # Get sync status for context
    sync_status = _get_sync_status(ctx)
    if sync_status:
        logger.debug(
            f"Database last synced: {sync_status.get('last_sync_time', 'unknown')}"
        )

    # Generate embedding for the agent's search query
    try:
        # Use embedding service from dependencies if available, otherwise create new
        if hasattr(ctx.deps, "embedding_service") and ctx.deps.embedding_service:
            embedding_service = ctx.deps.embedding_service
        else:
            embedding_service = EmbeddingService(ctx.deps.embedding_model)
        embedding = await embedding_service.generate_embedding(sanitized_query)
    except (TimeoutError, ConnectionError) as e:
        logger.warning(f"Embedding generation failed, will retry: {e}")
        raise ModelRetry(f"Embedding generation temporarily failed: {e}") from e
    except Exception as e:
        logger.error(f"Permanent embedding error: {e}")
        return []

    # Perform vector similarity search in ChromaDB
    try:
        results = await _search_chromadb_vector(ctx, embedding, sanitized_query)
        return results if results else []
    except Exception as e:
        logger.error(f"Vector search failed: {e}")
        return []


async def _search_chromadb_vector(
    ctx: RunContext[SearchKnowledgebaseDependencies], embedding: List[float], query: str
) -> List[DocumentResult]:
    """
    Perform vector similarity search in ChromaDB.

    Args:
        ctx: PydanticAI context with dependencies
        embedding: Query embedding vector
        query: Original query text for logging

    Returns:
        List of matching documents
    """
    try:
        # Check if ChromaDB client and collection are available
        if (
            not hasattr(ctx.deps, "chromadb_collection")
            or not ctx.deps.chromadb_collection
        ):
            logger.error("ChromaDB collection not available in dependencies")
            return []

        collection = ctx.deps.chromadb_collection

        # Get configuration settings
        max_results = 3
        if hasattr(ctx.deps, "config") and ctx.deps.config:
            max_results = getattr(ctx.deps.config, "max_search_results", 3)

        # Query ChromaDB with embedding
        # ChromaDB returns results sorted by distance (smaller = more similar)
        results = collection.query(
            query_embeddings=[embedding],
            n_results=max_results,
            include=["documents", "metadatas", "distances"],
        )

        if not results or not results.get("documents"):
            logger.info(f"No vector search results for query: {query}")
            return []

        # Extract and format results
        documents = results["documents"][0]  # First query's results
        metadatas = (
            results["metadatas"][0] if "metadatas" in results else [{}] * len(documents)
        )
        distances = (
            results["distances"][0] if "distances" in results else [0] * len(documents)
        )

        # Filter by similarity threshold (ChromaDB uses distance, not similarity)
        # Convert distance to similarity score (1 - distance for cosine)
        threshold = 0.3  # Distance threshold (lower is better)
        if hasattr(ctx.deps, "config") and ctx.deps.config:
            similarity_threshold = getattr(ctx.deps.config, "similarity_threshold", 0.7)
            threshold = 1 - similarity_threshold  # Convert similarity to distance

        formatted_results = []
        for doc, metadata, distance in zip(documents, metadatas, distances):
            if distance <= threshold:
                title = metadata.get("title", "Untitled") if metadata else "Untitled"
                formatted_results.append(
                    DocumentResult(
                        title=title, content=_format_document_xml(title, doc)
                    )
                )

        # Return top 2 results
        return formatted_results[:2]

    except Exception as e:
        logger.error(f"ChromaDB vector search error: {e}")
        raise


def _format_document_xml(title: str, content: str) -> str:
    """
    Format document content as XML block.

    Args:
        title: Document title
        content: Document content

    Returns:
        Formatted content with XML tags
    """
    # Ensure we have valid strings
    title = title or "Untitled"
    content = content or ""

    # Format as XML block
    return f"<Start of {title} document>{content}</Start of {title} document>"


def _get_sync_status(
    ctx: RunContext[SearchKnowledgebaseDependencies],
) -> Optional[Dict[str, Any]]:
    """
    Get sync status information from metadata.

    Args:
        ctx: PydanticAI context with dependencies

    Returns:
        Dictionary with sync status information or None
    """
    try:
        # Get ChromaDB path from dependencies
        if hasattr(ctx.deps, "chromadb_path") and ctx.deps.chromadb_path:
            metadata_file = Path(ctx.deps.chromadb_path) / "sync_metadata.json"
            if metadata_file.exists():
                with open(metadata_file, "r") as f:
                    metadata = json.load(f)
                    return {
                        "last_sync_time": metadata.get("last_sync_time"),
                        "document_count": metadata.get("document_count"),
                        "version": metadata.get("version", "unknown"),
                    }
    except Exception as e:
        logger.debug(f"Could not read sync status: {e}")

    return None
