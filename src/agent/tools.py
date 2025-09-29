"""
MRtrix3 documentation search tool for PydanticAI agent using ChromaDB.

This module implements the search_knowledgebase tool that allows the AI agent
to autonomously retrieve MRtrix3 documentation context from the local ChromaDB
database when needed to answer questions.
"""

import asyncio
import logging
import re
import json
from typing import List, Optional, Dict, Any, Union, Tuple
from pathlib import Path
from pydantic_ai import RunContext, ModelRetry
from .models import DocumentResult, SearchKnowledgebaseDependencies
from .embedding_service import EmbeddingService
from .session_logger import get_session_logger

logger = logging.getLogger("agent.tools")


async def search_knowledgebase(
    ctx: RunContext[SearchKnowledgebaseDependencies], queries: Union[str, List[str]]
) -> List[DocumentResult]:
    """
    Tool to search MRtrix3 documentation in local ChromaDB.

    Call this when you needs additional context to answer questions
    about MRtrix3 commands, tutorials, guides, or references.

    Args:
        ctx: PydanticAI context with dependencies including ChromaDB client,
             embedding service, and rate limiter
        queries: Single query string or list of natural language search queries

    Returns:
        List of matching documents formatted with XML blocks and query separators
    """
    # Normalize input to list of queries
    if isinstance(queries, str):
        query_list = [queries]
    else:
        query_list = queries

    # Validate input
    if not query_list:
        logger.warning("No queries provided")
        return []

    # Filter out empty queries and prepare valid ones
    valid_queries = []
    for original_query in query_list:
        if original_query and original_query.strip():
            query = original_query.strip()[:500]  # Limit query length
            sanitized_query = re.sub(
                r"[^\w\s\-.,!?]", " ", query
            )  # Keep alphanumeric and basic punctuation
            valid_queries.append((original_query, sanitized_query))
        else:
            logger.warning("Empty query in list, skipping")

    if not valid_queries:
        logger.warning("No valid queries after filtering")
        return []

    # Log sync status once
    sync_status = _get_sync_status(ctx)
    if sync_status:
        logger.debug(
            f"Database last synced: {sync_status.get('last_sync_time', 'unknown')}"
        )

    # Create or get embedding service
    if hasattr(ctx.deps, "embedding_service") and ctx.deps.embedding_service:
        embedding_service = ctx.deps.embedding_service
    else:
        embedding_service = EmbeddingService(
            ctx.deps.embedding_model,
            api_key=getattr(ctx.deps, "embedding_api_key", None),
        )

    # Process all queries in parallel
    all_results = await _process_queries_parallel(ctx, valid_queries, embedding_service)

    return all_results


async def _process_queries_parallel(
    ctx: RunContext[SearchKnowledgebaseDependencies],
    valid_queries: List[Tuple[str, str]],
    embedding_service: EmbeddingService,
) -> List[DocumentResult]:
    """
    Process multiple queries in parallel for faster search.

    Args:
        ctx: PydanticAI context with dependencies
        valid_queries: List of (original_query, sanitized_query) tuples
        embedding_service: Service for generating embeddings

    Returns:
        List of DocumentResults for all queries
    """
    # Create tasks for each query
    tasks = []
    for original_query, sanitized_query in valid_queries:
        task = _process_single_query(
            ctx, original_query, sanitized_query, embedding_service
        )
        tasks.append(task)

    # Execute all tasks concurrently
    results = await asyncio.gather(*tasks, return_exceptions=True)

    # Process results and handle exceptions
    all_results = []
    for idx, result in enumerate(results):
        original_query, _ = valid_queries[idx]
        display_query = original_query.strip()[:100]

        if isinstance(result, Exception):
            # Re-raise ModelRetry exceptions (they signal temporary failures needing retry)
            if isinstance(result, ModelRetry):
                raise result
            # Handle other failed queries
            logger.error(f"Query processing failed for '{original_query}': {result}")
            all_results.append(
                DocumentResult(
                    title=f"No results for query: {display_query}",
                    content=f"--- Results from query: {display_query} ---\nNo matching documents found.\n--- End of results from query: {display_query} ---",
                )
            )
        else:
            # Add successful result
            all_results.append(result)

    return all_results


async def _process_single_query(
    ctx: RunContext[SearchKnowledgebaseDependencies],
    original_query: str,
    sanitized_query: str,
    embedding_service: EmbeddingService,
) -> DocumentResult:
    """
    Process a single query including embedding generation and search.

    Args:
        ctx: PydanticAI context with dependencies
        original_query: Original query text for display
        sanitized_query: Sanitized query for processing
        embedding_service: Service for generating embeddings

    Returns:
        DocumentResult with search results for this query
    """
    # Log the RAG search query using session logger
    session_logger = get_session_logger()
    if session_logger:
        with session_logger.rag_search(sanitized_query):
            pass  # Context manager handles logging

    # Generate embedding for the query
    try:
        embedding = await embedding_service.generate_embedding(sanitized_query)
    except (TimeoutError, ConnectionError) as e:
        logger.warning(
            f"Embedding generation failed for query '{original_query}', will retry: {e}"
        )
        raise ModelRetry(f"Embedding generation temporarily failed: {e}") from e
    except Exception as e:
        logger.error(f"Permanent embedding error for query '{original_query}': {e}")
        raise

    # Perform vector similarity search in ChromaDB
    try:
        query_results = await _search_chromadb_vector(ctx, embedding, sanitized_query)

        # Format results with query separators (use original query for display)
        display_query = original_query.strip()[
            :100
        ]  # Truncate for display if very long

        if query_results:
            # Add query separator at the beginning
            separator_start = f"--- Results from query: {display_query} ---"
            separator_end = f"--- End of results from query: {display_query} ---"

            # Combine all results for this query
            combined_content = separator_start + "\n"
            for result in query_results:
                combined_content += result.content + "\n"
            combined_content += separator_end

            # Create a single DocumentResult with all results for this query
            return DocumentResult(
                title=f"Results for query: {display_query}",
                content=combined_content,
            )
        else:
            # Even if no results, add a marker for this query
            return DocumentResult(
                title=f"No results for query: {display_query}",
                content=f"--- Results from query: {display_query} ---\nNo matching documents found.\n--- End of results from query: {display_query} ---",
            )

    except Exception as e:
        logger.error(f"Vector search failed for query '{original_query}': {e}")
        raise


async def _search_chromadb_vector(
    ctx: RunContext[SearchKnowledgebaseDependencies],
    embedding: List[float],
    sanitized_query: str,
) -> List[DocumentResult]:
    """
    Perform vector similarity search in ChromaDB.

    Args:
        ctx: PydanticAI context with dependencies
        embedding: Query embedding vector
        sanitized_query: Sanitized query text for logging

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
        max_results = 2
        if hasattr(ctx.deps, "config") and ctx.deps.config:
            max_results = getattr(ctx.deps.config, "max_search_results", 2)

        # Query ChromaDB with embedding
        # ChromaDB returns results sorted by distance (smaller = more similar)
        results = collection.query(
            query_embeddings=[embedding],
            n_results=max_results,
            include=["documents", "metadatas", "distances"],
        )

        if not results or not results.get("documents"):
            logger.info(f"No vector search results for query: {sanitized_query}")
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
                similarity_score = 1 - distance  # Convert distance to similarity

                # Log individual document retrieval with score
                logger.debug(
                    f"Retrieved: '{title}' (similarity: {similarity_score:.3f})"
                )

                formatted_results.append(
                    DocumentResult(
                        title=title, content=_format_document_xml(title, doc)
                    )
                )

        # Log retrieved document titles for monitoring
        # All results are returned (max 2 by configuration)
        if formatted_results:
            # Log search results using session logger
            session_logger = get_session_logger()
            if session_logger:
                session_logger.log_rag_results(formatted_results)
        else:
            # Log no results using session logger
            session_logger = get_session_logger()
            if session_logger:
                session_logger.log_rag_results([])

        # Return all results (already limited to max 2 by configuration)
        return formatted_results

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
