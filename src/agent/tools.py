"""
MRtrix3 documentation search tool for PydanticAI agent.

This module implements the search_knowledgebase tool that allows the AI agent
to autonomously retrieve MRtrix3 documentation context when needed to answer questions.
"""

import logging
import re
from typing import List
from pydantic_ai import RunContext, ModelRetry
from .models import DocumentResult, SearchKnowledgebaseDependencies
from .embedding_service import EmbeddingService
from .circuit_breaker import supabase_circuit_breaker, CircuitBreakerError

logger = logging.getLogger("agent.tools")


async def search_knowledgebase(
    ctx: RunContext[SearchKnowledgebaseDependencies], query: str
) -> List[DocumentResult]:
    """
    Tool to search MRtrix3 documentation.

    Call this when you needs additional context to answer questions
    about MRtrix3 commands, tutorials, guides, or references.

    Args:
        ctx: PydanticAI context with dependencies including Supabase client,
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
        # Fall back to BM25 search
        return await _bm25_fallback(ctx, sanitized_query)

    # Perform cosine similarity search using RPC function with circuit breaker
    try:

        async def rpc_call():
            return await ctx.deps.supabase_client.rpc(
                "match_documents",
                {
                    "query_embedding": embedding,
                    "match_threshold": getattr(
                        ctx.deps.config, "similarity_threshold", 0.7
                    )
                    if hasattr(ctx.deps, "config")
                    else 0.7,
                    "match_count": getattr(ctx.deps.config, "max_search_results", 3)
                    if hasattr(ctx.deps, "config")
                    else 3,
                },
            ).execute()

        results = await supabase_circuit_breaker.call(rpc_call)

        if results.data and len(results.data) > 0:
            return _format_results(results.data[:2])  # Return top-2 from top-3

    except CircuitBreakerError as e:
        logger.warning(f"Circuit breaker open for vector search: {e}")
        # Fall back to BM25 search
    except Exception as e:
        logger.warning(f"Vector search failed, falling back to BM25: {e}")

    # Fall back to BM25 keyword search
    return await _bm25_fallback(ctx, sanitized_query)


async def _bm25_fallback(
    ctx: RunContext[SearchKnowledgebaseDependencies], query: str
) -> List[DocumentResult]:
    """
    BM25 fallback search using keywords column.

    Args:
        ctx: PydanticAI context with dependencies
        query: Natural language search query

    Returns:
        List of matching documents

    Raises:
        ModelRetry: For transient database errors
    """
    try:
        # Sanitize query for safe SQL LIKE pattern
        # Escape SQL wildcard characters and limit length
        safe_query = query.replace("%", "\\%").replace("_", "\\_").replace("[", "\\[")
        safe_query = safe_query[:100]  # Limit pattern length for performance

        async def bm25_query():
            # Use parameterized query pattern with escaped wildcards
            table = ctx.deps.supabase_client.from_("documents")

            # Use safe pattern with escaped query
            pattern = f"%{safe_query}%"
            return (
                await table.select("title, content")
                .ilike("keywords", pattern)
                .limit(3)
                .execute()
            )

        results = await supabase_circuit_breaker.call(bm25_query)

        if results.data:
            return _format_results(results.data[:2])  # Return top-2

        logger.info(f"No results found for query: {query}")
        return []

    except CircuitBreakerError as e:
        logger.error(f"Circuit breaker open for BM25 search: {e}")
        raise ModelRetry(f"Database temporarily unavailable: {e}") from e
    except Exception as e:
        logger.error(f"BM25 search failed: {e}")
        raise ModelRetry(f"Search temporarily unavailable: {e}") from e


def _format_results(results: List[dict]) -> List[DocumentResult]:
    """
    Format search results into DocumentResult models with XML blocks.

    Args:
        results: Raw database results

    Returns:
        List of formatted DocumentResult objects
    """
    formatted = []
    for result in results:
        # Filter internal database fields from agent view
        title = result.get("title", "Untitled") or "Untitled"
        content = result.get("content", "") or ""

        # Format as XML block
        formatted_content = (
            f"<Start of {title} document>{content}</Start of {title} document>"
        )

        formatted.append(DocumentResult(title=title, content=formatted_content))

    return formatted
