"""Embedding generation service for agent search queries."""

import os
import asyncio
import logging
from typing import List
import google.generativeai as genai
from google.api_core import retry, exceptions
from .rate_limiter import embedding_limiter

logger = logging.getLogger("agent.embedding")


class EmbeddingService:
    """Service for generating embeddings using Google Gemini API."""

    def __init__(self, model_name: str = None, api_key: str = None):
        """
        Initialize embedding service.

        Args:
            model_name: Gemini embedding model name. Defaults to EMBEDDING_MODEL env var.
            api_key: Google API key for embeddings. Defaults to GOOGLE_API_KEY_EMBEDDING env var.
        """
        self.model_name = model_name or os.getenv("EMBEDDING_MODEL")
        if not self.model_name:
            raise ValueError(
                "EMBEDDING_MODEL not found in environment and no model_name provided"
            )
        api_key = api_key or os.getenv("GOOGLE_API_KEY_EMBEDDING")
        if not api_key:
            raise ValueError("API key for embeddings not provided")

        genai.configure(api_key=api_key)

    async def generate_embedding(self, text: str) -> List[float]:
        """
        Generate 768-dimensional embedding for text.

        Args:
            text: Text to embed.

        Returns:
            768-dimensional embedding vector.

        Raises:
            TimeoutError: If API request times out.
            ConnectionError: If network connection fails.
        """
        # Apply rate limiting before making API call
        await embedding_limiter.acquire()

        try:
            # Use retry decorator for transient errors
            @retry.Retry(
                predicate=retry.if_exception_type(
                    exceptions.ServiceUnavailable,
                    exceptions.DeadlineExceeded,
                ),
                deadline=30.0,
            )
            def _generate_sync():
                result = genai.embed_content(
                    model=self.model_name,
                    content=text,
                    task_type="retrieval_query",
                    output_dimensionality=768,
                )
                return result["embedding"]

            # Run sync function in executor to avoid blocking the event loop
            loop = asyncio.get_event_loop()
            embedding = await loop.run_in_executor(None, _generate_sync)
            logger.debug(f"Generated embedding for query: {text[:50]}...")
            return embedding

        except exceptions.DeadlineExceeded as e:
            logger.error(f"Embedding generation timed out: {e}")
            raise TimeoutError("Embedding generation timed out") from e
        except (exceptions.ServiceUnavailable, exceptions.NetworkError) as e:
            logger.error(f"Network error during embedding generation: {e}")
            raise ConnectionError("Network error during embedding") from e
        except Exception as e:
            logger.error(f"Unexpected error generating embedding: {e}")
            raise
