#!/usr/bin/env python3
"""
MRtrix3 Documentation Embedding Generator
Generates embeddings for documents using Google's Gemini embedding model.
"""

import os
import asyncio
import logging
from typing import Dict, Any, List, Optional
from dotenv import load_dotenv
import google.generativeai as genai
from tenacity import retry, stop_after_attempt, wait_exponential

# Load environment variables
load_dotenv()

# Configuration
GOOGLE_API_KEY_EMBEDDING = os.getenv("GOOGLE_API_KEY_EMBEDDING")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL")
if not EMBEDDING_MODEL:
    raise ValueError("EMBEDDING_MODEL not found in environment")
EMBEDDING_DIMENSIONS = int(os.getenv("EMBEDDING_DIMENSIONS", "768"))

# Configure logging
logger = logging.getLogger(__name__)


class EmbeddingGenerator:
    """Generate embeddings for MRtrix3 documentation"""

    def __init__(self):
        """Initialize the embedding generator with Gemini configuration"""
        # Configure Gemini for embeddings
        genai.configure(api_key=GOOGLE_API_KEY_EMBEDDING)
        self.model = EMBEDDING_MODEL

    def prepare_embedding_text(self, doc: Dict[str, Any]) -> str:
        """
        Prepare natural language text for embedding generation.

        Args:
            doc: Document dictionary with fields like title, doc_type, concepts, error_types

        Returns:
            Natural language string describing the document
        """
        # Start with basic document type and title
        text_parts = [f"This is a {doc['doc_type']} document titled {doc['title']}."]

        # Add concepts if available
        if doc.get("concepts") and doc["concepts"]:
            concepts_str = ", ".join(doc["concepts"])
            text_parts.append(f"Concepts covered in this file are {concepts_str}.")

        # Add error types for command documents
        if doc.get("doc_type") == "command" and doc.get("error_types"):
            error_types = doc["error_types"]
            if error_types:
                # Extract error messages from the JSONB structure
                error_messages = []
                if isinstance(error_types, dict):
                    for func, msg in error_types.items():
                        error_messages.append(msg)

                if error_messages:
                    error_str = ", ".join(
                        error_messages[:5]
                    )  # Limit to first 5 for brevity
                    text_parts.append(f"Error types in this document are: {error_str}.")

        # Add synopsis if available
        if doc.get("synopsis"):
            text_parts.append(doc["synopsis"])

        return " ".join(text_parts)

    @retry(
        wait=wait_exponential(multiplier=1, min=2, max=10), stop=stop_after_attempt(3)
    )
    async def generate_embedding(self, text: str) -> Optional[List[float]]:
        """
        Generate embedding vector for the given text using Gemini.

        Args:
            text: Text to generate embedding for

        Returns:
            List of floats representing the embedding vector, or None if failed
        """
        try:
            # Use asyncio.to_thread to run the synchronous API call in a thread
            result = await asyncio.to_thread(
                genai.embed_content,
                model=self.model,
                content=text,
                task_type="retrieval_document",  # Optimize for document retrieval
                output_dimensionality=EMBEDDING_DIMENSIONS,  # Set to 768 dimensions
            )

            # Extract embedding from result
            if result and "embedding" in result:
                embedding = result["embedding"]

                # Validate dimensions
                if len(embedding) != EMBEDDING_DIMENSIONS:
                    return None

                return embedding
            else:
                return None

        except Exception:
            return None

    async def generate_document_embedding(
        self, doc: Dict[str, Any]
    ) -> Optional[List[float]]:
        """
        Generate embedding for a document using its metadata.

        Args:
            doc: Document dictionary containing title, doc_type, concepts, etc.

        Returns:
            Embedding vector or None if generation failed
        """
        # Prepare natural language text
        embedding_text = self.prepare_embedding_text(doc)

        # Generate and return embedding
        return await self.generate_embedding(embedding_text)

    async def batch_generate_embeddings(
        self, documents: List[Dict[str, Any]], max_concurrent: int = 10
    ) -> List[Optional[List[float]]]:
        """
        Generate embeddings for multiple documents concurrently.

        Args:
            documents: List of document dictionaries
            max_concurrent: Maximum number of concurrent API calls

        Returns:
            List of embeddings (or None for failed generations)
        """
        embeddings = []

        # Process in batches to respect rate limits
        for i in range(0, len(documents), max_concurrent):
            batch = documents[i : i + max_concurrent]

            # Create tasks for concurrent processing
            tasks = [self.generate_document_embedding(doc) for doc in batch]

            # Wait for batch to complete
            batch_results = await asyncio.gather(*tasks, return_exceptions=True)

            # Handle results
            for result in batch_results:
                if isinstance(result, Exception):
                    embeddings.append(None)
                else:
                    embeddings.append(result)

        return embeddings


# Convenience function for integration
async def add_embeddings_to_documents(
    documents: List[Dict[str, Any]], max_concurrent: int = 10
) -> List[Dict[str, Any]]:
    """
    Add embeddings to a list of documents in place.

    Args:
        documents: List of document dictionaries to add embeddings to
        max_concurrent: Maximum concurrent API calls

    Returns:
        The same documents list with 'content_embedding' field added
    """
    generator = EmbeddingGenerator()

    # Generate embeddings for all documents
    embeddings = await generator.batch_generate_embeddings(documents, max_concurrent)

    # Add embeddings to documents
    success_count = 0
    for doc, embedding in zip(documents, embeddings):
        if embedding:
            doc["content_embedding"] = embedding
            success_count += 1
        else:
            doc["content_embedding"] = None

    return documents


# Example usage for testing
if __name__ == "__main__":

    async def test():
        # Test document
        test_doc = {
            "title": "mrconvert",
            "doc_type": "command",
            "concepts": ["diffusion", "tensor", "b-value"],
            "synopsis": "Convert between different MRI image formats",
            "error_types": {
                "run": "axis supplied to option -axes is out of bounds",
                "validate": "Input image expected to be 4D",
            },
        }

        generator = EmbeddingGenerator()

        # Test text preparation
        text = generator.prepare_embedding_text(test_doc)
        print(f"Embedding text: {text}\n")

        # Test embedding generation
        embedding = await generator.generate_document_embedding(test_doc)
        if embedding:
            print(f"Generated embedding with {len(embedding)} dimensions")
            print(f"First 10 values: {embedding[:10]}")
        else:
            print("Failed to generate embedding")

    asyncio.run(test())
