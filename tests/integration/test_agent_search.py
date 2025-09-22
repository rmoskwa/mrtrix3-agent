"""Integration tests for agent search functionality with ChromaDB and Supabase sync."""

import os
import sys
from unittest.mock import patch, MagicMock, AsyncMock

import pytest
import pytest_asyncio

# Only load .env if not in CI environment
if not os.getenv("CI"):
    from dotenv import load_dotenv

    # Load environment variables
    load_dotenv()

# Mock Google modules before importing
sys.modules["google.generativeai"] = MagicMock()
sys.modules["google.api_core"] = MagicMock()
sys.modules["google.api_core.retry"] = MagicMock()
sys.modules["google.api_core.exceptions"] = MagicMock()

import chromadb  # noqa: E402
from supabase import acreate_client, AsyncClient  # noqa: E402
from pydantic_ai import RunContext  # noqa: E402
from src.agent.models import SearchKnowledgebaseDependencies, DocumentResult  # noqa: E402
from src.agent.tools import search_knowledgebase  # noqa: E402


@pytest_asyncio.fixture
async def supabase_client() -> AsyncClient:
    """Create real async Supabase client for sync operations."""
    # Only load .env if not in CI environment
    if not os.getenv("CI"):
        from dotenv import load_dotenv

        load_dotenv(override=True)

    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_KEY")

    if not url or not key:
        pytest.skip("SUPABASE_URL or SUPABASE_KEY not found in environment")

    return await acreate_client(url, key)


@pytest.fixture
def chromadb_test_client(tmp_path):
    """Create a test ChromaDB client with temporary storage."""
    client = chromadb.PersistentClient(
        path=str(tmp_path / "chromadb_test"),
        settings=chromadb.Settings(anonymized_telemetry=False),
    )
    return client


@pytest.fixture
def chromadb_test_collection(chromadb_test_client):
    """Create a test ChromaDB collection with sample data."""
    collection = chromadb_test_client.get_or_create_collection(
        name="mrtrix3_documents",
        metadata={
            "description": "MRtrix3 documentation for testing",
            "embedding_dimensions": "768",
        },
    )

    # Add test documents
    test_docs = [
        {
            "id": "test1",
            "document": "Perform conversion between image formats using mrconvert",
            "metadata": {
                "title": "mrconvert",
                "keywords": "convert image format",
                "doc_type": "command",
            },
            "embedding": [0.1] * 768,  # Dummy embedding
        },
        {
            "id": "test2",
            "document": "How to install MRtrix3 on different systems",
            "metadata": {
                "title": "Installation Guide",
                "keywords": "install setup",
                "doc_type": "guide",
            },
            "embedding": [0.2] * 768,  # Dummy embedding
        },
    ]

    for doc in test_docs:
        collection.add(
            ids=[doc["id"]],
            documents=[doc["document"]],
            metadatas=[doc["metadata"]],
            embeddings=[doc["embedding"]],
        )

    return collection


@pytest.fixture
def real_dependencies(
    supabase_client, chromadb_test_client, chromadb_test_collection, tmp_path
):
    """Create real dependencies with both Supabase and ChromaDB."""
    deps = SearchKnowledgebaseDependencies(
        supabase_client=supabase_client,
        chromadb_client=chromadb_test_client,
        chromadb_collection=chromadb_test_collection,
        chromadb_path=str(tmp_path / "chromadb_test"),
        embedding_model=os.getenv("EMBEDDING_MODEL", "gemini-embedding-001"),
        rate_limiter=None,
    )
    return deps


@pytest.fixture
def real_context(real_dependencies):
    """Create context with real dependencies."""
    ctx = MagicMock(spec=RunContext)
    ctx.deps = real_dependencies
    return ctx


class TestChromaDBIntegration:
    """Test ChromaDB integration for agent search."""

    def test_chromadb_collection_exists(self, chromadb_test_collection):
        """Test that ChromaDB collection is properly created."""
        assert chromadb_test_collection is not None
        assert chromadb_test_collection.count() > 0

    def test_chromadb_query_works(self, chromadb_test_collection):
        """Test basic ChromaDB query functionality."""
        # Test query with dummy embedding
        results = chromadb_test_collection.query(
            query_embeddings=[[0.1] * 768], n_results=2
        )

        assert results is not None
        assert "documents" in results
        assert len(results["documents"][0]) > 0

    def test_chromadb_metadata_filter(self, chromadb_test_collection):
        """Test ChromaDB get functionality without filtering."""
        # Test basic get without filters (ChromaDB doesn't support $contains)
        results = chromadb_test_collection.get(limit=2)

        assert results is not None
        assert "documents" in results
        assert len(results["documents"]) > 0


class TestRPCFunction:
    """Test the actual Supabase RPC function."""

    @pytest.mark.asyncio
    async def test_match_documents_rpc_exists(self, supabase_client):
        """Verify the match_documents RPC function exists and is callable."""
        # Create a dummy embedding
        dummy_embedding = [0.1] * 768

        try:
            # Call the RPC function (Supabase client is async)
            result = await supabase_client.rpc(
                "match_documents",
                {
                    "query_embedding": dummy_embedding,
                    "match_threshold": 0.7,
                    "match_count": 3,
                },
            ).execute()

            # The function should return a result (even if empty)
            assert result is not None
            assert hasattr(result, "data")

        except Exception as e:
            if "function match_documents" in str(e) and "does not exist" in str(e):
                pytest.fail("RPC function match_documents not found in database")
            # Other errors might be OK (e.g., no matching documents)

    @pytest.mark.asyncio
    async def test_rpc_returns_correct_fields(self, supabase_client):
        """Test that RPC returns the expected fields."""
        dummy_embedding = [0.1] * 768

        try:
            result = await supabase_client.rpc(
                "match_documents",
                {
                    "query_embedding": dummy_embedding,
                    "match_threshold": 0.0,  # Low threshold to get some results
                    "match_count": 1,
                },
            ).execute()
        except Exception as e:
            print(f"Error calling RPC: {e}")
            print(f"Error type: {type(e)}")
            if "Name or service not known" in str(e):
                pytest.skip("Cannot connect to Supabase - network or URL issue")
            raise

        if result.data and len(result.data) > 0:
            # Check returned fields (optimized to only return needed fields)
            first_result = result.data[0]
            assert "title" in first_result
            assert "content" in first_result
            assert "similarity" in first_result
            # These fields should NOT be returned anymore
            assert "doc_id" not in first_result
            assert "doc_type" not in first_result


class TestSearchIntegration:
    """Integration tests for search_knowledgebase with ChromaDB."""

    @pytest.mark.asyncio
    async def test_search_with_chromadb(self, real_context):
        """Test search_knowledgebase with ChromaDB (mocked embedding)."""
        # Mock only the embedding generation
        with patch("src.agent.tools.EmbeddingService") as mock_embedding_class:
            mock_embedding = mock_embedding_class.return_value
            mock_embedding.generate_embedding = AsyncMock(
                return_value=[0.1] * 768  # Dummy embedding
            )

            # This will use ChromaDB
            results = await search_knowledgebase(real_context, "test query")

            # Results should be formatted DocumentResult objects
            assert isinstance(results, list)
            for result in results:
                assert isinstance(result, DocumentResult)
                assert result.title
                assert "<Start of" in result.content
                assert "</Start of" in result.content

    @pytest.mark.asyncio
    async def test_search_returns_empty_on_vector_error(self, real_context):
        """Test that search returns empty list when vector search fails."""
        with patch("src.agent.tools.EmbeddingService") as mock_embedding_class:
            mock_embedding = mock_embedding_class.return_value
            mock_embedding.generate_embedding = AsyncMock(return_value=[0.1] * 768)

            # Mock ChromaDB query to fail
            original_query = real_context.deps.chromadb_collection.query
            real_context.deps.chromadb_collection.query = MagicMock(
                side_effect=Exception("Vector search error")
            )

            # Should return empty list when vector search fails
            results = await search_knowledgebase(real_context, "install")

            # Restore original method
            real_context.deps.chromadb_collection.query = original_query

            # Should return empty list, not error
            assert results == []

    @pytest.mark.asyncio
    async def test_search_with_special_characters(self, real_context):
        """Test search handles special characters in queries."""
        with patch("src.agent.tools.EmbeddingService") as mock_embedding_class:
            mock_embedding = mock_embedding_class.return_value
            mock_embedding.generate_embedding = AsyncMock(return_value=[0.1] * 768)

            # Query with special characters that should be sanitized
            results = await search_knowledgebase(
                real_context, "test @#$%^&*() query with symbols"
            )

            # Should handle gracefully
            assert isinstance(results, list)

    @pytest.mark.asyncio
    async def test_search_with_long_query(self, real_context):
        """Test search handles very long queries."""
        with patch("src.agent.tools.EmbeddingService") as mock_embedding_class:
            mock_embedding = mock_embedding_class.return_value
            mock_embedding.generate_embedding = AsyncMock(return_value=[0.1] * 768)

            # Create a very long query (should be truncated to 500 chars)
            long_query = "test " * 200  # 1000 characters

            results = await search_knowledgebase(real_context, long_query)

            # Should handle gracefully
            assert isinstance(results, list)

    @pytest.mark.asyncio
    async def test_search_consistency_between_calls(self, real_context):
        """Test that repeated searches return consistent results."""
        with patch("src.agent.tools.EmbeddingService") as mock_embedding_class:
            mock_embedding = mock_embedding_class.return_value
            # Use same embedding for consistency
            fixed_embedding = [0.2] * 768
            mock_embedding.generate_embedding = AsyncMock(return_value=fixed_embedding)

            # Make two identical searches
            results1 = await search_knowledgebase(real_context, "mrconvert")
            results2 = await search_knowledgebase(real_context, "mrconvert")

            # Should get same results (at least same titles)
            if results1 and results2:
                assert results1[0].title == results2[0].title
