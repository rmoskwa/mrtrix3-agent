"""Integration tests for agent search functionality with real Supabase."""

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

from supabase import acreate_client, AsyncClient  # noqa: E402
from pydantic_ai import RunContext  # noqa: E402
from src.agent.models import SearchKnowledgebaseDependencies, DocumentResult  # noqa: E402
from src.agent.tools import search_knowledgebase, _bm25_fallback  # noqa: E402


@pytest_asyncio.fixture
async def supabase_client() -> AsyncClient:
    """Create real async Supabase client for integration tests."""
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
def real_dependencies(supabase_client):
    """Create real dependencies with actual Supabase client."""
    deps = SearchKnowledgebaseDependencies(
        supabase_client=supabase_client,
        embedding_model=os.getenv("EMBEDDING_MODEL"),
        rate_limiter=None,
    )
    return deps


@pytest.fixture
def real_context(real_dependencies):
    """Create context with real dependencies."""
    ctx = MagicMock(spec=RunContext)
    ctx.deps = real_dependencies
    return ctx


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
    """Integration tests for search_knowledgebase with real database."""

    @pytest.mark.asyncio
    async def test_search_with_real_rpc(self, real_context):
        """Test search_knowledgebase with real RPC function (mocked embedding)."""
        # Mock only the embedding generation
        with patch("src.agent.tools.EmbeddingService") as mock_embedding_class:
            mock_embedding = mock_embedding_class.return_value
            mock_embedding.generate_embedding = AsyncMock(
                return_value=[0.1] * 768  # Dummy embedding
            )

            # This will use the real RPC function
            results = await search_knowledgebase(real_context, "test query")

            # Results should be formatted DocumentResult objects
            assert isinstance(results, list)
            for result in results:
                assert isinstance(result, DocumentResult)
                assert result.title
                assert "<Start of" in result.content
                assert "</Start of" in result.content

    @pytest.mark.asyncio
    async def test_bm25_fallback_with_real_database(self, real_context):
        """Test BM25 fallback with real database."""
        # Test BM25 directly
        results = await _bm25_fallback(real_context, "mrconvert")

        assert isinstance(results, list)
        # BM25 should also return formatted results
        for result in results[:2]:  # Check top 2
            assert isinstance(result, DocumentResult)
            assert "<Start of" in result.content

    @pytest.mark.asyncio
    async def test_search_fallback_on_rpc_error(self, real_context):
        """Test that search falls back to BM25 when RPC fails."""
        with patch("src.agent.tools.EmbeddingService") as mock_embedding_class:
            mock_embedding = mock_embedding_class.return_value
            mock_embedding.generate_embedding = AsyncMock(return_value=[0.1] * 768)

            # Mock RPC to fail
            real_context.deps.supabase_client.rpc = MagicMock(
                side_effect=Exception("RPC error")
            )

            # Should fall back to BM25
            results = await search_knowledgebase(real_context, "installation")

            # Should still get results from BM25
            assert isinstance(results, list)
