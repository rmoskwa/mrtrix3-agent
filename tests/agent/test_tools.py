"""Unit tests for the search_knowledgebase tool."""

import sys
from unittest.mock import AsyncMock, Mock, patch, MagicMock

import pytest
from pydantic_ai import RunContext, ModelRetry

# Mock the google.generativeai module before importing tools
sys.modules["google.generativeai"] = MagicMock()
sys.modules["google.api_core"] = MagicMock()
sys.modules["google.api_core.retry"] = MagicMock()
sys.modules["google.api_core.exceptions"] = MagicMock()

from src.agent.models import SearchKnowledgebaseDependencies, DocumentResult  # noqa: E402
from src.agent.tools import search_knowledgebase, _bm25_fallback, _format_results  # noqa: E402


@pytest.fixture
def mock_dependencies():
    """Create mock dependencies for testing."""
    deps = Mock(spec=SearchKnowledgebaseDependencies)
    deps.supabase_client = AsyncMock()
    deps.embedding_model = Mock()
    deps.rate_limiter = None
    return deps


@pytest.fixture
def mock_context(mock_dependencies):
    """Create mock RunContext with dependencies."""
    ctx = Mock(spec=RunContext)
    ctx.deps = mock_dependencies
    return ctx


@pytest.fixture
def sample_results():
    """Sample search results from database."""
    return [
        {
            "title": "mrconvert",
            "content": "Convert image files between different formats.",
            "similarity": 0.85,  # RPC returns similarity, but we don't use it
        },
        {
            "title": "Installation Guide",
            "content": "Instructions for installing MRtrix3.",
            "similarity": 0.79,
        },
    ]


class TestSearchKnowledgebase:
    """Test cases for search_knowledgebase tool."""

    @pytest.mark.asyncio
    async def test_successful_search(self, mock_context, sample_results):
        """Test successful semantic search with vector embeddings."""
        # Mock embedding generation
        with patch("src.agent.tools.EmbeddingService") as mock_embedding_class:
            mock_embedding = mock_embedding_class.return_value
            mock_embedding.generate_embedding = AsyncMock(return_value=[0.1] * 768)

            # Mock RPC call
            mock_rpc = Mock()
            mock_rpc.execute = AsyncMock(return_value=Mock(data=sample_results))
            mock_context.deps.supabase_client.rpc = Mock(return_value=mock_rpc)

            # Also mock BM25 as fallback (in case RPC mock fails)
            mock_table = Mock()
            mock_table.select = Mock(return_value=mock_table)
            mock_table.ilike = Mock(return_value=mock_table)
            mock_table.limit = Mock(return_value=mock_table)
            mock_table.execute = AsyncMock(return_value=Mock(data=sample_results))
            mock_context.deps.supabase_client.from_ = Mock(return_value=mock_table)

            # Execute search
            results = await search_knowledgebase(mock_context, "How to convert images?")

            # Verify results
            assert len(results) == 2
            assert isinstance(results[0], DocumentResult)
            assert "mrconvert" in results[0].title
            assert "<Start of mrconvert document>" in results[0].content

    @pytest.mark.asyncio
    async def test_embedding_timeout_retry(self, mock_context):
        """Test that embedding timeout triggers ModelRetry."""
        with patch("src.agent.tools.EmbeddingService") as mock_embedding_class:
            mock_embedding = mock_embedding_class.return_value
            mock_embedding.generate_embedding = AsyncMock(
                side_effect=TimeoutError("API timeout")
            )

            # Should raise ModelRetry for transient error
            with pytest.raises(ModelRetry) as exc_info:
                await search_knowledgebase(mock_context, "test query")

            assert "temporarily failed" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_bm25_fallback_activation(self, mock_context, sample_results):
        """Test BM25 fallback when vector search fails."""
        with patch("src.agent.tools.EmbeddingService") as mock_embedding_class:
            mock_embedding = mock_embedding_class.return_value
            mock_embedding.generate_embedding = AsyncMock(return_value=[0.1] * 768)

            # Mock RPC failure
            mock_rpc_result = Mock()
            mock_rpc_result.execute = AsyncMock(
                side_effect=Exception("Vector search error")
            )
            mock_context.deps.supabase_client.rpc = Mock(return_value=mock_rpc_result)

            # Mock BM25 fallback
            mock_table = Mock()
            mock_table.select = Mock(return_value=mock_table)
            mock_table.ilike = Mock(return_value=mock_table)
            mock_table.limit = Mock(return_value=mock_table)
            mock_table.execute = AsyncMock(return_value=Mock(data=sample_results))
            mock_context.deps.supabase_client.from_ = Mock(return_value=mock_table)

            # Execute search
            results = await search_knowledgebase(mock_context, "convert images")

            # Verify BM25 was called
            mock_context.deps.supabase_client.from_.assert_called_with("documents")
            assert len(results) == 2

    @pytest.mark.asyncio
    async def test_no_results_found(self, mock_context):
        """Test handling when no results are found."""
        with patch("src.agent.tools.EmbeddingService") as mock_embedding_class:
            mock_embedding = mock_embedding_class.return_value
            mock_embedding.generate_embedding = AsyncMock(return_value=[0.1] * 768)

            # Mock empty results
            mock_rpc_result = Mock()
            mock_rpc_result.execute = AsyncMock(return_value=Mock(data=[]))
            mock_context.deps.supabase_client.rpc = Mock(return_value=mock_rpc_result)

            # Mock empty BM25 results too
            mock_table = Mock()
            mock_table.select = Mock(return_value=mock_table)
            mock_table.ilike = Mock(return_value=mock_table)
            mock_table.limit = Mock(return_value=mock_table)
            mock_table.execute = AsyncMock(return_value=Mock(data=[]))
            mock_context.deps.supabase_client.from_ = Mock(return_value=mock_table)

            results = await search_knowledgebase(mock_context, "nonexistent topic")
            assert results == []

    @pytest.mark.asyncio
    async def test_search_with_large_results(self, mock_context):
        """Test search handles large result sets properly."""
        with patch("src.agent.tools.EmbeddingService") as mock_embedding_class:
            mock_embedding = mock_embedding_class.return_value
            mock_embedding.generate_embedding = AsyncMock(return_value=[0.1] * 768)

            # Mock large result set
            large_results = [
                {"title": f"Doc{i}", "content": f"Content {i}"} for i in range(10)
            ]
            mock_rpc_result = Mock()
            mock_rpc_result.execute = AsyncMock(return_value=Mock(data=large_results))
            mock_context.deps.supabase_client.rpc = Mock(return_value=mock_rpc_result)

            results = await search_knowledgebase(mock_context, "test query")

            # Should only return top 2 results even if more are available
            assert len(results) == 2
            assert results[0].title == "Doc0"

    @pytest.mark.asyncio
    async def test_search_with_empty_query(self, mock_context):
        """Test search with empty query string."""
        with patch("src.agent.tools.EmbeddingService") as mock_embedding_class:
            mock_embedding = mock_embedding_class.return_value
            mock_embedding.generate_embedding = AsyncMock(return_value=[0.1] * 768)

            mock_rpc_result = Mock()
            mock_rpc_result.execute = AsyncMock(return_value=Mock(data=[]))
            mock_context.deps.supabase_client.rpc = Mock(return_value=mock_rpc_result)

            mock_table = Mock()
            mock_table.select = Mock(return_value=mock_table)
            mock_table.ilike = Mock(return_value=mock_table)
            mock_table.limit = Mock(return_value=mock_table)
            mock_table.execute = AsyncMock(return_value=Mock(data=[]))
            mock_context.deps.supabase_client.from_ = Mock(return_value=mock_table)

            results = await search_knowledgebase(mock_context, "")
            assert results == []


class TestBM25Fallback:
    """Test cases for BM25 fallback search."""

    @pytest.mark.asyncio
    async def test_bm25_successful_search(self, mock_context, sample_results):
        """Test successful BM25 keyword search."""
        mock_table = Mock()
        mock_table.select = Mock(return_value=mock_table)
        mock_table.ilike = Mock(return_value=mock_table)
        mock_table.limit = Mock(return_value=mock_table)
        mock_table.execute = AsyncMock(return_value=Mock(data=sample_results))
        mock_context.deps.supabase_client.from_ = Mock(return_value=mock_table)

        results = await _bm25_fallback(mock_context, "installation guide")

        assert len(results) == 2
        assert isinstance(results[0], DocumentResult)

    @pytest.mark.asyncio
    async def test_bm25_database_error(self, mock_context):
        """Test BM25 raises ModelRetry on database error."""
        mock_context.deps.supabase_client.from_ = AsyncMock(
            side_effect=Exception("Database connection error")
        )

        with pytest.raises(ModelRetry) as exc_info:
            await _bm25_fallback(mock_context, "test query")

        assert "temporarily unavailable" in str(exc_info.value)


class TestFormatResults:
    """Test cases for result formatting."""

    def test_format_results_with_xml(self, sample_results):
        """Test formatting results into XML blocks."""
        formatted = _format_results(sample_results)

        assert len(formatted) == 2
        assert formatted[0].title == "mrconvert"
        assert "<Start of mrconvert document>" in formatted[0].content
        assert "</Start of mrconvert document>" in formatted[0].content

    def test_format_empty_results(self):
        """Test formatting empty results."""
        formatted = _format_results([])
        assert formatted == []

    def test_format_missing_fields(self):
        """Test formatting with missing fields."""
        results = [{"doc_id": "test"}]
        formatted = _format_results(results)

        assert len(formatted) == 1
        assert formatted[0].title == "Untitled"
        assert (
            formatted[0].content
            == "<Start of Untitled document></Start of Untitled document>"
        )

    def test_format_results_with_none_values(self):
        """Test formatting with None values in fields."""
        # _format_results should handle None by converting to "Untitled" and empty content
        results = [{"title": None, "content": None}]
        formatted = _format_results(results)

        assert len(formatted) == 1
        assert formatted[0].title == "Untitled"
        assert "<Start of Untitled document>" in formatted[0].content

    def test_format_results_with_special_characters(self):
        """Test formatting with special characters in content."""
        results = [{"title": "test&<>", "content": "content with & < > characters"}]
        formatted = _format_results(results)

        assert len(formatted) == 1
        assert formatted[0].title == "test&<>"
        assert "content with & < > characters" in formatted[0].content
