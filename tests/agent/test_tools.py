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
from src.agent.tools import (  # noqa: E402
    search_knowledgebase,
    _keyword_fallback_chromadb,
    _format_document_xml,
)


@pytest.fixture
def mock_dependencies():
    """Create mock dependencies for testing."""
    deps = Mock(spec=SearchKnowledgebaseDependencies)
    deps.supabase_client = AsyncMock()
    deps.embedding_model = Mock()
    deps.rate_limiter = None
    # Add ChromaDB mocks
    deps.chromadb_client = Mock()
    deps.chromadb_collection = Mock()
    deps.chromadb_path = "/mock/chromadb/path"
    deps.embedding_service = None
    deps.config = Mock(max_search_results=3, similarity_threshold=0.7)
    return deps


@pytest.fixture
def mock_context(mock_dependencies):
    """Create mock context for testing."""
    ctx = Mock(spec=RunContext)
    ctx.deps = mock_dependencies
    return ctx


@pytest.fixture
def sample_results():
    """Sample search results for testing."""
    return [
        {"title": "mrconvert", "content": "Perform conversion between image formats"},
        {"title": "Installation Guide", "content": "How to install MRtrix3"},
    ]


@pytest.fixture
def sample_chromadb_results():
    """Sample ChromaDB query results."""
    return {
        "documents": [
            ["Perform conversion between image formats", "How to install MRtrix3"]
        ],
        "metadatas": [
            [
                {
                    "title": "mrconvert",
                    "keywords": "convert image format",
                    "source_url": "/docs/mrconvert",
                    "synopsis": "Convert images",
                },
                {
                    "title": "Installation Guide",
                    "keywords": "install setup",
                    "source_url": "/docs/install",
                    "synopsis": "Installation instructions",
                },
            ]
        ],
        "distances": [[0.2, 0.3]],
    }


class TestSearchKnowledgebase:
    """Test cases for the main search_knowledgebase function."""

    @pytest.mark.asyncio
    async def test_successful_vector_search(
        self, mock_context, sample_chromadb_results
    ):
        """Test successful vector search with ChromaDB."""
        with patch("src.agent.tools.EmbeddingService") as mock_embedding_class:
            mock_embedding = mock_embedding_class.return_value
            mock_embedding.generate_embedding = AsyncMock(return_value=[0.1] * 768)

            # Mock ChromaDB collection query
            mock_context.deps.chromadb_collection.query = Mock(
                return_value=sample_chromadb_results
            )

            # Execute search
            results = await search_knowledgebase(mock_context, "convert images")

            # Verify results
            assert len(results) == 2
            assert isinstance(results[0], DocumentResult)
            assert results[0].title == "mrconvert"
            assert "<Start of mrconvert document>" in results[0].content
            assert "</Start of mrconvert document>" in results[0].content

    @pytest.mark.asyncio
    async def test_embedding_generation_failure(self, mock_context):
        """Test that embedding failures trigger retry."""
        with patch("src.agent.tools.EmbeddingService") as mock_embedding_class:
            mock_embedding = mock_embedding_class.return_value
            mock_embedding.generate_embedding = AsyncMock(
                side_effect=TimeoutError("Embedding timeout")
            )

            with pytest.raises(ModelRetry) as exc_info:
                await search_knowledgebase(mock_context, "test query")

            assert "temporarily failed" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_keyword_fallback_activation(self, mock_context):
        """Test keyword fallback when vector search fails."""
        with patch("src.agent.tools.EmbeddingService") as mock_embedding_class:
            mock_embedding = mock_embedding_class.return_value
            mock_embedding.generate_embedding = AsyncMock(return_value=[0.1] * 768)

            # Mock ChromaDB collection query failure
            mock_context.deps.chromadb_collection.query = Mock(
                side_effect=Exception("Vector search error")
            )

            # Mock keyword search with get()
            mock_context.deps.chromadb_collection.get = Mock(
                return_value={
                    "documents": ["Perform conversion between image formats"],
                    "metadatas": [{"title": "mrconvert", "keywords": "convert"}],
                }
            )

            # Execute search
            results = await search_knowledgebase(mock_context, "convert images")

            # Verify keyword search was called
            mock_context.deps.chromadb_collection.get.assert_called_once()
            assert len(results) > 0

    @pytest.mark.asyncio
    async def test_no_results_found(self, mock_context):
        """Test handling when no results are found."""
        with patch("src.agent.tools.EmbeddingService") as mock_embedding_class:
            mock_embedding = mock_embedding_class.return_value
            mock_embedding.generate_embedding = AsyncMock(return_value=[0.1] * 768)

            # Mock empty results from ChromaDB
            mock_context.deps.chromadb_collection.query = Mock(
                return_value={"documents": [[]], "metadatas": [[]], "distances": [[]]}
            )

            # Mock empty keyword search results too
            mock_context.deps.chromadb_collection.get = Mock(
                return_value={"documents": [], "metadatas": []}
            )

            results = await search_knowledgebase(mock_context, "nonexistent topic")
            assert results == []

    @pytest.mark.asyncio
    async def test_search_with_large_results(self, mock_context):
        """Test search handles large result sets properly."""
        with patch("src.agent.tools.EmbeddingService") as mock_embedding_class:
            mock_embedding = mock_embedding_class.return_value
            mock_embedding.generate_embedding = AsyncMock(return_value=[0.1] * 768)

            # Mock large result set
            large_docs = [f"Content {i}" for i in range(10)]
            large_metadata = [
                {"title": f"Doc{i}", "keywords": f"key{i}"} for i in range(10)
            ]
            large_distances = [0.1 * i for i in range(10)]

            mock_context.deps.chromadb_collection.query = Mock(
                return_value={
                    "documents": [large_docs],
                    "metadatas": [large_metadata],
                    "distances": [large_distances],
                }
            )

            results = await search_knowledgebase(mock_context, "test query")

            # Should only return top 2 results even if more are available
            assert len(results) == 2
            assert results[0].title == "Doc0"

    @pytest.mark.asyncio
    async def test_search_with_empty_query(self, mock_context):
        """Test search with empty query string."""
        results = await search_knowledgebase(mock_context, "")
        assert results == []

    @pytest.mark.asyncio
    async def test_no_chromadb_collection(self, mock_context):
        """Test handling when ChromaDB collection is not available."""
        mock_context.deps.chromadb_collection = None

        with patch("src.agent.tools.EmbeddingService") as mock_embedding_class:
            mock_embedding = mock_embedding_class.return_value
            mock_embedding.generate_embedding = AsyncMock(return_value=[0.1] * 768)

            results = await search_knowledgebase(mock_context, "test query")
            assert results == []


class TestKeywordFallback:
    """Test cases for keyword fallback search."""

    @pytest.mark.asyncio
    async def test_keyword_successful_search(self, mock_context):
        """Test successful keyword search."""
        # Mock ChromaDB get() for keyword search
        mock_context.deps.chromadb_collection.get = Mock(
            return_value={
                "documents": ["Installation instructions", "Convert images"],
                "metadatas": [
                    {"title": "Installation Guide", "keywords": "install setup"},
                    {"title": "mrconvert", "keywords": "convert image"},
                ],
            }
        )

        results = await _keyword_fallback_chromadb(mock_context, "installation guide")

        assert len(results) == 2
        assert isinstance(results[0], DocumentResult)
        assert results[0].title == "Installation Guide"

    @pytest.mark.asyncio
    async def test_keyword_database_error(self, mock_context):
        """Test keyword search raises ModelRetry on database error."""
        mock_context.deps.chromadb_collection.get = Mock(
            side_effect=Exception("Database connection error")
        )

        with pytest.raises(ModelRetry) as exc_info:
            await _keyword_fallback_chromadb(mock_context, "test query")

        assert "temporarily unavailable" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_keyword_no_collection(self, mock_context):
        """Test keyword search when collection is not available."""
        mock_context.deps.chromadb_collection = None

        results = await _keyword_fallback_chromadb(mock_context, "test")
        assert results == []


class TestFormatDocumentXML:
    """Test cases for document XML formatting."""

    def test_format_document_with_content(self):
        """Test formatting document with title and content."""
        formatted = _format_document_xml("Test Document", "This is the content")

        assert (
            formatted
            == "<Start of Test Document document>This is the content</Start of Test Document document>"
        )

    def test_format_document_empty_content(self):
        """Test formatting with empty content."""
        formatted = _format_document_xml("Test", "")
        assert formatted == "<Start of Test document></Start of Test document>"

    def test_format_document_none_values(self):
        """Test formatting with None values."""
        formatted = _format_document_xml(None, None)
        assert formatted == "<Start of Untitled document></Start of Untitled document>"

    def test_format_document_special_characters(self):
        """Test formatting with special characters."""
        formatted = _format_document_xml("Test & Demo", "Content with <tags>")
        assert "<Start of Test & Demo document>" in formatted
        assert "Content with <tags>" in formatted


class TestSearchWithSpecialCases:
    """Test special edge cases in search."""

    @pytest.mark.asyncio
    async def test_search_sanitizes_query(self, mock_context):
        """Test that search sanitizes special characters in query."""
        with patch("src.agent.tools.EmbeddingService") as mock_embedding_class:
            mock_embedding = mock_embedding_class.return_value
            mock_embedding.generate_embedding = AsyncMock(return_value=[0.1] * 768)

            mock_context.deps.chromadb_collection.query = Mock(
                return_value={"documents": [[]], "metadatas": [[]], "distances": [[]]}
            )

            # Mock keyword fallback too
            mock_context.deps.chromadb_collection.get = Mock(
                return_value={"documents": [], "metadatas": []}
            )

            # Query with special characters
            await search_knowledgebase(
                mock_context, "test @#$%^&*() query with <script>alert('xss')</script>"
            )

            # Verify the embedding was called with sanitized query
            call_args = mock_embedding.generate_embedding.call_args[0][0]
            assert "<script>" not in call_args
            assert "alert" in call_args  # Basic text preserved

    @pytest.mark.asyncio
    async def test_search_truncates_long_query(self, mock_context):
        """Test that very long queries are truncated."""
        with patch("src.agent.tools.EmbeddingService") as mock_embedding_class:
            mock_embedding = mock_embedding_class.return_value
            mock_embedding.generate_embedding = AsyncMock(return_value=[0.1] * 768)

            mock_context.deps.chromadb_collection.query = Mock(
                return_value={"documents": [[]], "metadatas": [[]], "distances": [[]]}
            )

            # Mock keyword fallback too
            mock_context.deps.chromadb_collection.get = Mock(
                return_value={"documents": [], "metadatas": []}
            )

            # Create a very long query (over 500 chars)
            long_query = "test " * 200  # 1000 characters

            await search_knowledgebase(mock_context, long_query)

            # Verify the query was truncated
            call_args = mock_embedding.generate_embedding.call_args[0][0]
            assert len(call_args) <= 500

    @pytest.mark.asyncio
    async def test_distance_threshold_filtering(self, mock_context):
        """Test that results are filtered by distance threshold."""
        with patch("src.agent.tools.EmbeddingService") as mock_embedding_class:
            mock_embedding = mock_embedding_class.return_value
            mock_embedding.generate_embedding = AsyncMock(return_value=[0.1] * 768)

            # Mock results with varying distances
            mock_context.deps.chromadb_collection.query = Mock(
                return_value={
                    "documents": [["Doc1", "Doc2", "Doc3"]],
                    "metadatas": [
                        [
                            {"title": "Close Match", "keywords": "test"},
                            {"title": "Medium Match", "keywords": "test"},
                            {"title": "Far Match", "keywords": "test"},
                        ]
                    ],
                    "distances": [
                        [0.1, 0.25, 0.5]
                    ],  # Only first two should pass threshold
                }
            )

            results = await search_knowledgebase(mock_context, "test")

            # Should filter out results with distance > 0.3 (similarity < 0.7)
            assert len(results) == 2
            assert results[0].title == "Close Match"
            assert results[1].title == "Medium Match"
