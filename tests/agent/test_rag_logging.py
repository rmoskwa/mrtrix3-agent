"""Test RAG search functionality."""

from unittest.mock import Mock, AsyncMock, patch
import pytest

from src.agent.tools import search_knowledgebase
from src.agent.models import SearchKnowledgebaseDependencies


@pytest.mark.asyncio
class TestRAGSearch:
    """Test RAG search functionality."""

    async def test_successful_search_returns_top_results(self):
        """Test that search returns top 2 results when documents are found."""
        # Setup mock dependencies
        mock_deps = Mock(spec=SearchKnowledgebaseDependencies)
        mock_collection = Mock()
        mock_embedding_service = AsyncMock()

        # Mock ChromaDB query results
        mock_collection.query.return_value = {
            "documents": [
                ["Document 1 content", "Document 2 content", "Document 3 content"]
            ],
            "metadatas": [
                [
                    {"title": "MRtrix3 Command Reference: dwi2tensor"},
                    {"title": "Tutorial: DTI Analysis"},
                    {"title": "Guide: Tensor Fitting"},
                ]
            ],
            "distances": [[0.1, 0.2, 0.25]],  # Low distances = high similarity
        }

        mock_deps.chromadb_collection = mock_collection
        mock_deps.embedding_service = mock_embedding_service
        mock_deps.embedding_model = "test-model"

        # Mock embedding generation
        mock_embedding_service.generate_embedding.return_value = [0.1] * 768

        # Create mock context
        ctx = Mock()
        ctx.deps = mock_deps
        ctx.retry = 0

        # Mock get_session_logger to return None (no session logging)
        with patch("src.agent.tools.get_session_logger", return_value=None):
            # Run the search
            results = await search_knowledgebase(ctx, "How do I calculate DTI tensors?")

        # Check that results were returned (now combined in a single result)
        assert len(results) == 1
        assert results[0].title == "Results for query: How do I calculate DTI tensors?"

        # Verify the content is properly formatted with query separators and XML tags
        assert (
            "--- Results from query: How do I calculate DTI tensors? ---"
            in results[0].content
        )
        assert (
            "--- End of results from query: How do I calculate DTI tensors? ---"
            in results[0].content
        )
        assert (
            "<Start of MRtrix3 Command Reference: dwi2tensor document>"
            in results[0].content
        )
        assert (
            "</Start of MRtrix3 Command Reference: dwi2tensor document>"
            in results[0].content
        )
        assert "<Start of Tutorial: DTI Analysis document>" in results[0].content

    async def test_no_documents_retrieved(self):
        """Test behavior when no documents are retrieved."""
        # Setup mock dependencies
        mock_deps = Mock(spec=SearchKnowledgebaseDependencies)
        mock_collection = Mock()
        mock_embedding_service = AsyncMock()

        # Mock ChromaDB query with no results
        mock_collection.query.return_value = {
            "documents": [[]],
            "metadatas": [[]],
            "distances": [[]],
        }

        mock_deps.chromadb_collection = mock_collection
        mock_deps.embedding_service = mock_embedding_service
        mock_deps.embedding_model = "test-model"

        # Mock embedding generation
        mock_embedding_service.generate_embedding.return_value = [0.1] * 768

        # Create mock context
        ctx = Mock()
        ctx.deps = mock_deps
        ctx.retry = 0

        # Mock get_session_logger to return None (no session logging)
        with patch("src.agent.tools.get_session_logger", return_value=None):
            # Run the search
            results = await search_knowledgebase(ctx, "Unknown query")

        # Check that a result with "No matching documents found" message is returned
        assert len(results) == 1
        assert results[0].title == "No results for query: Unknown query"
        assert "No matching documents found" in results[0].content
        assert "--- Results from query: Unknown query ---" in results[0].content

    async def test_filters_by_distance_threshold(self):
        """Test that results are filtered by distance threshold."""
        # Setup mock dependencies
        mock_deps = Mock(spec=SearchKnowledgebaseDependencies)
        mock_collection = Mock()
        mock_embedding_service = AsyncMock()

        # Mock ChromaDB query results with varying distances
        mock_collection.query.return_value = {
            "documents": [["Doc 1", "Doc 2", "Doc 3"]],
            "metadatas": [
                [
                    {"title": "Close Match"},
                    {"title": "Medium Match"},
                    {"title": "Far Match"},
                ]
            ],
            "distances": [[0.1, 0.25, 0.5]],  # Last one exceeds default threshold
        }

        mock_deps.chromadb_collection = mock_collection
        mock_deps.embedding_service = mock_embedding_service
        mock_deps.embedding_model = "test-model"

        # Mock embedding generation
        mock_embedding_service.generate_embedding.return_value = [0.1] * 768

        # Create mock context
        ctx = Mock()
        ctx.deps = mock_deps
        ctx.retry = 0

        # Mock get_session_logger to return None
        with patch("src.agent.tools.get_session_logger", return_value=None):
            # Run the search
            results = await search_knowledgebase(ctx, "test query")

        # Should return 1 combined result with first 2 documents (the third has distance > threshold)
        assert len(results) == 1
        assert results[0].title == "Results for query: test query"
        assert "<Start of Close Match document>" in results[0].content
        assert "<Start of Medium Match document>" in results[0].content
        assert "<Start of Far Match document>" not in results[0].content

    async def test_handles_empty_query(self):
        """Test that empty queries return empty results."""
        # Create mock context
        ctx = Mock()
        ctx.deps = Mock(spec=SearchKnowledgebaseDependencies)

        # Mock get_session_logger to return None
        with patch("src.agent.tools.get_session_logger", return_value=None):
            # Run with empty query
            results = await search_knowledgebase(ctx, "")

        # Should return empty list
        assert len(results) == 0

    async def test_sanitizes_query_input(self):
        """Test that queries are sanitized for safety."""
        # Setup mock dependencies
        mock_deps = Mock(spec=SearchKnowledgebaseDependencies)
        mock_collection = Mock()
        mock_embedding_service = AsyncMock()

        mock_collection.query.return_value = {
            "documents": [[]],
            "metadatas": [[]],
            "distances": [[]],
        }

        mock_deps.chromadb_collection = mock_collection
        mock_deps.embedding_service = mock_embedding_service
        mock_deps.embedding_model = "test-model"

        # Mock embedding generation
        mock_embedding_service.generate_embedding.return_value = [0.1] * 768

        # Create mock context
        ctx = Mock()
        ctx.deps = mock_deps
        ctx.retry = 0

        # Mock get_session_logger to return None
        with patch("src.agent.tools.get_session_logger", return_value=None):
            # Run with query containing special characters
            malicious_query = "test'; DROP TABLE--"
            await search_knowledgebase(ctx, malicious_query)

        # Verify the query was sanitized when passed to embedding service
        called_query = mock_embedding_service.generate_embedding.call_args[0][0]
        # Special characters should be replaced with spaces
        assert "'" not in called_query  # Single quote removed
        assert ";" not in called_query  # Semicolon removed
        assert "test" in called_query  # Normal text preserved
