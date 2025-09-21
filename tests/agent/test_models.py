"""Unit tests for MRtrix3 agent Pydantic models."""

import pytest
from pydantic import ValidationError

from src.agent.models import (
    AgentConfiguration,
    DocumentResult,
    SearchKnowledgebaseDependencies,
    SearchToolParameters,
)


class TestDocumentResult:
    """Test DocumentResult model."""

    def test_valid_document_result(self):
        """Test creating a valid DocumentResult instance."""
        doc = DocumentResult(
            title="MRtrix3 Command Guide",
            content="This is the content of the document.",
        )
        assert doc.title == "MRtrix3 Command Guide"
        assert doc.content == "This is the content of the document."

    def test_missing_required_fields(self):
        """Test that missing required fields raise validation error."""
        with pytest.raises(ValidationError):
            DocumentResult(title="Only Title")

        with pytest.raises(ValidationError):
            DocumentResult(content="Only Content")


class TestSearchToolParameters:
    """Test SearchToolParameters model."""

    def test_valid_search_parameters(self):
        """Test creating valid SearchToolParameters."""
        params = SearchToolParameters(query="How to use mrconvert?")
        assert params.query == "How to use mrconvert?"

    def test_missing_query(self):
        """Test that missing query raises validation error."""
        with pytest.raises(ValidationError):
            SearchToolParameters()


class TestSearchKnowledgebaseDependencies:
    """Test SearchKnowledgebaseDependencies model."""

    def test_valid_dependencies(self):
        """Test creating valid SearchKnowledgebaseDependencies with mock objects."""
        deps = SearchKnowledgebaseDependencies(
            supabase_client="mock_client",
            embedding_model="mock_model",
            rate_limiter="mock_limiter",
        )
        assert deps.supabase_client == "mock_client"
        assert deps.embedding_model == "mock_model"
        assert deps.rate_limiter == "mock_limiter"

    def test_missing_dependencies(self):
        """Test that missing required dependencies raise validation error."""
        with pytest.raises(ValidationError):
            SearchKnowledgebaseDependencies()
            # Missing supabase_client which is required


class TestAgentConfiguration:
    """Test AgentConfiguration model."""

    def test_default_configuration(self):
        """Test AgentConfiguration with default values."""
        config = AgentConfiguration()
        assert config.model_name == "gemini-2.5-flash"
        assert config.embedding_model == "gemini-embedding-001"
        assert config.embedding_dimensions == 768
        assert config.max_search_results == 3
        assert config.return_top_n == 2
        assert config.system_prompt == ""

    def test_custom_configuration(self):
        """Test AgentConfiguration with custom values."""
        config = AgentConfiguration(
            model_name="custom-model",
            embedding_model="custom-embedding",
            embedding_dimensions=512,
            max_search_results=5,
            return_top_n=3,
            system_prompt="Custom system prompt",
        )
        assert config.model_name == "custom-model"
        assert config.embedding_model == "custom-embedding"
        assert config.embedding_dimensions == 512
        assert config.max_search_results == 5
        assert config.return_top_n == 3
        assert config.system_prompt == "Custom system prompt"

    def test_invalid_type_configuration(self):
        """Test that invalid types raise validation error."""
        with pytest.raises(ValidationError):
            AgentConfiguration(embedding_dimensions="not_an_int")

        with pytest.raises(ValidationError):
            AgentConfiguration(max_search_results="not_an_int")
