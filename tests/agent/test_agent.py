"""Unit tests for MRtrix3 Assistant agent."""

import os
from unittest.mock import MagicMock, AsyncMock, patch

import pytest
from pydantic_ai.models.test import TestModel

from src.agent.agent import MRtrixAssistant
from src.agent.dependencies import validate_environment
from src.agent.models import SearchKnowledgebaseDependencies


class TestMRtrixAssistant:
    """Test MRtrixAssistant class."""

    def test_agent_initialization(self):
        """Test MRtrixAssistant initialization with mock dependencies."""
        mock_deps = SearchKnowledgebaseDependencies(
            supabase_client=MagicMock(),
            embedding_model=os.getenv("EMBEDDING_MODEL"),
            rate_limiter=MagicMock(),
        )

        assistant = MRtrixAssistant(dependencies=mock_deps)

        assert assistant.dependencies == mock_deps
        assert assistant.agent is not None
        assert "MRtrix3 documentation assistant" in assistant.system_prompt

    @pytest.mark.asyncio
    async def test_agent_run_with_test_model(self):
        """Test agent run method using PydanticAI TestModel."""
        # Mock the embedding service to prevent actual API calls
        with patch("src.agent.tools.EmbeddingService") as mock_embedding_class:
            mock_embedding = mock_embedding_class.return_value
            mock_embedding.generate_embedding = AsyncMock(
                return_value=[0.1] * 768  # Return dummy embedding
            )

            # Create a mock Supabase client with proper async mocks
            mock_supabase = MagicMock()

            # Mock RPC call
            mock_rpc_result = MagicMock()
            mock_rpc_result.execute = AsyncMock(return_value=MagicMock(data=[]))
            mock_supabase.rpc = MagicMock(return_value=mock_rpc_result)

            # Mock BM25 fallback table operations
            mock_table = MagicMock()
            mock_table.select = MagicMock(return_value=mock_table)
            mock_table.ilike = MagicMock(return_value=mock_table)
            mock_table.limit = MagicMock(return_value=mock_table)
            mock_table.execute = AsyncMock(return_value=MagicMock(data=[]))
            mock_supabase.from_ = MagicMock(return_value=mock_table)

            mock_deps = SearchKnowledgebaseDependencies(
                supabase_client=mock_supabase,
                embedding_model=os.getenv("EMBEDDING_MODEL"),
                rate_limiter=MagicMock(),
            )

            assistant = MRtrixAssistant(dependencies=mock_deps)

            test_model = TestModel()

            with assistant.agent.override(model=test_model):
                result = await assistant.run("What is MRtrix3?")

            assert result is not None


class TestEnvironmentValidation:
    """Test environment validation function."""

    def test_validate_environment_success(self):
        """Test successful environment validation."""
        env_vars = {
            "SUPABASE_URL": "https://test.supabase.co",
            "SUPABASE_KEY": "test_key",
            "GEMINI_API_KEY": "gemini_key",
            "GOOGLE_API_KEY_EMBEDDING": "google_key",
            "EMBEDDING_MODEL": "gemini-embedding-001",
            "EMBEDDING_DIMENSIONS": "768",
        }

        with patch.dict(os.environ, env_vars, clear=True):
            result = validate_environment()

        assert result["SUPABASE_URL"] == "https://test.supabase.co"
        assert result["SUPABASE_KEY"] == "test_key"
        assert result["GEMINI_API_KEY"] == "gemini_key"
        assert result["GOOGLE_API_KEY_EMBEDDING"] == "google_key"
        assert result["EMBEDDING_MODEL"] == "gemini-embedding-001"
        assert result["EMBEDDING_DIMENSIONS"] == 768

    def test_validate_environment_missing_required(self):
        """Test validation fails when required variables are missing."""
        env_vars = {
            "SUPABASE_URL": "https://test.supabase.co",
            "GEMINI_API_KEY": "gemini_key",
            "GOOGLE_API_KEY_EMBEDDING": "google_key",
            "EMBEDDING_MODEL": "gemini-embedding-001",
            "EMBEDDING_DIMENSIONS": "768",
        }

        with patch.dict(os.environ, env_vars, clear=False):
            with patch.dict(os.environ, {"SUPABASE_KEY": ""}, clear=False):
                with pytest.raises(ValueError, match="SUPABASE_KEY is not set"):
                    validate_environment()

    def test_validate_environment_wrong_embedding_model(self):
        """Test validation fails with wrong embedding model."""
        env_vars = {
            "SUPABASE_URL": "https://test.supabase.co",
            "SUPABASE_KEY": "test_key",
            "GEMINI_API_KEY": "gemini_key",
            "GOOGLE_API_KEY_EMBEDDING": "google_key",
            "EMBEDDING_MODEL": "wrong-model",
            "EMBEDDING_DIMENSIONS": "768",
        }

        with patch.dict(os.environ, env_vars, clear=True):
            with pytest.raises(ValueError, match="EMBEDDING_MODEL must be"):
                validate_environment()

    def test_validate_environment_wrong_dimensions(self):
        """Test validation fails with wrong embedding dimensions."""
        env_vars = {
            "SUPABASE_URL": "https://test.supabase.co",
            "SUPABASE_KEY": "test_key",
            "GEMINI_API_KEY": "gemini_key",
            "GOOGLE_API_KEY_EMBEDDING": "google_key",
            "EMBEDDING_MODEL": "gemini-embedding-001",
            "EMBEDDING_DIMENSIONS": "512",
        }

        with patch.dict(os.environ, env_vars, clear=True):
            with pytest.raises(ValueError, match="EMBEDDING_DIMENSIONS must be"):
                validate_environment()
