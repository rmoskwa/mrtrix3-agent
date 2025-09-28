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
                result = await assistant.run("What is MRtrix3?", message_history=[])

            assert result is not None

    @pytest.mark.asyncio
    async def test_agent_with_search_tool_call(self):
        """Test agent triggering search tool with TestModel."""
        # Mock the embedding service
        with patch("src.agent.tools.EmbeddingService") as mock_embedding_class:
            mock_embedding = mock_embedding_class.return_value
            mock_embedding.generate_embedding = AsyncMock(return_value=[0.1] * 768)

            # Mock successful search results
            mock_supabase = MagicMock()
            test_results = [
                {
                    "title": "MRtrix3 Overview",
                    "content": "MRtrix3 is a software package...",
                },
                {"title": "Installation", "content": "Installing MRtrix3..."},
            ]

            mock_rpc_result = MagicMock()
            mock_rpc_result.execute = AsyncMock(
                return_value=MagicMock(data=test_results)
            )
            mock_supabase.rpc = MagicMock(return_value=mock_rpc_result)

            mock_deps = SearchKnowledgebaseDependencies(
                supabase_client=mock_supabase,
                embedding_model="gemini-embedding-001",
                rate_limiter=None,
            )

            assistant = MRtrixAssistant(dependencies=mock_deps)

            # Use TestModel with custom output
            test_model = TestModel(
                custom_output_text="I found information about MRtrix3 in the documentation."
            )

            with assistant.agent.override(model=test_model):
                result = await assistant.run(
                    "How do I install MRtrix3?", message_history=[]
                )

            assert result is not None
            assert "MRtrix3" in str(result)

    @pytest.mark.asyncio
    async def test_agent_handles_empty_search_results(self):
        """Test agent behavior when search returns no results."""
        with patch("src.agent.tools.EmbeddingService") as mock_embedding_class:
            mock_embedding = mock_embedding_class.return_value
            mock_embedding.generate_embedding = AsyncMock(return_value=[0.1] * 768)

            # Mock empty search results
            mock_supabase = MagicMock()
            mock_rpc_result = MagicMock()
            mock_rpc_result.execute = AsyncMock(return_value=MagicMock(data=[]))
            mock_supabase.rpc = MagicMock(return_value=mock_rpc_result)

            # Mock BM25 fallback also returns empty
            mock_table = MagicMock()
            mock_table.select = MagicMock(return_value=mock_table)
            mock_table.ilike = MagicMock(return_value=mock_table)
            mock_table.limit = MagicMock(return_value=mock_table)
            mock_table.execute = AsyncMock(return_value=MagicMock(data=[]))
            mock_supabase.from_ = MagicMock(return_value=mock_table)

            mock_deps = SearchKnowledgebaseDependencies(
                supabase_client=mock_supabase,
                embedding_model="gemini-embedding-001",
                rate_limiter=None,
            )

            assistant = MRtrixAssistant(dependencies=mock_deps)

            test_model = TestModel(
                custom_output_text="I couldn't find any documentation on that topic."
            )

            with assistant.agent.override(model=test_model):
                result = await assistant.run("Unknown topic xyz123", message_history=[])

            assert result is not None
            result_str = str(result).lower()
            assert "couldn't find" in result_str or "no" in result_str


class TestEnvironmentValidation:
    """Test environment validation function."""

    def test_validate_environment_success(self):
        """Test successful environment validation."""
        env_vars = {
            "GOOGLE_API_KEY": "google_key",
            "GOOGLE_API_KEY_EMBEDDING": "google_key_embed",
        }

        with patch("src.agent.dependencies.load_dotenv"):
            with patch.dict(os.environ, env_vars, clear=True):
                result = validate_environment()

        # Supabase credentials are now hardcoded
        assert result["SUPABASE_URL"].startswith("https://")
        assert result["SUPABASE_URL"].endswith(".supabase.co")
        assert result["SUPABASE_KEY"] is not None
        assert result["GOOGLE_API_KEY"] == "google_key"
        assert result["GOOGLE_API_KEY_EMBEDDING"] == "google_key_embed"
        # Embedding config is hardcoded
        assert result["EMBEDDING_MODEL"] == "gemini-embedding-001"
        assert result["EMBEDDING_DIMENSIONS"] == 768

    def test_validate_environment_missing_required(self):
        """Test validation fails when required variables are missing."""
        # Clear all environment variables to ensure GOOGLE_API_KEY is missing
        with patch("src.agent.dependencies.load_dotenv"):
            with patch.dict(os.environ, {}, clear=True):
                with pytest.raises(
                    ValueError, match="Required API key GOOGLE_API_KEY is not set"
                ):
                    validate_environment()

    def test_validate_environment_uses_hardcoded_embedding_model(self):
        """Test that validation always uses hardcoded embedding model."""
        env_vars = {
            "GOOGLE_API_KEY": "google_key",
            "EMBEDDING_MODEL": "wrong-model",  # This should be ignored
            "EMBEDDING_DIMENSIONS": "512",  # This should be ignored
        }

        with patch("src.agent.dependencies.load_dotenv"):
            with patch.dict(os.environ, env_vars, clear=True):
                result = validate_environment()
                # Should use hardcoded values regardless of env
                assert result["EMBEDDING_MODEL"] == "gemini-embedding-001"
                assert result["EMBEDDING_DIMENSIONS"] == 768

    def test_validate_environment_default_embedding_key(self):
        """Test that GOOGLE_API_KEY_EMBEDDING defaults to GOOGLE_API_KEY."""
        env_vars = {
            "GOOGLE_API_KEY": "google_key",
            # Not providing GOOGLE_API_KEY_EMBEDDING
        }

        with patch("src.agent.dependencies.load_dotenv"):
            with patch.dict(os.environ, env_vars, clear=True):
                result = validate_environment()
                # Should default to GOOGLE_API_KEY
                assert result["GOOGLE_API_KEY_EMBEDDING"] == "google_key"
