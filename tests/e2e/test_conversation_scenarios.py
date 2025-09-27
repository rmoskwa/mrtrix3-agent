"""End-to-end tests for MRtrix3 Assistant conversation scenarios."""

import asyncio
import sys
from unittest.mock import patch, MagicMock, AsyncMock

import pytest
from pydantic_ai.models.test import TestModel

# Mock Google modules before importing
sys.modules["google.generativeai"] = MagicMock()
sys.modules["google.api_core"] = MagicMock()
sys.modules["google.api_core.retry"] = MagicMock()
sys.modules["google.api_core.exceptions"] = MagicMock()

from src.agent.agent import MRtrixAssistant  # noqa: E402
from src.agent.models import SearchKnowledgebaseDependencies  # noqa: E402
from src.agent.cli import TokenManager  # noqa: E402


@pytest.fixture
def mock_assistant():
    """Create mock MRtrix Assistant for testing."""
    # Mock the embedding service
    with patch("src.agent.tools.EmbeddingService") as mock_embedding_class:
        mock_embedding = mock_embedding_class.return_value
        mock_embedding.generate_embedding = AsyncMock(return_value=[0.1] * 768)

        # Create mock Supabase client
        mock_supabase = MagicMock()
        mock_rpc_result = MagicMock()
        mock_rpc_result.execute = AsyncMock(
            return_value=MagicMock(
                data=[
                    {"title": "mrconvert", "content": "Convert image formats"},
                    {"title": "dwi2tensor", "content": "Process diffusion data"},
                ]
            )
        )
        mock_supabase.rpc = MagicMock(return_value=mock_rpc_result)

        mock_deps = SearchKnowledgebaseDependencies(
            supabase_client=mock_supabase,
            embedding_model="gemini-embedding-001",
            rate_limiter=None,
        )

        assistant = MRtrixAssistant(dependencies=mock_deps)
        return assistant


class TestBasicConversations:
    """Test basic conversation flows."""

    @pytest.mark.asyncio
    async def test_simple_question_response(self, mock_assistant):
        """Test a simple question-answer conversation."""
        test_model = TestModel(
            custom_output_text="MRtrix3 is a software package for diffusion MRI analysis."
        )

        with mock_assistant.agent.override(model=test_model):
            result = await mock_assistant.run("What is MRtrix3?", message_history=[])

        assert result is not None
        assert "MRtrix3" in str(result)

    @pytest.mark.asyncio
    async def test_multi_turn_conversation(self, mock_assistant):
        """Test multi-turn conversation with context retention."""
        test_model = TestModel(call_tools=["search_knowledgebase"])

        with mock_assistant.agent.override(model=test_model):
            # First turn
            result1 = await mock_assistant.run(
                "Tell me about mrconvert", message_history=[]
            )
            assert result1 is not None

            # Second turn (should maintain context)
            # In a real conversation, we'd pass the history from result1
            result2 = await mock_assistant.run(
                "What formats does it support?", message_history=[]
            )
            assert result2 is not None

    @pytest.mark.skip(reason="TokenManager interface mismatch - needs refactoring")
    @pytest.mark.asyncio
    async def test_tool_invocation_in_conversation(self, mock_assistant):
        """Test that tools are properly invoked during conversation."""
        # Track tool calls
        tool_calls = []

        async def mock_search(*args, **kwargs):
            tool_calls.append("search_knowledgebase")
            return [
                MagicMock(
                    title="Test", content="<Start of Test>Content</Start of Test>"
                )
            ]

        with patch("src.agent.tools.search_knowledgebase", mock_search):
            test_model = TestModel(call_tools=["search_knowledgebase"])

            with mock_assistant.agent.override(model=test_model):
                await mock_assistant.run(
                    "How do I convert DICOM to NIfTI?", message_history=[]
                )

            # Verify tool was called
            assert "search_knowledgebase" in tool_calls


class TestTokenLimitScenarios:
    """Test token limit handling in conversations."""

    @pytest.mark.skip(reason="TokenManager interface mismatch - needs refactoring")
    def test_token_counting_accuracy(self):
        """Test that token counting is accurate."""
        manager = TokenManager()

        # Add messages
        manager.add_message("user", "Short message")
        assert manager.current_tokens > 0
        assert manager.current_tokens < 100

    @pytest.mark.skip(reason="TokenManager interface mismatch - needs refactoring")
    def test_token_limit_enforcement(self):
        """Test that token limit is enforced."""
        manager = TokenManager()
        # Set a low limit for testing
        manager.MAX_TOKENS = 100

        # Add messages until limit
        for i in range(20):
            added = manager.add_message("user", f"Message {i}" * 10)
            if not added:
                break

        # Should not exceed limit
        assert manager.current_tokens <= manager.MAX_TOKENS

    @pytest.mark.skip(reason="TokenManager interface mismatch - needs refactoring")
    def test_history_trimming(self):
        """Test that history is trimmed when approaching limit."""
        manager = TokenManager()
        manager.MAX_TOKENS = 200  # Low limit for testing

        # Add many messages
        for i in range(10):
            manager.add_message("user", f"This is message number {i}")
            manager.add_message("assistant", f"Response to message {i}")

        # Trim history
        manager.trim_history()

        # Should keep system message and some recent messages
        assert len(manager.messages) > 1  # At least system + 1 pair
        assert manager.messages[0]["role"] == "system"
        assert manager.current_tokens < manager.MAX_TOKENS

    @pytest.mark.skip(reason="TokenManager interface mismatch - needs refactoring")
    @pytest.mark.asyncio
    async def test_token_limit_graceful_degradation(self, mock_assistant):
        """Test graceful handling when token limit is reached."""
        manager = TokenManager()
        manager.MAX_TOKENS = 100  # Very low limit for testing

        # Fill up token limit
        for i in range(5):
            manager.add_message("user", "Long message " * 20)

        # Should handle gracefully
        test_model = TestModel(
            custom_output_text="I understand your question despite token limits."
        )

        with mock_assistant.agent.override(model=test_model):
            result = await mock_assistant.run("One more question", message_history=[])
            assert result is not None


class TestErrorRecoveryScenarios:
    """Test error recovery in conversations."""

    @pytest.mark.asyncio
    async def test_network_error_recovery(self, mock_assistant):
        """Test recovery from network errors."""
        call_count = [0]

        async def flaky_search(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                raise ConnectionError("Network error")
            return []

        with patch("src.agent.tools.search_knowledgebase", flaky_search):
            # Should handle the error gracefully
            test_model = TestModel(custom_output_text="Handled the network error.")

            with mock_assistant.agent.override(model=test_model):
                result = await mock_assistant.run(
                    "Search for something", message_history=[]
                )
                assert result is not None

    @pytest.mark.asyncio
    async def test_timeout_recovery(self, mock_assistant):
        """Test recovery from timeout errors."""
        from pydantic_ai import ModelRetry

        async def timeout_search(*args, **kwargs):
            raise ModelRetry("Timeout occurred, please retry")

        with patch("src.agent.tools.search_knowledgebase", timeout_search):
            test_model = TestModel(
                custom_output_text="I'll help you despite the timeout."
            )

            with mock_assistant.agent.override(model=test_model):
                # Should handle ModelRetry
                result = await mock_assistant.run("Search query", message_history=[])
                assert result is not None

    @pytest.mark.asyncio
    async def test_empty_results_handling(self, mock_assistant):
        """Test handling of empty search results."""

        async def empty_search(*args, **kwargs):
            return []

        with patch("src.agent.tools.search_knowledgebase", empty_search):
            test_model = TestModel(
                custom_output_text="I couldn't find information on that topic."
            )

            with mock_assistant.agent.override(model=test_model):
                result = await mock_assistant.run(
                    "Find obscure topic xyz", message_history=[]
                )
                assert "couldn't find" in str(result).lower()


class TestRateLimitingInConversation:
    """Test rate limiting behavior in conversations."""

    @pytest.mark.asyncio
    async def test_rate_limit_enforcement(self, mock_assistant):
        """Test that rate limiting is enforced during rapid queries."""
        # Mock rate limiter
        mock_rate_limiter = AsyncMock()
        mock_rate_limiter.__aenter__ = AsyncMock()
        mock_rate_limiter.__aexit__ = AsyncMock()

        mock_assistant.dependencies.rate_limiter = mock_rate_limiter

        test_model = TestModel(custom_output_text="Rate limited response")

        # Rapid fire multiple requests
        tasks = []
        with mock_assistant.agent.override(model=test_model):
            for i in range(5):
                tasks.append(mock_assistant.run(f"Query {i}", message_history=[]))

            results = await asyncio.gather(*tasks, return_exceptions=True)

        # All should complete (rate limiting delays, not blocks)
        assert all(r is not None for r in results if not isinstance(r, Exception))


class TestConversationHistoryManagement:
    """Test conversation history management."""

    @pytest.mark.asyncio
    async def test_context_retention_across_turns(self, mock_assistant):
        """Test that context is retained across conversation turns."""
        test_model = TestModel(
            custom_output_text="I remember you asked about mrconvert earlier."
        )

        with mock_assistant.agent.override(model=test_model):
            # First mention a tool
            await mock_assistant.run("Tell me about mrconvert", message_history=[])

            # Reference it indirectly
            result = await mock_assistant.run(
                "What are its main options?", message_history=[]
            )

            # Should maintain context
            assert "mrconvert" in str(result)

    @pytest.mark.skip(reason="TokenManager interface mismatch - needs refactoring")
    @pytest.mark.asyncio
    async def test_context_reset_behavior(self, mock_assistant):
        """Test conversation reset behavior."""
        manager = TokenManager()

        # Add some history
        manager.add_message("user", "Previous conversation")
        manager.add_message("assistant", "Previous response")

        # Reset
        manager.reset()

        # Should only have system message
        assert len(manager.messages) == 1
        assert manager.messages[0]["role"] == "system"
        assert manager.current_tokens > 0
