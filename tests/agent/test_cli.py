"""Unit tests for CLI interface."""

import os
from unittest.mock import Mock, patch, AsyncMock
import pytest

from src.agent.cli import TokenManager, start_conversation


@pytest.mark.asyncio
class TestTokenManager:
    """Test TokenManager functionality."""

    async def test_token_counting(self):
        """Test token counting with mocked Gemini API."""
        with patch("google.generativeai.GenerativeModel") as MockModel:
            mock_model = Mock()
            mock_result = Mock()
            mock_result.total_tokens = 100
            mock_model.count_tokens.return_value = mock_result
            MockModel.return_value = mock_model

            manager = TokenManager()
            tokens = await manager.count_tokens("test message")

            assert tokens == 100
            mock_model.count_tokens.assert_called_once_with("test message")

    async def test_token_counting_fallback(self):
        """Test token counting fallback when API fails."""
        with patch("google.generativeai.GenerativeModel") as MockModel:
            mock_model = Mock()
            mock_model.count_tokens.side_effect = Exception("API error")
            MockModel.return_value = mock_model

            manager = TokenManager()
            tokens = await manager.count_tokens("test message" * 10)

            assert tokens > 0
            assert tokens == len("test message" * 10) // 4

    async def test_add_message_within_limit(self):
        """Test adding messages within token limit."""
        with patch("google.generativeai.GenerativeModel") as MockModel:
            mock_model = Mock()
            mock_result = Mock()
            mock_result.total_tokens = 100
            mock_model.count_tokens.return_value = mock_result
            MockModel.return_value = mock_model

            manager = TokenManager()
            result = await manager.add_message("test message")

            assert result is True
            assert manager.total_tokens == 100
            assert len(manager.message_history) == 1

    async def test_add_message_exceeds_limit(self):
        """Test handling when message exceeds token limit."""
        with patch("google.generativeai.GenerativeModel") as MockModel:
            mock_model = Mock()
            mock_result = Mock()
            mock_result.total_tokens = TokenManager.MAX_TOKENS + 1
            mock_model.count_tokens.return_value = mock_result
            MockModel.return_value = mock_model

            manager = TokenManager()
            result = await manager.add_message("huge message")

            assert result is False

    async def test_trim_history(self):
        """Test trimming old messages when approaching limit."""
        with patch("google.generativeai.GenerativeModel") as MockModel:
            mock_model = Mock()
            mock_result = Mock()
            mock_result.total_tokens = 100
            mock_model.count_tokens.return_value = mock_result
            MockModel.return_value = mock_model

            manager = TokenManager()
            manager.total_tokens = int(TokenManager.MAX_TOKENS * 0.9)
            manager.message_history = [("old message", 10000) for _ in range(10)]

            await manager._trim_history()

            # Check that trimming removed messages - since it's at 0.9 * MAX (450,000)
            # and trim will bring it to <= 0.8 * MAX (400,000)
            assert manager.total_tokens <= TokenManager.MAX_TOKENS * 0.8
            assert len(manager.message_history) < 10  # Some messages should be removed

    async def test_reset(self):
        """Test resetting token manager."""
        manager = TokenManager()
        manager.total_tokens = 1000
        manager.message_history = [("msg", 100)]

        manager.reset()

        assert manager.total_tokens == 0
        assert len(manager.message_history) == 0


@pytest.mark.asyncio
class TestStartConversation:
    """Test main conversation loop."""

    @patch.dict(os.environ, {"COLLECT_LOGS": "false"})
    @patch("asyncio.get_event_loop")
    @patch("src.agent.cli.ThreadPoolExecutor")
    @patch("src.agent.cli.create_async_dependencies")
    @patch("src.agent.cli.MRtrixAssistant")
    @patch("src.agent.cli.TokenManager")
    @patch("src.agent.cli.console")
    async def test_exit_command(
        self,
        mock_console,
        MockTokenManager,
        MockAssistant,
        mock_create_deps,
        MockExecutor,
        mock_get_loop,
    ):
        """Test /exit command handling."""
        # Setup mock executor
        mock_executor = Mock()
        MockExecutor.return_value = mock_executor
        mock_executor.shutdown = Mock()

        # Setup mock loop
        mock_loop = AsyncMock()
        mock_get_loop.return_value = mock_loop
        mock_loop.run_in_executor.side_effect = ["/exit"]

        mock_deps = AsyncMock()
        mock_create_deps.return_value = mock_deps

        mock_agent = AsyncMock()
        MockAssistant.return_value = mock_agent

        mock_token_mgr = AsyncMock()
        MockTokenManager.return_value = mock_token_mgr

        await start_conversation()

        mock_create_deps.assert_called_once()
        mock_agent.run.assert_not_called()
        mock_executor.shutdown.assert_called_once_with(wait=False, cancel_futures=True)

    @patch.dict(os.environ, {"COLLECT_LOGS": "false"})
    @patch("asyncio.get_event_loop")
    @patch("src.agent.cli.ThreadPoolExecutor")
    @patch("src.agent.cli.create_async_dependencies")
    @patch("src.agent.cli.MRtrixAssistant")
    @patch("src.agent.cli.TokenManager")
    @patch("src.agent.cli.console")
    async def test_ctrl_c_exit(
        self,
        mock_console,
        MockTokenManager,
        MockAssistant,
        mock_create_deps,
        MockExecutor,
        mock_get_loop,
    ):
        """Test Ctrl+C exits immediately."""
        # Setup mock executor
        mock_executor = Mock()
        MockExecutor.return_value = mock_executor
        mock_executor.shutdown = Mock()

        # Setup mock loop
        mock_loop = AsyncMock()
        mock_get_loop.return_value = mock_loop
        mock_loop.run_in_executor.side_effect = KeyboardInterrupt()

        mock_deps = AsyncMock()
        mock_create_deps.return_value = mock_deps

        mock_agent = AsyncMock()
        MockAssistant.return_value = mock_agent

        mock_token_mgr = AsyncMock()
        MockTokenManager.return_value = mock_token_mgr

        await start_conversation()

        # Verify executor was properly shut down
        mock_executor.shutdown.assert_called_once_with(wait=False, cancel_futures=True)
        assert mock_console.print.call_count >= 2  # Welcome message and instruction

    @patch.dict(os.environ, {"COLLECT_LOGS": "false"})
    @patch("asyncio.get_event_loop")
    @patch("src.agent.cli.ThreadPoolExecutor")
    @patch("src.agent.cli.create_async_dependencies")
    @patch("src.agent.cli.MRtrixAssistant")
    @patch("src.agent.cli.TokenManager")
    @patch("src.agent.cli.console")
    async def test_conversation_flow(
        self,
        mock_console,
        MockTokenManager,
        MockAssistant,
        mock_create_deps,
        MockExecutor,
        mock_get_loop,
    ):
        """Test normal conversation flow."""
        # Setup mock executor
        mock_executor = Mock()
        MockExecutor.return_value = mock_executor
        mock_executor.shutdown = Mock()

        # Setup mock loop to return input values
        mock_loop = AsyncMock()
        mock_get_loop.return_value = mock_loop
        mock_loop.run_in_executor.side_effect = ["Hello", "/exit"]

        mock_deps = AsyncMock()
        mock_create_deps.return_value = mock_deps

        mock_agent = AsyncMock()
        mock_result = Mock()
        mock_result.output = "Hello! I'm the MRtrix3 Assistant."
        mock_result.all_messages.return_value = []  # Add the all_messages method
        mock_agent.run.return_value = mock_result
        MockAssistant.return_value = mock_agent

        mock_token_mgr = AsyncMock()
        mock_token_mgr.add_message.return_value = True
        MockTokenManager.return_value = mock_token_mgr

        await start_conversation()

        mock_agent.run.assert_called_once_with("Hello", message_history=[])
        assert mock_token_mgr.add_message.call_count == 2
        mock_executor.shutdown.assert_called_once_with(wait=False, cancel_futures=True)

    @patch.dict(os.environ, {"COLLECT_LOGS": "false"})
    @patch("asyncio.get_event_loop")
    @patch("src.agent.cli.ThreadPoolExecutor")
    @patch("src.agent.cli.create_async_dependencies")
    @patch("src.agent.cli.MRtrixAssistant")
    @patch("src.agent.cli.TokenManager")
    @patch("src.agent.cli.console")
    async def test_token_limit_handling(
        self,
        mock_console,
        MockTokenManager,
        MockAssistant,
        mock_create_deps,
        MockExecutor,
        mock_get_loop,
    ):
        """Test handling when token limit is reached."""
        # Setup mock executor
        mock_executor = Mock()
        MockExecutor.return_value = mock_executor
        mock_executor.shutdown = Mock()

        # Setup mock loop
        mock_loop = AsyncMock()
        mock_get_loop.return_value = mock_loop
        mock_loop.run_in_executor.side_effect = ["Long message", "/exit"]

        mock_deps = AsyncMock()
        mock_create_deps.return_value = mock_deps

        mock_agent = AsyncMock()
        mock_result = Mock()
        mock_result.output = "Response"
        mock_result.all_messages.return_value = []  # Add the all_messages method
        mock_agent.run.return_value = mock_result
        MockAssistant.return_value = mock_agent

        mock_token_mgr = AsyncMock()
        mock_token_mgr.add_message.side_effect = [False, True, True]
        mock_token_mgr.reset = Mock()
        MockTokenManager.return_value = mock_token_mgr

        await start_conversation()

        mock_token_mgr.reset.assert_called_once()
        mock_console.print.assert_any_call(
            "[yellow]Token limit reached. Starting new session.[/yellow]"
        )

    @patch.dict(os.environ, {"COLLECT_LOGS": "false"})
    @patch("asyncio.get_event_loop")
    @patch("src.agent.cli.ThreadPoolExecutor")
    @patch("src.agent.cli.create_async_dependencies")
    @patch("src.agent.cli.MRtrixAssistant")
    @patch("src.agent.cli.TokenManager")
    @patch("src.agent.cli.console")
    async def test_error_handling(
        self,
        mock_console,
        MockTokenManager,
        MockAssistant,
        mock_create_deps,
        MockExecutor,
        mock_get_loop,
    ):
        """Test error handling in conversation loop."""
        # Setup mock executor
        mock_executor = Mock()
        MockExecutor.return_value = mock_executor
        mock_executor.shutdown = Mock()

        # Setup mock loop
        mock_loop = AsyncMock()
        mock_get_loop.return_value = mock_loop
        mock_loop.run_in_executor.side_effect = ["Error test", "/exit"]

        mock_deps = AsyncMock()
        mock_create_deps.return_value = mock_deps

        mock_agent = AsyncMock()
        mock_agent.run.side_effect = Exception("Test error")
        MockAssistant.return_value = mock_agent

        mock_token_mgr = AsyncMock()
        mock_token_mgr.add_message.return_value = True
        MockTokenManager.return_value = mock_token_mgr

        await start_conversation()

        # Check that the error was handled and a user-friendly message was displayed
        # The exact message depends on the error mapper, but it should contain helpful text
        calls = [str(call) for call in mock_console.print.call_args_list]
        error_message_found = any(
            "unexpected issue" in str(call).lower() or "try again" in str(call).lower()
            for call in calls
        )
        assert (
            error_message_found
        ), f"Expected user-friendly error message, got calls: {calls}"

    @patch.dict(os.environ, {"COLLECT_LOGS": "false"})
    @patch("asyncio.get_event_loop")
    @patch("src.agent.cli.ThreadPoolExecutor")
    @patch("src.agent.cli.create_async_dependencies")
    @patch("src.agent.cli.MRtrixAssistant")
    @patch("src.agent.cli.TokenManager")
    @patch("src.agent.cli.console")
    async def test_empty_input_handling(
        self,
        mock_console,
        MockTokenManager,
        MockAssistant,
        mock_create_deps,
        MockExecutor,
        mock_get_loop,
    ):
        """Test handling of empty input."""
        # Setup mock executor
        mock_executor = Mock()
        MockExecutor.return_value = mock_executor
        mock_executor.shutdown = Mock()

        # Setup mock loop
        mock_loop = AsyncMock()
        mock_get_loop.return_value = mock_loop
        mock_loop.run_in_executor.side_effect = ["", "Hello", "/exit"]

        mock_deps = AsyncMock()
        mock_create_deps.return_value = mock_deps

        mock_agent = AsyncMock()
        mock_result = Mock()
        mock_result.output = "Response"
        mock_result.all_messages.return_value = []  # Add the all_messages method
        mock_agent.run.return_value = mock_result
        MockAssistant.return_value = mock_agent

        mock_token_mgr = AsyncMock()
        mock_token_mgr.add_message.return_value = True
        MockTokenManager.return_value = mock_token_mgr

        await start_conversation()

        mock_agent.run.assert_called_once_with("Hello", message_history=[])


@pytest.mark.asyncio
class TestEnvironmentVariables:
    """Test environment variable configuration."""

    @patch.dict(os.environ, {"COLLECT_LOGS": "false"})
    @patch("asyncio.get_event_loop")
    @patch("src.agent.cli.ThreadPoolExecutor")
    @patch("src.agent.cli.create_async_dependencies")
    @patch("src.agent.cli.MRtrixAssistant")
    @patch("src.agent.cli.TokenManager")
    @patch("src.agent.cli.console")
    async def test_modes_disabled_by_default(
        self,
        mock_console,
        MockTokenManager,
        MockAssistant,
        mock_create_deps,
        MockExecutor,
        mock_get_loop,
    ):
        """Test that logging is disabled when COLLECT_LOGS=false."""
        # Setup mock executor
        mock_executor = Mock()
        MockExecutor.return_value = mock_executor
        mock_executor.shutdown = Mock()

        # Setup mock loop
        mock_loop = AsyncMock()
        mock_get_loop.return_value = mock_loop
        mock_loop.run_in_executor.side_effect = ["/exit"]

        mock_deps = AsyncMock()
        mock_create_deps.return_value = mock_deps

        mock_agent = AsyncMock()
        mock_result = Mock()
        mock_result.output = "Response"
        mock_result.all_messages.return_value = []  # Add the all_messages method
        mock_agent.run.return_value = mock_result
        MockAssistant.return_value = mock_agent

        mock_token_mgr = AsyncMock()
        mock_token_mgr.add_message.return_value = True
        MockTokenManager.return_value = mock_token_mgr

        await start_conversation()

        # Should not print session logging message when COLLECT_LOGS=false
        found_log_message = False
        for call in mock_console.print.call_args_list:
            if call[0] and len(call[0]) > 0:
                if "Session logging to:" in str(call[0][0]):
                    found_log_message = True
                    break
        assert (
            not found_log_message
        ), "Should not show session logging message when disabled"
