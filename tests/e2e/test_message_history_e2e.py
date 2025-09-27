"""End-to-end tests for message history preservation across agent interactions."""

import asyncio
import os
from unittest.mock import Mock, patch, AsyncMock
import pytest
from pydantic_ai.messages import ModelRequest, ModelResponse, UserPromptPart, TextPart

from src.agent.agent import MRtrixAssistant
from src.agent.cli import start_conversation
from src.agent.slash_commands import SlashCommandHandler


@pytest.mark.e2e
class TestMessageHistoryEndToEnd:
    """End-to-end tests for message history preservation across multiple interactions."""

    @pytest.mark.asyncio
    async def test_conversation_history_accumulation(self):
        """Test that conversation history accumulates correctly across multiple agent calls."""
        # Mock dependencies
        mock_deps = AsyncMock()
        agent = MRtrixAssistant(dependencies=mock_deps)

        # Create mock message history
        initial_history = []

        with patch.object(agent.agent, "run") as mock_run:
            # First interaction
            mock_result_1 = Mock()
            mock_result_1.output = "First response"
            mock_result_1.all_messages.return_value = [
                ModelRequest(parts=[UserPromptPart(content="First question")]),
                ModelResponse(parts=[TextPart(content="First response")]),
            ]
            mock_run.return_value = mock_result_1

            result_1 = await agent.run(
                "First question", message_history=initial_history
            )
            first_history = result_1.all_messages()

            # Second interaction with accumulated history
            mock_result_2 = Mock()
            mock_result_2.output = "Second response"
            mock_result_2.all_messages.return_value = first_history + [
                ModelRequest(parts=[UserPromptPart(content="Second question")]),
                ModelResponse(parts=[TextPart(content="Second response")]),
            ]
            mock_run.return_value = mock_result_2

            result_2 = await agent.run("Second question", message_history=first_history)
            second_history = result_2.all_messages()

            # Verify history accumulation
            assert len(second_history) == 4  # 2 questions + 2 responses
            assert len(first_history) == 2  # 1 question + 1 response

            # Verify calls were made with correct history
            assert mock_run.call_count == 2
            first_call = mock_run.call_args_list[0]
            second_call = mock_run.call_args_list[1]

            assert first_call[1]["message_history"] == initial_history
            assert second_call[1]["message_history"] == first_history

    @pytest.mark.asyncio
    async def test_message_history_with_slash_commands(self):
        """Test message history preservation when slash commands modify input."""
        mock_deps = AsyncMock()
        agent = MRtrixAssistant(dependencies=mock_deps)
        slash_handler = SlashCommandHandler()

        # Simulate /sharefile command transforming input
        with patch("subprocess.run") as mock_subprocess:
            mock_result = Mock()
            mock_result.returncode = 0
            mock_result.stdout = "<user file information>\n{test: data}\n</user file information>\nHow to process?"
            mock_subprocess.return_value = mock_result

            command_result = slash_handler.process_command(
                "/sharefile /test/file.nii How to process?"
            )
            processed_input = command_result.agent_input

            # Mock agent response
            with patch.object(agent.agent, "run") as mock_run:
                mock_agent_result = Mock()
                mock_agent_result.output = "Processing steps for your file"
                mock_agent_result.all_messages.return_value = [
                    ModelRequest(parts=[UserPromptPart(content=processed_input)]),
                    ModelResponse(
                        parts=[TextPart(content="Processing steps for your file")]
                    ),
                ]
                mock_run.return_value = mock_agent_result

                # Run agent with processed input
                result = await agent.run(processed_input, message_history=[])

                # Verify the processed input (not original slash command) is in history
                history = result.all_messages()
                assert len(history) == 2
                user_message = history[0]
                assert user_message.parts[0].content == processed_input
                assert "/sharefile" not in user_message.parts[0].content
                assert "<user file information>" in user_message.parts[0].content

    @pytest.mark.asyncio
    async def test_full_conversation_flow_with_history_persistence(self):
        """Test complete conversation flow maintaining history across multiple exchanges."""
        with (
            patch("src.agent.cli.create_async_dependencies") as mock_deps_factory,
            patch("src.agent.cli.ThreadPoolExecutor") as MockExecutor,
            patch("asyncio.get_event_loop") as mock_get_loop,
            patch("src.agent.cli.console"),
            patch.dict(os.environ, {"COLLECT_LOGS": "false"}),
        ):
            # Setup mocks
            mock_deps = AsyncMock()
            mock_deps_factory.return_value = mock_deps

            mock_executor = Mock()
            mock_executor.shutdown = Mock()
            MockExecutor.return_value = mock_executor

            mock_loop = AsyncMock()
            mock_get_loop.return_value = mock_loop

            # Mock conversation: regular message, then /help, then another message, then /exit
            conversation_inputs = [
                "Hello, what can you help with?",
                "/help",
                "Tell me about dwi2fod",
                "/exit",
            ]
            mock_loop.run_in_executor.side_effect = conversation_inputs

            # Mock agent and responses
            with (
                patch("src.agent.cli.MRtrixAssistant") as MockAssistant,
                patch("src.agent.cli.TokenManager") as MockTokenManager,
            ):
                mock_agent = AsyncMock()
                mock_token_mgr = AsyncMock()
                mock_token_mgr.add_message.return_value = True

                # Setup progressive message history
                first_response = Mock()
                first_response.output = "I can help with MRtrix3 commands"
                first_response.all_messages.return_value = [
                    ModelRequest(
                        parts=[UserPromptPart(content="Hello, what can you help with?")]
                    ),
                    ModelResponse(
                        parts=[TextPart(content="I can help with MRtrix3 commands")]
                    ),
                ]

                second_response = Mock()
                second_response.output = "dwi2fod generates FOD images"
                second_response.all_messages.return_value = [
                    ModelRequest(
                        parts=[UserPromptPart(content="Hello, what can you help with?")]
                    ),
                    ModelResponse(
                        parts=[TextPart(content="I can help with MRtrix3 commands")]
                    ),
                    ModelRequest(
                        parts=[UserPromptPart(content="Tell me about dwi2fod")]
                    ),
                    ModelResponse(
                        parts=[TextPart(content="dwi2fod generates FOD images")]
                    ),
                ]

                mock_agent.run.side_effect = [first_response, second_response]
                MockAssistant.return_value = mock_agent
                MockTokenManager.return_value = mock_token_mgr

                await start_conversation()

                # Verify agent was called twice (excluding /help and /exit)
                assert mock_agent.run.call_count == 2

                # Verify first call had empty history
                first_call = mock_agent.run.call_args_list[0]
                assert first_call[0][0] == "Hello, what can you help with?"
                assert first_call[1]["message_history"] == []

                # Verify second call had accumulated history
                second_call = mock_agent.run.call_args_list[1]
                assert second_call[0][0] == "Tell me about dwi2fod"
                # The history should contain the previous exchange
                passed_history = second_call[1]["message_history"]
                assert len(passed_history) == 2  # Previous question and answer

    @pytest.mark.asyncio
    async def test_message_history_with_sharefile_integration(self):
        """Test message history preservation with /sharefile command integration."""
        with (
            patch("src.agent.cli.create_async_dependencies") as mock_deps_factory,
            patch("src.agent.cli.ThreadPoolExecutor") as MockExecutor,
            patch("asyncio.get_event_loop") as mock_get_loop,
            patch("src.agent.cli.console"),
            patch("subprocess.run") as mock_subprocess,
            patch.dict(os.environ, {"COLLECT_LOGS": "false"}),
        ):
            # Setup basic CLI mocks
            mock_deps = AsyncMock()
            mock_deps_factory.return_value = mock_deps
            mock_executor = Mock()
            mock_executor.shutdown = Mock()
            MockExecutor.return_value = mock_executor
            mock_loop = AsyncMock()
            mock_get_loop.return_value = mock_loop

            # Mock /sharefile subprocess call
            mock_subprocess_result = Mock()
            mock_subprocess_result.returncode = 0
            mock_subprocess_result.stdout = "<user file information>\n{format: 'NIfTI'}\n</user file information>\nAnalyze this scan"
            mock_subprocess.return_value = mock_subprocess_result

            # Conversation: regular message, then /sharefile, then /exit
            conversation_inputs = [
                "What is MRtrix3?",
                "/sharefile /data/scan.nii Analyze this scan",
                "/exit",
            ]
            mock_loop.run_in_executor.side_effect = conversation_inputs

            with (
                patch("src.agent.cli.MRtrixAssistant") as MockAssistant,
                patch("src.agent.cli.TokenManager") as MockTokenManager,
            ):
                mock_agent = AsyncMock()
                mock_token_mgr = AsyncMock()
                mock_token_mgr.add_message.return_value = True

                # First response (regular message)
                first_response = Mock()
                first_response.output = "MRtrix3 is neuroimaging software"
                first_response.all_messages.return_value = [
                    ModelRequest(parts=[UserPromptPart(content="What is MRtrix3?")]),
                    ModelResponse(
                        parts=[TextPart(content="MRtrix3 is neuroimaging software")]
                    ),
                ]

                # Second response (processed /sharefile input)
                processed_input = "<user file information>\n{format: 'NIfTI'}\n</user file information>\nAnalyze this scan"
                second_response = Mock()
                second_response.output = "Based on the NIfTI file information..."
                second_response.all_messages.return_value = [
                    ModelRequest(parts=[UserPromptPart(content="What is MRtrix3?")]),
                    ModelResponse(
                        parts=[TextPart(content="MRtrix3 is neuroimaging software")]
                    ),
                    ModelRequest(parts=[UserPromptPart(content=processed_input)]),
                    ModelResponse(
                        parts=[
                            TextPart(content="Based on the NIfTI file information...")
                        ]
                    ),
                ]

                mock_agent.run.side_effect = [first_response, second_response]
                MockAssistant.return_value = mock_agent
                MockTokenManager.return_value = mock_token_mgr

                await start_conversation()

                # Verify both agent calls
                assert mock_agent.run.call_count == 2

                # Verify first call
                first_call = mock_agent.run.call_args_list[0]
                assert first_call[0][0] == "What is MRtrix3?"
                assert first_call[1]["message_history"] == []

                # Verify second call with processed input and accumulated history
                second_call = mock_agent.run.call_args_list[1]
                assert second_call[0][0] == processed_input
                # Should have history from first exchange
                passed_history = second_call[1]["message_history"]
                assert len(passed_history) == 2  # First question and answer

    @pytest.mark.asyncio
    async def test_token_limit_reset_clears_history(self):
        """Test that hitting token limit resets conversation history."""
        with (
            patch("src.agent.cli.create_async_dependencies") as mock_deps_factory,
            patch("src.agent.cli.ThreadPoolExecutor") as MockExecutor,
            patch("asyncio.get_event_loop") as mock_get_loop,
            patch("src.agent.cli.console") as mock_console,
            patch.dict(os.environ, {"COLLECT_LOGS": "false"}),
        ):
            mock_deps = AsyncMock()
            mock_deps_factory.return_value = mock_deps
            mock_executor = Mock()
            mock_executor.shutdown = Mock()
            MockExecutor.return_value = mock_executor
            mock_loop = AsyncMock()
            mock_get_loop.return_value = mock_loop

            # Simulate conversation that hits token limit
            conversation_inputs = [
                "First message",
                "Second message that exceeds token limit",
                "/exit",
            ]
            mock_loop.run_in_executor.side_effect = conversation_inputs

            with (
                patch("src.agent.cli.MRtrixAssistant") as MockAssistant,
                patch("src.agent.cli.TokenManager") as MockTokenManager,
            ):
                mock_agent = AsyncMock()
                mock_token_mgr = AsyncMock()

                # First message succeeds, second fails due to token limit
                mock_token_mgr.add_message.side_effect = [True, True, False, True, True]
                mock_token_mgr.reset = Mock()

                first_response = Mock()
                first_response.output = "First response"
                first_response.all_messages.return_value = [
                    ModelRequest(parts=[UserPromptPart(content="First message")]),
                    ModelResponse(parts=[TextPart(content="First response")]),
                ]

                second_response = Mock()
                second_response.output = "Second response"
                second_response.all_messages.return_value = [
                    ModelRequest(
                        parts=[
                            UserPromptPart(
                                content="Second message that exceeds token limit"
                            )
                        ]
                    ),
                    ModelResponse(parts=[TextPart(content="Second response")]),
                ]

                mock_agent.run.side_effect = [first_response, second_response]
                MockAssistant.return_value = mock_agent
                MockTokenManager.return_value = mock_token_mgr

                await start_conversation()

                # Verify token manager was reset
                mock_token_mgr.reset.assert_called_once()

                # Verify token limit message was displayed
                token_limit_message_found = any(
                    "Token limit reached" in str(call)
                    for call in mock_console.print.call_args_list
                )
                assert token_limit_message_found

                # Verify second agent call had empty history due to reset
                second_call = mock_agent.run.call_args_list[1]
                assert second_call[1]["message_history"] == []

    @pytest.mark.asyncio
    async def test_concurrent_message_processing(self):
        """Test message history consistency under concurrent processing scenarios."""
        mock_deps = AsyncMock()
        agent = MRtrixAssistant(dependencies=mock_deps)

        # Simulate multiple concurrent agent runs
        with patch.object(agent.agent, "run") as mock_run:
            responses = []
            for i in range(3):
                mock_result = Mock()
                mock_result.output = f"Response {i + 1}"
                mock_result.all_messages.return_value = [
                    ModelRequest(parts=[UserPromptPart(content=f"Question {i + 1}")]),
                    ModelResponse(parts=[TextPart(content=f"Response {i + 1}")]),
                ]
                responses.append(mock_result)

            mock_run.side_effect = responses

            # Run multiple agent calls
            tasks = []
            for i in range(3):
                task = agent.run(f"Question {i + 1}", message_history=[])
                tasks.append(task)

            results = await asyncio.gather(*tasks)

            # Verify all calls completed
            assert len(results) == 3
            assert mock_run.call_count == 3

            # Each call should have been made with empty history since we're testing concurrency
            for i, call in enumerate(mock_run.call_args_list):
                assert call[0][0] == f"Question {i + 1}"
                assert call[1]["message_history"] == []

    @pytest.mark.asyncio
    async def test_error_recovery_preserves_history(self):
        """Test that errors during agent processing don't corrupt message history."""
        with (
            patch("src.agent.cli.create_async_dependencies") as mock_deps_factory,
            patch("src.agent.cli.ThreadPoolExecutor") as MockExecutor,
            patch("asyncio.get_event_loop") as mock_get_loop,
            patch("src.agent.cli.console") as mock_console,
            patch.dict(os.environ, {"COLLECT_LOGS": "false"}),
        ):
            mock_deps = AsyncMock()
            mock_deps_factory.return_value = mock_deps
            mock_executor = Mock()
            mock_executor.shutdown = Mock()
            MockExecutor.return_value = mock_executor
            mock_loop = AsyncMock()
            mock_get_loop.return_value = mock_loop

            conversation_inputs = [
                "First successful message",
                "Message that causes error",
                "Recovery message",
                "/exit",
            ]
            mock_loop.run_in_executor.side_effect = conversation_inputs

            with (
                patch("src.agent.cli.MRtrixAssistant") as MockAssistant,
                patch("src.agent.cli.TokenManager") as MockTokenManager,
            ):
                mock_agent = AsyncMock()
                mock_token_mgr = AsyncMock()
                mock_token_mgr.add_message.return_value = True

                # First call succeeds
                first_response = Mock()
                first_response.output = "First response"
                first_response.all_messages.return_value = [
                    ModelRequest(
                        parts=[UserPromptPart(content="First successful message")]
                    ),
                    ModelResponse(parts=[TextPart(content="First response")]),
                ]

                # Second call fails
                # Third call succeeds with history intact
                third_response = Mock()
                third_response.output = "Recovery response"
                third_response.all_messages.return_value = [
                    ModelRequest(
                        parts=[UserPromptPart(content="First successful message")]
                    ),
                    ModelResponse(parts=[TextPart(content="First response")]),
                    ModelRequest(parts=[UserPromptPart(content="Recovery message")]),
                    ModelResponse(parts=[TextPart(content="Recovery response")]),
                ]

                mock_agent.run.side_effect = [
                    first_response,
                    Exception("Simulated error"),
                    third_response,
                ]
                MockAssistant.return_value = mock_agent
                MockTokenManager.return_value = mock_token_mgr

                await start_conversation()

                # Verify error was handled gracefully
                error_message_found = any(
                    "unexpected issue" in str(call).lower()
                    or "try again" in str(call).lower()
                    for call in mock_console.print.call_args_list
                )
                assert error_message_found

                # Verify third call had correct history (from first successful exchange)
                assert mock_agent.run.call_count == 3
                third_call = mock_agent.run.call_args_list[2]
                assert third_call[0][0] == "Recovery message"
                # History should still contain first successful exchange
                passed_history = third_call[1]["message_history"]
                assert len(passed_history) == 2  # First question and answer


@pytest.mark.asyncio
async def test_message_history_integration_with_token_manager():
    """Test integration between message history and TokenManager."""
    from src.agent.cli import TokenManager

    # Mock the Gemini model to avoid API calls
    with patch("src.agent.cli.genai.GenerativeModel") as MockModel:
        mock_model = Mock()
        mock_result = Mock()
        mock_result.total_tokens = 10  # Simulate token count
        mock_model.count_tokens.return_value = mock_result
        MockModel.return_value = mock_model

        token_manager = TokenManager()

        # Create sample messages as strings
        sample_messages = [
            "Short question",
            "Short answer",
        ]

        # Add messages to token manager
        for msg in sample_messages:
            success = await token_manager.add_message(msg)
            assert success is True

        # Verify messages were added (check internal state)
        assert len(token_manager.message_history) == 2
        assert token_manager.message_history[0][0] == "Short question"
        assert token_manager.message_history[1][0] == "Short answer"

        # Add a message that would exceed limit
        # Set token count to exceed limit
        mock_result.total_tokens = 600000
        long_message = "x" * 1000
        success = await token_manager.add_message(long_message)
        assert success is False  # Should fail due to token limit

        # History should remain unchanged
        assert len(token_manager.message_history) == 2

        # Reset and verify cleanup
        token_manager.reset()
        assert len(token_manager.message_history) == 0
        assert token_manager.total_tokens == 0
