"""End-to-end tests for conversation flow with mixed slash commands and regular queries."""

import os
from unittest.mock import Mock, patch, AsyncMock
import pytest
from pydantic_ai.messages import ModelRequest, ModelResponse, UserPromptPart, TextPart

from src.agent.cli import start_conversation


@pytest.mark.e2e
class TestConversationFlowEndToEnd:
    """End-to-end tests for realistic conversation flows mixing slash commands and regular queries."""

    @pytest.mark.asyncio
    async def test_mixed_conversation_flow(self):
        """Test conversation mixing regular queries, /help, and /sharefile commands."""
        with (
            patch("src.agent.cli.create_async_dependencies") as mock_deps_factory,
            patch("src.agent.cli.ThreadPoolExecutor") as MockExecutor,
            patch("asyncio.get_event_loop") as mock_get_loop,
            patch("src.agent.cli.console"),
            patch("src.agent.slash_commands.console") as mock_slash_console,
            patch("subprocess.run") as mock_subprocess,
            patch.dict(os.environ, {"COLLECT_LOGS": "false"}),
        ):
            # Setup basic mocks
            mock_deps = AsyncMock()
            mock_deps_factory.return_value = mock_deps
            mock_executor = Mock()
            mock_executor.shutdown = Mock()
            MockExecutor.return_value = mock_executor
            mock_loop = AsyncMock()
            mock_get_loop.return_value = mock_loop

            # Mock /sharefile subprocess
            mock_subprocess_result = Mock()
            mock_subprocess_result.returncode = 0
            mock_subprocess_result.stdout = '<user file information>\n{"format": "NIfTI"}\n</user file information>\n<user_provided_filepath>\n/data/dwi.nii\n</user_provided_filepath>\nHow should I preprocess this DWI data?'
            mock_subprocess.return_value = mock_subprocess_result

            # Complex conversation flow
            conversation_inputs = [
                "What is MRtrix3 used for?",  # Regular query
                "/help",  # Slash command (no agent call)
                "Tell me about dwi2fod",  # Regular query
                "/sharefile /data/dwi.nii How should I preprocess this DWI data?",  # Slash command -> agent call
                "What about fiber tracking?",  # Follow-up regular query
                "/exit",  # Exit
            ]
            mock_loop.run_in_executor.side_effect = conversation_inputs

            with (
                patch("src.agent.cli.MRtrixAssistant") as MockAssistant,
                patch("src.agent.cli.TokenManager") as MockTokenManager,
            ):
                mock_agent = AsyncMock()
                mock_token_mgr = AsyncMock()
                mock_token_mgr.add_message.return_value = True

                # Setup responses for agent calls (3 total: regular, /sharefile processed, follow-up)

                # First response - regular query about MRtrix3
                first_response = Mock()
                first_response.output = (
                    "MRtrix3 is software for analyzing diffusion MRI data"
                )
                first_response.all_messages.return_value = [
                    ModelRequest(
                        parts=[UserPromptPart(content="What is MRtrix3 used for?")]
                    ),
                    ModelResponse(
                        parts=[
                            TextPart(
                                content="MRtrix3 is software for analyzing diffusion MRI data"
                            )
                        ]
                    ),
                ]

                # Second response - dwi2fod query with history
                second_response = Mock()
                second_response.output = (
                    "dwi2fod estimates fiber orientation distributions from DWI data"
                )
                second_response.all_messages.return_value = [
                    ModelRequest(
                        parts=[UserPromptPart(content="What is MRtrix3 used for?")]
                    ),
                    ModelResponse(
                        parts=[
                            TextPart(
                                content="MRtrix3 is software for analyzing diffusion MRI data"
                            )
                        ]
                    ),
                    ModelRequest(
                        parts=[UserPromptPart(content="Tell me about dwi2fod")]
                    ),
                    ModelResponse(
                        parts=[
                            TextPart(
                                content="dwi2fod estimates fiber orientation distributions from DWI data"
                            )
                        ]
                    ),
                ]

                # Third response - /sharefile processed input with accumulated history
                processed_input = '<user file information>\n{"format": "NIfTI"}\n</user file information>\n<user_provided_filepath>\n/data/dwi.nii\n</user_provided_filepath>\nHow should I preprocess this DWI data?'
                third_response = Mock()
                third_response.output = "For DWI preprocessing, I recommend: 1) Denoising with dwidenoise, 2) Gibbs ringing removal with mrdegibbs, 3) Motion and distortion correction"
                third_response.all_messages.return_value = [
                    ModelRequest(
                        parts=[UserPromptPart(content="What is MRtrix3 used for?")]
                    ),
                    ModelResponse(
                        parts=[
                            TextPart(
                                content="MRtrix3 is software for analyzing diffusion MRI data"
                            )
                        ]
                    ),
                    ModelRequest(
                        parts=[UserPromptPart(content="Tell me about dwi2fod")]
                    ),
                    ModelResponse(
                        parts=[
                            TextPart(
                                content="dwi2fod estimates fiber orientation distributions from DWI data"
                            )
                        ]
                    ),
                    ModelRequest(parts=[UserPromptPart(content=processed_input)]),
                    ModelResponse(
                        parts=[
                            TextPart(
                                content="For DWI preprocessing, I recommend: 1) Denoising with dwidenoise, 2) Gibbs ringing removal with mrdegibbs, 3) Motion and distortion correction"
                            )
                        ]
                    ),
                ]

                # Fourth response - follow-up query with full history
                fourth_response = Mock()
                fourth_response.output = "For fiber tracking, use tckgen with appropriate tracking algorithms"
                fourth_response.all_messages.return_value = [
                    ModelRequest(
                        parts=[UserPromptPart(content="What is MRtrix3 used for?")]
                    ),
                    ModelResponse(
                        parts=[
                            TextPart(
                                content="MRtrix3 is software for analyzing diffusion MRI data"
                            )
                        ]
                    ),
                    ModelRequest(
                        parts=[UserPromptPart(content="Tell me about dwi2fod")]
                    ),
                    ModelResponse(
                        parts=[
                            TextPart(
                                content="dwi2fod estimates fiber orientation distributions from DWI data"
                            )
                        ]
                    ),
                    ModelRequest(parts=[UserPromptPart(content=processed_input)]),
                    ModelResponse(
                        parts=[
                            TextPart(
                                content="For DWI preprocessing, I recommend: 1) Denoising with dwidenoise, 2) Gibbs ringing removal with mrdegibbs, 3) Motion and distortion correction"
                            )
                        ]
                    ),
                    ModelRequest(
                        parts=[UserPromptPart(content="What about fiber tracking?")]
                    ),
                    ModelResponse(
                        parts=[
                            TextPart(
                                content="For fiber tracking, use tckgen with appropriate tracking algorithms"
                            )
                        ]
                    ),
                ]

                mock_agent.run.side_effect = [
                    first_response,
                    second_response,
                    third_response,
                    fourth_response,
                ]
                MockAssistant.return_value = mock_agent
                MockTokenManager.return_value = mock_token_mgr

                await start_conversation()

                # Verify agent was called 4 times (excluding /help and /exit)
                assert mock_agent.run.call_count == 4

                # Verify call sequence and history accumulation
                calls = mock_agent.run.call_args_list

                # First call - empty history
                assert calls[0][0][0] == "What is MRtrix3 used for?"
                assert calls[0][1]["message_history"] == []

                # Second call - has first exchange in history
                assert calls[1][0][0] == "Tell me about dwi2fod"
                assert len(calls[1][1]["message_history"]) == 2

                # Third call - /sharefile processed input, has previous history
                assert calls[2][0][0] == processed_input
                assert len(calls[2][1]["message_history"]) == 4

                # Fourth call - follow-up, has all previous history
                assert calls[3][0][0] == "What about fiber tracking?"
                assert len(calls[3][1]["message_history"]) == 6

                # Verify /help was shown (check both console instances)
                help_displayed = any(
                    "Available Commands" in str(call)
                    for call in mock_slash_console.print.call_args_list
                )
                assert help_displayed

    @pytest.mark.asyncio
    async def test_conversation_with_multiple_sharefile_commands(self):
        """Test conversation with multiple /sharefile commands and different files."""
        with (
            patch("src.agent.cli.create_async_dependencies") as mock_deps_factory,
            patch("src.agent.cli.ThreadPoolExecutor") as MockExecutor,
            patch("asyncio.get_event_loop") as mock_get_loop,
            patch("src.agent.cli.console"),
            patch("src.agent.slash_commands.console"),
            patch("subprocess.run") as mock_subprocess,
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
                "/sharefile /data/anatomical.nii This is a T1-weighted image, how should I segment it?",
                "Can you be more specific about the segmentation process?",
                "/sharefile /data/dwi.nii How should I process this DWI data for tractography?",
                "What about quality control for these processes?",
                "/exit",
            ]
            mock_loop.run_in_executor.side_effect = conversation_inputs

            # Mock different /sharefile outputs for different files
            def mock_subprocess_side_effect(*args, **kwargs):
                command_args = args[0]
                if "/data/anatomical.nii" in command_args:
                    result = Mock()
                    result.returncode = 0
                    result.stdout = '<user file information>\n{"format": "NIfTI", "type": "T1"}\n</user file information>\n<user_provided_filepath>\n/data/anatomical.nii\n</user_provided_filepath>\nThis is a T1-weighted image, how should I segment it?'
                    return result
                elif "/data/dwi.nii" in command_args:
                    result = Mock()
                    result.returncode = 0
                    result.stdout = '<user file information>\n{"format": "NIfTI", "type": "DWI"}\n</user file information>\n<user_provided_filepath>\n/data/dwi.nii\n</user_provided_filepath>\nHow should I process this DWI data for tractography?'
                    return result

            mock_subprocess.side_effect = mock_subprocess_side_effect

            with (
                patch("src.agent.cli.MRtrixAssistant") as MockAssistant,
                patch("src.agent.cli.TokenManager") as MockTokenManager,
            ):
                mock_agent = AsyncMock()
                mock_token_mgr = AsyncMock()
                mock_token_mgr.add_message.return_value = True

                # Response sequence
                anatomical_input = '<user file information>\n{"format": "NIfTI", "type": "T1"}\n</user file information>\n<user_provided_filepath>\n/data/anatomical.nii\n</user_provided_filepath>\nThis is a T1-weighted image, how should I segment it?'
                dwi_input = '<user file information>\n{"format": "NIfTI", "type": "DWI"}\n</user file information>\n<user_provided_filepath>\n/data/dwi.nii\n</user_provided_filepath>\nHow should I process this DWI data for tractography?'

                responses = [
                    # First /sharefile response
                    Mock(
                        output="For T1 segmentation, use 5ttgen with FSL or FreeSurfer",
                        all_messages=lambda: [
                            ModelRequest(
                                parts=[UserPromptPart(content=anatomical_input)]
                            ),
                            ModelResponse(
                                parts=[
                                    TextPart(
                                        content="For T1 segmentation, use 5ttgen with FSL or FreeSurfer"
                                    )
                                ]
                            ),
                        ],
                    ),
                    # Follow-up question
                    Mock(
                        output="5ttgen fsl creates 5-tissue-type segmentation: GM, WM, CSF, pathological, cortical GM",
                        all_messages=lambda: [
                            ModelRequest(
                                parts=[UserPromptPart(content=anatomical_input)]
                            ),
                            ModelResponse(
                                parts=[
                                    TextPart(
                                        content="For T1 segmentation, use 5ttgen with FSL or FreeSurfer"
                                    )
                                ]
                            ),
                            ModelRequest(
                                parts=[
                                    UserPromptPart(
                                        content="Can you be more specific about the segmentation process?"
                                    )
                                ]
                            ),
                            ModelResponse(
                                parts=[
                                    TextPart(
                                        content="5ttgen fsl creates 5-tissue-type segmentation: GM, WM, CSF, pathological, cortical GM"
                                    )
                                ]
                            ),
                        ],
                    ),
                    # Second /sharefile response
                    Mock(
                        output="DWI processing steps: 1) dwidenoise, 2) mrdegibbs, 3) dwipreproc, 4) dwi2fod, 5) tckgen",
                        all_messages=lambda: [
                            ModelRequest(
                                parts=[UserPromptPart(content=anatomical_input)]
                            ),
                            ModelResponse(
                                parts=[
                                    TextPart(
                                        content="For T1 segmentation, use 5ttgen with FSL or FreeSurfer"
                                    )
                                ]
                            ),
                            ModelRequest(
                                parts=[
                                    UserPromptPart(
                                        content="Can you be more specific about the segmentation process?"
                                    )
                                ]
                            ),
                            ModelResponse(
                                parts=[
                                    TextPart(
                                        content="5ttgen fsl creates 5-tissue-type segmentation: GM, WM, CSF, pathological, cortical GM"
                                    )
                                ]
                            ),
                            ModelRequest(parts=[UserPromptPart(content=dwi_input)]),
                            ModelResponse(
                                parts=[
                                    TextPart(
                                        content="DWI processing steps: 1) dwidenoise, 2) mrdegibbs, 3) dwipreproc, 4) dwi2fod, 5) tckgen"
                                    )
                                ]
                            ),
                        ],
                    ),
                    # Quality control follow-up
                    Mock(
                        output="For QC, check outputs at each step: inspect images, verify tensor fits, validate tract counts",
                        all_messages=lambda: [
                            ModelRequest(
                                parts=[UserPromptPart(content=anatomical_input)]
                            ),
                            ModelResponse(
                                parts=[
                                    TextPart(
                                        content="For T1 segmentation, use 5ttgen with FSL or FreeSurfer"
                                    )
                                ]
                            ),
                            ModelRequest(
                                parts=[
                                    UserPromptPart(
                                        content="Can you be more specific about the segmentation process?"
                                    )
                                ]
                            ),
                            ModelResponse(
                                parts=[
                                    TextPart(
                                        content="5ttgen fsl creates 5-tissue-type segmentation: GM, WM, CSF, pathological, cortical GM"
                                    )
                                ]
                            ),
                            ModelRequest(parts=[UserPromptPart(content=dwi_input)]),
                            ModelResponse(
                                parts=[
                                    TextPart(
                                        content="DWI processing steps: 1) dwidenoise, 2) mrdegibbs, 3) dwipreproc, 4) dwi2fod, 5) tckgen"
                                    )
                                ]
                            ),
                            ModelRequest(
                                parts=[
                                    UserPromptPart(
                                        content="What about quality control for these processes?"
                                    )
                                ]
                            ),
                            ModelResponse(
                                parts=[
                                    TextPart(
                                        content="For QC, check outputs at each step: inspect images, verify tensor fits, validate tract counts"
                                    )
                                ]
                            ),
                        ],
                    ),
                ]

                mock_agent.run.side_effect = responses
                MockAssistant.return_value = mock_agent
                MockTokenManager.return_value = mock_token_mgr

                await start_conversation()

                # Verify 4 agent calls total
                assert mock_agent.run.call_count == 4

                # Verify subprocess was called twice for different files
                assert mock_subprocess.call_count == 2

                calls = mock_agent.run.call_args_list

                # First call - anatomical file
                assert calls[0][0][0] == anatomical_input
                assert calls[0][1]["message_history"] == []

                # Second call - follow-up with history
                assert (
                    calls[1][0][0]
                    == "Can you be more specific about the segmentation process?"
                )
                assert len(calls[1][1]["message_history"]) == 2

                # Third call - DWI file with accumulated history
                assert calls[2][0][0] == dwi_input
                assert len(calls[2][1]["message_history"]) == 4

                # Fourth call - QC follow-up with full history
                assert (
                    calls[3][0][0] == "What about quality control for these processes?"
                )
                assert len(calls[3][1]["message_history"]) == 6

    @pytest.mark.asyncio
    async def test_conversation_flow_with_token_limit_reset(self):
        """Test conversation flow when token limit is hit and history resets."""
        with (
            patch("src.agent.cli.create_async_dependencies") as mock_deps_factory,
            patch("src.agent.cli.ThreadPoolExecutor") as MockExecutor,
            patch("asyncio.get_event_loop") as mock_get_loop,
            patch("src.agent.cli.console") as mock_console,
            patch("src.agent.slash_commands.console") as mock_slash_console,
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
                "First message",
                "/help",  # Should not trigger token limit
                "Second message that exceeds token limit",
                "Third message after reset",
                "/exit",
            ]
            mock_loop.run_in_executor.side_effect = conversation_inputs

            with (
                patch("src.agent.cli.MRtrixAssistant") as MockAssistant,
                patch("src.agent.cli.TokenManager") as MockTokenManager,
            ):
                mock_agent = AsyncMock()
                mock_token_mgr = AsyncMock()

                # First message succeeds, second fails token limit, then succeeds after reset
                mock_token_mgr.add_message.side_effect = [
                    True,
                    True,  # First exchange succeeds
                    False,
                    True,
                    True,  # Second fails, then succeeds after reset
                    True,
                    True,  # Third exchange succeeds
                ]
                mock_token_mgr.reset = Mock()

                responses = [
                    Mock(
                        output="First response",
                        all_messages=lambda: [
                            ModelRequest(
                                parts=[UserPromptPart(content="First message")]
                            ),
                            ModelResponse(parts=[TextPart(content="First response")]),
                        ],
                    ),
                    Mock(
                        output="Second response",
                        all_messages=lambda: [
                            ModelRequest(
                                parts=[
                                    UserPromptPart(
                                        content="Second message that exceeds token limit"
                                    )
                                ]
                            ),
                            ModelResponse(parts=[TextPart(content="Second response")]),
                        ],
                    ),
                    Mock(
                        output="Third response",
                        all_messages=lambda: [
                            ModelRequest(
                                parts=[
                                    UserPromptPart(content="Third message after reset")
                                ]
                            ),
                            ModelResponse(parts=[TextPart(content="Third response")]),
                        ],
                    ),
                ]

                mock_agent.run.side_effect = responses
                MockAssistant.return_value = mock_agent
                MockTokenManager.return_value = mock_token_mgr

                await start_conversation()

                # Verify token manager was reset
                mock_token_mgr.reset.assert_called_once()

                # Verify token limit warning was shown
                token_warning_shown = any(
                    "Token limit reached" in str(call)
                    for call in mock_console.print.call_args_list
                )
                assert token_warning_shown

                # Verify help was shown (not affected by token limit)
                help_shown = any(
                    "Available Commands" in str(call)
                    for call in mock_slash_console.print.call_args_list
                )
                assert help_shown

                # Verify agent calls had correct history
                calls = mock_agent.run.call_args_list

                # First call - empty history
                assert calls[0][1]["message_history"] == []

                # Second call after token limit reset - empty history
                assert calls[1][1]["message_history"] == []

                # Third call - should have history from second call
                assert len(calls[2][1]["message_history"]) == 2

    @pytest.mark.asyncio
    async def test_empty_input_handling_in_mixed_conversation(self):
        """Test handling of empty inputs within mixed conversation flow."""
        with (
            patch("src.agent.cli.create_async_dependencies") as mock_deps_factory,
            patch("src.agent.cli.ThreadPoolExecutor") as MockExecutor,
            patch("asyncio.get_event_loop") as mock_get_loop,
            patch("src.agent.cli.console"),
            patch("src.agent.slash_commands.console") as mock_slash_console,
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
                "First valid message",
                "",  # Empty input - should be skipped
                "   ",  # Whitespace only - should be skipped
                "/help",  # Valid slash command
                "",  # Another empty input
                "Second valid message",
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

                responses = [
                    Mock(
                        output="First response",
                        all_messages=lambda: [
                            ModelRequest(
                                parts=[UserPromptPart(content="First valid message")]
                            ),
                            ModelResponse(parts=[TextPart(content="First response")]),
                        ],
                    ),
                    Mock(
                        output="Second response",
                        all_messages=lambda: [
                            ModelRequest(
                                parts=[UserPromptPart(content="First valid message")]
                            ),
                            ModelResponse(parts=[TextPart(content="First response")]),
                            ModelRequest(
                                parts=[UserPromptPart(content="Second valid message")]
                            ),
                            ModelResponse(parts=[TextPart(content="Second response")]),
                        ],
                    ),
                ]

                mock_agent.run.side_effect = responses
                MockAssistant.return_value = mock_agent
                MockTokenManager.return_value = mock_token_mgr

                await start_conversation()

                # Should only have 2 agent calls (empty inputs skipped)
                assert mock_agent.run.call_count == 2

                # Verify help was still displayed
                help_shown = any(
                    "Available Commands" in str(call)
                    for call in mock_slash_console.print.call_args_list
                )
                assert help_shown

                # Verify proper history accumulation despite empty inputs
                calls = mock_agent.run.call_args_list
                assert (
                    calls[1][1]["message_history"]
                    and len(calls[1][1]["message_history"]) == 2
                )
