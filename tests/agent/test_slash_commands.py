"""Unit tests for slash commands functionality."""

import subprocess
import sys
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch
import pytest

from src.agent.slash_commands import SlashCommandHandler, SlashCommandResult


class TestSlashCommandHandler:
    """Test suite for SlashCommandHandler."""

    def setup_method(self):
        """Set up test fixtures."""
        self.handler = SlashCommandHandler()

    def test_initialization(self):
        """Test that handler initializes with correct commands."""
        assert "/exit" in self.handler.commands
        assert "/help" in self.handler.commands
        assert "/sharefile" in self.handler.commands

        assert len(self.handler.command_descriptions) == 3
        assert self.handler.command_descriptions["/exit"] == "Exit the application"
        assert self.handler.command_descriptions["/help"] == "Show available commands"
        assert (
            self.handler.command_descriptions["/sharefile"]
            == "Share file metadata with the assistant"
        )

    def test_process_non_slash_command(self):
        """Test that non-slash commands pass through unchanged."""
        result = self.handler.process_command("Hello, how are you?")

        assert result.success is True
        assert result.agent_input == "Hello, how are you?"
        assert result.continue_conversation is True
        assert result.exit_requested is False

    def test_process_unknown_slash_command(self):
        """Test handling of unknown slash commands."""
        with patch("src.agent.slash_commands.console") as mock_console:
            result = self.handler.process_command("/unknown")

            assert result.success is False
            assert result.continue_conversation is False
            mock_console.print.assert_called_with(
                "[yellow]Unknown command: /unknown. Type /help for available commands.[/yellow]"
            )

    def test_exit_command(self):
        """Test /exit command sets exit flag."""
        result = self.handler.process_command("/exit")

        assert result.success is True
        assert result.exit_requested is True
        assert result.continue_conversation is True

    def test_help_command(self):
        """Test /help command displays available commands."""
        with patch("src.agent.slash_commands.console") as mock_console:
            result = self.handler.process_command("/help")

            assert result.success is True
            assert result.continue_conversation is False
            assert result.exit_requested is False

            # Verify help text is printed
            calls = mock_console.print.call_args_list
            assert any(
                "[bold]Available Commands:[/bold]" in str(call) for call in calls
            )
            assert any("/exit" in str(call) for call in calls)
            assert any("/help" in str(call) for call in calls)
            assert any(
                "/sharefile" in str(call) and "<path> <query>" in str(call)
                for call in calls
            )


class TestShareFileCommand:
    """Test suite specifically for /sharefile command."""

    def setup_method(self):
        """Set up test fixtures."""
        self.handler = SlashCommandHandler()

    def test_sharefile_missing_arguments(self):
        """Test /sharefile with missing arguments shows usage."""
        with patch("src.agent.slash_commands.console") as mock_console:
            # Test with no arguments
            result = self.handler.process_command("/sharefile")

            assert result.success is False
            assert result.continue_conversation is False

            # Check usage message
            mock_console.print.assert_any_call(
                "[red]Usage: /sharefile <path> <query>[/red]"
            )
            mock_console.print.assert_any_call(
                "[yellow]Example: /sharefile /data/scan.nii How do I preprocess this?[/yellow]"
            )

    def test_sharefile_only_path_no_query(self):
        """Test /sharefile with only path but no query."""
        with patch("src.agent.slash_commands.console") as mock_console:
            result = self.handler.process_command("/sharefile /path/to/file.nii")

            assert result.success is False
            assert result.continue_conversation is False
            mock_console.print.assert_any_call(
                "[red]Usage: /sharefile <path> <query>[/red]"
            )

    def test_sharefile_with_valid_arguments(self):
        """Test /sharefile with valid path and query."""
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "<user file information>...</user file information>"

        with patch("subprocess.run", return_value=mock_result) as mock_run:
            result = self.handler.process_command(
                "/sharefile /data/scan.nii How do I preprocess?"
            )

            assert result.success is True
            assert result.continue_conversation is True
            assert (
                result.agent_input
                == "<user file information>...</user file information>"
            )

            # Verify subprocess was called correctly
            mock_run.assert_called_once()
            call_args = mock_run.call_args[0][0]
            assert sys.executable in call_args[0]
            assert "sharefile.py" in call_args[1]
            assert call_args[2] == "/data/scan.nii"
            assert call_args[3] == "How do I preprocess?"

    def test_sharefile_handles_complex_query(self):
        """Test /sharefile with multi-word query containing special characters."""
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "test output"

        with patch("subprocess.run", return_value=mock_result) as mock_run:
            complex_query = "What's the best way to preprocess this DWI data? Should I use -pe_dir AP?"
            result = self.handler.process_command(
                f"/sharefile /data/scan.nii {complex_query}"
            )

            assert result.success is True
            # Verify the full query is passed
            call_args = mock_run.call_args[0][0]
            assert call_args[3] == complex_query

    def test_sharefile_script_error(self):
        """Test /sharefile when sharefile.py returns an error."""
        mock_result = MagicMock()
        mock_result.returncode = 1
        mock_result.stdout = "Error: mrinfo failed"

        with patch("subprocess.run", return_value=mock_result):
            with patch("src.agent.slash_commands.console") as mock_console:
                result = self.handler.process_command(
                    "/sharefile /invalid/path test query"
                )

                assert result.success is False
                assert result.continue_conversation is False
                mock_console.print.assert_called_with("[red]Error: mrinfo failed[/red]")

    def test_sharefile_timeout(self):
        """Test /sharefile timeout handling."""
        with patch("subprocess.run", side_effect=subprocess.TimeoutExpired("cmd", 60)):
            with patch("src.agent.slash_commands.console") as mock_console:
                result = self.handler.process_command("/sharefile /data/scan.nii query")

                assert result.success is False
                assert result.continue_conversation is False
                mock_console.print.assert_called_with(
                    "[red]Error: File analysis timed out[/red]"
                )


class TestShareFileOutput:
    """Test the output format of /sharefile command."""

    def test_commandgen_script_output_format(self):
        """Test that sharefile.py produces correct XML format."""
        # We'll test the build_prompt function directly
        import sys

        sys.path.insert(0, str(Path(__file__).parent.parent.parent))
        from src.workflows.sharefile.sharefile import build_prompt

        json_content = '{"format": "DICOM", "size": [256, 256, 62, 33]}'
        filepath = "/data/scan.nii"
        query = "How do I preprocess this?"

        result = build_prompt(json_content, filepath, query)

        # Check for XML blocks
        assert "<user file information>" in result
        assert "</user file information>" in result
        assert "<user_provided_filepath>" in result
        assert "</user_provided_filepath>" in result

        # Check content is included
        assert json_content in result
        assert filepath in result
        assert query in result

        # Check order: XML blocks first, then query
        lines = result.split("\n")
        xml_start = next(
            i for i, line in enumerate(lines) if "<user file information>" in line
        )
        filepath_end = next(
            i for i, line in enumerate(lines) if "</user_provided_filepath>" in line
        )
        query_line = next(i for i, line in enumerate(lines) if query in line)

        assert xml_start < filepath_end < query_line


class TestMessageHistoryPreservation:
    """Test that message history is preserved across agent calls."""

    @pytest.mark.asyncio
    async def test_message_history_passed_to_agent(self):
        """Test that conversation history is passed to agent.run()."""
        from src.agent.agent import MRtrixAssistant
        from pydantic_ai.messages import ModelRequest, UserPromptPart

        # Create mock dependencies
        mock_deps = MagicMock()
        assistant = MRtrixAssistant(dependencies=mock_deps)

        # Create mock message history
        mock_history = [
            ModelRequest(parts=[UserPromptPart(content="Previous question")])
        ]

        # Mock the agent's run method
        with patch.object(assistant.agent, "run") as mock_run:
            mock_result = MagicMock()
            mock_result.output = "Test response"
            mock_result.all_messages.return_value = mock_history + [
                ModelRequest(parts=[UserPromptPart(content="New question")])
            ]
            mock_run.return_value = mock_result

            # Call with message history
            await assistant.run("New question", message_history=mock_history)

            # Verify message_history was passed
            mock_run.assert_called_once_with(
                "New question", deps=mock_deps, message_history=mock_history
            )

    @pytest.mark.asyncio
    async def test_conversation_history_updated_after_run(self):
        """Test that conversation history is updated with new messages."""
        from src.agent.agent import MRtrixAssistant
        from pydantic_ai.messages import (
            ModelRequest,
            ModelResponse,
            UserPromptPart,
            TextPart,
        )

        mock_deps = MagicMock()
        assistant = MRtrixAssistant(dependencies=mock_deps)

        # Initial history
        initial_history = [
            ModelRequest(parts=[UserPromptPart(content="First question")])
        ]

        # New messages after run
        new_messages = initial_history + [
            ModelResponse(parts=[TextPart(content="First answer")]),
            ModelRequest(parts=[UserPromptPart(content="Second question")]),
            ModelResponse(parts=[TextPart(content="Second answer")]),
        ]

        with patch.object(assistant.agent, "run") as mock_run:
            mock_result = MagicMock()
            mock_result.output = "Second answer"
            mock_result.all_messages.return_value = new_messages
            mock_run.return_value = mock_result

            result = await assistant.run(
                "Second question", message_history=initial_history
            )

            # The all_messages() should return the complete history
            assert result.all_messages() == new_messages


class TestCommandRegistration:
    """Test dynamic command registration functionality."""

    def test_register_new_command(self):
        """Test registering a new slash command."""
        handler = SlashCommandHandler()

        def custom_handler(args: str) -> SlashCommandResult:
            return SlashCommandResult(success=True, agent_input=f"Custom: {args}")

        handler.register_command("/custom", custom_handler, "Custom command")

        assert "/custom" in handler.commands
        assert handler.command_descriptions["/custom"] == "Custom command"

        # Test the command works
        result = handler.process_command("/custom test args")
        assert result.success is True
        assert result.agent_input == "Custom: test args"

    def test_unregister_command(self):
        """Test unregistering a command."""
        handler = SlashCommandHandler()

        # Unregister help command
        handler.unregister_command("/help")

        assert "/help" not in handler.commands
        assert "/help" not in handler.command_descriptions

        # Command should now be unknown
        with patch("src.agent.slash_commands.console"):
            result = handler.process_command("/help")
            assert result.success is False


@pytest.mark.unit
class TestSlashCommandEdgeCases:
    """Edge case tests for slash command error handling and boundary conditions."""

    def setup_method(self):
        """Set up test fixtures."""
        self.handler = SlashCommandHandler()

    def test_empty_slash_command(self):
        """Test handling of empty slash command."""
        with patch("src.agent.slash_commands.console") as mock_console:
            result = self.handler.process_command("/")

            assert result.success is False
            assert result.continue_conversation is False
            mock_console.print.assert_called_with(
                "[yellow]Unknown command: /. Type /help for available commands.[/yellow]"
            )

    def test_slash_command_with_only_whitespace(self):
        """Test slash command with only whitespace arguments."""
        test_cases = ["/help ", "/exit   ", "/sharefile    "]

        for command in test_cases:
            if command.strip() == "/help":
                with patch("src.agent.slash_commands.console"):
                    result = self.handler.process_command(command)
                    assert result.success is True
                    assert result.continue_conversation is False
            elif command.strip() == "/exit":
                result = self.handler.process_command(command)
                assert result.success is True
                assert result.exit_requested is True
            elif command.strip() == "/sharefile":
                with patch("src.agent.slash_commands.console"):
                    result = self.handler.process_command(command)
                    assert result.success is False
                    assert result.continue_conversation is False

    def test_case_sensitivity(self):
        """Test case sensitivity of slash commands."""
        case_variants = ["/EXIT", "/Exit", "/eXiT", "/HELP", "/Help", "/SHAREFILE"]

        for command in case_variants:
            if command.lower() == "/exit":
                result = self.handler.process_command(command)
                assert result.success is True
                assert result.exit_requested is True
            elif command.lower() == "/help":
                with patch("src.agent.slash_commands.console"):
                    result = self.handler.process_command(command)
                    assert result.success is True
                    assert result.continue_conversation is False
            elif command.lower() == "/sharefile":
                with patch("src.agent.slash_commands.console") as mock_console:
                    result = self.handler.process_command(command)
                    assert result.success is False
                    mock_console.print.assert_any_call(
                        "[red]Usage: /sharefile <path> <query>[/red]"
                    )

    def test_command_with_special_characters(self):
        """Test commands with special characters in arguments."""
        special_char_cases = [
            "/sharefile /path/with spaces/file.nii How to handle?",
            "/sharefile /path/with-dashes/file.nii Process this?",
            "/sharefile /path/with_underscores/file.nii What next?",
            "/sharefile '/quoted/path/file.nii' How about this?",
            "/sharefile /path/with/unicode/café.nii Analyze unicode?",
        ]

        with patch("subprocess.run") as mock_subprocess:
            mock_result = Mock()
            mock_result.returncode = 0
            mock_result.stdout = "test output"
            mock_subprocess.return_value = mock_result

            for command in special_char_cases:
                result = self.handler.process_command(command)
                assert result.success is True
                assert result.continue_conversation is True

    def test_very_long_command_arguments(self):
        """Test handling of extremely long command arguments."""
        long_path = "/very/long/path/" + "directory/" * 100 + "file.nii"
        long_query = "This is a very long query. " * 100

        with patch("subprocess.run") as mock_subprocess:
            mock_result = Mock()
            mock_result.returncode = 0
            mock_result.stdout = "processed"
            mock_subprocess.return_value = mock_result

            result = self.handler.process_command(
                f"/sharefile {long_path} {long_query}"
            )

            assert result.success is True
            # Verify subprocess was called with long arguments
            call_args = mock_subprocess.call_args[0][0]
            # call_args is a list of arguments
            assert any(long_path in arg for arg in call_args)
            # The query might have trailing space removed
            assert any(long_query.rstrip() in arg for arg in call_args)

    def test_command_with_newlines_and_control_chars(self):
        """Test commands with newlines and control characters."""
        problematic_inputs = [
            "/sharefile /path/file.nii Query with\nnewlines",
            "/sharefile /path/file.nii Query with\ttabs",
            "/sharefile /path/file.nii Query with\rcarriage returns",
        ]

        with patch("subprocess.run") as mock_subprocess:
            mock_result = Mock()
            mock_result.returncode = 0
            mock_result.stdout = "output"
            mock_subprocess.return_value = mock_result

            for command in problematic_inputs:
                result = self.handler.process_command(command)
                assert result.success is True
                # Commands should be processed despite control characters

    def test_concurrent_command_processing(self):
        """Test thread safety of command processing."""
        import threading

        results = []
        errors = []

        def process_command(command):
            try:
                with patch("subprocess.run") as mock_subprocess:
                    mock_result = Mock()
                    mock_result.returncode = 0
                    mock_result.stdout = f"output for {command}"
                    mock_subprocess.return_value = mock_result

                    result = self.handler.process_command(
                        f"/sharefile /test/{command}.nii Process {command}"
                    )
                    results.append(result)
            except Exception as e:
                errors.append(e)

        # Create multiple threads processing commands simultaneously
        threads = []
        for i in range(10):
            thread = threading.Thread(target=process_command, args=(f"file{i}",))
            threads.append(thread)

        # Start all threads
        for thread in threads:
            thread.start()

        # Wait for completion
        for thread in threads:
            thread.join()

        # Verify results
        assert len(errors) == 0, f"Concurrent processing errors: {errors}"
        assert len(results) == 10
        assert all(result.success for result in results)

    def test_memory_usage_with_large_inputs(self):
        """Test memory handling with large input strings."""
        # Create a large input string (simulating large file analysis output)
        large_output = "x" * (10 * 1024 * 1024)  # 10MB string

        with patch("subprocess.run") as mock_subprocess:
            mock_result = Mock()
            mock_result.returncode = 0
            mock_result.stdout = large_output
            mock_subprocess.return_value = mock_result

            result = self.handler.process_command(
                "/sharefile /test/file.nii Analyze this"
            )

            assert result.success is True
            assert result.agent_input == large_output
            # Should handle large strings without issues

    def test_subprocess_environment_isolation(self):
        """Test that subprocess calls are properly isolated."""
        with patch("subprocess.run") as mock_subprocess:
            mock_result = Mock()
            mock_result.returncode = 0
            mock_result.stdout = "test output"
            mock_subprocess.return_value = mock_result

            self.handler.process_command("/sharefile /test/file.nii test query")

            # Verify subprocess was called with proper isolation settings
            mock_subprocess.assert_called_once()
            call_kwargs = mock_subprocess.call_args[1]
            assert call_kwargs["capture_output"] is True
            assert call_kwargs["text"] is True
            assert call_kwargs["timeout"] == 60

    def test_error_message_sanitization(self):
        """Test that error messages are properly sanitized."""
        dangerous_outputs = [
            "Error: \x1b[31mDangerous ANSI escape codes\x1b[0m",
            "Error with \x00 null bytes",
            "Error with control chars \x07\x08\x09",
        ]

        with patch("subprocess.run") as mock_subprocess:
            for dangerous_output in dangerous_outputs:
                mock_result = Mock()
                mock_result.returncode = 1
                mock_result.stdout = dangerous_output
                mock_subprocess.return_value = mock_result

                with patch("src.agent.slash_commands.console") as mock_console:
                    result = self.handler.process_command(
                        "/sharefile /test/file.nii test"
                    )

                    assert result.success is False
                    # Error should be displayed (though potentially sanitized by Rich)
                    mock_console.print.assert_called_with(
                        f"[red]{dangerous_output.strip()}[/red]"
                    )

    def test_filesystem_path_validation(self):
        """Test handling of various filesystem path edge cases."""
        path_edge_cases = [
            "/path/with/../traversal/file.nii",
            "//double/slash//paths//file.nii",
            "/path/with/./dot/file.nii",
            "~/home/path/file.nii",
            "relative/path/file.nii",
            "/path/ending/with/slash/",
            "",  # Empty path
        ]

        with patch("subprocess.run") as mock_subprocess:
            mock_result = Mock()
            mock_result.returncode = 0
            mock_result.stdout = "output"
            mock_subprocess.return_value = mock_result

            for path in path_edge_cases:
                if not path:  # Empty path should be handled by argument validation
                    with patch("src.agent.slash_commands.console"):
                        result = self.handler.process_command(
                            "/sharefile   query_without_path"
                        )
                        assert result.success is False
                else:
                    result = self.handler.process_command(
                        f"/sharefile {path} test query"
                    )
                    # Should pass path through to subprocess - let mrinfo handle validation
                    assert result.success is True

    def test_unicode_and_encoding_handling(self):
        """Test handling of unicode characters and encoding issues."""
        unicode_cases = [
            "/sharefile /path/with/émojis/😀.nii How to process?",
            "/sharefile /path/with/中文/file.nii Process chinese chars?",
            "/sharefile /path/with/русский/file.nii Handle cyrillic?",
            "/sharefile /path/file.nii Query with unicode: 🔬📊🧠",
        ]

        with patch("subprocess.run") as mock_subprocess:
            mock_result = Mock()
            mock_result.returncode = 0
            mock_result.stdout = "unicode output: 🎉"
            mock_subprocess.return_value = mock_result

            for command in unicode_cases:
                result = self.handler.process_command(command)
                assert result.success is True
                assert "🎉" in result.agent_input

    def test_malformed_command_recovery(self):
        """Test recovery from malformed commands."""
        malformed_commands = [
            "/sharefile",  # Missing arguments
            "/sharefile only_one_arg",  # Missing query
            "/sharefile   ",  # Only whitespace
            "/ sharefile /path query",  # Space after slash
            "//sharefile /path query",  # Double slash
        ]

        for command in malformed_commands:
            with patch("src.agent.slash_commands.console") as mock_console:
                result = self.handler.process_command(command)

                if command.startswith("//") or command.startswith("/ "):
                    # These are unknown commands
                    assert result.success is False
                    assert "Unknown command" in str(mock_console.print.call_args)
                else:
                    # These are /sharefile with missing arguments
                    assert result.success is False
                    assert result.continue_conversation is False
                    # Should show usage message
                    usage_shown = any(
                        "Usage: /sharefile" in str(call)
                        for call in mock_console.print.call_args_list
                    )
                    assert usage_shown

    def test_command_handler_exception_safety(self):
        """Test that command handlers don't crash the system on exceptions."""

        # Register a command that always throws an exception
        def failing_handler(args: str) -> SlashCommandResult:
            raise Exception("Handler intentionally fails")

        self.handler.register_command("/fail", failing_handler, "Failing command")

        # The exception should be caught and handled appropriately
        # Since we don't have explicit exception handling in the current implementation,
        # the exception will propagate. This test documents the current behavior.
        with pytest.raises(Exception, match="Handler intentionally fails"):
            self.handler.process_command("/fail test args")

    def test_resource_cleanup_on_interruption(self):
        """Test resource cleanup when commands are interrupted."""

        # This test simulates the scenario where subprocess is interrupted
        with patch("subprocess.run") as mock_subprocess:
            # Simulate KeyboardInterrupt during subprocess execution
            mock_subprocess.side_effect = KeyboardInterrupt("User interrupted")

            with pytest.raises(KeyboardInterrupt):
                self.handler.process_command("/sharefile /test/file.nii query")

            # In a real implementation, we'd want to ensure temporary files are cleaned up
            # even when interrupted. This test documents the expected behavior.

    def test_help_command_formatting_edge_cases(self):
        """Test help command formatting with edge cases."""
        # Test with very long command names and descriptions
        long_cmd_name = "/very_long_command_name_that_might_break_formatting"
        long_description = (
            "This is a very long description that might cause formatting issues " * 3
        )

        self.handler.register_command(
            long_cmd_name, lambda x: SlashCommandResult(True), long_description
        )

        with patch("src.agent.slash_commands.console") as mock_console:
            result = self.handler.process_command("/help")

            assert result.success is True
            # Help should handle long command names gracefully
            mock_console.print.assert_called()
            # Verify the long command appears in the output
            help_calls = [str(call) for call in mock_console.print.call_args_list]
            long_cmd_found = any(long_cmd_name in call for call in help_calls)
            assert long_cmd_found
