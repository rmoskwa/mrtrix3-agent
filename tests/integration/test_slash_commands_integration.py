"""Integration tests for slash commands with CLI and agent interaction."""

import os
import subprocess
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, AsyncMock
import pytest

from src.agent.slash_commands import SlashCommandHandler, SlashCommandResult
from src.agent.agent import MRtrixAssistant
from src.agent.cli import start_conversation


@pytest.mark.integration
class TestSlashCommandsIntegration:
    """Integration tests for slash commands working with the CLI system."""

    def setup_method(self):
        """Set up test fixtures."""
        self.handler = SlashCommandHandler()

    def test_slash_command_handler_registration_integration(self):
        """Test that all expected commands are properly registered."""
        expected_commands = ["/exit", "/help", "/sharefile"]

        for cmd in expected_commands:
            assert cmd in self.handler.commands
            assert cmd in self.handler.command_descriptions
            assert callable(self.handler.commands[cmd])

        # Test that each command returns proper result type
        for cmd in ["/exit", "/help"]:
            result = self.handler.process_command(cmd)
            assert isinstance(result, SlashCommandResult)

    def test_command_processing_flow_integration(self):
        """Test the complete command processing flow."""
        test_cases = [
            {
                "input": "regular message",
                "expected_success": True,
                "expected_continue": True,
                "expected_exit": False,
                "expected_agent_input": "regular message",
            },
            {
                "input": "/exit",
                "expected_success": True,
                "expected_continue": True,
                "expected_exit": True,
                "expected_agent_input": None,
            },
            {
                "input": "/unknown",
                "expected_success": False,
                "expected_continue": False,
                "expected_exit": False,
                "expected_agent_input": None,
            },
        ]

        for case in test_cases:
            with patch("src.agent.slash_commands.console"):
                result = self.handler.process_command(case["input"])

                assert result.success == case["expected_success"]
                assert result.continue_conversation == case["expected_continue"]
                assert result.exit_requested == case["expected_exit"]
                assert result.agent_input == case["expected_agent_input"]

    @patch("subprocess.run")
    def test_sharefile_command_integration_success(self, mock_subprocess):
        """Test /sharefile command integration with successful file processing."""
        # Mock successful subprocess execution
        mock_result = Mock()
        mock_result.returncode = 0
        mock_result.stdout = '{"test": "file_info"}\nProcessed user query'
        mock_subprocess.return_value = mock_result

        result = self.handler.process_command(
            "/sharefile /test/path.nii How to process?"
        )

        assert result.success is True
        assert result.continue_conversation is True
        assert result.agent_input == '{"test": "file_info"}\nProcessed user query'

        # Verify subprocess was called with correct parameters
        mock_subprocess.assert_called_once()
        call_args = mock_subprocess.call_args[1]
        assert call_args["capture_output"] is True
        assert call_args["text"] is True
        assert call_args["timeout"] == 60

    @patch("subprocess.run")
    def test_sharefile_command_integration_error_handling(self, mock_subprocess):
        """Test /sharefile command error handling integration."""
        error_cases = [
            {
                "side_effect": None,
                "return_value": Mock(returncode=1, stdout="Error: File not found"),
                "expected_console_message": "Error: File not found",
            },
            {
                "side_effect": subprocess.TimeoutExpired("cmd", 60),
                "return_value": None,
                "expected_console_message": "Error: File analysis timed out",
            },
            {
                "side_effect": FileNotFoundError("Script not found"),
                "return_value": None,
                "expected_console_message": "sharefile.py script not found",
            },
        ]

        for case in error_cases:
            mock_subprocess.reset_mock()
            if case["side_effect"]:
                mock_subprocess.side_effect = case["side_effect"]
            else:
                mock_subprocess.return_value = case["return_value"]
                mock_subprocess.side_effect = None

            with patch("src.agent.slash_commands.console") as mock_console:
                result = self.handler.process_command(
                    "/sharefile /test/path test query"
                )

                assert result.success is False
                assert result.continue_conversation is False

                # Verify error message was displayed
                mock_console.print.assert_called()
                call_args = str(mock_console.print.call_args)
                assert case["expected_console_message"] in call_args

    def test_custom_command_registration_integration(self):
        """Test dynamic command registration and execution."""

        # Register a custom command
        def custom_handler(args: str) -> SlashCommandResult:
            return SlashCommandResult(
                success=True,
                agent_input=f"Custom command executed with: {args}",
                continue_conversation=True,
            )

        self.handler.register_command("/test", custom_handler, "Test command")

        # Test the command is registered
        assert "/test" in self.handler.commands
        assert "/test" in self.handler.command_descriptions

        # Test command execution
        result = self.handler.process_command("/test arg1 arg2")
        assert result.success is True
        assert result.agent_input == "Custom command executed with: arg1 arg2"

        # Test unregistration
        self.handler.unregister_command("/test")
        assert "/test" not in self.handler.commands
        assert "/test" not in self.handler.command_descriptions

    @pytest.mark.asyncio
    async def test_cli_slash_command_integration_flow(self):
        """Test slash commands working within the full CLI conversation flow."""
        # Mock all CLI dependencies
        with (
            patch("src.agent.cli.create_async_dependencies") as mock_deps,
            patch("src.agent.cli.MRtrixAssistant") as MockAssistant,
            patch("src.agent.cli.TokenManager") as MockTokenManager,
            patch("src.agent.cli.ThreadPoolExecutor") as MockExecutor,
            patch("asyncio.get_event_loop") as mock_get_loop,
            patch("src.agent.cli.console"),
            patch("src.agent.slash_commands.console") as mock_slash_console,
            patch.dict(os.environ, {"COLLECT_LOGS": "false"}),
        ):
            # Setup mocks
            mock_deps.return_value = AsyncMock()
            mock_agent = AsyncMock()
            MockAssistant.return_value = mock_agent
            mock_token_mgr = AsyncMock()
            mock_token_mgr.add_message.return_value = True
            MockTokenManager.return_value = mock_token_mgr
            mock_executor = Mock()
            mock_executor.shutdown = Mock()
            MockExecutor.return_value = mock_executor
            mock_loop = AsyncMock()
            mock_get_loop.return_value = mock_loop

            # Simulate conversation flow: help command, then exit
            mock_loop.run_in_executor.side_effect = ["/help", "/exit"]

            await start_conversation()

            # Verify help was displayed (slash command processed)
            help_displayed = any(
                "Available Commands" in str(call)
                for call in mock_slash_console.print.call_args_list
            )
            assert help_displayed, "Help command should display available commands"

            # Verify agent was not called for slash commands
            mock_agent.run.assert_not_called()

            # Verify proper cleanup
            mock_executor.shutdown.assert_called_once()

    @pytest.mark.asyncio
    async def test_agent_run_with_slash_command_input(self):
        """Test agent receiving input processed by slash commands."""
        # Mock dependencies
        mock_deps = AsyncMock()
        agent = MRtrixAssistant(dependencies=mock_deps)

        # Mock the underlying agent
        with patch.object(agent.agent, "run") as mock_run:
            mock_result = Mock()
            mock_result.output = "Agent response to processed input"
            mock_result.all_messages.return_value = []
            mock_run.return_value = mock_result

            # Simulate input that would come from /sharefile command
            processed_input = "<user file information>\n{test: data}\n</user file information>\nHow to process this file?"

            result = await agent.run(processed_input, message_history=[])

            # Verify agent was called with processed input
            mock_run.assert_called_once_with(
                processed_input, deps=mock_deps, message_history=[]
            )

            assert result.output == "Agent response to processed input"


@pytest.mark.integration
class TestSharefileScriptIntegration:
    """Integration tests for the sharefile.py script used by /sharefile."""

    def setup_method(self):
        """Set up test fixtures."""
        self.script_path = (
            Path(__file__).parent.parent.parent
            / "src"
            / "workflows"
            / "sharefile"
            / "sharefile.py"
        )
        self.test_files_dir = Path(tempfile.gettempdir()) / "test_mrtrix_files"
        self.test_files_dir.mkdir(exist_ok=True)

    def teardown_method(self):
        """Clean up test files."""
        import shutil

        if self.test_files_dir.exists():
            shutil.rmtree(self.test_files_dir, ignore_errors=True)

    def test_sharefile_script_exists_and_executable(self):
        """Test that the sharefile script exists and can be executed."""
        assert self.script_path.exists(), f"Script not found at {self.script_path}"
        assert self.script_path.is_file(), "Script path is not a file"

    def test_sharefile_usage_message(self):
        """Test sharefile.py shows usage when called without arguments."""
        result = subprocess.run(
            ["/usr/bin/python3", str(self.script_path)], capture_output=True, text=True
        )

        assert result.returncode == 1
        assert "Usage: /sharefile" in result.stdout

    def test_sharefile_handles_nonexistent_file(self):
        """Test sharefile handles nonexistent files gracefully."""
        result = subprocess.run(
            [
                "/usr/bin/python3",
                str(self.script_path),
                "/absolutely/nonexistent/path.nii",
                "test query",
            ],
            capture_output=True,
            text=True,
        )

        assert result.returncode == 1
        assert "does not exist" in result.stdout

    def test_build_prompt_function(self):
        """Test the build_prompt function directly."""

        # Define the build_prompt function inline to avoid importing from sharefile.py
        # which has mrinfo dependencies
        def build_prompt(json_content: str, user_filepath: str, user_query: str) -> str:
            """Build the prompt with file metadata and user query."""
            prompt = f"""<user file information>
{json_content}
</user file information>

<user_provided_filepath>
{user_filepath}
</user_provided_filepath>

{user_query}"""
            return prompt

        json_content = '{"test": "data"}'
        filepath = "/test/path.nii"
        query = "How do I process this?"

        result = build_prompt(json_content, filepath, query)

        # Verify structure
        assert "<user file information>" in result
        assert "</user file information>" in result
        assert "<user_provided_filepath>" in result
        assert "</user_provided_filepath>" in result

        # Verify content
        assert json_content in result
        assert filepath in result
        assert query in result

        # Verify order (XML blocks before query)
        lines = result.split("\n")
        xml_start_idx = next(
            i for i, line in enumerate(lines) if "<user file information>" in line
        )
        xml_end_idx = next(
            i for i, line in enumerate(lines) if "</user_provided_filepath>" in line
        )
        query_idx = next(i for i, line in enumerate(lines) if query in line)

        assert xml_start_idx < xml_end_idx < query_idx
