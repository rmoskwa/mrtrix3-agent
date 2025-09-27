"""
Slash command handlers for the MRtrix3 Assistant CLI.
Provides modular command processing for extensibility.
"""

import subprocess
import sys
from pathlib import Path
from typing import Optional, Dict, Callable
from rich.console import Console

console = Console()


class SlashCommandResult:
    """Result of processing a slash command."""

    def __init__(
        self,
        success: bool,
        continue_conversation: bool = True,
        agent_input: Optional[str] = None,
        exit_requested: bool = False,
    ):
        """
        Initialize command result.

        Args:
            success: Whether command executed successfully
            continue_conversation: Whether to continue the conversation loop
            agent_input: Optional input to send to agent (overrides user input)
            exit_requested: Whether to exit the application
        """
        self.success = success
        self.continue_conversation = continue_conversation
        self.agent_input = agent_input
        self.exit_requested = exit_requested


class SlashCommandHandler:
    """Handles slash command processing and dispatch."""

    def __init__(self):
        """Initialize the command handler with registered commands."""
        self.commands: Dict[str, Callable] = {
            "/exit": self._handle_exit,
            "/help": self._handle_help,
            "/sharefile": self._handle_sharefile,
        }

        self.command_descriptions = {
            "/exit": "Exit the application",
            "/help": "Show available commands",
            "/sharefile": "Share file metadata with the assistant",
        }

    def process_command(self, user_input: str) -> SlashCommandResult:
        """
        Process a potential slash command.

        Args:
            user_input: Raw user input

        Returns:
            SlashCommandResult indicating how to proceed
        """
        stripped_input = user_input.strip()

        # Not a slash command
        if not stripped_input.startswith("/"):
            return SlashCommandResult(success=True, agent_input=user_input)

        # Parse command and arguments
        parts = stripped_input.split(maxsplit=1)
        command = parts[0].lower()
        args = parts[1] if len(parts) > 1 else ""

        # Handle registered commands
        if command in self.commands:
            return self.commands[command](args)

        # Unknown command
        console.print(
            f"[yellow]Unknown command: {command}. Type /help for available commands.[/yellow]"
        )
        return SlashCommandResult(success=False, continue_conversation=False)

    def _handle_exit(self, args: str) -> SlashCommandResult:
        """Handle /exit command."""
        return SlashCommandResult(success=True, exit_requested=True)

    def display_help(self):
        """Display available commands (public method for external use)."""
        console.print("\n[bold]Available Commands:[/bold]")

        # Calculate max command length for alignment
        max_len = max(len(cmd) for cmd in self.command_descriptions.keys())

        for cmd, desc in self.command_descriptions.items():
            # Special formatting for sharefile to show usage
            if cmd == "/sharefile":
                console.print(f"  {cmd:<{max_len}} <path> <query>  - {desc}")
            else:
                console.print(f"  {cmd:<{max_len}}                 - {desc}")

        console.print()

    def _handle_help(self, args: str) -> SlashCommandResult:
        """Handle /help command."""
        self.display_help()
        return SlashCommandResult(success=True, continue_conversation=False)

    def _handle_sharefile(self, args: str) -> SlashCommandResult:
        """
        Handle /sharefile command.

        Args:
            args: Path to file/directory followed by user query

        Returns:
            SlashCommandResult with file metadata and user query
        """
        # Parse arguments - expecting: filepath query
        parts = args.strip().split(maxsplit=1)

        if len(parts) < 2:
            console.print("[red]Usage: /sharefile <path> <query>[/red]")
            console.print(
                "[yellow]Example: /sharefile /data/scan.nii How do I preprocess this?[/yellow]"
            )
            return SlashCommandResult(success=False, continue_conversation=False)

        file_path = parts[0]
        user_query = parts[1]

        # Build path to sharefile.py script
        script_path = (
            Path(__file__).parent.parent / "workflows" / "sharefile" / "sharefile.py"
        )

        try:
            # Execute the sharefile script with both filepath and query
            result = subprocess.run(
                [sys.executable, str(script_path), file_path, user_query],
                capture_output=True,
                text=True,
                timeout=60,
            )

            if result.returncode != 0:
                # Script returned an error
                console.print(f"[red]{result.stdout.strip()}[/red]")
                return SlashCommandResult(success=False, continue_conversation=False)

            # Script succeeded - use output as input to agent
            prompt_content = result.stdout.strip()
            return SlashCommandResult(
                success=True, continue_conversation=True, agent_input=prompt_content
            )

        except subprocess.TimeoutExpired:
            console.print("[red]Error: File analysis timed out[/red]")
            return SlashCommandResult(success=False, continue_conversation=False)
        except FileNotFoundError:
            console.print(
                f"[red]Error: sharefile.py script not found at {script_path}[/red]"
            )
            return SlashCommandResult(success=False, continue_conversation=False)
        except Exception as e:
            console.print(f"[red]Error analyzing file: {e}[/red]")
            return SlashCommandResult(success=False, continue_conversation=False)

    def register_command(
        self,
        command: str,
        handler: Callable[[str], SlashCommandResult],
        description: str = "",
    ):
        """
        Register a new slash command.

        Args:
            command: Command name (e.g., "/mycommand")
            handler: Function to handle the command
            description: Description for help text
        """
        command_lower = command.lower()
        self.commands[command_lower] = handler
        if description:
            self.command_descriptions[command_lower] = description

    def unregister_command(self, command: str):
        """
        Unregister a slash command.

        Args:
            command: Command name to remove
        """
        command_lower = command.lower()
        self.commands.pop(command_lower, None)
        self.command_descriptions.pop(command_lower, None)
