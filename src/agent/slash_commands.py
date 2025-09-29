"""
Slash command handlers for the MRtrix3 Assistant CLI.
Provides modular command processing for extensibility.
"""

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

        try:
            # Import and run the sharefile module directly instead of subprocess
            # This ensures it works when installed via PyPI
            from src.workflows.sharefile import sharefile

            # Call the main function directly with arguments
            import sys as sys_module

            original_argv = sys_module.argv
            try:
                # Simulate command line arguments
                sys_module.argv = ["sharefile", file_path, user_query]

                # Capture stdout
                from io import StringIO
                import contextlib

                stdout_capture = StringIO()
                with contextlib.redirect_stdout(stdout_capture):
                    # Run the sharefile main function
                    exit_code = 0
                    try:
                        sharefile.main()
                    except SystemExit as e:
                        exit_code = e.code if e.code is not None else 0

                output = stdout_capture.getvalue()

                if exit_code != 0:
                    # Script returned an error
                    console.print(f"[red]{output.strip()}[/red]")
                    return SlashCommandResult(
                        success=False, continue_conversation=False
                    )

                # Script succeeded - use output as input to agent
                prompt_content = output.strip()
                return SlashCommandResult(
                    success=True, continue_conversation=True, agent_input=prompt_content
                )
            finally:
                # Restore original argv
                sys_module.argv = original_argv

        except ImportError:
            console.print(
                "[red]Error: sharefile module not found. Please ensure the package is properly installed.[/red]"
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
