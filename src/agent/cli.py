"""MRtrix3 Assistant CLI interface."""

# Suppress absl logging warning - must be set before importing
import os

os.environ["GRPC_VERBOSITY"] = "ERROR"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["GRPC_ENABLE_FORK_SUPPORT"] = "0"  # Disable fork support to reduce warnings

import asyncio
import logging
import os as os_module
import sys
import warnings
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import threading
import select
import termios

try:
    import readline  # Enable readline support for arrow keys

    readline.parse_and_bind("tab: complete")  # Use readline to enable tab completion
except ImportError:
    pass  # readline not available on all platforms

from dotenv import load_dotenv

env_path = Path(__file__).parent.parent.parent / ".env"
load_dotenv(env_path, override=False)  # Load environment variables


class StderrFilter:
    """Filters stderr at the file descriptor level to suppress ALTS warnings."""

    def __init__(self):
        self.original_stderr_fd = None
        self.pipe_read = None
        self.pipe_write = None
        self.filter_thread = None
        self.stop_filtering = False

    def start(self):
        """Start filtering stderr."""
        # Save original stderr
        self.original_stderr_fd = os_module.dup(sys.stderr.fileno())

        # Create pipe for capturing stderr
        self.pipe_read, self.pipe_write = os_module.pipe()

        # Make read end non-blocking
        import fcntl

        flags = fcntl.fcntl(self.pipe_read, fcntl.F_GETFL)
        fcntl.fcntl(self.pipe_read, fcntl.F_SETFL, flags | os_module.O_NONBLOCK)

        # Redirect stderr to our pipe
        os_module.dup2(self.pipe_write, sys.stderr.fileno())
        os_module.close(self.pipe_write)

        # Start filter thread
        self.stop_filtering = False
        self.filter_thread = threading.Thread(target=self._filter_loop, daemon=True)
        self.filter_thread.start()

    def _filter_loop(self):
        """Background thread that filters stderr output."""
        while not self.stop_filtering:
            try:
                # Use select with timeout to check for data
                ready, _, _ = select.select([self.pipe_read], [], [], 0.1)
                if ready:
                    data = os_module.read(self.pipe_read, 4096)
                    if data:
                        # Decode and filter
                        text = data.decode("utf-8", errors="ignore")

                        # Filter out ALTS warnings line by line
                        lines = text.split("\n")
                        filtered_lines = []
                        for line in lines:
                            if not any(
                                msg in line
                                for msg in [
                                    "ALTS creds ignored",
                                    "All log messages before absl::InitializeLog()",
                                    "alts_credentials.cc",
                                ]
                            ):
                                filtered_lines.append(line)

                        # Write filtered output to original stderr
                        filtered_text = "\n".join(filtered_lines)
                        if filtered_text.strip():
                            os_module.write(
                                self.original_stderr_fd, filtered_text.encode("utf-8")
                            )
            except (OSError, BlockingIOError):
                pass
            except Exception:
                pass  # Ignore errors in filter thread

    def stop(self):
        """Stop filtering and restore original stderr."""
        if self.filter_thread:
            self.stop_filtering = True
            self.filter_thread.join(timeout=1)

        if self.original_stderr_fd is not None:
            # Restore original stderr
            os_module.dup2(self.original_stderr_fd, sys.stderr.fileno())
            os_module.close(self.original_stderr_fd)

        if self.pipe_read is not None:
            os_module.close(self.pipe_read)


# Create global stderr filter instance (but don't start it yet)
stderr_filter = StderrFilter()

# Now safe to import other modules that may use environment variables
import google.generativeai as genai  # noqa: E402
from rich.console import Console, ConsoleOptions, RenderResult  # noqa: E402
from rich.markdown import Markdown, CodeBlock  # noqa: E402
from rich.progress import Progress, SpinnerColumn, TextColumn  # noqa: E402
from rich.syntax import Syntax  # noqa: E402
from rich.text import Text  # noqa: E402
from rich.panel import Panel  # noqa: E402
from rich.theme import Theme  # noqa: E402
from pyfiglet import figlet_format  # noqa: E402

from src.agent.agent import MRtrixAssistant  # noqa: E402
from src.agent.async_dependencies import create_async_dependencies  # noqa: E402
from src.agent.error_messages import get_user_friendly_message  # noqa: E402
from src.agent.sync_manager import DatabaseSyncManager  # noqa: E402
from src.agent.dependencies import validate_environment  # noqa: E402
from src.agent.local_storage_manager import LocalDatabaseManager  # noqa: E402
from src.agent.session_logger import (  # noqa: E402
    initialize_session_logger,
    cleanup_session_logger,
)
from src.agent.slash_commands import SlashCommandHandler  # noqa: E402

# Set up logging - only show warnings and above by default
logging.basicConfig(level=logging.WARNING, format="%(message)s")
logger = logging.getLogger("agent.cli")

# Suppress verbose HTTP and AI service logging
logging.getLogger("httpcore").setLevel(logging.ERROR)
logging.getLogger("httpx").setLevel(logging.ERROR)
logging.getLogger("hpack").setLevel(logging.ERROR)
logging.getLogger("google_genai").setLevel(logging.ERROR)
logging.getLogger("agent.tools").setLevel(logging.WARNING)
logging.getLogger("agent.embedding").setLevel(logging.WARNING)
logging.getLogger("markdown_it").setLevel(logging.ERROR)
# Suppress ChromaDB's internal logging
logging.getLogger("chromadb").setLevel(logging.ERROR)
logging.getLogger("chromadb.config").setLevel(logging.ERROR)

# Suppress ALTS warnings
warnings.filterwarnings("ignore", message="ALTS creds ignored")

# Create custom theme for inline code (no background, bold)
custom_theme = Theme(
    {
        "markdown.code": "bold bright_blue",  # Inline code - bold blue with no background
    }
)

console = Console(theme=custom_theme)


def prettier_code_blocks():
    """Make rich code blocks prettier and easier to copy."""

    class SimpleCodeBlock(CodeBlock):
        def __rich_console__(
            self, console: Console, options: ConsoleOptions
        ) -> RenderResult:
            code = str(self.text).rstrip()
            width = options.max_width if options.max_width else 80

            # Create dynamic border based on console width
            border_width = min(width - len(self.lexer_name) - 4, 60)

            # Top border with language label
            yield Text(
                f"╭─[ {self.lexer_name} ]" + "─" * border_width, style="bright_black"
            )

            # Create syntax with highlighting
            syntax = Syntax(
                code,
                self.lexer_name,
                theme="monokai",  # Better contrast theme
                background_color=None,  # Use terminal's native background
                word_wrap=True,
                line_numbers=False,
                indent_guides=False,
                code_width=width,
            )

            # Render the syntax and apply bold to the entire output
            from rich.style import Style
            from rich.segment import Segment

            bold_style = Style(bold=True)

            # Render the syntax and apply bold to each segment
            for segment in console.render(syntax, options):
                # Apply bold to each segment
                if segment.text:
                    # Create a new segment with bold style
                    new_style = (segment.style or Style()) + bold_style
                    yield Segment(segment.text, new_style)
                else:
                    yield segment

            # Bottom border
            yield Text(
                "╰" + "─" * (border_width + len(self.lexer_name) + 5),
                style="bright_black",
            )

    # Apply custom styles to Markdown elements
    Markdown.elements["fence"] = SimpleCodeBlock


# Apply prettier code blocks on module load
prettier_code_blocks()


async def get_user_input(loop, executor) -> str:
    """Get user input with styled prompt."""
    # Add spacing
    console.print()

    # Simple "User" label in green
    console.print("[bold green]User[/bold green]")
    console.print("[bold green]----[/bold green]")

    console.print("[bright_green]▶ [/bright_green]", end="")
    sys.stdout.flush()  # Ensure prompt is visible before input

    try:
        user_input = await loop.run_in_executor(executor, input)
        return user_input
    except (KeyboardInterrupt, asyncio.CancelledError):
        # Handle interruption gracefully
        raise KeyboardInterrupt()


class TokenManager:
    """Manages conversation token count with 500k limit."""

    MAX_TOKENS = 500000

    def __init__(self, model_name: str = "gemini-2.5-flash"):
        """Initialize token manager with specified model.

        Args:
            model_name: Name of the Gemini model for token counting
        """
        self.model = genai.GenerativeModel(model_name)
        self.total_tokens = 0
        self.message_history = []

    async def count_tokens(self, text: str) -> int:
        """Count tokens using Gemini API.

        Args:
            text: Text to count tokens for

        Returns:
            Number of tokens in the text
        """
        try:
            count_result = await asyncio.to_thread(self.model.count_tokens, text)
            return count_result.total_tokens
        except Exception as e:
            logger.warning(f"Token counting failed: {e}. Estimating tokens.")
            return len(text) // 4

    async def add_message(self, message: str) -> bool:
        """Add message and check if within limit.

        Args:
            message: Message to add to conversation

        Returns:
            True if message fits within limit, False if limit exceeded
        """
        tokens = await self.count_tokens(message)

        if self.total_tokens + tokens > self.MAX_TOKENS:
            await self._trim_history()
            if self.total_tokens + tokens > self.MAX_TOKENS:
                return False

        self.total_tokens += tokens
        self.message_history.append((message, tokens))
        return True

    async def _trim_history(self):
        """Trim oldest messages to make room in token window."""
        while self.message_history and self.total_tokens > self.MAX_TOKENS * 0.8:
            _, tokens = self.message_history.pop(0)
            self.total_tokens -= tokens

    def reset(self):
        """Reset token manager for new conversation."""
        self.total_tokens = 0
        self.message_history = []


async def start_conversation():
    """Main conversation loop."""
    # Single variable controls logging (to file only, never to terminal)
    collect_logs = os.getenv("COLLECT_LOGS", "false").lower() == "true"

    # Initialize session logging
    session_logger = initialize_session_logger(collect_logs=collect_logs)

    if collect_logs:
        # Don't set root logger level here - let session logger handle it
        # This prevents debug messages from appearing in the console
        pass

    # Start without title - MRtrixBot will be shown after sync

    # Run sync check before creating dependencies
    local_manager = None
    try:
        env_vars = validate_environment()

        # Initialize local database manager with health check
        local_manager = LocalDatabaseManager(env_vars["CHROMADB_PATH"])

        # Acquire lock for exclusive access
        if not local_manager.lock_manager.acquire_lock():
            console.print("[yellow]Warning: Another instance may be running[/yellow]")
            console.print("Continuing anyway...\n")

        # Perform health check
        is_healthy, issues = local_manager.health_check()
        if not is_healthy:
            console.print(f"[yellow]Database health issues detected: {issues}[/yellow]")
            if local_manager.recover_database():
                console.print("[green]Database recovered successfully[/green]")
            else:
                console.print("[yellow]Continuing with potential issues...[/yellow]")

        # Initialize collection with schema migration if needed
        local_manager.initialize_collection()

        # Clean up old temp files
        cleaned = local_manager.cleanup_temp_files(older_than_hours=24)
        if cleaned > 0:
            console.print(f"[dim]Cleaned up {cleaned} temporary files[/dim]")

        # Create sync manager and check for updates
        from supabase import create_client

        supabase_client = create_client(
            env_vars["SUPABASE_URL"], env_vars["SUPABASE_KEY"]
        )

        sync_manager = DatabaseSyncManager(
            supabase_client=supabase_client,
            chromadb_client=local_manager.client,
            chromadb_path=env_vars["CHROMADB_PATH"],
        )

        # Perform sync on startup
        sync_manager.sync_on_startup()

    except Exception as e:
        console.print(f"[yellow]Warning: Database initialization issue: {e}[/yellow]")
        console.print("Continuing with local database...\n")
    finally:
        # Release lock when done
        if local_manager:
            local_manager.lock_manager.release_lock()

    # Display MRtrixBot with ASCII art for larger text
    console.print()
    # Create ASCII art text with a smaller font for better fit
    ascii_text = figlet_format("MRtrixBot", font="big")
    # Remove trailing newline if present
    ascii_text = ascii_text.rstrip()
    panel = Panel(
        f"[bold blue]{ascii_text}[/bold blue]",
        style="bold blue",
        border_style="red",
        expand=False,
        padding=(1, 2),
    )
    console.print(panel, justify="center")
    console.print()

    try:
        deps = await create_async_dependencies()
    except Exception as e:
        error_msg = get_user_friendly_message(e, "connecting to the knowledge base")
        console.print(f"[red]{error_msg}[/red]")
        return

    agent = MRtrixAssistant(dependencies=deps)
    token_manager = TokenManager()
    slash_handler = SlashCommandHandler()

    # Display available commands at startup
    slash_handler.display_help()

    # Conversation history for maintaining context
    conversation_history = []

    # Create a thread pool executor for input handling
    executor = ThreadPoolExecutor(max_workers=1)
    loop = asyncio.get_event_loop()

    try:
        while True:
            try:
                # Get user input with styled prompt
                user_input = await get_user_input(loop, executor)

                if not user_input.strip():
                    continue

                # Process potential slash commands
                command_result = slash_handler.process_command(user_input)

                # Handle command results
                if command_result.exit_requested:
                    break

                if not command_result.continue_conversation:
                    continue

                # Use agent_input if provided by command, otherwise use original user_input
                processing_input = command_result.agent_input or user_input

                if not await token_manager.add_message(processing_input):
                    console.print(
                        "[yellow]Token limit reached. Starting new session.[/yellow]"
                    )
                    token_manager.reset()
                    conversation_history = []  # Reset conversation history too
                    await token_manager.add_message(processing_input)

                # Token count is logged to file if logging is enabled

                # Log the user's query using session logger
                if session_logger:
                    session_logger.log_user_query(processing_input)

                # Run the agent - use non-streaming for reliability
                console.print("\n[bold red]Assistant:[/bold red]")
                console.print("[bold red]----------[/bold red]")

                # Show a spinner while the agent is thinking
                with Progress(
                    SpinnerColumn(),
                    TextColumn("[progress.description]{task.description}"),
                    console=console,
                    transient=True,  # Remove spinner when done
                ) as progress:
                    task = progress.add_task("[cyan]Thinking...", total=None)

                    # Run the agent
                    result = await agent.run(
                        processing_input, message_history=conversation_history
                    )

                    # Update spinner to show we're processing the response
                    if result.all_messages() and len(result.all_messages()) > 4:
                        # Multiple tool calls were made
                        progress.update(
                            task, description="[cyan]Organizing search results..."
                        )
                    else:
                        progress.update(
                            task, description="[cyan]Finalizing response..."
                        )

                    # Mark task as complete
                    progress.update(task, completed=100)

                # Get the complete response - check multiple sources
                full_response = ""

                # First try result.output (may be incomplete with multiple tool calls)
                if result.output:
                    full_response = result.output
                    logger.debug(
                        f"Got response from result.output: {len(full_response)} chars"
                    )

                # Also check the message history for the actual response
                # PydanticAI often splits the response across multiple TextParts in messages
                # We need to reconstruct the complete response from all parts
                all_text_parts = []
                assistant_message_parts = []

                if result.all_messages():
                    logger.debug(
                        f"Checking {len(result.all_messages())} messages for complete response..."
                    )

                    # Find where the NEW messages start (after the previous conversation_history)
                    prev_history_length = (
                        len(conversation_history) if conversation_history else 0
                    )

                    # Only process NEW messages from this conversation turn
                    new_messages = result.all_messages()[prev_history_length:]
                    logger.debug(
                        f"Processing {len(new_messages)} new messages from current turn (skipping {prev_history_length} previous)"
                    )

                    # Collect text parts only from the NEW assistant message(s)
                    for idx, msg in enumerate(new_messages):
                        msg_parts = []
                        if hasattr(msg, "parts"):
                            for part in msg.parts:
                                part_type = (
                                    part.__class__.__name__
                                    if hasattr(part, "__class__")
                                    else ""
                                )
                                if part_type == "TextPart" and hasattr(part, "content"):
                                    if part.content and len(part.content.strip()) > 10:
                                        msg_parts.append(part.content)
                                        all_text_parts.append(part.content)
                                        logger.debug(
                                            f"New message {idx} TextPart: {len(part.content)} chars"
                                        )

                        # If this message had text parts, it's likely an assistant response
                        if msg_parts:
                            assistant_message_parts.append((idx, msg_parts))

                # Try to reconstruct the complete response
                if assistant_message_parts:
                    # Check if we have response parts spread across multiple messages
                    # This happens when the agent makes multiple tool calls
                    if len(assistant_message_parts) > 1:
                        logger.debug(
                            f"Found response parts across {len(assistant_message_parts)} messages"
                        )

                        # Combine ALL assistant message parts in order
                        all_parts_combined = []
                        for _, parts in assistant_message_parts:
                            all_parts_combined.extend(parts)

                        # Join all parts with proper spacing
                        complete_response = "\n".join(all_parts_combined)
                        logger.debug(
                            f"Reconstructed response from {len(all_parts_combined)} total parts: {len(complete_response)} chars"
                        )

                        # Always use the reconstructed response when fragmented
                        # (PydanticAI's result.output is unreliable with multiple tool calls)
                        logger.debug(
                            "Using fully reconstructed response (fragmented across messages)"
                        )
                        full_response = complete_response

                    else:
                        # Single message with multiple parts
                        last_assistant_idx, last_assistant_parts = (
                            assistant_message_parts[-1]
                        )
                        if len(last_assistant_parts) > 1:
                            logger.debug(
                                f"Found multi-part response in single message {last_assistant_idx}"
                            )
                            reconstructed = "\n".join(last_assistant_parts)
                            # Use reconstructed version for multi-part single messages
                            full_response = reconstructed

                # Now display the complete response
                # We have the full response, so we can render it properly with Markdown
                if full_response:
                    # Check if response is very long and might cause display issues
                    if len(full_response) > 5000:
                        # Split into chunks for very long responses
                        lines = full_response.split("\n")
                        current_chunk = ""
                        chunks = []

                        for line in lines:
                            if (
                                len(current_chunk) + len(line) + 1 > 4000
                            ):  # Keep chunks under 4000 chars
                                if current_chunk:
                                    chunks.append(current_chunk)
                                current_chunk = line
                            else:
                                current_chunk = (
                                    current_chunk + "\n" + line
                                    if current_chunk
                                    else line
                                )

                        if current_chunk:
                            chunks.append(current_chunk)

                        # Render each chunk with Markdown
                        for chunk in chunks:
                            console.print(Markdown(chunk))
                    else:
                        # Normal length - render with Markdown formatting
                        console.print(Markdown(full_response))

                # Update conversation history with complete message list from result
                conversation_history = result.all_messages()

                # Add newline after response
                console.print()

                # Log the complete response
                logger.debug(f"Response length: {len(full_response)} characters")
                if full_response and len(full_response) > 1000:
                    logger.debug(
                        f"Response preview (last 200 chars): ...{full_response[-200:]}"
                    )

                await token_manager.add_message(full_response)

                # Log Gemini's response using session logger
                if session_logger:
                    session_logger.log_gemini_response(full_response)

            except (KeyboardInterrupt, EOFError):
                print()  # Print newline for clean exit
                break  # Exit the loop cleanly

            except Exception as e:
                # Log the actual error for debugging
                logger.error(f"Error processing request: {e}", exc_info=True)

                # Get user-friendly message
                user_message = get_user_friendly_message(e, "processing your request")
                console.print(f"\n[yellow]{user_message}[/yellow]\n")

    finally:
        # Properly shutdown the executor
        executor.shutdown(wait=False, cancel_futures=True)

        # Cleanup resources properly
        try:
            if hasattr(deps, "cleanup"):
                await deps.cleanup()
            elif hasattr(deps, "supabase_client"):
                # Ensure Supabase client is closed even if no explicit cleanup method
                if hasattr(deps.supabase_client, "close"):
                    await deps.supabase_client.close()
        except Exception:
            # Ignore cleanup errors during exit
            pass

        # Clean up session logging
        if session_logger:
            cleanup_session_logger()


async def main():
    """Entry point for CLI application."""
    # Validate setup before starting
    from .validate_setup import check_setup

    success, message = check_setup()
    if not success:
        console.print(f"[red]❌ Setup Error:[/red] {message}")
        sys.exit(1)

    # Save original terminal settings
    original_term_settings = None
    try:
        original_term_settings = termios.tcgetattr(sys.stdin)
    except (OSError, termios.error):
        pass  # Not a terminal or termios not available

    try:
        # Start stderr filtering only when running as main CLI
        stderr_filter.start()

        gemini_key = os.getenv("GOOGLE_API_KEY")
        if not gemini_key:
            console.print("[red]❌ Error: GOOGLE_API_KEY not found[/red]")
            console.print(
                "[yellow]Please run 'mrtrixBot-setup' to configure your API key.[/yellow]"
            )
            sys.exit(1)

        genai.configure(api_key=gemini_key)

        await start_conversation()
    finally:
        # Always restore terminal settings
        if original_term_settings:
            try:
                termios.tcsetattr(sys.stdin, termios.TCSANOW, original_term_settings)
            except (OSError, termios.error):
                pass


def run():
    """Entry point for the mrtrixBot command."""
    # Save terminal settings at the very start
    original_settings = None
    try:
        original_settings = termios.tcgetattr(sys.stdin)
    except (OSError, termios.error):
        pass

    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        # Exit cleanly on Ctrl+C
        print()  # Print newline for clean terminal prompt
        os_module._exit(0)  # Use os._exit for immediate termination
    except SystemExit:
        # Let SystemExit propagate normally
        raise
    except Exception:
        # Exit silently on any other exception during shutdown
        pass
    finally:
        # Restore terminal settings before any cleanup
        if original_settings:
            try:
                termios.tcsetattr(sys.stdin, termios.TCSANOW, original_settings)
                # Also reset terminal to sane state
                sys.stdout.write("\033[0m")  # Reset all attributes
                sys.stdout.flush()
            except (OSError, termios.error):
                pass

        # Clean up stderr filter
        stderr_filter.stop()


if __name__ == "__main__":
    run()
