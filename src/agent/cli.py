"""MRtrix3 Assistant CLI interface."""

# Suppress absl logging warning - must be set before importing
import os

os.environ["GRPC_VERBOSITY"] = "ERROR"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import asyncio
import logging
import sys
import warnings
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from dotenv import load_dotenv

env_path = Path(__file__).parent.parent.parent / ".env"
load_dotenv(env_path, override=False)  # Load environment variables

# Now safe to import other modules that may use environment variables
import google.generativeai as genai  # noqa: E402
from rich.console import Console  # noqa: E402
from rich.markdown import Markdown  # noqa: E402

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
logging.getLogger("markdown_it").setLevel(
    logging.ERROR
)  # Suppress markdown parsing debug

# Suppress ALTS warnings
warnings.filterwarnings("ignore", message="ALTS creds ignored")

console = Console()


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
        # Set appropriate logging levels for collection
        logging.getLogger().setLevel(logging.DEBUG)
        logging.getLogger("agent.cli").setLevel(logging.INFO)
        logging.getLogger("agent.tools").setLevel(logging.INFO)

    console.print("[bold blue]MRtrix3 Assistant[/bold blue]")
    console.print("Initializing local database...\n")

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

    console.print("Ask me anything about MRtrix3! Type '/exit' or Ctrl+C to quit.\n")

    try:
        deps = await create_async_dependencies()
    except Exception as e:
        error_msg = get_user_friendly_message(e, "connecting to the knowledge base")
        console.print(f"[red]{error_msg}[/red]")
        return

    agent = MRtrixAssistant(dependencies=deps)
    token_manager = TokenManager()

    # Create a thread pool executor for input handling
    executor = ThreadPoolExecutor(max_workers=1)
    loop = asyncio.get_event_loop()

    try:
        while True:
            try:
                # Use executor for non-blocking input
                user_input = await loop.run_in_executor(executor, input, "You: ")

                if not user_input.strip():
                    continue

                if user_input.strip() == "/exit":
                    break

                if not await token_manager.add_message(user_input):
                    console.print(
                        "[yellow]Token limit reached. Starting new session.[/yellow]"
                    )
                    token_manager.reset()
                    await token_manager.add_message(user_input)

                # Token count is logged to file if logging is enabled

                # Log the user's query using session logger
                if session_logger:
                    session_logger.log_user_query(user_input)

                result = await agent.run(user_input)
                response_text = result.output

                await token_manager.add_message(response_text)

                # Log Gemini's response using session logger
                if session_logger:
                    session_logger.log_gemini_response(response_text)

                console.print("\n[bold cyan]Assistant:[/bold cyan]")
                console.print(Markdown(response_text))
                console.print()

            except (KeyboardInterrupt, EOFError):
                print()
                break

            except Exception as e:
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
    gemini_key = os.getenv("GOOGLE_API_KEY")
    if not gemini_key:
        console.print("[red]Error: GOOGLE_API_KEY not found in environment[/red]")
        console.print(f"[yellow]Attempted to load .env from: {env_path}[/yellow]")
        console.print(f"[yellow]File exists: {env_path.exists()}[/yellow]")
        sys.exit(1)

    genai.configure(api_key=gemini_key)

    await start_conversation()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except (KeyboardInterrupt, SystemExit):
        # Exit silently on Ctrl+C or system exit
        pass
    except Exception:
        # Exit silently on any other exception during shutdown
        pass
    finally:
        # Force exit to avoid thread cleanup issues
        import os

        os._exit(0)
