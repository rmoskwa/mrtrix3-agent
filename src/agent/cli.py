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

import google.generativeai as genai
from dotenv import load_dotenv
from rich.console import Console
from rich.markdown import Markdown

from src.agent.agent import MRtrixAssistant
from src.agent.async_dependencies import create_async_dependencies
from src.agent.error_messages import get_user_friendly_message, log_and_get_message

# Load .env file from project root
env_path = Path(__file__).parent.parent.parent / ".env"
load_dotenv(env_path, override=True)

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
    debug_mode = os.getenv("DEBUG_MODE", "false").lower() == "true"
    verbose = os.getenv("VERBOSE_MODE", "false").lower() == "true"

    if debug_mode:
        logging.getLogger().setLevel(logging.DEBUG)

    console.print("[bold blue]MRtrix3 Assistant[/bold blue]")
    console.print("Ask me anything about MRtrix3! Type '/exit' or Ctrl+C to quit.\n")

    try:
        deps = await create_async_dependencies()
    except Exception as e:
        error_msg = get_user_friendly_message(e, "connecting to the knowledge base")
        console.print(f"[red]{error_msg}[/red]")
        if debug_mode:
            console.print(f"[dim]Debug: {e}[/dim]")
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

                # Only show token count in verbose mode
                if verbose:
                    console.print(
                        f"[dim]Tokens used: {token_manager.total_tokens}/{TokenManager.MAX_TOKENS}[/dim]"
                    )

                result = await agent.run(user_input)
                response_text = result.output

                await token_manager.add_message(response_text)

                console.print("\n[bold cyan]Assistant:[/bold cyan]")
                console.print(Markdown(response_text))
                console.print()

            except (KeyboardInterrupt, EOFError):
                print()
                break

            except Exception as e:
                # Log error with context and get user-friendly message
                user_message = log_and_get_message(
                    e,
                    severity="error",
                    user_query=user_input if "user_input" in locals() else None,
                    tool_name="conversation_loop",
                )

                if debug_mode:
                    import traceback

                    traceback.print_exc()
                    console.print(f"\n[red]Debug: {e}[/red]")

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
