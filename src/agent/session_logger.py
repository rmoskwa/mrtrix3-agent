"""Centralized session logging module for MRtrix3 Agent.

This module provides file-only logging for the application.
All logging output goes to files in monitoring/logs/, never to the terminal.
"""

import logging
from pathlib import Path
from datetime import datetime
from typing import Optional, TextIO, Any, List
from contextlib import contextmanager


class SessionLogger:
    """Manages session logging to files only (no terminal output)."""

    def __init__(self):
        """Initialize the session logger."""
        self.log_file: Optional[TextIO] = None
        self.log_file_path: Optional[Path] = None
        self.file_handler: Optional[logging.FileHandler] = None
        self.enabled = False

    def initialize(self, collect_logs: bool = False) -> Optional[Path]:
        """Initialize logging to file.

        Args:
            collect_logs: Whether to collect logs to a file

        Returns:
            Path to log file if logging is enabled, None otherwise
        """
        self.enabled = collect_logs

        if not collect_logs:
            return None

        # Create logs directory
        logs_dir = Path("monitoring/logs")
        logs_dir.mkdir(parents=True, exist_ok=True)

        # Create timestamped log file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file_path = logs_dir / f"session_{timestamp}.txt"

        try:
            # Open log file
            self.log_file = open(self.log_file_path, "w", encoding="utf-8")

            # Set up logging handlers (file only, no console)
            self._setup_logging_handlers()

            # Log session start
            self._log_session_start()

            # Session logging is active but no console output

            return self.log_file_path

        except Exception as e:
            print(f"[Warning: Could not create log file: {e}]")
            self.cleanup()
            return None

    def _setup_logging_handlers(self):
        """Configure Python logging to file only."""
        # Add file handler for Python logging
        self.file_handler = logging.FileHandler(
            self.log_file_path, mode="a", encoding="utf-8"
        )
        self.file_handler.setLevel(logging.DEBUG)
        self.file_handler.setFormatter(logging.Formatter("%(message)s"))

        # Configure loggers to use file handler
        for logger_name in [
            "agent.cli",
            "agent.tools",
            "agent.embedding",
            "agent.agent",
        ]:
            logger = logging.getLogger(logger_name)
            logger.setLevel(logging.DEBUG)
            logger.addHandler(self.file_handler)
            # Disable propagation to prevent console output
            logger.propagate = False

    def _log_session_start(self):
        """Log the session start message to file."""
        self.log_file.write(
            f"[Session started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}]\n"
        )
        self.log_file.write(f"[Log file: {self.log_file_path}]\n\n")
        self.log_file.flush()

    def cleanup(self):
        """Clean up logging resources."""
        # Remove file handler from loggers
        if self.file_handler:
            for logger_name in [
                "agent.cli",
                "agent.tools",
                "agent.embedding",
                "agent.agent",
            ]:
                logger = logging.getLogger(logger_name)
                logger.removeHandler(self.file_handler)
            self.file_handler.flush()
            self.file_handler.close()

        # Close log file
        if self.log_file:
            self.log_file.write(
                f"\n[Session ended: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}]\n"
            )
            self.log_file.close()
            # Session log saved but no console output

    @contextmanager
    def rag_search(self, query: str):
        """Context manager for RAG search logging.

        Args:
            query: The search query being executed
        """
        if self.enabled:
            # Log RAG search start
            logger = logging.getLogger("agent.tools")
            logger.info("\n" + "=" * 60)
            logger.info(f"🔍 GEMINI RAG SEARCH QUERY: {query}")
            logger.info("=" * 60)

        try:
            yield
        finally:
            if self.log_file:
                self.log_file.flush()

    def log_rag_results(self, results: List[Any]):
        """Log RAG search results.

        Args:
            results: List of search results
        """
        if self.enabled:
            logger = logging.getLogger("agent.tools")
            logger.info("\n" + "=" * 60)
            logger.info(f"📚 RAG SEARCH RESULTS: Retrieved {len(results)} documents")
            logger.info("=" * 60)
            for i, result in enumerate(results, 1):
                if hasattr(result, "title"):
                    logger.info(f"  {i}. {result.title}")
            logger.info("=" * 60 + "\n")

    def log_user_query(self, query: str):
        """Log user query.

        Args:
            query: The user's input query
        """
        if self.enabled:
            logger = logging.getLogger("agent.cli")
            logger.info("\n" + "=" * 60)
            logger.info(f"🔵 USER QUERY: {query}")
            logger.info("=" * 60 + "\n")

    def log_gemini_response(self, response: str):
        """Log Gemini's response.

        Args:
            response: The response text (full response will be logged)
        """
        if self.enabled:
            logger = logging.getLogger("agent.cli")
            logger.info("\n" + "=" * 60)
            logger.info("🤖 GEMINI RESPONSE:")
            logger.info("=" * 60)
            logger.info(f"[Response length: {len(response)} characters]")
            # Log the full response for debugging
            logger.info(response)
            logger.info("=" * 60 + "\n")


# Global session logger instance
_session_logger: Optional[SessionLogger] = None


def get_session_logger() -> Optional[SessionLogger]:
    """Get the global session logger instance."""
    return _session_logger


def initialize_session_logger(collect_logs: bool = False) -> Optional[SessionLogger]:
    """Initialize the global session logger.

    Args:
        collect_logs: Whether to collect logs to a file

    Returns:
        The initialized SessionLogger instance, or None if logging is disabled
    """
    global _session_logger

    if _session_logger:
        # Clean up existing logger
        _session_logger.cleanup()

    _session_logger = SessionLogger()
    log_path = _session_logger.initialize(collect_logs)

    if log_path is None:
        _session_logger = None

    return _session_logger


def cleanup_session_logger():
    """Clean up the global session logger."""
    global _session_logger

    if _session_logger:
        _session_logger.cleanup()
        _session_logger = None
