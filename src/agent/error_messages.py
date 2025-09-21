"""
User-friendly error message mappings for the MRtrix3 agent.

Maps technical errors to friendly messages that don't expose internal details.
"""

import logging
from typing import Dict, Optional

logger = logging.getLogger(__name__)


class ErrorMessageMapper:
    """Maps technical errors to user-friendly messages."""

    # Default error messages for common error types
    ERROR_MESSAGES: Dict[type, str] = {
        ConnectionError: "I'm having trouble accessing the knowledge base. Please try again in a moment.",
        TimeoutError: "The request is taking longer than expected. Please try again.",
        ValueError: "I encountered an issue with your request. Could you rephrase it?",
        MemoryError: "I'm running low on resources. Please try a simpler query.",
        KeyError: "I couldn't find what you're looking for. Please check your request.",
    }

    # Specific error patterns and their messages
    ERROR_PATTERNS: Dict[str, str] = {
        "rate limit": "I'm processing too many requests right now. Please wait a moment and try again.",
        "circuit breaker": "The knowledge base is temporarily unavailable. Please try again in a few seconds.",
        "embedding": "I'm having trouble understanding your query. Let me try another approach.",
        "supabase": "I'm having trouble accessing the documentation database. Please try again.",
        "api key": "There's a configuration issue. Please contact support.",
        "network": "There seems to be a network issue. Please check your connection.",
        "authentication": "There's an authentication issue. Please check your credentials.",
        "not found": "I couldn't find any relevant documentation for your query.",
        "invalid": "Your request contains invalid parameters. Please check and try again.",
        "quota": "API quota exceeded. Please try again later.",
    }

    # Severity-based messages for logging context
    SEVERITY_MESSAGES: Dict[str, str] = {
        "critical": "A critical error occurred. Please contact support if this persists.",
        "error": "An error occurred while processing your request. Please try again.",
        "warning": "The system is experiencing some issues but your request will be processed.",
        "info": "Your request is being processed.",
    }

    @classmethod
    def get_user_message(
        cls,
        error: Exception,
        context: Optional[str] = None,
        include_retry_hint: bool = True,
    ) -> str:
        """
        Get user-friendly message for an error.

        Args:
            error: The exception that occurred
            context: Optional context about where the error occurred
            include_retry_hint: Whether to include retry suggestion

        Returns:
            User-friendly error message
        """
        # Check for specific error patterns in the error message
        error_str = str(error).lower()
        for pattern, message in cls.ERROR_PATTERNS.items():
            if pattern in error_str:
                logger.debug(f"Matched error pattern '{pattern}' for: {error}")
                return cls._format_message(message, include_retry_hint)

        # Check for error type
        error_type = type(error)
        if error_type in cls.ERROR_MESSAGES:
            logger.debug(f"Matched error type '{error_type.__name__}' for: {error}")
            return cls._format_message(
                cls.ERROR_MESSAGES[error_type], include_retry_hint
            )

        # Generic fallback message
        logger.warning(f"No specific mapping for error: {error}")
        base_message = "I encountered an unexpected issue"

        if context:
            base_message += f" while {context}"

        return cls._format_message(base_message, include_retry_hint)

    @classmethod
    def _format_message(cls, base_message: str, include_retry_hint: bool) -> str:
        """
        Format message with optional retry hint.

        Args:
            base_message: The base error message
            include_retry_hint: Whether to add retry suggestion

        Returns:
            Formatted message
        """
        if include_retry_hint and not base_message.endswith(
            ("Please try again.", "Please try again later.", "try again in a moment.")
        ):
            return f"{base_message} Please try again."
        return base_message

    @classmethod
    def log_error_with_context(
        cls,
        error: Exception,
        severity: str = "error",
        user_query: Optional[str] = None,
        tool_name: Optional[str] = None,
        additional_context: Optional[Dict] = None,
    ) -> None:
        """
        Log error with structured context while preserving privacy.

        Args:
            error: The exception that occurred
            severity: Log severity level (debug, info, warning, error, critical)
            user_query: The user's query (will be truncated for privacy)
            tool_name: Name of the tool where error occurred
            additional_context: Additional context to log
        """
        # Build safe context for logging
        log_context = {
            "error_type": type(error).__name__,
            "error_message": str(error)[:200],  # Truncate long errors
        }

        if user_query:
            # Only log first 50 chars of query for privacy
            log_context["query_preview"] = user_query[:50] + "..."

        if tool_name:
            log_context["tool"] = tool_name

        if additional_context:
            # Filter out sensitive keys
            safe_keys = ["attempt", "retry_count", "duration", "state"]
            log_context.update(
                {k: v for k, v in additional_context.items() if k in safe_keys}
            )

        # Log at appropriate level
        log_message = f"Error handled: {log_context}"

        if severity == "critical":
            logger.critical(log_message)
        elif severity == "error":
            logger.error(log_message)
        elif severity == "warning":
            logger.warning(log_message)
        elif severity == "info":
            logger.info(log_message)
        else:
            logger.debug(log_message)


# Convenience functions for common use cases
def get_user_friendly_message(error: Exception, context: Optional[str] = None) -> str:
    """
    Get user-friendly error message.

    Args:
        error: The exception that occurred
        context: Optional context about the operation

    Returns:
        User-friendly error message
    """
    return ErrorMessageMapper.get_user_message(error, context)


def log_and_get_message(
    error: Exception,
    severity: str = "error",
    user_query: Optional[str] = None,
    tool_name: Optional[str] = None,
) -> str:
    """
    Log error and get user-friendly message.

    Args:
        error: The exception that occurred
        severity: Log severity level
        user_query: The user's query
        tool_name: Name of the tool

    Returns:
        User-friendly error message
    """
    ErrorMessageMapper.log_error_with_context(error, severity, user_query, tool_name)
    return ErrorMessageMapper.get_user_message(error)
