"""
Structured logging configuration for developer monitoring.

This module provides structured logging separate from user-facing logs,
enabling detailed monitoring and debugging capabilities for developers.
"""

import json
import logging
import os
import sys
import uuid
from datetime import datetime
from typing import Any, Dict, Optional


class StructuredFormatter(logging.Formatter):
    """Custom formatter for structured logging output."""

    def __init__(self, format_type: str = "text"):
        """
        Initialize the structured formatter.

        Args:
            format_type: Output format ('json' or 'text')
        """
        super().__init__()
        self.format_type = format_type

    def format(self, record: logging.LogRecord) -> str:
        """
        Format the log record.

        Args:
            record: The log record to format

        Returns:
            Formatted log string
        """
        # Extract additional fields from record
        extra_fields = {}
        for key, value in record.__dict__.items():
            if key not in logging.LogRecord.__dict__ and not key.startswith("_"):
                extra_fields[key] = value

        # Build log entry
        log_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }

        # Add extra fields
        if extra_fields:
            log_entry["extra"] = extra_fields

        # Add exception info if present
        if record.exc_info:
            log_entry["exception"] = self.formatException(record.exc_info)

        # Format output
        if self.format_type == "json":
            return json.dumps(log_entry, default=str)
        else:
            # Text format with structure
            text_parts = [
                f"[{log_entry['timestamp']}]",
                f"[{log_entry['level']:8s}]",
                f"[{log_entry['logger']}]",
                f"{log_entry['message']}",
            ]

            if extra_fields:
                text_parts.append(f"| {extra_fields}")

            if "exception" in log_entry:
                text_parts.append(f"\n{log_entry['exception']}")

            return " ".join(text_parts)


class RequestIdFilter(logging.Filter):
    """Add request ID to log records for correlation."""

    def __init__(self):
        """Initialize the request ID filter."""
        super().__init__()
        self._request_id = None

    def set_request_id(self, request_id: Optional[str] = None):
        """
        Set the current request ID.

        Args:
            request_id: Request ID to use, generates new one if None
        """
        self._request_id = request_id or str(uuid.uuid4())

    def clear_request_id(self):
        """Clear the current request ID."""
        self._request_id = None

    def filter(self, record: logging.LogRecord) -> bool:
        """
        Add request ID to log record.

        Args:
            record: The log record to filter

        Returns:
            Always True to pass the record
        """
        if self._request_id:
            record.request_id = self._request_id
        return True


# Global request ID filter instance
_request_id_filter = RequestIdFilter()

# Logger cache to avoid recreating loggers
_loggers: Dict[str, logging.Logger] = {}


def configure_structured_logging(
    log_level: str = "INFO",
    format_type: str = "text",
    output_file: Optional[str] = None,
) -> None:
    """
    Configure structured logging for monitoring.

    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
        format_type: Output format ('json' or 'text')
        output_file: Optional file path for log output
    """
    # Get logging configuration from environment
    env_level = os.getenv("MONITORING_LOG_LEVEL", log_level).upper()
    env_format = os.getenv("MONITORING_LOG_FORMAT", format_type).lower()

    # Create formatter
    formatter = StructuredFormatter(format_type=env_format)

    # Configure root logger for monitoring namespace
    monitoring_logger = logging.getLogger("monitoring")
    monitoring_logger.setLevel(getattr(logging, env_level))
    monitoring_logger.handlers.clear()

    # Add console handler
    console_handler = logging.StreamHandler(sys.stderr)
    console_handler.setFormatter(formatter)
    console_handler.addFilter(_request_id_filter)
    monitoring_logger.addHandler(console_handler)

    # Add file handler if specified
    if output_file:
        file_handler = logging.FileHandler(output_file)
        file_handler.setFormatter(formatter)
        file_handler.addFilter(_request_id_filter)
        monitoring_logger.addHandler(file_handler)

    # Prevent propagation to root logger
    monitoring_logger.propagate = False


def get_logger(name: str) -> logging.Logger:
    """
    Get a structured logger instance.

    Args:
        name: Logger name (usually __name__)

    Returns:
        Configured logger instance
    """
    # Ensure logger is under monitoring namespace
    if not name.startswith("monitoring"):
        name = f"monitoring.{name}"

    # Return cached logger if available
    if name in _loggers:
        return _loggers[name]

    # Create new logger
    logger = logging.getLogger(name)
    _loggers[name] = logger

    return logger


def set_log_level(level: str) -> None:
    """
    Set the logging level for all monitoring loggers.

    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR)
    """
    level_obj = getattr(logging, level.upper())
    monitoring_logger = logging.getLogger("monitoring")
    monitoring_logger.setLevel(level_obj)


def set_request_id(request_id: Optional[str] = None) -> str:
    """
    Set the current request ID for log correlation.

    Args:
        request_id: Request ID to use, generates new one if None

    Returns:
        The request ID that was set
    """
    if request_id is None:
        request_id = str(uuid.uuid4())
    _request_id_filter.set_request_id(request_id)
    return request_id


def clear_request_id() -> None:
    """Clear the current request ID."""
    _request_id_filter.clear_request_id()


def log_tool_invocation(
    logger: logging.Logger,
    tool_name: str,
    phase: str,
    duration: Optional[float] = None,
    **kwargs: Any,
) -> None:
    """
    Log a tool invocation with structured data.

    Args:
        logger: Logger to use
        tool_name: Name of the tool being invoked
        phase: Phase of invocation ('start', 'end', 'error')
        duration: Duration in seconds (for 'end' phase)
        **kwargs: Additional fields to log
    """
    extra = {
        "tool_name": tool_name,
        "phase": phase,
    }

    if duration is not None:
        extra["duration_ms"] = int(duration * 1000)

    # Add any additional fields
    extra.update(kwargs)

    # Log at appropriate level
    if phase == "error":
        logger.error(f"Tool invocation failed: {tool_name}", extra=extra)
    elif phase == "start":
        logger.debug(f"Tool invocation started: {tool_name}", extra=extra)
    else:
        logger.info(f"Tool invocation completed: {tool_name}", extra=extra)


# Auto-configure on import if monitoring is enabled
if os.getenv("ENABLE_MONITORING", "false").lower() == "true":
    configure_structured_logging()
