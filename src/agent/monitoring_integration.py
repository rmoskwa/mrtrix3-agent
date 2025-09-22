"""
Helper module for integrating monitoring with agent modules.

This module provides utilities for agent modules to use monitoring
capabilities when enabled, while maintaining clean separation of concerns.
"""

import asyncio
import os
import logging
from typing import Optional, Any, Callable
from functools import wraps
import time


# Dynamic monitoring check - evaluates at runtime, not import time
def is_monitoring_enabled() -> bool:
    """Check if monitoring is enabled at runtime."""
    return os.getenv("ENABLE_MONITORING", "false").lower() == "true"


# For backward compatibility, set initial value
MONITORING_ENABLED = is_monitoring_enabled()

# Initialize monitoring components if enabled
monitoring_available = False
get_monitoring_logger = None
log_tool_invocation = None
set_request_id = None
clear_request_id = None
MetricsCollector = None
get_global_collector = None


# Lazy initialization function
def _initialize_monitoring():
    """Initialize monitoring components if not already done."""
    global monitoring_available, get_monitoring_logger, log_tool_invocation
    global set_request_id, clear_request_id, MetricsCollector, get_global_collector

    if monitoring_available:
        return True  # Already initialized

    if not is_monitoring_enabled():
        return False  # Monitoring not enabled

    try:
        from monitoring.logging_config import (
            get_logger as _get_monitoring_logger,
            log_tool_invocation as _log_tool_invocation,
            set_request_id as _set_request_id,
            clear_request_id as _clear_request_id,
        )
        from monitoring.metrics import (
            MetricsCollector as _MetricsCollector,
            get_global_collector as _get_global_collector,
        )

        # Make available at module level
        get_monitoring_logger = _get_monitoring_logger
        log_tool_invocation = _log_tool_invocation
        set_request_id = _set_request_id
        clear_request_id = _clear_request_id
        MetricsCollector = _MetricsCollector
        get_global_collector = _get_global_collector

        monitoring_available = True
        return True
    except ImportError:
        monitoring_available = False
        return False


# Try initial initialization
if MONITORING_ENABLED:
    try:
        from monitoring.logging_config import (
            get_logger as _get_monitoring_logger,
            log_tool_invocation as _log_tool_invocation,
            set_request_id as _set_request_id,
            clear_request_id as _clear_request_id,
        )
        from monitoring.metrics import (
            MetricsCollector as _MetricsCollector,
            get_global_collector as _get_global_collector,
        )

        # Make available at module level
        get_monitoring_logger = _get_monitoring_logger
        log_tool_invocation = _log_tool_invocation
        set_request_id = _set_request_id
        clear_request_id = _clear_request_id
        MetricsCollector = _MetricsCollector
        get_global_collector = _get_global_collector

        monitoring_available = True
    except ImportError:
        monitoring_available = False
        print("Warning: Monitoring enabled but modules not available")


def get_dual_logger(name: str) -> tuple:
    """
    Get both user-facing and monitoring loggers.

    Args:
        name: Logger name

    Returns:
        Tuple of (user_logger, monitoring_logger)
    """
    # Always get user-facing logger
    user_logger = logging.getLogger(name)

    # Try to initialize monitoring if needed
    _initialize_monitoring()

    # Get monitoring logger if available
    monitoring_logger = None
    if monitoring_available and get_monitoring_logger:
        monitoring_logger = get_monitoring_logger(name)

    return user_logger, monitoring_logger


def log_dual(
    user_logger: logging.Logger,
    monitoring_logger: Optional[logging.Logger],
    level: str,
    user_message: str,
    monitoring_message: Optional[str] = None,
    **monitoring_extra: Any,
) -> None:
    """
    Log to both user and monitoring loggers.

    Args:
        user_logger: User-facing logger
        monitoring_logger: Monitoring logger (if available)
        level: Log level
        user_message: Message for user
        monitoring_message: Optional detailed message for monitoring
        **monitoring_extra: Extra fields for monitoring
    """
    # Log to user logger (simple message)
    log_func = getattr(user_logger, level.lower())
    log_func(user_message)

    # Log to monitoring logger if available (detailed)
    if monitoring_logger:
        monitoring_msg = monitoring_message or user_message
        log_func = getattr(monitoring_logger, level.lower())
        log_func(monitoring_msg, extra=monitoring_extra)


def track_tool_invocation(tool_name: str):
    """
    Decorator to track tool invocations with monitoring.

    Args:
        tool_name: Name of the tool

    Returns:
        Decorator function
    """

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            # Try to initialize monitoring if needed
            _initialize_monitoring()

            if monitoring_available:
                metrics = get_global_collector()
                monitoring_logger = get_monitoring_logger(func.__module__)

                # Log start
                log_tool_invocation(
                    monitoring_logger,
                    tool_name,
                    "start",
                    input_args=str(args)[:100],
                    input_kwargs=str(kwargs)[:100],
                )

                # Track timing
                start_time = time.time()
                metrics.record_counter("request_count")

                try:
                    result = await func(*args, **kwargs)

                    # Log success
                    duration = time.time() - start_time
                    log_tool_invocation(
                        monitoring_logger,
                        tool_name,
                        "end",
                        duration=duration,
                        success=True,
                    )
                    metrics.record_latency(f"tool_{tool_name}", duration)

                    return result

                except Exception as e:
                    # Log error
                    duration = time.time() - start_time
                    log_tool_invocation(
                        monitoring_logger,
                        tool_name,
                        "error",
                        duration=duration,
                        error_type=type(e).__name__,
                        error_message=str(e)[:200],
                    )
                    metrics.record_counter("error_count")
                    raise
            else:
                # No monitoring, just execute
                return await func(*args, **kwargs)

        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            # Try to initialize monitoring if needed
            _initialize_monitoring()

            if monitoring_available:
                metrics = get_global_collector()
                monitoring_logger = get_monitoring_logger(func.__module__)

                # Log start
                log_tool_invocation(
                    monitoring_logger,
                    tool_name,
                    "start",
                    input_args=str(args)[:100],
                    input_kwargs=str(kwargs)[:100],
                )

                # Track timing
                start_time = time.time()
                metrics.record_counter("request_count")

                try:
                    result = func(*args, **kwargs)

                    # Log success
                    duration = time.time() - start_time
                    log_tool_invocation(
                        monitoring_logger,
                        tool_name,
                        "end",
                        duration=duration,
                        success=True,
                    )
                    metrics.record_latency(f"tool_{tool_name}", duration)

                    return result

                except Exception as e:
                    # Log error
                    duration = time.time() - start_time
                    log_tool_invocation(
                        monitoring_logger,
                        tool_name,
                        "error",
                        duration=duration,
                        error_type=type(e).__name__,
                        error_message=str(e)[:200],
                    )
                    metrics.record_counter("error_count")
                    raise
            else:
                # No monitoring, just execute
                return func(*args, **kwargs)

        # Return appropriate wrapper based on function type
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper

    return decorator


def record_metric(
    metric_type: str,
    name: str,
    value: float,
    **tags: str,
) -> None:
    """
    Record a metric if monitoring is enabled.

    Args:
        metric_type: Type of metric (counter, gauge, latency)
        name: Metric name
        value: Metric value
        **tags: Additional tags
    """
    if monitoring_available:
        metrics = get_global_collector()

        if metric_type == "counter":
            metrics.record_counter(name, value, **tags)
        elif metric_type == "gauge":
            metrics.record_gauge(name, value, **tags)
        elif metric_type == "latency":
            metrics.record_latency(name, value, **tags)


def set_monitoring_request_id(request_id: Optional[str] = None) -> Optional[str]:
    """
    Set request ID for monitoring correlation.

    Args:
        request_id: Optional request ID

    Returns:
        Request ID that was set (if monitoring enabled)
    """
    if monitoring_available:
        return set_request_id(request_id)
    return None


def clear_monitoring_request_id() -> None:
    """Clear request ID for monitoring."""
    if monitoring_available:
        clear_request_id()
