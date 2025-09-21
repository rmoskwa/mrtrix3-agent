"""
Circuit breaker pattern for handling Supabase connection failures.

Implements a state machine with three states:
- CLOSED: Normal operation, requests pass through
- OPEN: Circuit is tripped, requests fail immediately
- HALF_OPEN: Testing recovery, limited requests allowed
"""

import asyncio
import logging
import time
from enum import Enum
from typing import Any, Callable, TypeVar, Coroutine, Optional
from functools import wraps
from datetime import datetime

logger = logging.getLogger(__name__)

T = TypeVar("T")


class CircuitState(Enum):
    """Circuit breaker states."""

    CLOSED = "closed"  # Normal operation
    OPEN = "open"  # Circuit tripped, failing fast
    HALF_OPEN = "half_open"  # Testing recovery


class CircuitBreakerError(Exception):
    """Raised when circuit breaker is open."""

    pass


class CircuitBreaker:
    """
    Circuit breaker for protecting against cascading failures.

    Monitors failures and opens circuit when threshold is exceeded.
    After timeout, enters half-open state to test recovery.
    """

    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: float = 30.0,
        half_open_max_calls: int = 1,
        reset_timeout: float = 60.0,
    ):
        """
        Initialize circuit breaker.

        Args:
            failure_threshold: Number of failures before opening circuit
            recovery_timeout: Seconds to wait before trying half-open
            half_open_max_calls: Max calls allowed in half-open state
            reset_timeout: Seconds of success before resetting failure count
        """
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.half_open_max_calls = half_open_max_calls
        self.reset_timeout = reset_timeout

        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.last_failure_time: Optional[float] = None
        self.last_success_time: Optional[float] = None
        self.circuit_opened_at: Optional[float] = None
        self.half_open_calls = 0
        self.lock = asyncio.Lock()

        logger.info(
            f"CircuitBreaker initialized: threshold={failure_threshold}, "
            f"recovery={recovery_timeout}s"
        )

    async def call(
        self, func: Callable[..., Coroutine[Any, Any, T]], *args, **kwargs
    ) -> T:
        """
        Execute function through circuit breaker.

        Args:
            func: Async function to execute
            *args: Positional arguments for func
            **kwargs: Keyword arguments for func

        Returns:
            Result from func

        Raises:
            CircuitBreakerError: If circuit is open
            Exception: Any exception from func
        """
        async with self.lock:
            current_state = await self._get_state()

            if current_state == CircuitState.OPEN:
                logger.warning("Circuit breaker is OPEN, rejecting call")
                raise CircuitBreakerError(
                    "Circuit breaker is open due to too many failures. "
                    "Please try again later."
                )

            if current_state == CircuitState.HALF_OPEN:
                if self.half_open_calls >= self.half_open_max_calls:
                    logger.warning("Half-open call limit reached, rejecting call")
                    raise CircuitBreakerError(
                        "Circuit breaker is testing recovery. Please try again later."
                    )
                self.half_open_calls += 1

        try:
            # Execute the function
            result = await func(*args, **kwargs)

            # Record success
            async with self.lock:
                await self._on_success()

            return result

        except Exception:
            # Record failure
            async with self.lock:
                await self._on_failure()
            raise

    async def _get_state(self) -> CircuitState:
        """
        Get current circuit state.

        Handles automatic transitions based on timeouts.
        """
        now = time.monotonic()

        # Check if we should reset failure count after success period
        if (
            self.state == CircuitState.CLOSED
            and self.last_success_time
            and self.failure_count > 0
            and now - self.last_success_time >= self.reset_timeout
        ):
            logger.info("Resetting failure count after success period")
            self.failure_count = 0

        # Check if we should transition from OPEN to HALF_OPEN
        if (
            self.state == CircuitState.OPEN
            and self.circuit_opened_at
            and now - self.circuit_opened_at >= self.recovery_timeout
        ):
            logger.info("Circuit breaker transitioning from OPEN to HALF_OPEN")
            self.state = CircuitState.HALF_OPEN
            self.half_open_calls = 0

        return self.state

    async def _on_success(self):
        """Handle successful call."""
        self.last_success_time = time.monotonic()

        if self.state == CircuitState.HALF_OPEN:
            # Success in half-open state closes the circuit
            logger.info("Circuit breaker closing after successful half-open call")
            self.state = CircuitState.CLOSED
            self.failure_count = 0
            self.half_open_calls = 0
            self.circuit_opened_at = None

    async def _on_failure(self):
        """Handle failed call."""
        self.failure_count += 1
        self.last_failure_time = time.monotonic()

        if self.state == CircuitState.HALF_OPEN:
            # Failure in half-open state reopens the circuit
            logger.warning("Circuit breaker reopening after failed half-open call")
            self.state = CircuitState.OPEN
            self.circuit_opened_at = time.monotonic()
            self.half_open_calls = 0

        elif (
            self.state == CircuitState.CLOSED
            and self.failure_count >= self.failure_threshold
        ):
            # Threshold exceeded, open circuit
            logger.error(f"Circuit breaker opening after {self.failure_count} failures")
            self.state = CircuitState.OPEN
            self.circuit_opened_at = time.monotonic()

    def get_status(self) -> dict:
        """
        Get current circuit breaker status.

        Returns:
            Dictionary with state, failure count, and timing info
        """
        return {
            "state": self.state.value,
            "failure_count": self.failure_count,
            "last_failure": (
                datetime.fromtimestamp(self.last_failure_time).isoformat()
                if self.last_failure_time
                else None
            ),
            "last_success": (
                datetime.fromtimestamp(self.last_success_time).isoformat()
                if self.last_success_time
                else None
            ),
            "circuit_opened_at": (
                datetime.fromtimestamp(self.circuit_opened_at).isoformat()
                if self.circuit_opened_at
                else None
            ),
        }

    async def reset(self):
        """Manually reset circuit breaker to closed state."""
        async with self.lock:
            logger.info("Manually resetting circuit breaker")
            self.state = CircuitState.CLOSED
            self.failure_count = 0
            self.half_open_calls = 0
            self.circuit_opened_at = None


def circuit_breaker(
    breaker: CircuitBreaker,
) -> Callable[
    [Callable[..., Coroutine[Any, Any, T]]], Callable[..., Coroutine[Any, Any, T]]
]:
    """
    Decorator to apply circuit breaker to async functions.

    Args:
        breaker: CircuitBreaker instance to use

    Returns:
        Decorated function protected by circuit breaker

    Example:
        >>> db_breaker = CircuitBreaker(failure_threshold=5)
        >>> @circuit_breaker(db_breaker)
        ... async def query_database():
        ...     return await db.query()
    """

    def decorator(
        func: Callable[..., Coroutine[Any, Any, T]],
    ) -> Callable[..., Coroutine[Any, Any, T]]:
        @wraps(func)
        async def wrapper(*args, **kwargs) -> T:
            return await breaker.call(func, *args, **kwargs)

        return wrapper

    return decorator


# Pre-configured circuit breaker for Supabase connections
supabase_circuit_breaker = CircuitBreaker(
    failure_threshold=5,  # Open after 5 failures
    recovery_timeout=30.0,  # Try recovery after 30 seconds
    half_open_max_calls=1,  # Allow 1 test call in half-open
    reset_timeout=60.0,  # Reset failure count after 60s of success
)
