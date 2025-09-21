"""
Rate limiter for PydanticAI tool calls.
Implements token bucket algorithm for API rate limiting.
"""

import asyncio
import time
from typing import Any, Callable, TypeVar, Coroutine
from functools import wraps
import logging

logger = logging.getLogger(__name__)

T = TypeVar("T")


class RateLimiter:
    """
    Token bucket rate limiter for API calls.

    Ensures API calls respect rate limits by implementing a token bucket algorithm.
    Tokens are replenished at a constant rate up to a maximum capacity.
    """

    def __init__(self, rate: float, per: float = 1.0):
        """
        Initialize rate limiter.

        Args:
            rate: Number of requests allowed
            per: Time period in seconds (default 1.0 for per-second)
        """
        self.rate = rate
        self.per = per
        self.allowance = rate
        self.last_check = time.monotonic()
        self.lock = asyncio.Lock()
        logger.info(f"RateLimiter initialized: {rate} requests per {per} seconds")

    async def acquire(self):
        """
        Wait if necessary to maintain rate limit.

        Uses token bucket algorithm to determine if a request can proceed.
        If not enough tokens are available, sleeps until tokens are replenished.
        """
        sleep_time = 0.0

        async with self.lock:
            current = time.monotonic()
            time_passed = current - self.last_check
            self.last_check = current

            # Replenish tokens based on time passed
            self.allowance += time_passed * (self.rate / self.per)
            if self.allowance > self.rate:
                self.allowance = self.rate

            # If not enough tokens, calculate wait time
            if self.allowance < 1.0:
                sleep_time = (1.0 - self.allowance) * (self.per / self.rate)
                logger.debug(f"Rate limit reached, sleeping for {sleep_time:.2f}s")
                self.allowance = 0.0
            else:
                self.allowance -= 1.0

        # Sleep outside the lock to avoid blocking other coroutines
        if sleep_time > 0:
            await asyncio.sleep(sleep_time)

    async def __aenter__(self):
        """Context manager entry."""
        await self.acquire()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        pass


def rate_limited(limiter: RateLimiter):
    """
    Decorator to apply rate limiting to async functions.

    Args:
        limiter: RateLimiter instance to use for throttling

    Returns:
        Decorated function that respects rate limits

    Example:
        >>> gemini_limiter = RateLimiter(rate=30, per=1.0)  # 30 req/s
        >>> @rate_limited(gemini_limiter)
        ... async def call_api():
        ...     return await api.request()
    """

    def decorator(
        func: Callable[..., Coroutine[Any, Any, T]],
    ) -> Callable[..., Coroutine[Any, Any, T]]:
        @wraps(func)
        async def wrapper(*args, **kwargs) -> T:
            await limiter.acquire()
            return await func(*args, **kwargs)

        return wrapper

    return decorator


# Pre-configured rate limiters for different APIs
gemini_flash_limiter = RateLimiter(rate=30, per=1.0)  # Gemini 2.5 Flash: 30 req/s
embedding_limiter = RateLimiter(
    rate=10, per=1.0
)  # Conservative embedding rate: 10 req/s
