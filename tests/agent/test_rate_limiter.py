"""Unit tests for rate limiter module."""

import asyncio
import time
from unittest.mock import patch
import pytest

from src.agent.rate_limiter import RateLimiter, rate_limited


class TestRateLimiter:
    """Test cases for RateLimiter class."""

    @pytest.mark.asyncio
    async def test_rate_limiter_init(self):
        """Test rate limiter initialization."""
        limiter = RateLimiter(rate=5, per=1.0)
        assert limiter.rate == 5
        assert limiter.per == 1.0
        assert limiter.allowance == 5

    @pytest.mark.asyncio
    async def test_single_request_passes_immediately(self):
        """Test that a single request passes immediately."""
        limiter = RateLimiter(rate=5, per=1.0)
        start = time.monotonic()
        await limiter.acquire()
        duration = time.monotonic() - start
        assert duration < 0.1  # Should be immediate

    @pytest.mark.asyncio
    async def test_rate_limiting_enforced(self):
        """Test that rate limiting is enforced when limit exceeded."""
        limiter = RateLimiter(rate=2, per=1.0)  # 2 requests per second

        # Make 3 requests rapidly - third should be delayed
        start = time.monotonic()
        await limiter.acquire()  # First request - immediate
        await limiter.acquire()  # Second request - immediate
        await limiter.acquire()  # Third request - should wait
        duration = time.monotonic() - start

        # Third request should have waited approximately 0.5 seconds
        # (need 0.5s to replenish 1 token at rate of 2/s)
        assert duration >= 0.4  # Allow some timing slack
        assert duration < 0.7

    @pytest.mark.asyncio
    async def test_token_replenishment(self):
        """Test that tokens are replenished over time."""
        limiter = RateLimiter(rate=10, per=1.0)  # 10 requests per second

        # Use all tokens
        for _ in range(10):
            await limiter.acquire()

        # Wait for tokens to replenish
        await asyncio.sleep(0.5)  # Should replenish 5 tokens

        # These should pass without delay
        start = time.monotonic()
        for _ in range(5):
            await limiter.acquire()
        duration = time.monotonic() - start

        assert duration < 0.1  # Should be immediate for replenished tokens

    @pytest.mark.asyncio
    async def test_concurrent_requests(self):
        """Test that rate limiter handles concurrent requests correctly."""
        limiter = RateLimiter(rate=5, per=1.0)
        call_times = []

        async def timed_acquire():
            await limiter.acquire()
            call_times.append(time.monotonic())

        # Start 10 concurrent requests
        start = time.monotonic()
        tasks = [timed_acquire() for _ in range(10)]
        await asyncio.gather(*tasks)

        # Sort call times since concurrent execution order isn't guaranteed
        call_times.sort()

        # First 5 should be relatively immediate
        for i in range(min(5, len(call_times))):
            assert call_times[i] - start < 0.2  # First batch immediate-ish

        # If we have more than 5 calls, later ones should be delayed
        if len(call_times) > 5:
            # The 6th call onwards should show some delay
            assert call_times[5] - start >= 0.1  # Some delay for rate limiting

    @pytest.mark.asyncio
    async def test_context_manager(self):
        """Test rate limiter as async context manager."""
        limiter = RateLimiter(rate=5, per=1.0)

        async with limiter:
            pass  # Should acquire and release

        # Should have consumed one token
        assert limiter.allowance == 4

    @pytest.mark.asyncio
    async def test_rate_limited_decorator(self):
        """Test the @rate_limited decorator."""
        limiter = RateLimiter(rate=3, per=1.0)
        call_count = 0
        call_times = []

        @rate_limited(limiter)
        async def api_call():
            nonlocal call_count
            call_count += 1
            call_times.append(time.monotonic())
            return f"Response {call_count}"

        # Make 5 calls
        start = time.monotonic()
        results = []
        for _ in range(5):
            result = await api_call()
            results.append(result)

        duration = time.monotonic() - start

        # Should have made all calls
        assert call_count == 5
        assert results == [
            "Response 1",
            "Response 2",
            "Response 3",
            "Response 4",
            "Response 5",
        ]

        # With rate of 3/second, 5 calls should show some delay
        # First 3 are immediate, 4th and 5th need to wait
        # More lenient check due to timing variations
        assert duration >= 0.3  # Should show some rate limiting delay

    @pytest.mark.asyncio
    async def test_rate_limited_decorator_with_exception(self):
        """Test that decorator preserves exceptions."""
        limiter = RateLimiter(rate=5, per=1.0)

        @rate_limited(limiter)
        async def failing_call():
            raise ValueError("Test error")

        with pytest.raises(ValueError, match="Test error"):
            await failing_call()

    @pytest.mark.asyncio
    async def test_preconfigured_limiters(self):
        """Test preconfigured rate limiters."""
        from src.agent.rate_limiter import gemini_flash_limiter, embedding_limiter

        assert gemini_flash_limiter.rate == 30
        assert gemini_flash_limiter.per == 1.0

        assert (
            embedding_limiter.rate == 20
        )  # Updated to match new rate limit for parallel processing
        assert embedding_limiter.per == 1.0

    @pytest.mark.asyncio
    @patch("time.monotonic")
    async def test_exact_rate_calculation(self, mock_time):
        """Test exact token bucket calculation with mocked time."""
        # Control time precisely
        current_time = [0.0]

        def get_time():
            return current_time[0]

        mock_time.side_effect = get_time

        limiter = RateLimiter(rate=10, per=2.0)  # 10 requests per 2 seconds = 5 req/s

        # Should start with 10 tokens
        assert limiter.allowance == 10

        # Use 5 tokens immediately
        for _ in range(5):
            await limiter.acquire()
        assert limiter.allowance == 5

        # Advance time by 1 second (should replenish 5 tokens)
        current_time[0] = 1.0
        await limiter.acquire()
        # Should have 5 - 1 + 5 = 9 tokens (capped at 10 but we used 1)
        assert limiter.allowance == 9

        # Use all remaining tokens
        for _ in range(9):
            await limiter.acquire()
        assert limiter.allowance == 0

        # Next request should calculate sleep time
        # Need 1 token, replenish rate is 5/s, so need 0.2s
        sleep_times = []

        async def mock_sleep(duration):
            sleep_times.append(duration)
            current_time[0] += duration

        with patch("asyncio.sleep", side_effect=mock_sleep):
            await limiter.acquire()

        # Should have slept for 0.2 seconds
        assert len(sleep_times) == 1
        assert abs(sleep_times[0] - 0.2) < 0.01
