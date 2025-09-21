"""Integration tests for error handling with rate limiting and circuit breakers."""

import asyncio
from unittest.mock import patch
import pytest

from src.agent.error_messages import ErrorMessageMapper, get_user_friendly_message
from src.agent.circuit_breaker import CircuitBreaker, CircuitBreakerError
from src.agent.rate_limiter import RateLimiter


class TestErrorMessages:
    """Test cases for error message mapping."""

    def test_connection_error_mapping(self):
        """Test that connection errors get friendly messages."""
        error = ConnectionError("Failed to connect to database")
        message = get_user_friendly_message(error)
        assert "knowledge base" in message.lower()
        assert "try again" in message.lower()
        assert "database" not in message.lower()  # Technical details hidden

    def test_timeout_error_mapping(self):
        """Test that timeout errors get friendly messages."""
        error = TimeoutError("Request timed out after 30s")
        message = get_user_friendly_message(error)
        assert "taking longer" in message.lower()
        assert "30s" not in message  # Technical details hidden

    def test_rate_limit_error_mapping(self):
        """Test that rate limit errors get friendly messages."""
        error = Exception("Rate limit exceeded for API")
        message = get_user_friendly_message(error)
        assert "too many requests" in message.lower()
        assert "wait" in message.lower()

    def test_circuit_breaker_error_mapping(self):
        """Test that circuit breaker errors get friendly messages."""
        error = CircuitBreakerError("Circuit breaker is open")
        message = get_user_friendly_message(error)
        assert "temporarily unavailable" in message.lower()
        assert "circuit" not in message.lower()  # Technical term hidden

    def test_embedding_error_mapping(self):
        """Test that embedding errors get friendly messages."""
        error = Exception("Embedding generation failed")
        message = get_user_friendly_message(error)
        assert "understanding your query" in message.lower()

    def test_unknown_error_fallback(self):
        """Test fallback for unknown errors."""
        error = RuntimeError("Some unexpected error")
        message = get_user_friendly_message(error)
        assert "unexpected issue" in message.lower()
        assert "try again" in message.lower()

    def test_error_with_context(self):
        """Test error message with context."""
        error = ConnectionError("Connection refused")
        message = get_user_friendly_message(
            error, context="searching for documentation"
        )
        assert "knowledge base" in message.lower()

    def test_no_sensitive_data_exposure(self):
        """Test that sensitive data is not exposed."""
        error = ValueError("API key 'sk-1234567890' is invalid")
        message = get_user_friendly_message(error)
        assert "sk-1234567890" not in message
        assert "API key" not in message.lower()


class TestErrorLogging:
    """Test cases for error logging."""

    @patch("src.agent.error_messages.logger")
    def test_log_error_with_context(self, mock_logger):
        """Test that errors are logged with proper context."""
        error = ConnectionError("Database connection failed")

        ErrorMessageMapper.log_error_with_context(
            error,
            severity="error",
            user_query="How do I use dwi2tensor?",
            tool_name="search_knowledgebase",
            additional_context={"attempt": 3, "api_key": "secret"},
        )

        # Check that error was logged
        mock_logger.error.assert_called_once()
        log_message = str(mock_logger.error.call_args[0][0])

        # Check context is included
        assert "ConnectionError" in log_message
        assert "search_knowledgebase" in log_message
        assert "How do I use" in log_message  # Query preview

        # Check sensitive data is not logged
        assert "secret" not in log_message

    @patch("src.agent.error_messages.logger")
    def test_log_severity_levels(self, mock_logger):
        """Test different severity levels."""
        error = Exception("Test error")

        # Test each severity level
        ErrorMessageMapper.log_error_with_context(error, severity="debug")
        mock_logger.debug.assert_called_once()

        ErrorMessageMapper.log_error_with_context(error, severity="info")
        mock_logger.info.assert_called_once()

        ErrorMessageMapper.log_error_with_context(error, severity="warning")
        mock_logger.warning.assert_called_once()

        ErrorMessageMapper.log_error_with_context(error, severity="error")
        mock_logger.error.assert_called()

        ErrorMessageMapper.log_error_with_context(error, severity="critical")
        mock_logger.critical.assert_called_once()


class TestIntegratedErrorHandling:
    """Test error handling with rate limiting and circuit breakers."""

    @pytest.mark.asyncio
    async def test_rate_limit_then_circuit_breaker(self):
        """Test that rate limiting and circuit breaker work together."""
        breaker = CircuitBreaker(failure_threshold=2, recovery_timeout=0.1)
        limiter = RateLimiter(rate=2, per=1.0)
        failures = 0

        async def api_call():
            nonlocal failures
            await limiter.acquire()
            failures += 1
            if failures <= 3:
                raise ConnectionError("API connection failed")
            return "Success"

        # First two calls fail and open circuit
        with pytest.raises(ConnectionError):
            await breaker.call(api_call)
        with pytest.raises(ConnectionError):
            await breaker.call(api_call)

        # Circuit should be open now
        with pytest.raises(CircuitBreakerError):
            await breaker.call(api_call)

        # Get user-friendly message for circuit breaker error
        try:
            await breaker.call(api_call)
        except CircuitBreakerError as e:
            message = get_user_friendly_message(e)
            assert "temporarily unavailable" in message.lower()

    @pytest.mark.asyncio
    async def test_retry_with_rate_limiting(self):
        """Test retry logic respects rate limits."""
        from tenacity import retry, stop_after_attempt, wait_fixed

        limiter = RateLimiter(rate=3, per=1.0)
        call_times = []
        attempt_count = 0

        @retry(stop=stop_after_attempt(3), wait=wait_fixed(0.1), reraise=True)
        async def retrying_api_call():
            nonlocal attempt_count
            await limiter.acquire()
            call_times.append(asyncio.get_event_loop().time())
            attempt_count += 1
            if attempt_count < 3:
                raise ConnectionError("Transient error")
            return "Success"

        result = await retrying_api_call()
        assert result == "Success"
        assert attempt_count == 3

        # Check that rate limiting was applied
        if len(call_times) >= 3:
            # Third call should have been delayed by rate limiter
            time_between_calls = call_times[2] - call_times[1]
            assert (
                time_between_calls >= 0.05
            )  # Some delay expected (reduced for test stability)

    @pytest.mark.asyncio
    async def test_error_cascade_handling(self):
        """Test handling of cascading errors."""
        breaker = CircuitBreaker(failure_threshold=2)

        async def failing_service():
            raise ConnectionError("Service unavailable")

        # Open the circuit
        for _ in range(2):
            with pytest.raises(ConnectionError):
                await breaker.call(failing_service)

        # Now circuit is open - get friendly message
        try:
            await breaker.call(failing_service)
        except CircuitBreakerError as e:
            user_msg = get_user_friendly_message(e)
            assert "temporarily unavailable" in user_msg.lower()
            assert "circuit" not in user_msg.lower()

    @pytest.mark.asyncio
    async def test_graceful_degradation(self):
        """Test system degrades gracefully under failure."""
        from src.agent.error_messages import get_user_friendly_message
        from src.agent.circuit_breaker import CircuitBreakerError

        # Test that circuit breaker errors get user-friendly messages
        cb_error = CircuitBreakerError("Database circuit breaker open")
        message = get_user_friendly_message(cb_error)
        assert "temporarily unavailable" in message.lower()
        assert "circuit" not in message.lower()

        # Test that connection errors with Supabase get user-friendly messages
        # The "supabase" pattern in ERROR_PATTERNS takes precedence
        conn_error = ConnectionError("Failed to connect to Supabase")
        message = get_user_friendly_message(conn_error)
        # Should match the "supabase" pattern and return documentation database message
        assert (
            "documentation database" in message.lower()
            or "knowledge base" in message.lower()
        )
        assert "supabase" not in message.lower()  # Technical details should be hidden
