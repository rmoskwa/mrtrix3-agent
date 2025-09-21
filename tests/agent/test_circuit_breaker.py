"""Unit tests for circuit breaker module."""

import asyncio
import pytest

from src.agent.circuit_breaker import (
    CircuitBreaker,
    CircuitState,
    CircuitBreakerError,
    circuit_breaker,
)


class TestCircuitBreaker:
    """Test cases for CircuitBreaker class."""

    @pytest.mark.asyncio
    async def test_circuit_breaker_init(self):
        """Test circuit breaker initialization."""
        breaker = CircuitBreaker(
            failure_threshold=3,
            recovery_timeout=10.0,
            half_open_max_calls=2,
            reset_timeout=30.0,
        )
        assert breaker.failure_threshold == 3
        assert breaker.recovery_timeout == 10.0
        assert breaker.half_open_max_calls == 2
        assert breaker.reset_timeout == 30.0
        assert breaker.state == CircuitState.CLOSED
        assert breaker.failure_count == 0

    @pytest.mark.asyncio
    async def test_successful_calls_pass_through(self):
        """Test that successful calls pass through when circuit is closed."""
        breaker = CircuitBreaker(failure_threshold=3)
        call_count = 0

        async def successful_function():
            nonlocal call_count
            call_count += 1
            return f"Success {call_count}"

        # Make successful calls
        result1 = await breaker.call(successful_function)
        result2 = await breaker.call(successful_function)
        result3 = await breaker.call(successful_function)

        assert result1 == "Success 1"
        assert result2 == "Success 2"
        assert result3 == "Success 3"
        assert breaker.state == CircuitState.CLOSED
        assert breaker.failure_count == 0

    @pytest.mark.asyncio
    async def test_circuit_opens_after_threshold(self):
        """Test that circuit opens after failure threshold is exceeded."""
        breaker = CircuitBreaker(failure_threshold=3)

        async def failing_function():
            raise ConnectionError("Database connection failed")

        # Make failing calls up to threshold
        for i in range(3):
            with pytest.raises(ConnectionError):
                await breaker.call(failing_function)

        # Circuit should now be open
        assert breaker.state == CircuitState.OPEN
        assert breaker.failure_count == 3

        # Next call should fail immediately with CircuitBreakerError
        with pytest.raises(CircuitBreakerError, match="Circuit breaker is open"):
            await breaker.call(failing_function)

    @pytest.mark.asyncio
    async def test_circuit_transitions_to_half_open(self):
        """Test that circuit transitions to half-open after recovery timeout."""
        breaker = CircuitBreaker(failure_threshold=2, recovery_timeout=0.1)

        async def failing_function():
            raise ConnectionError("Database connection failed")

        # Open the circuit
        for _ in range(2):
            with pytest.raises(ConnectionError):
                await breaker.call(failing_function)

        assert breaker.state == CircuitState.OPEN

        # Wait for recovery timeout
        await asyncio.sleep(0.15)

        # Circuit should transition to half-open on next check
        async def successful_function():
            return "Success"

        result = await breaker.call(successful_function)
        assert result == "Success"
        assert breaker.state == CircuitState.CLOSED  # Successful call closes circuit

    @pytest.mark.asyncio
    async def test_half_open_failure_reopens_circuit(self):
        """Test that failure in half-open state reopens the circuit."""
        breaker = CircuitBreaker(failure_threshold=2, recovery_timeout=0.1)

        async def failing_function():
            raise ConnectionError("Database connection failed")

        # Open the circuit
        for _ in range(2):
            with pytest.raises(ConnectionError):
                await breaker.call(failing_function)

        assert breaker.state == CircuitState.OPEN

        # Wait for recovery timeout
        await asyncio.sleep(0.15)

        # Next failure should reopen circuit
        with pytest.raises(ConnectionError):
            await breaker.call(failing_function)

        assert breaker.state == CircuitState.OPEN

    @pytest.mark.asyncio
    async def test_half_open_max_calls_limit(self):
        """Test that half-open state limits number of test calls."""
        breaker = CircuitBreaker(
            failure_threshold=2,
            recovery_timeout=0.1,
            half_open_max_calls=1,
        )

        async def failing_function():
            raise ConnectionError("Database connection failed")

        async def successful_function():
            return "Success"

        # Open the circuit
        for _ in range(2):
            with pytest.raises(ConnectionError):
                await breaker.call(failing_function)

        # Wait for recovery timeout
        await asyncio.sleep(0.15)

        # First call in half-open should succeed
        result = await breaker.call(successful_function)
        assert result == "Success"

    @pytest.mark.asyncio
    async def test_reset_timeout_clears_failure_count(self):
        """Test that failure count is reset after success period."""
        breaker = CircuitBreaker(failure_threshold=3, reset_timeout=0.1)

        async def failing_function():
            raise ConnectionError("Database connection failed")

        async def successful_function():
            return "Success"

        # One failure
        with pytest.raises(ConnectionError):
            await breaker.call(failing_function)
        assert breaker.failure_count == 1

        # Successful call
        await breaker.call(successful_function)

        # Wait for reset timeout
        await asyncio.sleep(0.15)

        # Failure count should be reset on next state check
        await breaker.call(successful_function)
        assert breaker.failure_count == 0

    @pytest.mark.asyncio
    async def test_manual_reset(self):
        """Test manual reset of circuit breaker."""
        breaker = CircuitBreaker(failure_threshold=2)

        async def failing_function():
            raise ConnectionError("Database connection failed")

        # Open the circuit
        for _ in range(2):
            with pytest.raises(ConnectionError):
                await breaker.call(failing_function)

        assert breaker.state == CircuitState.OPEN
        assert breaker.failure_count == 2

        # Manually reset
        await breaker.reset()

        assert breaker.state == CircuitState.CLOSED
        assert breaker.failure_count == 0

    @pytest.mark.asyncio
    async def test_get_status(self):
        """Test getting circuit breaker status."""
        breaker = CircuitBreaker(failure_threshold=2)

        async def failing_function():
            raise ConnectionError("Database connection failed")

        # Initial status
        status = breaker.get_status()
        assert status["state"] == "closed"
        assert status["failure_count"] == 0
        assert status["last_failure"] is None
        assert status["last_success"] is None

        # After failure
        with pytest.raises(ConnectionError):
            await breaker.call(failing_function)

        status = breaker.get_status()
        assert status["state"] == "closed"
        assert status["failure_count"] == 1
        assert status["last_failure"] is not None

    @pytest.mark.asyncio
    async def test_circuit_breaker_decorator(self):
        """Test the @circuit_breaker decorator."""
        breaker = CircuitBreaker(failure_threshold=2)
        call_count = 0
        failure_count = 0

        @circuit_breaker(breaker)
        async def api_call(should_fail=False):
            nonlocal call_count, failure_count
            call_count += 1
            if should_fail:
                failure_count += 1
                raise ConnectionError("API connection failed")
            return f"Response {call_count}"

        # Successful calls
        result1 = await api_call()
        result2 = await api_call()
        assert result1 == "Response 1"
        assert result2 == "Response 2"

        # Failing calls to open circuit
        with pytest.raises(ConnectionError):
            await api_call(should_fail=True)
        with pytest.raises(ConnectionError):
            await api_call(should_fail=True)

        # Circuit should be open
        with pytest.raises(CircuitBreakerError):
            await api_call()

        assert call_count == 4  # 2 success + 2 failures
        assert failure_count == 2

    @pytest.mark.asyncio
    async def test_concurrent_calls_with_circuit_breaker(self):
        """Test circuit breaker handles concurrent calls correctly."""
        breaker = CircuitBreaker(failure_threshold=3)

        async def function_with_delay(fail=False):
            await asyncio.sleep(0.01)  # Small delay
            if fail:
                raise ConnectionError("Failed")
            return "Success"

        # Create mix of successful and failing calls
        tasks = []
        for i in range(6):
            should_fail = i >= 3  # Last 3 calls will fail
            tasks.append(breaker.call(function_with_delay, fail=should_fail))

        # Gather results
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Check results
        success_count = sum(1 for r in results if r == "Success")
        connection_error_count = sum(
            1 for r in results if isinstance(r, ConnectionError)
        )

        assert success_count == 3  # First 3 should succeed
        assert connection_error_count >= 3  # At least 3 failures before circuit opens

        # Circuit should be open after threshold
        assert breaker.state == CircuitState.OPEN

    @pytest.mark.asyncio
    async def test_preconfigured_supabase_breaker(self):
        """Test preconfigured Supabase circuit breaker."""
        from src.agent.circuit_breaker import supabase_circuit_breaker

        assert supabase_circuit_breaker.failure_threshold == 5
        assert supabase_circuit_breaker.recovery_timeout == 30.0
        assert supabase_circuit_breaker.half_open_max_calls == 1
        assert supabase_circuit_breaker.reset_timeout == 60.0
