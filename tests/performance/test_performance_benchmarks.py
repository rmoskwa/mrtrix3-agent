"""
Performance benchmarks for MRtrix3 Agent.

These tests measure and validate performance characteristics
of critical components.
"""

import asyncio
import gc
import sys
import time
from pathlib import Path

import psutil
import pytest


sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from src.agent.cache import CacheManager, TTLCache
from src.agent.rate_limiter import RateLimiter
from src.agent.circuit_breaker import CircuitBreaker
from monitoring.metrics import MetricsCollector


class TestSearchPerformance:
    """Benchmarks for search operations."""

    @pytest.mark.benchmark
    def test_cache_lookup_performance(self):
        """Test cache lookup performance."""
        cache = TTLCache(max_size=1000)

        # Populate cache
        for i in range(1000):
            cache.set(f"key_{i}", f"value_{i}")

        # Measure lookup time
        start = time.perf_counter()
        for i in range(10000):
            cache.get(f"key_{i % 1000}")
        end = time.perf_counter()

        total_time = end - start
        avg_lookup_time_us = (total_time / 10000) * 1_000_000

        # Should be very fast (< 10 microseconds per lookup)
        assert (
            avg_lookup_time_us < 10
        ), f"Cache lookup too slow: {avg_lookup_time_us:.2f}μs"

    def test_embedding_cache_performance(self):
        """Test embedding cache performance with real-world sizes."""
        manager = CacheManager(embedding_cache_size=500)

        # Create realistic embedding (768 dimensions)
        embedding = [0.1] * 768

        # Populate cache
        for i in range(500):
            manager.cache_embedding(f"query_{i}", embedding, "test_model")

        # Measure retrieval time
        start = time.perf_counter()
        hits = 0
        for _ in range(1000):
            for i in range(100):
                result = manager.get_embedding(f"query_{i}", "test_model")
                if result:
                    hits += 1
        end = time.perf_counter()

        total_time = end - start
        operations = 100000  # 1000 * 100
        avg_time_us = (total_time / operations) * 1_000_000

        assert hits == 100000, "All queries should hit cache"
        assert avg_time_us < 20, f"Embedding cache lookup too slow: {avg_time_us:.2f}μs"

    def test_search_result_cache_performance(self):
        """Test search result cache performance."""
        manager = CacheManager(search_cache_size=200)

        # Create realistic search results
        results = [
            {
                "id": f"doc_{j}",
                "content": f"Content for document {j}" * 100,  # ~2KB per result
                "score": 0.95 - (j * 0.01),
            }
            for j in range(10)
        ]

        # Populate cache
        for i in range(200):
            manager.cache_search_results(
                query=f"query_{i}",
                results=results,
                search_type="semantic",
            )

        # Measure retrieval time
        start = time.perf_counter()
        for _ in range(1000):
            for i in range(50):
                manager.get_search_results(f"query_{i}", "semantic")
        end = time.perf_counter()

        total_time = end - start
        operations = 50000  # 1000 * 50
        avg_time_us = (total_time / operations) * 1_000_000

        assert avg_time_us < 50, f"Search cache lookup too slow: {avg_time_us:.2f}μs"


class TestRateLimiterStress:
    """Stress tests for rate limiter."""

    @pytest.mark.asyncio
    async def test_rate_limiter_throughput(self):
        """Test rate limiter can handle target throughput."""
        limiter = RateLimiter(rate=100, per=1.0)

        # Attempt 100 requests rapidly
        start = time.perf_counter()
        success_count = 0

        for _ in range(100):
            await (
                limiter.acquire()
            )  # acquire() doesn't return boolean, just waits if needed
            success_count += 1

        end = time.perf_counter()
        duration = end - start

        # All 100 should succeed
        assert success_count == 100
        # Should complete within reasonable time (allowing for rate limiting)
        assert duration < 2.0, f"Rate limiter took too long: {duration:.3f}s"

    @pytest.mark.asyncio
    async def test_rate_limiter_concurrent_load(self):
        """Test rate limiter under concurrent load."""
        limiter = RateLimiter(rate=50, per=1.0)
        success_count = 0
        lock = asyncio.Lock()

        async def make_request():
            nonlocal success_count
            await (
                limiter.acquire()
            )  # acquire() doesn't return boolean, just waits if needed
            async with lock:
                success_count += 1
            await asyncio.sleep(0.001)  # Simulate work

        # Create 100 concurrent tasks
        tasks = [make_request() for _ in range(100)]
        start = time.perf_counter()
        await asyncio.gather(*tasks)
        end = time.perf_counter()

        # All 100 requests should eventually complete
        assert success_count == 100
        # Should complete within reasonable time (50 req/s means 100 requests need ~2s)
        assert (end - start) < 3.0, f"Rate limiter took too long: {end - start:.3f}s"


class TestCircuitBreakerPerformance:
    """Performance tests for circuit breaker."""

    @pytest.mark.asyncio
    async def test_circuit_breaker_overhead(self):
        """Test circuit breaker overhead on successful calls."""
        breaker = CircuitBreaker(
            failure_threshold=5,
            recovery_timeout=1,
        )

        async def fast_operation():
            return "success"

        # Measure overhead
        iterations = 10000

        # Baseline without circuit breaker
        start = time.perf_counter()
        for _ in range(iterations):
            await fast_operation()
        baseline_time = time.perf_counter() - start

        # With circuit breaker
        start = time.perf_counter()
        for _ in range(iterations):
            await breaker.call(fast_operation)
        breaker_time = time.perf_counter() - start

        overhead_ms = ((breaker_time - baseline_time) / iterations) * 1000
        assert (
            overhead_ms < 0.1
        ), f"Circuit breaker overhead too high: {overhead_ms:.3f}ms"

    @pytest.mark.asyncio
    async def test_circuit_breaker_state_transitions(self):
        """Test circuit breaker state transition performance."""
        breaker = CircuitBreaker(
            failure_threshold=3,
            recovery_timeout=0.1,
        )

        call_count = 0

        async def flaky_operation():
            nonlocal call_count
            call_count += 1
            if call_count <= 3:
                raise ValueError("Simulated failure")
            return "success"

        # Trigger failures to open circuit
        for _ in range(3):
            try:
                await breaker.call(flaky_operation)
            except ValueError:
                pass

        # Circuit should be open
        from src.agent.circuit_breaker import CircuitState

        assert await breaker._get_state() == CircuitState.OPEN

        # Wait for recovery
        await asyncio.sleep(0.11)

        # Should transition to half-open and then closed
        result = await breaker.call(flaky_operation)
        assert result == "success"
        assert await breaker._get_state() == CircuitState.CLOSED


class TestMetricsCollectorPerformance:
    """Performance tests for metrics collection."""

    def test_metrics_collection_overhead(self):
        """Test overhead of metrics collection."""
        collector = MetricsCollector()
        iterations = 10000

        # Measure counter recording
        start = time.perf_counter()
        for i in range(iterations):
            collector.record_counter("test_counter", 1)
        counter_time = time.perf_counter() - start

        # Measure gauge recording
        start = time.perf_counter()
        for i in range(iterations):
            collector.record_gauge("test_gauge", i)
        gauge_time = time.perf_counter() - start

        # Measure latency recording
        start = time.perf_counter()
        for i in range(iterations):
            collector.record_latency("test_latency", 0.001)
        latency_time = time.perf_counter() - start

        # Average overhead per operation
        counter_overhead_us = (counter_time / iterations) * 1_000_000
        gauge_overhead_us = (gauge_time / iterations) * 1_000_000
        latency_overhead_us = (latency_time / iterations) * 1_000_000

        assert (
            counter_overhead_us < 20
        ), f"Counter overhead: {counter_overhead_us:.2f}μs"
        assert gauge_overhead_us < 20, f"Gauge overhead: {gauge_overhead_us:.2f}μs"
        assert (
            latency_overhead_us < 20
        ), f"Latency overhead: {latency_overhead_us:.2f}μs"

    def test_metrics_timer_overhead(self):
        """Test overhead of Timer context manager."""
        collector = MetricsCollector()

        def timed_operation():
            time.sleep(0.001)
            return "result"

        # Baseline without timer
        start = time.perf_counter()
        for _ in range(100):
            timed_operation()
        baseline_time = time.perf_counter() - start

        # With timer
        start = time.perf_counter()
        for _ in range(100):
            with collector.start_timer("test_op"):
                timed_operation()
        timer_time = time.perf_counter() - start

        overhead_ms = ((timer_time - baseline_time) / 100) * 1000
        assert overhead_ms < 1, f"Timer overhead: {overhead_ms:.2f}ms"


class TestMemoryUsage:
    """Memory usage benchmarks."""

    def test_cache_memory_usage(self):
        """Test memory usage of cache with realistic data."""
        process = psutil.Process()
        gc.collect()
        memory_before = process.memory_info().rss / 1024 / 1024  # MB

        # Create cache with realistic embeddings
        cache = CacheManager(
            embedding_cache_size=1000,
            search_cache_size=500,
        )

        # Add 1000 embeddings (768-dim each)
        embedding = [0.1] * 768
        for i in range(1000):
            cache.cache_embedding(f"query_{i}", embedding, "model")

        # Add 500 search results (10 results each, ~2KB per result)
        results = [
            {"id": f"doc_{j}", "content": "x" * 2000, "score": 0.9} for j in range(10)
        ]
        for i in range(500):
            cache.cache_search_results(f"search_{i}", results, "semantic")

        gc.collect()
        memory_after = process.memory_info().rss / 1024 / 1024  # MB
        memory_used = memory_after - memory_before

        # Should use less than 100MB for this cache size
        assert memory_used < 100, f"Cache using too much memory: {memory_used:.1f}MB"

    def test_metrics_memory_usage(self):
        """Test memory usage of metrics collection."""
        process = psutil.Process()
        gc.collect()
        memory_before = process.memory_info().rss / 1024 / 1024

        collector = MetricsCollector()

        # Simulate heavy metrics collection
        for i in range(10000):
            collector.record_counter(f"counter_{i % 100}", 1)
            collector.record_gauge(f"gauge_{i % 50}", i)
            collector.record_latency(f"latency_{i % 20}", 0.001)

        gc.collect()
        memory_after = process.memory_info().rss / 1024 / 1024
        memory_used = memory_after - memory_before

        # Should use less than 50MB
        assert memory_used < 50, f"Metrics using too much memory: {memory_used:.1f}MB"


class TestConcurrentPerformance:
    """Benchmarks for concurrent operations."""

    @pytest.mark.asyncio
    async def test_concurrent_cache_access(self):
        """Test cache performance under concurrent access."""
        manager = CacheManager()

        # Populate cache
        for i in range(100):
            manager.cache_embedding(f"key_{i}", [0.1] * 768, "model")

        async def access_cache(iterations):
            for i in range(iterations):
                key = f"key_{i % 100}"
                manager.get_embedding(key, "model")

        # Run concurrent tasks
        tasks = [access_cache(1000) for _ in range(10)]
        start = time.perf_counter()
        await asyncio.gather(*tasks)
        end = time.perf_counter()

        total_ops = 10000  # 10 tasks * 1000 iterations
        ops_per_second = total_ops / (end - start)

        # Should handle at least 100k ops/sec
        assert (
            ops_per_second > 100000
        ), f"Concurrent cache ops too slow: {ops_per_second:.0f} ops/s"

    @pytest.mark.asyncio
    async def test_concurrent_metrics_collection(self):
        """Test metrics collection under concurrent load."""
        collector = MetricsCollector()

        async def collect_metrics(iterations):
            for i in range(iterations):
                collector.record_counter("test", 1)
                collector.record_latency("op", 0.001)
                await asyncio.sleep(0)  # Yield control

        # Run concurrent tasks
        tasks = [collect_metrics(100) for _ in range(20)]
        start = time.perf_counter()
        await asyncio.gather(*tasks)
        end = time.perf_counter()

        duration = end - start
        total_ops = 2000  # 20 tasks * 100 iterations
        ops_per_second = total_ops / duration

        # Should handle at least 10k ops/sec
        assert (
            ops_per_second > 10000
        ), f"Metrics collection too slow: {ops_per_second:.0f} ops/s"


# Performance baselines for documentation
PERFORMANCE_BASELINES = {
    "cache_lookup": {"target_us": 10, "description": "Single cache lookup"},
    "embedding_cache": {"target_us": 20, "description": "Embedding cache retrieval"},
    "search_cache": {"target_us": 50, "description": "Search result cache retrieval"},
    "rate_limiter": {
        "target_ms": 100,
        "description": "Rate limiter overhead for 100 requests",
    },
    "circuit_breaker": {
        "target_ms": 0.1,
        "description": "Circuit breaker overhead per call",
    },
    "metrics_recording": {"target_us": 10, "description": "Single metric recording"},
    "cache_memory": {
        "target_mb": 100,
        "description": "Memory for 1000 embeddings + 500 search results",
    },
    "metrics_memory": {
        "target_mb": 50,
        "description": "Memory for 10000 metric entries",
    },
    "concurrent_cache_ops": {
        "target_ops_per_sec": 100000,
        "description": "Concurrent cache operations",
    },
    "concurrent_metrics_ops": {
        "target_ops_per_sec": 10000,
        "description": "Concurrent metrics collection",
    },
}
