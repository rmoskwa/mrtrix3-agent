"""
Standalone performance benchmarks for MRtrix3 Agent.

This module provides executable benchmarks for measuring
system performance outside of the test framework.
"""

import asyncio
import gc
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

import psutil

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.agent.cache import CacheManager, TTLCache  # noqa: E402
from src.agent.rate_limiter import RateLimiter  # noqa: E402
from src.agent.circuit_breaker import CircuitBreaker  # noqa: E402
from monitoring.metrics import MetricsCollector  # noqa: E402


class BenchmarkRunner:
    """Runs and reports performance benchmarks."""

    def __init__(self, output_dir: str = "./benchmark_results"):
        """
        Initialize the benchmark runner.

        Args:
            output_dir: Directory for benchmark results
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.results = {}

    def run_benchmark(
        self,
        name: str,
        func: callable,
        iterations: int = 1000,
        warmup: int = 100,
    ) -> Dict[str, Any]:
        """
        Run a single benchmark.

        Args:
            name: Benchmark name
            func: Function to benchmark
            iterations: Number of iterations
            warmup: Number of warmup iterations

        Returns:
            Benchmark results
        """
        print(f"Running benchmark: {name}")

        # Warmup
        for _ in range(warmup):
            func()

        # Collect garbage before timing
        gc.collect()

        # Run benchmark
        times = []
        for _ in range(iterations):
            start = time.perf_counter()
            func()
            end = time.perf_counter()
            times.append(end - start)

        # Calculate statistics
        times.sort()
        total_time = sum(times)
        avg_time = total_time / iterations
        min_time = times[0]
        max_time = times[-1]
        p50 = times[int(len(times) * 0.50)]
        p95 = times[int(len(times) * 0.95)]
        p99 = times[int(len(times) * 0.99)]

        result = {
            "name": name,
            "iterations": iterations,
            "total_time_s": total_time,
            "avg_time_ms": avg_time * 1000,
            "min_time_ms": min_time * 1000,
            "max_time_ms": max_time * 1000,
            "p50_ms": p50 * 1000,
            "p95_ms": p95 * 1000,
            "p99_ms": p99 * 1000,
            "ops_per_second": 1 / avg_time if avg_time > 0 else 0,
        }

        self.results[name] = result
        return result

    async def run_async_benchmark(
        self,
        name: str,
        func: callable,
        iterations: int = 1000,
        warmup: int = 100,
    ) -> Dict[str, Any]:
        """
        Run an async benchmark.

        Args:
            name: Benchmark name
            func: Async function to benchmark
            iterations: Number of iterations
            warmup: Number of warmup iterations

        Returns:
            Benchmark results
        """
        print(f"Running async benchmark: {name}")

        # Warmup
        for _ in range(warmup):
            await func()

        # Collect garbage before timing
        gc.collect()

        # Run benchmark
        times = []
        for _ in range(iterations):
            start = time.perf_counter()
            await func()
            end = time.perf_counter()
            times.append(end - start)

        # Calculate statistics
        times.sort()
        total_time = sum(times)
        avg_time = total_time / iterations
        min_time = times[0]
        max_time = times[-1]
        p50 = times[int(len(times) * 0.50)]
        p95 = times[int(len(times) * 0.95)]
        p99 = times[int(len(times) * 0.99)]

        result = {
            "name": name,
            "iterations": iterations,
            "total_time_s": total_time,
            "avg_time_ms": avg_time * 1000,
            "min_time_ms": min_time * 1000,
            "max_time_ms": max_time * 1000,
            "p50_ms": p50 * 1000,
            "p95_ms": p95 * 1000,
            "p99_ms": p99 * 1000,
            "ops_per_second": 1 / avg_time if avg_time > 0 else 0,
        }

        self.results[name] = result
        return result

    def save_results(self) -> Path:
        """
        Save benchmark results to file.

        Returns:
            Path to results file
        """
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        filename = f"benchmark_results_{timestamp}.json"
        filepath = self.output_dir / filename

        with open(filepath, "w") as f:
            json.dump(
                {
                    "timestamp": datetime.utcnow().isoformat(),
                    "system_info": self._get_system_info(),
                    "results": self.results,
                },
                f,
                indent=2,
            )

        return filepath

    def _get_system_info(self) -> Dict[str, Any]:
        """Get system information."""
        return {
            "cpu_count": psutil.cpu_count(),
            "memory_gb": psutil.virtual_memory().total / (1024**3),
            "python_version": sys.version,
            "platform": sys.platform,
        }

    def print_summary(self) -> None:
        """Print benchmark summary."""
        print("\n" + "=" * 60)
        print("BENCHMARK SUMMARY")
        print("=" * 60)

        for name, result in self.results.items():
            print(f"\n{name}:")
            if isinstance(result, dict):
                if "iterations" in result:
                    print(f"  Iterations: {result['iterations']}")
                    print(f"  Avg time: {result['avg_time_ms']:.3f} ms")
                    print(f"  P50: {result['p50_ms']:.3f} ms")
                    print(f"  P95: {result['p95_ms']:.3f} ms")
                    print(f"  P99: {result['p99_ms']:.3f} ms")
                    print(f"  Ops/sec: {result['ops_per_second']:.0f}")
                else:
                    # Memory results or other metrics
                    for key, value in result.items():
                        print(f"  {key}: {value:.2f}")


def benchmark_cache_operations():
    """Benchmark cache operations."""
    cache = TTLCache(max_size=1000)

    # Populate cache
    for i in range(1000):
        cache.set(f"key_{i}", f"value_{i}")

    # Benchmark function
    counter = 0

    def cache_lookup():
        nonlocal counter
        cache.get(f"key_{counter % 1000}")
        counter += 1

    return cache_lookup


def benchmark_embedding_cache():
    """Benchmark embedding cache."""
    manager = CacheManager(embedding_cache_size=500)

    # Populate with realistic embeddings
    embedding = [0.1] * 768
    for i in range(500):
        manager.cache_embedding(f"query_{i}", embedding, "model")

    counter = 0

    def embedding_lookup():
        nonlocal counter
        manager.get_embedding(f"query_{counter % 500}", "model")
        counter += 1

    return embedding_lookup


def benchmark_metrics_collection():
    """Benchmark metrics collection."""
    collector = MetricsCollector()
    counter = 0

    def record_metrics():
        nonlocal counter
        collector.record_counter("test_counter", 1)
        collector.record_gauge("test_gauge", counter)
        collector.record_latency("test_op", 0.001)
        counter += 1

    return record_metrics


async def benchmark_rate_limiter():
    """Benchmark rate limiter."""
    limiter = RateLimiter(rate=1000, per=1.0)

    async def acquire():
        await limiter.acquire()

    return acquire


async def benchmark_circuit_breaker():
    """Benchmark circuit breaker."""
    breaker = CircuitBreaker(
        failure_threshold=100,
        recovery_timeout=60,
    )

    async def operation():
        return "success"

    async def wrapped_operation():
        await breaker.call(operation)

    return wrapped_operation


def measure_memory_usage():
    """Measure memory usage of components."""
    process = psutil.Process()
    results = {}

    # Baseline
    gc.collect()
    baseline_mb = process.memory_info().rss / (1024 * 1024)

    # Cache memory
    cache = CacheManager(embedding_cache_size=1000, search_cache_size=500)
    embedding = [0.1] * 768
    for i in range(1000):
        cache.cache_embedding(f"query_{i}", embedding, "model")

    gc.collect()
    cache_mb = process.memory_info().rss / (1024 * 1024)
    results["cache_memory_mb"] = cache_mb - baseline_mb

    # Metrics memory
    collector = MetricsCollector()
    for i in range(10000):
        collector.record_counter(f"counter_{i % 100}", 1)
        collector.record_gauge(f"gauge_{i % 50}", i)

    gc.collect()
    metrics_mb = process.memory_info().rss / (1024 * 1024)
    results["metrics_memory_mb"] = metrics_mb - cache_mb

    return results


async def main():
    """Run all benchmarks."""
    runner = BenchmarkRunner()

    # Synchronous benchmarks
    print("Running synchronous benchmarks...")

    cache_func = benchmark_cache_operations()
    runner.run_benchmark("cache_lookup", cache_func, iterations=10000)

    embedding_func = benchmark_embedding_cache()
    runner.run_benchmark("embedding_cache", embedding_func, iterations=5000)

    metrics_func = benchmark_metrics_collection()
    runner.run_benchmark("metrics_collection", metrics_func, iterations=10000)

    # Async benchmarks
    print("\nRunning async benchmarks...")

    rate_limiter_func = await benchmark_rate_limiter()
    await runner.run_async_benchmark("rate_limiter", rate_limiter_func, iterations=1000)

    circuit_breaker_func = await benchmark_circuit_breaker()
    await runner.run_async_benchmark(
        "circuit_breaker", circuit_breaker_func, iterations=10000
    )

    # Memory benchmarks
    print("\nMeasuring memory usage...")
    memory_results = measure_memory_usage()
    runner.results["memory_usage"] = memory_results

    # Save and print results
    runner.print_summary()
    results_file = runner.save_results()
    print(f"\nResults saved to: {results_file}")


if __name__ == "__main__":
    asyncio.run(main())
