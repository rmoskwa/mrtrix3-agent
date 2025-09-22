"""
Metrics collection system for monitoring.

This module provides metrics collection and export functionality
for tracking performance and usage patterns.
"""

import json
import os
import time
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional


@dataclass
class Metric:
    """Individual metric data point."""

    name: str
    value: float
    timestamp: float = field(default_factory=time.time)
    tags: Dict[str, str] = field(default_factory=dict)
    metric_type: str = "gauge"  # gauge, counter, histogram


@dataclass
class MetricsSummary:
    """Summary of collected metrics."""

    total_requests: int = 0
    average_latency_ms: float = 0.0
    p95_latency_ms: float = 0.0
    p99_latency_ms: float = 0.0
    total_tokens_used: int = 0
    cache_hit_ratio: float = 0.0
    error_rate: float = 0.0
    search_success_rate: float = 0.0
    period_start: Optional[str] = None
    period_end: Optional[str] = None


class MetricsCollector:
    """Collects and manages metrics for monitoring."""

    def __init__(self, export_path: Optional[str] = None):
        """
        Initialize the metrics collector.

        Args:
            export_path: Path for metrics export files
        """
        self.export_path = Path(
            export_path or os.getenv("METRICS_EXPORT_PATH", "./metrics/")
        )
        self.metrics: List[Metric] = []
        self.counters: Dict[str, float] = defaultdict(float)
        self.gauges: Dict[str, float] = {}
        self.histograms: Dict[str, List[float]] = defaultdict(list)
        self.start_time = time.time()

        # Specific tracking for key metrics
        self.request_count = 0
        self.error_count = 0
        self.cache_hits = 0
        self.cache_misses = 0
        self.search_successes = 0
        self.search_failures = 0
        self.token_usage = 0

    def record_counter(self, name: str, value: float = 1.0, **tags: str) -> None:
        """
        Record a counter metric (cumulative).

        Args:
            name: Metric name
            value: Value to increment by
            **tags: Additional tags for the metric
        """
        self.counters[name] += value
        metric = Metric(
            name=name, value=self.counters[name], tags=tags, metric_type="counter"
        )
        self.metrics.append(metric)

        # Update specific counters
        if name == "request_count":
            self.request_count += int(value)
        elif name == "error_count":
            self.error_count += int(value)
        elif name == "cache_hit":
            self.cache_hits += int(value)
        elif name == "cache_miss":
            self.cache_misses += int(value)
        elif name == "search_success":
            self.search_successes += int(value)
        elif name == "search_failure":
            self.search_failures += int(value)
        elif name == "tokens_used":
            self.token_usage += int(value)

    def record_gauge(self, name: str, value: float, **tags: str) -> None:
        """
        Record a gauge metric (point-in-time value).

        Args:
            name: Metric name
            value: Current value
            **tags: Additional tags for the metric
        """
        self.gauges[name] = value
        metric = Metric(name=name, value=value, tags=tags, metric_type="gauge")
        self.metrics.append(metric)

    def record_latency(self, name: str, duration_seconds: float, **tags: str) -> None:
        """
        Record a latency measurement.

        Args:
            name: Metric name
            duration_seconds: Duration in seconds
            **tags: Additional tags for the metric
        """
        duration_ms = duration_seconds * 1000
        self.histograms[name].append(duration_ms)
        metric = Metric(
            name=f"{name}_latency",
            value=duration_ms,
            tags=tags,
            metric_type="histogram",
        )
        self.metrics.append(metric)

    def start_timer(self, name: str) -> "Timer":
        """
        Start a timer for measuring latency.

        Args:
            name: Name for the timer

        Returns:
            Timer context manager
        """
        return Timer(self, name)

    def calculate_percentile(self, values: List[float], percentile: float) -> float:
        """
        Calculate percentile of values.

        Args:
            values: List of values
            percentile: Percentile to calculate (0-100)

        Returns:
            Percentile value
        """
        if not values:
            return 0.0

        sorted_values = sorted(values)
        index = int(len(sorted_values) * percentile / 100)
        index = min(index, len(sorted_values) - 1)
        return sorted_values[index]

    def get_summary(self) -> MetricsSummary:
        """
        Get summary of collected metrics.

        Returns:
            Metrics summary
        """
        summary = MetricsSummary()

        # Calculate period
        if self.metrics:
            summary.period_start = datetime.fromtimestamp(self.start_time).isoformat()
            summary.period_end = datetime.utcnow().isoformat()

        # Basic counts
        summary.total_requests = self.request_count
        summary.total_tokens_used = self.token_usage

        # Calculate latencies
        all_latencies = []
        for latencies in self.histograms.values():
            all_latencies.extend(latencies)

        if all_latencies:
            summary.average_latency_ms = sum(all_latencies) / len(all_latencies)
            summary.p95_latency_ms = self.calculate_percentile(all_latencies, 95)
            summary.p99_latency_ms = self.calculate_percentile(all_latencies, 99)

        # Calculate rates
        if self.request_count > 0:
            summary.error_rate = self.error_count / self.request_count

        total_cache_ops = self.cache_hits + self.cache_misses
        if total_cache_ops > 0:
            summary.cache_hit_ratio = self.cache_hits / total_cache_ops

        total_searches = self.search_successes + self.search_failures
        if total_searches > 0:
            summary.search_success_rate = self.search_successes / total_searches

        return summary

    def export(self, filename: Optional[str] = None) -> Path:
        """
        Export metrics to file.

        Args:
            filename: Optional filename, auto-generated if not provided

        Returns:
            Path to exported file
        """
        # Create export directory if needed
        self.export_path.mkdir(parents=True, exist_ok=True)

        # Generate filename if not provided
        if filename is None:
            timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            filename = f"metrics_{timestamp}.json"

        export_file = self.export_path / filename

        # Prepare export data
        export_data = {
            "summary": asdict(self.get_summary()),
            "counters": dict(self.counters),
            "gauges": dict(self.gauges),
            "histograms": {
                name: {
                    "values": values,
                    "count": len(values),
                    "average": sum(values) / len(values) if values else 0,
                    "min": min(values) if values else 0,
                    "max": max(values) if values else 0,
                    "p50": self.calculate_percentile(values, 50),
                    "p95": self.calculate_percentile(values, 95),
                    "p99": self.calculate_percentile(values, 99),
                }
                for name, values in self.histograms.items()
            },
            "raw_metrics": [
                asdict(m) for m in self.metrics[-1000:]
            ],  # Last 1000 metrics
        }

        # Write to file
        with open(export_file, "w") as f:
            json.dump(export_data, f, indent=2, default=str)

        return export_file

    def reset(self) -> None:
        """Reset all metrics."""
        self.metrics.clear()
        self.counters.clear()
        self.gauges.clear()
        self.histograms.clear()
        self.start_time = time.time()
        self.request_count = 0
        self.error_count = 0
        self.cache_hits = 0
        self.cache_misses = 0
        self.search_successes = 0
        self.search_failures = 0
        self.token_usage = 0


class Timer:
    """Context manager for timing operations."""

    def __init__(self, collector: MetricsCollector, name: str):
        """
        Initialize timer.

        Args:
            collector: Metrics collector to record to
            name: Name for the timer
        """
        self.collector = collector
        self.name = name
        self.start_time = None
        self.tags = {}

    def add_tags(self, **tags: str) -> "Timer":
        """
        Add tags to the timer.

        Args:
            **tags: Tags to add

        Returns:
            Self for chaining
        """
        self.tags.update(tags)
        return self

    def __enter__(self) -> "Timer":
        """Start the timer."""
        self.start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Stop timer and record latency."""
        if self.start_time is not None:
            duration = time.time() - self.start_time
            self.collector.record_latency(self.name, duration, **self.tags)


# Global metrics collector instance
_global_collector: Optional[MetricsCollector] = None


def get_global_collector() -> MetricsCollector:
    """
    Get the global metrics collector instance.

    Returns:
        Global metrics collector
    """
    global _global_collector
    if _global_collector is None:
        _global_collector = MetricsCollector()
    return _global_collector


def export_metrics(filename: Optional[str] = None) -> Path:
    """
    Export global metrics.

    Args:
        filename: Optional filename

    Returns:
        Path to exported file
    """
    collector = get_global_collector()
    return collector.export(filename)


def get_metrics_summary() -> MetricsSummary:
    """
    Get summary of global metrics.

    Returns:
        Metrics summary
    """
    collector = get_global_collector()
    return collector.get_summary()


def reset_global_metrics() -> None:
    """Reset global metrics collector."""
    collector = get_global_collector()
    collector.reset()
