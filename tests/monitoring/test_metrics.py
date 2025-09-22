"""
Tests for metrics collection system.
"""

import json
import time
from unittest.mock import patch

import pytest

from monitoring.metrics import (
    Metric,
    MetricsCollector,
    Timer,
    export_metrics,
    get_global_collector,
    get_metrics_summary,
    reset_global_metrics,
)


class TestMetric:
    """Tests for Metric dataclass."""

    def test_metric_creation(self):
        """Test creating a metric."""
        metric = Metric(
            name="test_metric",
            value=42.5,
            tags={"environment": "test"},
            metric_type="gauge",
        )

        assert metric.name == "test_metric"
        assert metric.value == 42.5
        assert metric.tags["environment"] == "test"
        assert metric.metric_type == "gauge"
        assert metric.timestamp > 0

    def test_metric_defaults(self):
        """Test metric with default values."""
        metric = Metric(name="test", value=1.0)

        assert metric.tags == {}
        assert metric.metric_type == "gauge"


class TestMetricsCollector:
    """Tests for MetricsCollector."""

    def test_record_counter(self):
        """Test recording counter metrics."""
        collector = MetricsCollector()

        collector.record_counter("requests", 1)
        collector.record_counter("requests", 2)

        assert collector.counters["requests"] == 3
        assert len(collector.metrics) == 2

    def test_record_gauge(self):
        """Test recording gauge metrics."""
        collector = MetricsCollector()

        collector.record_gauge("memory", 100.5)
        collector.record_gauge("memory", 150.2)

        assert collector.gauges["memory"] == 150.2  # Latest value
        assert len(collector.metrics) == 2

    def test_record_latency(self):
        """Test recording latency metrics."""
        collector = MetricsCollector()

        collector.record_latency("api_call", 0.5)
        collector.record_latency("api_call", 0.3)
        collector.record_latency("api_call", 0.7)

        assert len(collector.histograms["api_call"]) == 3
        assert collector.histograms["api_call"][0] == 500  # Convert to ms
        assert collector.histograms["api_call"][1] == 300
        assert collector.histograms["api_call"][2] == 700

    def test_calculate_percentile(self):
        """Test percentile calculation."""
        collector = MetricsCollector()

        values = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        p50 = collector.calculate_percentile(values, 50)
        p95 = collector.calculate_percentile(values, 95)

        assert p50 == 6  # 50th percentile of 10 items is 6th element (index 5)
        assert p95 == 10  # 95th percentile of 10 items is 10th element (index 9)

        # Test with empty list
        assert collector.calculate_percentile([], 50) == 0.0

    def test_get_summary(self):
        """Test getting metrics summary."""
        collector = MetricsCollector()

        # Record some metrics
        collector.record_counter("request_count", 5)
        collector.record_counter("error_count", 1)
        collector.record_counter("cache_hit", 3)
        collector.record_counter("cache_miss", 2)
        collector.record_counter("search_success", 4)
        collector.record_counter("search_failure", 1)
        collector.record_counter("tokens_used", 1000)

        collector.record_latency("test", 0.1)
        collector.record_latency("test", 0.2)
        collector.record_latency("test", 0.3)

        summary = collector.get_summary()

        assert summary.total_requests == 5
        assert summary.error_rate == 0.2  # 1/5
        assert summary.cache_hit_ratio == 0.6  # 3/5
        assert summary.search_success_rate == 0.8  # 4/5
        assert summary.total_tokens_used == 1000
        assert summary.average_latency_ms == pytest.approx(200, rel=1e-2)

    def test_export(self, tmp_path):
        """Test exporting metrics."""
        export_path = tmp_path / "test_metrics"
        collector = MetricsCollector(export_path=str(export_path))

        # Record some metrics
        collector.record_counter("test_counter", 10)
        collector.record_gauge("test_gauge", 42.5)
        collector.record_latency("test_latency", 0.5)

        # Export metrics
        export_file = collector.export()

        assert export_file.exists()
        assert export_file.parent == export_path

        # Check exported data
        with open(export_file) as f:
            data = json.load(f)

        assert "summary" in data
        assert "counters" in data
        assert "gauges" in data
        assert "histograms" in data
        assert data["counters"]["test_counter"] == 10
        assert data["gauges"]["test_gauge"] == 42.5

    def test_reset(self):
        """Test resetting metrics."""
        collector = MetricsCollector()

        collector.record_counter("test", 5)
        collector.record_gauge("memory", 100)
        collector.record_latency("api", 0.1)

        collector.reset()

        assert len(collector.metrics) == 0
        assert len(collector.counters) == 0
        assert len(collector.gauges) == 0
        assert len(collector.histograms) == 0
        assert collector.request_count == 0

    def test_timer_context_manager(self):
        """Test Timer context manager."""
        collector = MetricsCollector()

        with collector.start_timer("test_operation") as timer:
            timer.add_tags(environment="test")
            time.sleep(0.01)  # Simulate work

        # Check that latency was recorded
        assert "test_operation" in collector.histograms
        assert len(collector.histograms["test_operation"]) == 1
        assert collector.histograms["test_operation"][0] >= 10  # At least 10ms

    def test_specific_metric_tracking(self):
        """Test tracking of specific metric types."""
        collector = MetricsCollector()

        # Test request counting
        collector.record_counter("request_count", 3)
        assert collector.request_count == 3

        # Test error counting
        collector.record_counter("error_count", 2)
        assert collector.error_count == 2

        # Test cache tracking
        collector.record_counter("cache_hit", 5)
        collector.record_counter("cache_miss", 3)
        assert collector.cache_hits == 5
        assert collector.cache_misses == 3

        # Test search tracking
        collector.record_counter("search_success", 7)
        collector.record_counter("search_failure", 1)
        assert collector.search_successes == 7
        assert collector.search_failures == 1

        # Test token tracking
        collector.record_counter("tokens_used", 500)
        assert collector.token_usage == 500


class TestGlobalFunctions:
    """Tests for global collector functions."""

    def test_get_global_collector(self):
        """Test getting global collector instance."""
        collector1 = get_global_collector()
        collector2 = get_global_collector()

        assert collector1 is collector2  # Same instance

    def test_export_global_metrics(self, tmp_path):
        """Test exporting global metrics."""
        with patch.dict("os.environ", {"METRICS_EXPORT_PATH": str(tmp_path)}):
            # Reset and get fresh collector
            reset_global_metrics()
            collector = get_global_collector()

            # Record some metrics
            collector.record_counter("global_test", 5)

            # Export
            export_file = export_metrics("test_export.json")

            assert export_file.exists()
            assert export_file.name == "test_export.json"

    def test_get_global_metrics_summary(self):
        """Test getting global metrics summary."""
        reset_global_metrics()
        collector = get_global_collector()

        collector.record_counter("request_count", 10)
        collector.record_latency("test", 0.1)

        summary = get_metrics_summary()

        assert summary.total_requests == 10
        assert summary.average_latency_ms > 0

    def test_reset_global_metrics(self):
        """Test resetting global metrics."""
        collector = get_global_collector()
        collector.record_counter("test", 5)

        reset_global_metrics()

        # After reset, should be clean
        summary = get_metrics_summary()
        assert summary.total_requests == 0


class TestTimer:
    """Tests for Timer context manager."""

    def test_timer_basic(self):
        """Test basic timer functionality."""
        collector = MetricsCollector()
        timer = Timer(collector, "test_timer")

        with timer:
            time.sleep(0.01)

        assert "test_timer" in collector.histograms
        assert collector.histograms["test_timer"][0] >= 10

    def test_timer_with_tags(self):
        """Test timer with tags."""
        collector = MetricsCollector()

        with Timer(collector, "tagged_timer").add_tags(service="test", endpoint="/api"):
            time.sleep(0.01)

        # Check metric was recorded
        metric = collector.metrics[-1]
        assert metric.name == "tagged_timer_latency"
        assert metric.tags["service"] == "test"
        assert metric.tags["endpoint"] == "/api"
