"""
Tests for specialized metric collectors.
"""

import asyncio
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from monitoring.collectors import (
    ChromaDBCollector,
    ChromaDBMetrics,
    SearchCollector,
    SearchMetrics,
    TokenUsageCollector,
    TokenMetrics,
    MemoryCollector,
    APICallCollector,
    get_chromadb_collector,
    get_search_collector,
    get_token_collector,
    get_memory_collector,
    get_api_collector,
)


class TestChromaDBCollector:
    """Tests for ChromaDB metrics collector."""

    def test_record_query(self):
        """Test recording ChromaDB queries."""
        collector = ChromaDBCollector()

        # Record a normal query
        collector.record_query(0.5, result_count=10, collection_name="test")

        assert collector.chroma_metrics.query_count == 1
        assert collector.chroma_metrics.total_query_time_ms == 500
        assert collector.chroma_metrics.average_query_time_ms == 500
        assert collector.chroma_metrics.slow_query_count == 0

        # Record a slow query
        collector.record_query(1.5, result_count=20)

        assert collector.chroma_metrics.query_count == 2
        assert collector.chroma_metrics.total_query_time_ms == 2000
        assert collector.chroma_metrics.average_query_time_ms == 1000
        assert collector.chroma_metrics.slow_query_count == 1

    def test_record_collection_size(self):
        """Test recording collection size."""
        collector = ChromaDBCollector()

        collector.record_collection_size(1000, "documents")

        assert collector.chroma_metrics.collection_size == 1000

    @patch("monitoring.collectors.psutil.Process")
    def test_record_memory_usage(self, mock_process_class):
        """Test recording memory usage."""
        mock_process = MagicMock()
        mock_process.memory_info.return_value = MagicMock(rss=104857600)  # 100 MB
        mock_process_class.return_value = mock_process

        collector = ChromaDBCollector()
        collector.record_memory_usage()

        assert collector.chroma_metrics.memory_usage_mb == 100.0

    def test_get_metrics(self):
        """Test getting ChromaDB metrics."""
        collector = ChromaDBCollector()

        collector.record_query(0.1, 5)
        collector.record_query(0.2, 10)
        collector.record_collection_size(500)

        metrics = collector.get_metrics()

        assert isinstance(metrics, ChromaDBMetrics)
        assert metrics.query_count == 2
        assert metrics.average_query_time_ms == 150
        assert metrics.collection_size == 500


class TestSearchCollector:
    """Tests for search metrics collector."""

    def test_record_search_success(self):
        """Test recording successful searches."""
        collector = SearchCollector()

        collector.record_search(
            0.3, result_count=5, success=True, search_type="semantic"
        )
        collector.record_search(0.2, result_count=0, success=True, search_type="bm25")

        assert collector.search_metrics.total_searches == 2
        assert collector.search_metrics.successful_searches == 2
        assert collector.search_metrics.failed_searches == 0
        assert collector.search_metrics.empty_result_searches == 1
        assert collector.search_metrics.average_results_returned == 2.5

    def test_record_search_failure(self):
        """Test recording failed searches."""
        collector = SearchCollector()

        collector.record_search(0.1, result_count=0, success=False)

        assert collector.search_metrics.total_searches == 1
        assert collector.search_metrics.successful_searches == 0
        assert collector.search_metrics.failed_searches == 1

    def test_get_metrics(self):
        """Test getting search metrics."""
        collector = SearchCollector()

        collector.record_search(0.1, 10, True)
        collector.record_search(0.2, 5, True)
        collector.record_search(0.3, 0, False)

        metrics = collector.get_metrics()

        assert isinstance(metrics, SearchMetrics)
        assert metrics.total_searches == 3
        assert metrics.successful_searches == 2
        assert metrics.failed_searches == 1
        assert metrics.average_search_time_ms == 200  # (100 + 200 + 300) / 3


class TestTokenUsageCollector:
    """Tests for token usage collector."""

    def test_record_token_usage(self):
        """Test recording token usage."""
        collector = TokenUsageCollector()

        collector.record_token_usage(100, 50, model="gemini-pro")
        collector.record_token_usage(200, 100, model="gemini-pro")

        assert collector.token_metrics.total_prompt_tokens == 300
        assert collector.token_metrics.total_completion_tokens == 150
        assert collector.token_metrics.total_tokens == 450
        assert collector.token_metrics.average_prompt_tokens == 150
        assert collector.token_metrics.average_completion_tokens == 75

    def test_record_conversation(self):
        """Test recording conversations."""
        collector = TokenUsageCollector()

        collector.record_conversation()
        collector.record_conversation()

        assert collector.token_metrics.conversations_tracked == 2

    def test_get_metrics(self):
        """Test getting token metrics."""
        collector = TokenUsageCollector()

        collector.record_token_usage(100, 50)
        collector.record_conversation()

        metrics = collector.get_metrics()

        assert isinstance(metrics, TokenMetrics)
        assert metrics.total_tokens == 150
        assert metrics.conversations_tracked == 1


class TestMemoryCollector:
    """Tests for memory collector."""

    @patch("monitoring.collectors.psutil.Process")
    def test_collect_memory_metrics(self, mock_process_class):
        """Test collecting memory metrics."""
        mock_process = MagicMock()
        mock_process.memory_info.return_value = MagicMock(
            rss=209715200,  # 200 MB
            vms=419430400,  # 400 MB
        )
        mock_process.memory_percent.return_value = 5.5
        mock_process_class.return_value = mock_process

        collector = MemoryCollector()
        metrics = collector.collect_memory_metrics()

        assert metrics["rss_mb"] == 200.0
        assert metrics["vms_mb"] == 400.0
        assert metrics["percent"] == 5.5

    @pytest.mark.asyncio
    @patch("monitoring.collectors.psutil.Process")
    async def test_monitor_memory(self, mock_process_class):
        """Test memory monitoring loop."""
        mock_process = MagicMock()
        mock_process.memory_info.return_value = MagicMock(rss=104857600)
        mock_process.memory_percent.return_value = 2.5
        mock_process_class.return_value = mock_process

        collector = MemoryCollector()

        # Run monitor for a short time
        monitor_task = asyncio.create_task(
            collector.monitor_memory(interval_seconds=0.01)
        )

        await asyncio.sleep(0.03)  # Let it collect a couple times
        monitor_task.cancel()

        try:
            await monitor_task
        except asyncio.CancelledError:
            pass

        # Should have collected metrics
        assert mock_process.memory_info.called


class TestAPICallCollector:
    """Tests for API call collector."""

    def test_record_api_call_success(self):
        """Test recording successful API calls."""
        collector = APICallCollector()

        collector.record_api_call(
            "gemini",
            "/generate",
            0.5,
            status_code=200,
        )
        collector.record_api_call(
            "gemini",
            "/embed",
            0.3,
            status_code=200,
        )

        assert "gemini" in collector.api_calls
        assert len(collector.api_calls["gemini"]) == 2
        assert collector.api_calls["gemini"][0] == 500  # ms
        assert collector.api_calls["gemini"][1] == 300

    def test_record_api_call_error(self):
        """Test recording API call errors."""
        collector = APICallCollector()

        collector.record_api_call(
            "supabase",
            "/query",
            1.0,
            status_code=500,
            error="Internal Server Error",
        )

        assert collector.api_errors["supabase"] == 1

    def test_get_api_stats(self):
        """Test getting API statistics."""
        collector = APICallCollector()

        collector.record_api_call("gemini", "/generate", 0.5)
        collector.record_api_call("gemini", "/generate", 0.7)
        collector.record_api_call("gemini", "/generate", 0.3, error="Timeout")

        stats = collector.get_api_stats()

        assert "gemini" in stats
        assert stats["gemini"]["call_count"] == 3
        assert stats["gemini"]["average_latency_ms"] == 500
        assert stats["gemini"]["min_latency_ms"] == 300
        assert stats["gemini"]["max_latency_ms"] == 700
        assert stats["gemini"]["error_count"] == 1


class TestGlobalCollectors:
    """Tests for global collector instances."""

    def test_get_chromadb_collector(self):
        """Test getting global ChromaDB collector."""
        collector1 = get_chromadb_collector()
        collector2 = get_chromadb_collector()

        assert collector1 is collector2
        assert isinstance(collector1, ChromaDBCollector)

    def test_get_search_collector(self):
        """Test getting global search collector."""
        collector1 = get_search_collector()
        collector2 = get_search_collector()

        assert collector1 is collector2
        assert isinstance(collector1, SearchCollector)

    def test_get_token_collector(self):
        """Test getting global token collector."""
        collector1 = get_token_collector()
        collector2 = get_token_collector()

        assert collector1 is collector2
        assert isinstance(collector1, TokenUsageCollector)

    def test_get_memory_collector(self):
        """Test getting global memory collector."""
        collector1 = get_memory_collector()
        collector2 = get_memory_collector()

        assert collector1 is collector2
        assert isinstance(collector1, MemoryCollector)

    def test_get_api_collector(self):
        """Test getting global API collector."""
        collector1 = get_api_collector()
        collector2 = get_api_collector()

        assert collector1 is collector2
        assert isinstance(collector1, APICallCollector)
