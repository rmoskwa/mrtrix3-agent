"""
Specialized metric collectors for different components.

This module provides specific collectors for various metrics like
ChromaDB performance, API calls, and memory usage.
"""

import asyncio
import os
import psutil
from dataclasses import dataclass
from typing import Dict, List, Optional, Any

from monitoring.metrics import MetricsCollector


@dataclass
class ChromaDBMetrics:
    """Metrics specific to ChromaDB operations."""

    query_count: int = 0
    total_query_time_ms: float = 0.0
    average_query_time_ms: float = 0.0
    slow_query_count: int = 0  # Queries > 1 second
    collection_size: int = 0
    memory_usage_mb: float = 0.0


@dataclass
class SearchMetrics:
    """Metrics for search operations."""

    total_searches: int = 0
    successful_searches: int = 0
    failed_searches: int = 0
    empty_result_searches: int = 0
    average_results_returned: float = 0.0
    average_search_time_ms: float = 0.0


@dataclass
class TokenMetrics:
    """Metrics for token usage."""

    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0
    total_tokens: int = 0
    average_prompt_tokens: float = 0.0
    average_completion_tokens: float = 0.0
    conversations_tracked: int = 0


class ChromaDBCollector:
    """Collects metrics for ChromaDB operations."""

    def __init__(self, metrics_collector: Optional[MetricsCollector] = None):
        """
        Initialize the ChromaDB collector.

        Args:
            metrics_collector: Optional metrics collector instance
        """
        self.metrics = metrics_collector or MetricsCollector()
        self.chroma_metrics = ChromaDBMetrics()
        self.query_times: List[float] = []

    def record_query(
        self,
        duration_seconds: float,
        result_count: int = 0,
        collection_name: Optional[str] = None,
    ) -> None:
        """
        Record a ChromaDB query.

        Args:
            duration_seconds: Query duration in seconds
            result_count: Number of results returned
            collection_name: Optional collection name
        """
        duration_ms = duration_seconds * 1000
        self.query_times.append(duration_ms)
        self.chroma_metrics.query_count += 1
        self.chroma_metrics.total_query_time_ms += duration_ms

        # Update average
        if self.chroma_metrics.query_count > 0:
            self.chroma_metrics.average_query_time_ms = (
                self.chroma_metrics.total_query_time_ms
                / self.chroma_metrics.query_count
            )

        # Check for slow query
        if duration_ms > 1000:
            self.chroma_metrics.slow_query_count += 1
            self.metrics.record_counter("chromadb_slow_queries")

        # Record in global metrics
        self.metrics.record_latency("chromadb_query", duration_seconds)
        self.metrics.record_gauge(
            "chromadb_result_count",
            result_count,
            collection=collection_name or "default",
        )

    def record_collection_size(
        self, size: int, collection_name: Optional[str] = None
    ) -> None:
        """
        Record ChromaDB collection size.

        Args:
            size: Collection size (number of documents)
            collection_name: Optional collection name
        """
        self.chroma_metrics.collection_size = size
        self.metrics.record_gauge(
            "chromadb_collection_size", size, collection=collection_name or "default"
        )

    def record_memory_usage(self) -> None:
        """Record current memory usage for ChromaDB."""
        process = psutil.Process(os.getpid())
        memory_mb = process.memory_info().rss / 1024 / 1024
        self.chroma_metrics.memory_usage_mb = memory_mb
        self.metrics.record_gauge("chromadb_memory_mb", memory_mb)

    def get_metrics(self) -> ChromaDBMetrics:
        """
        Get collected ChromaDB metrics.

        Returns:
            ChromaDB metrics summary
        """
        return self.chroma_metrics


class SearchCollector:
    """Collects metrics for search operations."""

    def __init__(self, metrics_collector: Optional[MetricsCollector] = None):
        """
        Initialize the search collector.

        Args:
            metrics_collector: Optional metrics collector instance
        """
        self.metrics = metrics_collector or MetricsCollector()
        self.search_metrics = SearchMetrics()
        self.search_times: List[float] = []
        self.results_counts: List[int] = []

    def record_search(
        self,
        duration_seconds: float,
        result_count: int,
        success: bool,
        search_type: Optional[str] = None,
    ) -> None:
        """
        Record a search operation.

        Args:
            duration_seconds: Search duration in seconds
            result_count: Number of results returned
            success: Whether search was successful
            search_type: Optional search type (semantic, bm25, hybrid)
        """
        duration_ms = duration_seconds * 1000
        self.search_times.append(duration_ms)
        self.results_counts.append(result_count)

        self.search_metrics.total_searches += 1

        if success:
            self.search_metrics.successful_searches += 1
            self.metrics.record_counter("search_success")
            if result_count == 0:
                self.search_metrics.empty_result_searches += 1
        else:
            self.search_metrics.failed_searches += 1
            self.metrics.record_counter("search_failure")

        # Update averages
        if self.search_metrics.total_searches > 0:
            self.search_metrics.average_search_time_ms = sum(self.search_times) / len(
                self.search_times
            )
            if self.results_counts:
                self.search_metrics.average_results_returned = sum(
                    self.results_counts
                ) / len(self.results_counts)

        # Record in global metrics
        self.metrics.record_latency(
            f"search_{search_type or 'default'}", duration_seconds
        )
        self.metrics.record_gauge("search_result_count", result_count)

    def get_metrics(self) -> SearchMetrics:
        """
        Get collected search metrics.

        Returns:
            Search metrics summary
        """
        return self.search_metrics


class TokenUsageCollector:
    """Collects metrics for token usage."""

    def __init__(self, metrics_collector: Optional[MetricsCollector] = None):
        """
        Initialize the token usage collector.

        Args:
            metrics_collector: Optional metrics collector instance
        """
        self.metrics = metrics_collector or MetricsCollector()
        self.token_metrics = TokenMetrics()
        self.prompt_tokens_list: List[int] = []
        self.completion_tokens_list: List[int] = []

    def record_token_usage(
        self,
        prompt_tokens: int,
        completion_tokens: int,
        model: Optional[str] = None,
    ) -> None:
        """
        Record token usage for an API call.

        Args:
            prompt_tokens: Number of prompt tokens
            completion_tokens: Number of completion tokens
            model: Optional model name
        """
        self.prompt_tokens_list.append(prompt_tokens)
        self.completion_tokens_list.append(completion_tokens)

        self.token_metrics.total_prompt_tokens += prompt_tokens
        self.token_metrics.total_completion_tokens += completion_tokens
        self.token_metrics.total_tokens += prompt_tokens + completion_tokens

        # Update averages
        if self.prompt_tokens_list:
            self.token_metrics.average_prompt_tokens = sum(
                self.prompt_tokens_list
            ) / len(self.prompt_tokens_list)
        if self.completion_tokens_list:
            self.token_metrics.average_completion_tokens = sum(
                self.completion_tokens_list
            ) / len(self.completion_tokens_list)

        # Record in global metrics
        self.metrics.record_counter("tokens_used", prompt_tokens + completion_tokens)
        self.metrics.record_gauge(
            "prompt_tokens", prompt_tokens, model=model or "unknown"
        )
        self.metrics.record_gauge(
            "completion_tokens", completion_tokens, model=model or "unknown"
        )

    def record_conversation(self) -> None:
        """Record a new conversation."""
        self.token_metrics.conversations_tracked += 1
        self.metrics.record_counter("conversations")

    def get_metrics(self) -> TokenMetrics:
        """
        Get collected token metrics.

        Returns:
            Token metrics summary
        """
        return self.token_metrics


class MemoryCollector:
    """Collects memory usage metrics."""

    def __init__(self, metrics_collector: Optional[MetricsCollector] = None):
        """
        Initialize the memory collector.

        Args:
            metrics_collector: Optional metrics collector instance
        """
        self.metrics = metrics_collector or MetricsCollector()
        self.process = psutil.Process(os.getpid())

    def collect_memory_metrics(self) -> Dict[str, float]:
        """
        Collect current memory metrics.

        Returns:
            Dictionary of memory metrics
        """
        memory_info = self.process.memory_info()
        memory_metrics = {
            "rss_mb": memory_info.rss / 1024 / 1024,
            "vms_mb": memory_info.vms / 1024 / 1024,
            "percent": self.process.memory_percent(),
        }

        # Record in global metrics
        for key, value in memory_metrics.items():
            self.metrics.record_gauge(f"memory_{key}", value)

        return memory_metrics

    async def monitor_memory(self, interval_seconds: int = 60) -> None:
        """
        Monitor memory usage periodically.

        Args:
            interval_seconds: Monitoring interval in seconds
        """
        while True:
            self.collect_memory_metrics()
            await asyncio.sleep(interval_seconds)


class APICallCollector:
    """Collects metrics for external API calls."""

    def __init__(self, metrics_collector: Optional[MetricsCollector] = None):
        """
        Initialize the API call collector.

        Args:
            metrics_collector: Optional metrics collector instance
        """
        self.metrics = metrics_collector or MetricsCollector()
        self.api_calls: Dict[str, List[float]] = {}
        self.api_errors: Dict[str, int] = {}

    def record_api_call(
        self,
        api_name: str,
        endpoint: str,
        duration_seconds: float,
        status_code: Optional[int] = None,
        error: Optional[str] = None,
    ) -> None:
        """
        Record an API call.

        Args:
            api_name: Name of the API (e.g., "gemini", "supabase")
            endpoint: API endpoint
            duration_seconds: Call duration in seconds
            status_code: Optional HTTP status code
            error: Optional error message
        """
        # Track latencies by API
        if api_name not in self.api_calls:
            self.api_calls[api_name] = []
        self.api_calls[api_name].append(duration_seconds * 1000)

        # Track errors
        if error:
            if api_name not in self.api_errors:
                self.api_errors[api_name] = 0
            self.api_errors[api_name] += 1
            self.metrics.record_counter(f"api_{api_name}_errors")

        # Record in global metrics
        self.metrics.record_latency(
            f"api_call_{api_name}",
            duration_seconds,
            endpoint=endpoint,
            status=str(status_code) if status_code else "unknown",
        )

        if status_code:
            self.metrics.record_gauge(f"api_{api_name}_status", status_code)

    def get_api_stats(self) -> Dict[str, Any]:
        """
        Get API call statistics.

        Returns:
            Dictionary of API statistics
        """
        stats = {}
        for api_name, latencies in self.api_calls.items():
            if latencies:
                stats[api_name] = {
                    "call_count": len(latencies),
                    "average_latency_ms": sum(latencies) / len(latencies),
                    "min_latency_ms": min(latencies),
                    "max_latency_ms": max(latencies),
                    "error_count": self.api_errors.get(api_name, 0),
                }
        return stats


# Global collector instances (created on demand)
_chromadb_collector: Optional[ChromaDBCollector] = None
_search_collector: Optional[SearchCollector] = None
_token_collector: Optional[TokenUsageCollector] = None
_memory_collector: Optional[MemoryCollector] = None
_api_collector: Optional[APICallCollector] = None


def get_chromadb_collector() -> ChromaDBCollector:
    """Get global ChromaDB collector instance."""
    global _chromadb_collector
    if _chromadb_collector is None:
        _chromadb_collector = ChromaDBCollector()
    return _chromadb_collector


def get_search_collector() -> SearchCollector:
    """Get global search collector instance."""
    global _search_collector
    if _search_collector is None:
        _search_collector = SearchCollector()
    return _search_collector


def get_token_collector() -> TokenUsageCollector:
    """Get global token usage collector instance."""
    global _token_collector
    if _token_collector is None:
        _token_collector = TokenUsageCollector()
    return _token_collector


def get_memory_collector() -> MemoryCollector:
    """Get global memory collector instance."""
    global _memory_collector
    if _memory_collector is None:
        _memory_collector = MemoryCollector()
    return _memory_collector


def get_api_collector() -> APICallCollector:
    """Get global API call collector instance."""
    global _api_collector
    if _api_collector is None:
        _api_collector = APICallCollector()
    return _api_collector
