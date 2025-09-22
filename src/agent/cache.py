"""
TTL-based caching layer for search results and embeddings.

This module provides caching functionality to improve performance
by avoiding redundant API calls and database queries.
"""

import hashlib
import json
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from src.agent.monitoring_integration import record_metric


@dataclass
class CacheEntry:
    """Represents a single cache entry with TTL."""

    key: str
    value: Any
    timestamp: float = field(default_factory=time.time)
    ttl_seconds: float = 3600  # Default 1 hour
    access_count: int = 0
    last_accessed: float = field(default_factory=time.time)

    def is_expired(self) -> bool:
        """Check if the cache entry has expired."""
        return time.time() - self.timestamp > self.ttl_seconds

    def access(self) -> Any:
        """Access the cache entry and update access stats."""
        self.access_count += 1
        self.last_accessed = time.time()
        return self.value


class TTLCache:
    """Time-to-live based cache implementation."""

    def __init__(
        self,
        max_size: int = 1000,
        default_ttl: float = 3600,
        enable_metrics: bool = True,
    ):
        """
        Initialize the TTL cache.

        Args:
            max_size: Maximum number of entries to store
            default_ttl: Default time-to-live in seconds
            enable_metrics: Whether to record cache metrics
        """
        self.max_size = max_size
        self.default_ttl = default_ttl
        self.enable_metrics = enable_metrics
        self.cache: Dict[str, CacheEntry] = {}
        self.hit_count = 0
        self.miss_count = 0
        self.eviction_count = 0

    def _generate_key(self, *args, **kwargs) -> str:
        """
        Generate a cache key from arguments.

        Args:
            *args: Positional arguments
            **kwargs: Keyword arguments

        Returns:
            Cache key string
        """
        # Create a string representation of all arguments
        key_parts = [str(arg) for arg in args]
        key_parts.extend(f"{k}={v}" for k, v in sorted(kwargs.items()))
        key_string = "|".join(key_parts)

        # Hash for consistent key length
        return hashlib.md5(key_string.encode()).hexdigest()

    def get(
        self,
        key: str,
        default: Optional[Any] = None,
    ) -> Optional[Any]:
        """
        Get a value from the cache.

        Args:
            key: Cache key
            default: Default value if not found or expired

        Returns:
            Cached value or default
        """
        if key in self.cache:
            entry = self.cache[key]

            if not entry.is_expired():
                self.hit_count += 1
                if self.enable_metrics:
                    record_metric("counter", "cache_hit", 1)
                return entry.access()

            # Remove expired entry
            del self.cache[key]

        self.miss_count += 1
        if self.enable_metrics:
            record_metric("counter", "cache_miss", 1)
        return default

    def set(
        self,
        key: str,
        value: Any,
        ttl: Optional[float] = None,
    ) -> None:
        """
        Set a value in the cache.

        Args:
            key: Cache key
            value: Value to cache
            ttl: Optional TTL override in seconds
        """
        # Check size limit
        if len(self.cache) >= self.max_size:
            self._evict_oldest()

        # Create cache entry
        entry = CacheEntry(
            key=key,
            value=value,
            ttl_seconds=ttl or self.default_ttl,
        )
        self.cache[key] = entry

    def _evict_oldest(self) -> None:
        """Evict the oldest cache entry."""
        if not self.cache:
            return

        # Find oldest entry by last access time
        oldest_key = min(
            self.cache.keys(),
            key=lambda k: self.cache[k].last_accessed,
        )

        del self.cache[oldest_key]
        self.eviction_count += 1
        if self.enable_metrics:
            record_metric("counter", "cache_eviction", 1)

    def clear(self) -> None:
        """Clear all cache entries."""
        self.cache.clear()

    def cleanup_expired(self) -> int:
        """
        Remove all expired entries.

        Returns:
            Number of entries removed
        """
        expired_keys = [key for key, entry in self.cache.items() if entry.is_expired()]

        for key in expired_keys:
            del self.cache[key]

        return len(expired_keys)

    def get_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.

        Returns:
            Dictionary of cache statistics
        """
        total_requests = self.hit_count + self.miss_count
        hit_ratio = self.hit_count / total_requests if total_requests > 0 else 0

        return {
            "size": len(self.cache),
            "max_size": self.max_size,
            "hit_count": self.hit_count,
            "miss_count": self.miss_count,
            "hit_ratio": hit_ratio,
            "eviction_count": self.eviction_count,
            "total_requests": total_requests,
        }


class EmbeddingCache(TTLCache):
    """Specialized cache for embeddings."""

    def cache_embedding(
        self,
        text: str,
        embedding: List[float],
        model: str,
        ttl: Optional[float] = None,
    ) -> None:
        """
        Cache an embedding.

        Args:
            text: Original text
            embedding: Embedding vector
            model: Model used to generate embedding
            ttl: Optional TTL override
        """
        key = self._generate_key(text, model=model)
        self.set(key, embedding, ttl)

    def get_embedding(
        self,
        text: str,
        model: str,
    ) -> Optional[List[float]]:
        """
        Get a cached embedding.

        Args:
            text: Original text
            model: Model name

        Returns:
            Cached embedding or None
        """
        key = self._generate_key(text, model=model)
        return self.get(key)


class SearchResultCache(TTLCache):
    """Specialized cache for search results."""

    def cache_search_results(
        self,
        query: str,
        results: List[Dict[str, Any]],
        search_type: str,
        filters: Optional[Dict[str, Any]] = None,
        ttl: Optional[float] = None,
    ) -> None:
        """
        Cache search results.

        Args:
            query: Search query
            results: Search results
            search_type: Type of search (semantic, bm25, hybrid)
            filters: Optional search filters
            ttl: Optional TTL override
        """
        key = self._generate_key(
            query,
            search_type=search_type,
            filters=json.dumps(filters, sort_keys=True) if filters else None,
        )
        self.set(key, results, ttl)

    def get_search_results(
        self,
        query: str,
        search_type: str,
        filters: Optional[Dict[str, Any]] = None,
    ) -> Optional[List[Dict[str, Any]]]:
        """
        Get cached search results.

        Args:
            query: Search query
            search_type: Type of search
            filters: Optional search filters

        Returns:
            Cached results or None
        """
        key = self._generate_key(
            query,
            search_type=search_type,
            filters=json.dumps(filters, sort_keys=True) if filters else None,
        )
        return self.get(key)


class CacheManager:
    """Manages multiple cache types."""

    def __init__(
        self,
        embedding_cache_size: int = 500,
        search_cache_size: int = 200,
        default_ttl: float = 3600,
    ):
        """
        Initialize the cache manager.

        Args:
            embedding_cache_size: Max size for embedding cache
            search_cache_size: Max size for search cache
            default_ttl: Default TTL in seconds
        """
        self.embedding_cache = EmbeddingCache(
            max_size=embedding_cache_size,
            default_ttl=default_ttl,
        )
        self.search_cache = SearchResultCache(
            max_size=search_cache_size,
            default_ttl=default_ttl,
        )

    def cache_embedding(
        self,
        text: str,
        embedding: List[float],
        model: str,
    ) -> None:
        """Cache an embedding."""
        self.embedding_cache.cache_embedding(text, embedding, model)

    def get_embedding(
        self,
        text: str,
        model: str,
    ) -> Optional[List[float]]:
        """Get a cached embedding."""
        return self.embedding_cache.get_embedding(text, model)

    def cache_search_results(
        self,
        query: str,
        results: List[Dict[str, Any]],
        search_type: str,
        filters: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Cache search results."""
        self.search_cache.cache_search_results(query, results, search_type, filters)

    def get_search_results(
        self,
        query: str,
        search_type: str,
        filters: Optional[Dict[str, Any]] = None,
    ) -> Optional[List[Dict[str, Any]]]:
        """Get cached search results."""
        return self.search_cache.get_search_results(query, search_type, filters)

    def clear_all(self) -> None:
        """Clear all caches."""
        self.embedding_cache.clear()
        self.search_cache.clear()

    def cleanup_expired(self) -> Dict[str, int]:
        """
        Clean up expired entries in all caches.

        Returns:
            Dictionary with cleanup counts for each cache
        """
        return {
            "embeddings": self.embedding_cache.cleanup_expired(),
            "search_results": self.search_cache.cleanup_expired(),
        }

    def get_stats(self) -> Dict[str, Dict[str, Any]]:
        """
        Get statistics for all caches.

        Returns:
            Dictionary with stats for each cache
        """
        return {
            "embeddings": self.embedding_cache.get_stats(),
            "search_results": self.search_cache.get_stats(),
        }


# Global cache manager instance
_cache_manager: Optional[CacheManager] = None


def get_cache_manager() -> CacheManager:
    """
    Get the global cache manager instance.

    Returns:
        Global cache manager
    """
    global _cache_manager
    if _cache_manager is None:
        _cache_manager = CacheManager()
    return _cache_manager


def reset_cache_manager() -> None:
    """Reset the global cache manager."""
    global _cache_manager
    if _cache_manager is not None:
        _cache_manager.clear_all()
    _cache_manager = None
