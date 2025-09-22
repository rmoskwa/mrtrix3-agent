"""
Tests for the caching layer.
"""

import time
from unittest.mock import patch


from src.agent.cache import (
    CacheEntry,
    CacheManager,
    EmbeddingCache,
    SearchResultCache,
    TTLCache,
    get_cache_manager,
    reset_cache_manager,
)


class TestCacheEntry:
    """Tests for CacheEntry."""

    def test_cache_entry_creation(self):
        """Test creating a cache entry."""
        entry = CacheEntry(
            key="test_key",
            value="test_value",
            ttl_seconds=60,
        )

        assert entry.key == "test_key"
        assert entry.value == "test_value"
        assert entry.ttl_seconds == 60
        assert entry.access_count == 0

    def test_is_expired(self):
        """Test checking if entry is expired."""
        # Create entry with very short TTL
        entry = CacheEntry(
            key="test",
            value="value",
            ttl_seconds=0.01,
        )

        # Should not be expired immediately
        assert not entry.is_expired()

        # Wait for expiration
        time.sleep(0.02)
        assert entry.is_expired()

    def test_access(self):
        """Test accessing cache entry."""
        entry = CacheEntry(
            key="test",
            value="test_value",
        )

        # First access
        value = entry.access()
        assert value == "test_value"
        assert entry.access_count == 1

        # Second access
        value = entry.access()
        assert value == "test_value"
        assert entry.access_count == 2
        assert entry.last_accessed > entry.timestamp


class TestTTLCache:
    """Tests for TTLCache."""

    def test_cache_initialization(self):
        """Test cache initialization."""
        cache = TTLCache(max_size=10, default_ttl=60)

        assert cache.max_size == 10
        assert cache.default_ttl == 60
        assert len(cache.cache) == 0

    def test_generate_key(self):
        """Test key generation."""
        cache = TTLCache()

        # Same arguments should generate same key
        key1 = cache._generate_key("test", 123, param="value")
        key2 = cache._generate_key("test", 123, param="value")
        assert key1 == key2

        # Different arguments should generate different keys
        key3 = cache._generate_key("test", 456, param="value")
        assert key1 != key3

    def test_set_and_get(self):
        """Test setting and getting cache values."""
        cache = TTLCache()

        # Set value
        cache.set("key1", "value1")
        assert len(cache.cache) == 1

        # Get existing value
        value = cache.get("key1")
        assert value == "value1"
        assert cache.hit_count == 1

        # Get non-existing value
        value = cache.get("key2", default="default")
        assert value == "default"
        assert cache.miss_count == 1

    def test_ttl_expiration(self):
        """Test TTL expiration."""
        cache = TTLCache()

        # Set value with short TTL
        cache.set("key1", "value1", ttl=0.01)

        # Should exist immediately
        assert cache.get("key1") == "value1"

        # Wait for expiration
        time.sleep(0.02)

        # Should be expired
        assert cache.get("key1") is None
        assert "key1" not in cache.cache

    def test_max_size_eviction(self):
        """Test eviction when max size is reached."""
        cache = TTLCache(max_size=3)

        # Fill cache
        cache.set("key1", "value1")
        cache.set("key2", "value2")
        cache.set("key3", "value3")

        assert len(cache.cache) == 3

        # Access key2 to update its last_accessed time
        cache.get("key2")
        time.sleep(0.01)

        # Adding another should evict oldest (key1 or key3)
        cache.set("key4", "value4")
        assert len(cache.cache) == 3
        assert cache.eviction_count == 1

        # key2 should still exist (was accessed recently)
        assert cache.get("key2") == "value2"

    def test_clear(self):
        """Test clearing the cache."""
        cache = TTLCache()

        cache.set("key1", "value1")
        cache.set("key2", "value2")

        assert len(cache.cache) == 2

        cache.clear()
        assert len(cache.cache) == 0

    def test_cleanup_expired(self):
        """Test cleaning up expired entries."""
        cache = TTLCache()

        # Set entries with different TTLs
        cache.set("key1", "value1", ttl=0.01)
        cache.set("key2", "value2", ttl=60)
        cache.set("key3", "value3", ttl=0.01)

        # Wait for some to expire
        time.sleep(0.02)

        # Cleanup expired
        removed = cache.cleanup_expired()
        assert removed == 2
        assert len(cache.cache) == 1
        assert cache.get("key2") == "value2"

    def test_get_stats(self):
        """Test getting cache statistics."""
        cache = TTLCache(max_size=10)

        # Perform some operations
        cache.set("key1", "value1")
        cache.get("key1")  # Hit
        cache.get("key2")  # Miss

        stats = cache.get_stats()

        assert stats["size"] == 1
        assert stats["max_size"] == 10
        assert stats["hit_count"] == 1
        assert stats["miss_count"] == 1
        assert stats["hit_ratio"] == 0.5
        assert stats["total_requests"] == 2

    @patch("src.agent.cache.record_metric")
    def test_metrics_recording(self, mock_record_metric):
        """Test that metrics are recorded when enabled."""
        cache = TTLCache(enable_metrics=True)

        cache.set("key1", "value1")
        cache.get("key1")  # Hit
        cache.get("key2")  # Miss

        # Check metrics were recorded
        mock_record_metric.assert_any_call("counter", "cache_hit", 1)
        mock_record_metric.assert_any_call("counter", "cache_miss", 1)


class TestEmbeddingCache:
    """Tests for EmbeddingCache."""

    def test_cache_embedding(self):
        """Test caching embeddings."""
        cache = EmbeddingCache()

        embedding = [0.1, 0.2, 0.3]
        cache.cache_embedding(
            text="test query",
            embedding=embedding,
            model="test_model",
        )

        # Retrieve cached embedding
        cached = cache.get_embedding("test query", "test_model")
        assert cached == embedding

        # Different model should not match
        cached = cache.get_embedding("test query", "other_model")
        assert cached is None

    def test_embedding_cache_key_generation(self):
        """Test that embedding cache keys are consistent."""
        cache = EmbeddingCache()

        embedding = [0.1, 0.2]
        cache.cache_embedding("test", embedding, "model1")

        # Same text and model should retrieve
        assert cache.get_embedding("test", "model1") == embedding

        # Different text should not retrieve
        assert cache.get_embedding("other", "model1") is None

        # Different model should not retrieve
        assert cache.get_embedding("test", "model2") is None


class TestSearchResultCache:
    """Tests for SearchResultCache."""

    def test_cache_search_results(self):
        """Test caching search results."""
        cache = SearchResultCache()

        results = [
            {"id": 1, "content": "result1"},
            {"id": 2, "content": "result2"},
        ]

        cache.cache_search_results(
            query="test query",
            results=results,
            search_type="semantic",
        )

        # Retrieve cached results
        cached = cache.get_search_results(
            query="test query",
            search_type="semantic",
        )
        assert cached == results

    def test_search_cache_with_filters(self):
        """Test search cache with filters."""
        cache = SearchResultCache()

        results = [{"id": 1, "content": "filtered"}]
        filters = {"category": "commands", "version": "3.0"}

        cache.cache_search_results(
            query="test",
            results=results,
            search_type="hybrid",
            filters=filters,
        )

        # Same filters should retrieve
        cached = cache.get_search_results(
            query="test",
            search_type="hybrid",
            filters=filters,
        )
        assert cached == results

        # Different filters should not retrieve
        cached = cache.get_search_results(
            query="test",
            search_type="hybrid",
            filters={"category": "guides"},
        )
        assert cached is None

        # No filters should not retrieve
        cached = cache.get_search_results(
            query="test",
            search_type="hybrid",
        )
        assert cached is None


class TestCacheManager:
    """Tests for CacheManager."""

    def test_cache_manager_initialization(self):
        """Test cache manager initialization."""
        manager = CacheManager(
            embedding_cache_size=100,
            search_cache_size=50,
            default_ttl=1800,
        )

        assert manager.embedding_cache.max_size == 100
        assert manager.search_cache.max_size == 50
        assert manager.embedding_cache.default_ttl == 1800

    def test_cache_embedding_through_manager(self):
        """Test caching embeddings through manager."""
        manager = CacheManager()

        embedding = [0.1, 0.2, 0.3]
        manager.cache_embedding("test", embedding, "model1")

        cached = manager.get_embedding("test", "model1")
        assert cached == embedding

    def test_cache_search_results_through_manager(self):
        """Test caching search results through manager."""
        manager = CacheManager()

        results = [{"id": 1, "content": "test"}]
        manager.cache_search_results(
            query="test",
            results=results,
            search_type="semantic",
        )

        cached = manager.get_search_results("test", "semantic")
        assert cached == results

    def test_clear_all(self):
        """Test clearing all caches."""
        manager = CacheManager()

        # Add data to both caches
        manager.cache_embedding("test", [0.1], "model")
        manager.cache_search_results("query", [{"id": 1}], "semantic")

        # Clear all
        manager.clear_all()

        # Both caches should be empty
        assert manager.get_embedding("test", "model") is None
        assert manager.get_search_results("query", "semantic") is None

    def test_cleanup_expired(self):
        """Test cleaning up expired entries."""
        manager = CacheManager()

        # Set entries with short TTL
        manager.embedding_cache.set("key1", [0.1], ttl=0.01)
        manager.search_cache.set("key2", [{"id": 1}], ttl=0.01)

        # Wait for expiration
        time.sleep(0.02)

        # Cleanup
        counts = manager.cleanup_expired()
        assert counts["embeddings"] == 1
        assert counts["search_results"] == 1

    def test_get_stats(self):
        """Test getting statistics."""
        manager = CacheManager()

        # Perform some operations
        manager.cache_embedding("test", [0.1], "model")
        manager.get_embedding("test", "model")  # Hit
        manager.get_embedding("other", "model")  # Miss

        stats = manager.get_stats()

        assert "embeddings" in stats
        assert "search_results" in stats
        assert stats["embeddings"]["hit_count"] == 1
        assert stats["embeddings"]["miss_count"] == 1


class TestGlobalCacheManager:
    """Tests for global cache manager."""

    def test_get_cache_manager(self):
        """Test getting global cache manager."""
        manager1 = get_cache_manager()
        manager2 = get_cache_manager()

        assert manager1 is manager2
        assert isinstance(manager1, CacheManager)

    def test_reset_cache_manager(self):
        """Test resetting global cache manager."""
        # Get manager and add data
        manager = get_cache_manager()
        manager.cache_embedding("test", [0.1], "model")

        # Reset
        reset_cache_manager()

        # Get new manager - should be different instance
        new_manager = get_cache_manager()
        assert manager is not new_manager

        # Should be empty
        assert new_manager.get_embedding("test", "model") is None
