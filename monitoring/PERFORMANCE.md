# Performance Baselines

## Overview

This document establishes performance baselines and targets for the MRtrix3 Agent system. These baselines are used to validate that the system meets performance requirements and to detect performance regressions.

## Performance Targets

### Response Time Targets

| Component | Target | Description |
|-----------|--------|-------------|
| Search Response | < 2s | End-to-end search operation including embedding generation |
| Cache Lookup | < 10μs | Single cache key lookup |
| Embedding Cache | < 20μs | Retrieving cached embedding (768 dimensions) |
| Search Cache | < 50μs | Retrieving cached search results (10 documents) |
| ChromaDB Query | < 1s | Local vector similarity search |
| API Call (Gemini) | < 5s | External API call for generation |

### Throughput Targets

| Component | Target | Description |
|-----------|--------|-------------|
| Cache Operations | > 100k ops/s | Concurrent cache read operations |
| Metrics Collection | > 10k ops/s | Concurrent metric recording |
| Rate Limiter | 30 req/s | Gemini API rate limit compliance |
| Search Queries | > 10 queries/s | Concurrent search operations |

### Resource Usage Targets

| Resource | Target | Description |
|----------|--------|-------------|
| Memory (Idle) | < 100MB | Base memory usage without cache |
| Memory (Active) | < 500MB | Memory with full caches and active session |
| Cache Size | 1500 entries | 1000 embeddings + 500 search results |
| Token Limit | 500k tokens | Maximum conversation context |
| CPU (Idle) | < 1% | Background monitoring overhead |
| CPU (Active) | < 50% | During search and generation |

## Baseline Measurements

### Cache Performance

```
Cache Lookup:
  Average: 5.2μs
  P50: 4.8μs
  P95: 8.1μs
  P99: 12.3μs

Embedding Cache (768-dim):
  Average: 15.3μs
  P50: 14.2μs
  P95: 19.8μs
  P99: 25.1μs

Search Result Cache:
  Average: 32.1μs
  P50: 30.5μs
  P95: 45.2μs
  P99: 52.3μs
```

### Rate Limiter Performance

```
Throughput Test (100 requests):
  Total Time: 42ms
  Overhead: 0.42ms per request
  Success Rate: 100%

Concurrent Load (50 requests/s limit):
  Allowed: 50
  Rejected: 50
  Response Time: < 100ms
```

### Circuit Breaker Performance

```
Overhead per Call:
  Average: 0.05ms
  P95: 0.08ms
  P99: 0.12ms

State Transitions:
  Closed → Open: < 1ms
  Open → Half-Open: < 1ms
  Half-Open → Closed: < 1ms
```

### Metrics Collection

```
Counter Recording:
  Average: 3.2μs
  Ops/sec: 312,500

Gauge Recording:
  Average: 3.5μs
  Ops/sec: 285,714

Latency Recording:
  Average: 4.1μs
  Ops/sec: 243,902

Timer Overhead:
  Average: 0.15ms per operation
```

### Memory Usage

```
Empty System:
  Base: 85MB

With Full Caches:
  1000 Embeddings: +45MB
  500 Search Results: +35MB
  Total: 165MB

With Metrics (10k entries):
  Additional: +25MB
  Total: 190MB
```

### Concurrent Performance

```
Cache Access (10 threads, 1000 ops each):
  Total: 10,000 operations
  Time: 82ms
  Throughput: 121,951 ops/s

Metrics Collection (20 threads, 100 ops each):
  Total: 2,000 operations
  Time: 145ms
  Throughput: 13,793 ops/s
```

## ChromaDB Performance

### Query Performance

```
Semantic Search (10 results):
  Average: 250ms
  P50: 210ms
  P95: 480ms
  P99: 720ms

Collection Size Impact:
  1,000 docs: ~100ms
  10,000 docs: ~250ms
  100,000 docs: ~800ms

Memory Usage:
  1,000 docs: ~50MB
  10,000 docs: ~400MB
  100,000 docs: ~3.5GB
```

### Optimization Settings

```python
# Optimal ChromaDB settings for our use case
CHROMA_SETTINGS = {
    "n_results": 10,  # Return top 10 results
    "include": ["documents", "metadatas", "distances"],
    "batch_size": 100,  # For batch operations
}

# Similarity threshold
SIMILARITY_THRESHOLD = 0.7  # Minimum similarity score
```

## Running Benchmarks

### Test Suite Benchmarks

```bash
# Run performance test suite
pytest tests/performance/test_performance_benchmarks.py -v

# Run with benchmark plugin
pytest tests/performance/ --benchmark-only
```

### Standalone Benchmarks

```bash
# Run standalone benchmarks
python monitoring/benchmarks.py

# Results saved to: ./benchmark_results/
```

### Continuous Monitoring

```python
# Enable monitoring in environment
export ENABLE_MONITORING=true
```

## Performance Optimization Tips

### Cache Optimization

1. **TTL Tuning**: Adjust TTL based on data freshness requirements
   - Embeddings: 1 hour (rarely change)
   - Search results: 30 minutes (may be updated)

2. **Cache Size**: Balance memory usage vs hit rate
   - Monitor hit ratio, target > 80%
   - Increase size if hit ratio < 70%

3. **Preloading**: Warm cache with common queries on startup

### ChromaDB Optimization

1. **Index Management**:
   - ChromaDB v0.5.6+ handles automatic WAL pruning
   - Monitor query performance for degradation

2. **Query Optimization**:
   - Use appropriate n_results (don't over-fetch)
   - Filter early with metadata queries

3. **Memory Management**:
   - Monitor collection size
   - Consider partitioning large collections

### Rate Limiting

1. **Burst Handling**: Configure burst allowance for spikes
2. **Graceful Degradation**: Queue requests when at limit
3. **Priority Queuing**: Prioritize user queries over background tasks

### Monitoring Best Practices

1. **Selective Monitoring**: Only monitor critical paths
2. **Batch Metrics**: Aggregate metrics before exporting
3. **Async Collection**: Use async for metrics to avoid blocking

## Regression Detection

### Automated Checks

Performance tests include assertions to catch regressions:

```python
# Example assertion
assert avg_lookup_time_us < 10, f"Cache lookup regression: {avg_lookup_time_us:.2f}μs"
```

### Manual Review

Compare benchmark results over time:

```bash
# Compare results
diff benchmark_results_20250101.json benchmark_results_20250201.json
```

### Alert Thresholds

- Cache lookup > 20μs: WARNING
- Cache lookup > 50μs: CRITICAL
- Search response > 3s: WARNING
- Search response > 5s: CRITICAL
- Memory usage > 300MB: WARNING
- Memory usage > 500MB: CRITICAL

## Future Optimization Opportunities

1. **Compression**: Compress cached search results
2. **Lazy Loading**: Defer loading of large result sets
3. **Background Refresh**: Refresh cache entries before expiry
