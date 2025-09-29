# Monitoring Architecture

## Overview

The monitoring module provides developer-focused observability tools for the MRtrix3 Agent, separate from user-facing functionality. This separation ensures clean architecture where monitoring can be enabled/disabled without affecting core agent operations.

## Components

### Structured Logging (`logging_config.py`)
- Centralized logging configuration for developer use
- Structured format with JSON output option
- Request ID correlation across log entries
- Separate from user-facing logs in `src/agent/`

### Metrics Collection (`metrics.py`)
- Response time tracking for tool invocations
- Token usage monitoring (prompt and completion)
- Search accuracy metrics
- Cache hit/miss ratios
- Export functionality for analysis

### Metric Collectors (`collectors.py`)
- Specialized collectors for different metric types
- ChromaDB performance monitoring
- API call tracking
- Memory usage profiling

### Performance Benchmarks (`benchmarks.py`)
- Standalone benchmark execution
- Search latency testing
- Concurrent request handling
- Token processing rate measurement
- Memory usage profiling

## Usage

### Enabling Monitoring

Set environment variable:
```bash
export COLLECT_LOGS=true
```
### Running Benchmarks

```bash
python monitoring/benchmarks.py
```

## Performance Baselines

See [PERFORMANCE.md](./PERFORMANCE.md) for established performance baselines and targets.

## Integration Points

### Agent Module Hooks
- Tool invocation start/end
- Search operations
- Cache operations
- Error occurrences

### ChromaDB Monitoring
- Query latency tracking
- Collection size monitoring
- Memory usage profiling
- Slow query logging (>1 second)

### Caching Metrics
- Cache hit/miss ratios
- TTL expiration events
- Memory usage
- Eviction statistics

## Data Flow

```
User Request
    ↓
Agent Module (src/agent/)
    ├→ User-facing logs (simple)
    └→ Monitoring hooks (if enabled)
        ├→ Structured logs (monitoring/logging_config.py)
        ├→ Metrics collection (monitoring/metrics.py)
        └→ Performance tracking (monitoring/collectors.py)
```

## File Structure

```
monitoring/
├── __init__.py           # Module exports
├── README.md            # This file
├── logging_config.py    # Structured logging configuration
├── metrics.py           # Metrics collection system
├── collectors.py        # Specialized metric collectors
├── benchmarks.py        # Performance benchmarks
└── PERFORMANCE.md       # Baseline performance documentation
```
