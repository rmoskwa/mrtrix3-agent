# Monitoring Architecture

## Overview

The monitoring module provides developer-focused observability tools for the MRtrix3 Agent, separate from user-facing functionality. This separation ensures clean architecture where monitoring can be enabled/disabled without affecting core agent operations.

## Architecture Principles

1. **Separation of Concerns**: Monitoring is developer infrastructure, not user functionality
2. **Non-intrusive**: Agent modules can function without monitoring enabled
3. **Performance-focused**: Minimal overhead when enabled, zero overhead when disabled
4. **Practical Scope**: Solo developer monitoring needs, not enterprise observability

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

### Optional Dashboard (`dashboard.py`)
- Real-time metrics viewing
- CLI-based dashboard for development
- Resource usage monitoring

## Usage

### Enabling Monitoring

Set environment variable:
```bash
export ENABLE_MONITORING=true
```

### Importing in Agent Modules

Agent modules should import monitoring conditionally:

```python
import os

if os.getenv("ENABLE_MONITORING", "false").lower() == "true":
    from monitoring import get_logger, MetricsCollector
    logger = get_logger(__name__)
    metrics = MetricsCollector()
else:
    logger = None
    metrics = None

# Use monitoring if available
if logger:
    logger.info("Operation started", extra={"request_id": request_id})
if metrics:
    metrics.record_latency("tool_invocation", duration)
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

## Testing

Unit tests for monitoring modules are located in:
- `tests/monitoring/` - Unit tests for all monitoring components
- `tests/performance/` - Performance benchmarks and tests

## Environment Variables

- `ENABLE_MONITORING`: Enable/disable monitoring (default: false)
- `MONITORING_LOG_LEVEL`: Log level for monitoring (default: INFO)
- `MONITORING_LOG_FORMAT`: Log format (json|text, default: text)
- `METRICS_EXPORT_PATH`: Path for metrics export (default: ./metrics/)

## Best Practices

1. Always check if monitoring is enabled before using
2. Use structured logging with consistent field names
3. Include request IDs for correlation
4. Keep metrics lightweight and actionable
5. Profile performance impact regularly
6. Document any new metrics in this README

## Limitations

- Memory-based metrics storage (not persistent)
- Single-instance monitoring (no distributed tracing)
- CLI-focused tooling (no web UI)
- Local file export only (no external services)
