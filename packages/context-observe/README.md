# context-observe

OpenTelemetry-native observability for ContextEngine.

## Features

- **ContextTracer**: Distributed tracing with OpenTelemetry
- **ContextMetrics**: Prometheus-compatible metrics
- **ContextEvents**: Structured logging with structlog
- **Decorators**: `@traced` and `@metered` for automatic instrumentation

## Installation

```bash
# Basic installation
pip install context-observe

# With Prometheus exporter
pip install context-observe[prometheus]

# With OTLP exporter
pip install context-observe[otlp]

# All exporters
pip install context-observe[all]
```

## Quick Start

```python
from context_observe import ContextTracer, ContextMetrics, ContextEvents

# Initialize components
tracer = ContextTracer(service_name="my-agent")
metrics = ContextMetrics()
events = ContextEvents()

# Use tracing
with tracer.trace_compression("dedupe", "session-123", original_tokens=1000):
    # ... compression logic
    pass

# Record metrics
metrics.record_compression("dedupe", original_tokens=1000, compressed_tokens=300, session_id="session-123")

# Log events
events.compression_completed("session-123", "dedupe", 1000, 300, duration_ms=50.5)
```

## Decorators

```python
from context_observe import traced, metered

@traced(operation_type="graph.query")
def query_nodes(filters):
    return db.query(filters)

@metered(operation="retrieval.semantic")
def search_memories(query):
    return index.search(query)
```

## Key Metrics

| Metric | Description |
|--------|-------------|
| `context.token_usage_ratio` | Current token usage (0.0-1.0) |
| `context.compression_ratio` | Compression ratio achieved |
| `context.cache_hits_total` | Cache hit count |
| `context.retrieval_latency_seconds` | Memory retrieval latency |
| `context.pre_rot_warnings_total` | Pre-rot warning events |

## License

MIT
