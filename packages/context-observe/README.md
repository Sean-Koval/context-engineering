# context-observe

> Observability package for ContextEngine: OpenTelemetry tracing, Prometheus metrics, and structured logging

## Installation

```bash
uv pip install -e packages/context-observe
```

## Overview

context-observe provides three pillars of observability:

```
┌─────────────────────────────────────────────────────────────────┐
│                     OBSERVABILITY STACK                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐  │
│  │  ContextTracer  │  │ ContextMetrics  │  │  ContextEvents  │  │
│  │                 │  │                 │  │                 │  │
│  │  OpenTelemetry  │  │   Prometheus    │  │    structlog    │  │
│  │    Tracing      │  │    Metrics      │  │    Logging      │  │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘  │
│          │                    │                    │             │
│          ▼                    ▼                    ▼             │
│     Jaeger/Zipkin        Prometheus/           JSON logs        │
│     Tempo/etc.           Grafana               ELK/etc.         │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

## Quick Start

```python
from context_observe import ContextTracer, ContextMetrics, ContextEvents, traced

# Initialize observability
tracer = ContextTracer(service_name="my-agent")
metrics = ContextMetrics(service_name="my-agent")
events = ContextEvents(service_name="my-agent")

# Use the @traced decorator
@traced(tracer)
def process_message(message: str) -> str:
    # Automatically creates a span
    return message.upper()

# Manual span creation
with tracer.span("compression") as span:
    span.set_attribute("strategy", "hierarchical")
    # ... do work ...

# Record metrics
metrics.record_compression(
    strategy="hierarchical",
    original_tokens=10000,
    compressed_tokens=2000,
    duration_seconds=0.5,
)

# Log events
events.compression_completed(
    strategy="hierarchical",
    tokens_saved=8000,
    compression_ratio=5.0,
)
```

## Components

### ContextTracer

OpenTelemetry-based distributed tracing.

```python
from context_observe import ContextTracer

tracer = ContextTracer(
    service_name="my-agent",
    endpoint="http://localhost:4317",  # OTLP endpoint
)

# Create spans
with tracer.span("operation_name") as span:
    span.set_attribute("key", "value")
    span.add_event("checkpoint", {"progress": 50})

    # Nested spans
    with tracer.span("sub_operation") as child:
        child.set_attribute("detail", "info")

# Convenience methods for common operations
with tracer.trace_graph_operation("add_node", node_type="MESSAGE"):
    graph.add_message(...)

with tracer.trace_compression("externalize", tokens=5000):
    strategy.compress(...)

with tracer.trace_retrieval("semantic_search", query="auth flow"):
    index.search(...)
```

#### Span Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `context.session_id` | string | Current session ID |
| `context.node_count` | int | Nodes in graph |
| `context.token_count` | int | Total tokens |
| `compression.strategy` | string | Strategy name |
| `compression.tier` | string | LOSSLESS/COMPACTION/SUMMARIZATION |
| `compression.tokens_saved` | int | Tokens compressed |
| `retrieval.query` | string | Search query |
| `retrieval.results` | int | Results returned |

### ContextMetrics

Prometheus-compatible metrics collection.

```python
from context_observe import ContextMetrics

metrics = ContextMetrics(
    service_name="my-agent",
    port=9090,  # Prometheus scrape port
)

# Record compression metrics
metrics.record_compression(
    strategy="schema_compression",
    original_tokens=10000,
    compressed_tokens=3000,
    duration_seconds=0.02,
)

# Record retrieval metrics
metrics.record_retrieval(
    method="semantic_search",
    results=5,
    duration_seconds=0.05,
)

# Record token budget
metrics.record_budget_usage(
    section="context",
    used=45000,
    total=100000,
)

# Record graph operations
metrics.record_graph_operation(
    operation="add_node",
    node_type="TOOL_RESULT",
    duration_seconds=0.001,
)

# Get snapshot of current metrics
snapshot = metrics.snapshot()
print(f"Total compressions: {snapshot.compression_count}")
print(f"Average latency: {snapshot.avg_compression_latency_ms}ms")
```

#### Available Metrics

| Metric | Type | Labels | Description |
|--------|------|--------|-------------|
| `context_compression_total` | Counter | strategy, tier | Total compressions |
| `context_compression_tokens_saved` | Counter | strategy | Tokens saved |
| `context_compression_ratio` | Histogram | strategy | Compression ratios |
| `context_compression_duration_seconds` | Histogram | strategy | Compression latency |
| `context_retrieval_total` | Counter | method | Total retrievals |
| `context_retrieval_duration_seconds` | Histogram | method | Retrieval latency |
| `context_budget_usage_ratio` | Gauge | section | Budget utilization |
| `context_graph_nodes_total` | Gauge | type | Nodes by type |
| `context_graph_operations_total` | Counter | operation | Graph operations |

### ContextEvents

Structured logging with structlog.

```python
from context_observe import ContextEvents

events = ContextEvents(
    service_name="my-agent",
    log_level="INFO",
    json_output=True,  # JSON format for log aggregators
)

# Session lifecycle
events.session_started(session_id="abc123")
events.session_ended(session_id="abc123", duration_seconds=300)

# Compression events
events.compression_started(strategy="hierarchical", target_tokens=50000)
events.compression_completed(
    strategy="hierarchical",
    tokens_saved=8000,
    compression_ratio=5.0,
    nodes_processed=25,
)

# Budget alerts
events.budget_warning(
    section="context",
    usage_ratio=0.65,
    threshold=0.65,
)
events.budget_critical(
    section="context",
    usage_ratio=0.85,
)

# Entity events
events.entity_extracted(
    entity_type="FILE",
    name="auth.py",
    importance=0.8,
)
events.entities_merged(
    entity_ids=["id1", "id2"],
    canonical_name="User",
)

# Errors
events.error(
    operation="compression",
    error_type="ValueError",
    message="Invalid strategy",
)
```

#### Log Output Format

```json
{
  "timestamp": "2026-01-12T15:30:00Z",
  "level": "info",
  "event": "compression_completed",
  "service": "my-agent",
  "session_id": "abc123",
  "strategy": "hierarchical",
  "tokens_saved": 8000,
  "compression_ratio": 5.0,
  "nodes_processed": 25
}
```

### @traced Decorator

Automatic tracing for functions and methods.

```python
from context_observe import traced, ContextTracer

tracer = ContextTracer(service_name="my-agent")

@traced(tracer)
def my_function(arg1: str, arg2: int) -> str:
    # Creates span with name "my_function"
    # Automatically records:
    # - Function arguments (optional)
    # - Return value (optional)
    # - Exceptions
    # - Duration
    return f"{arg1}: {arg2}"

@traced(tracer, name="custom_name", record_args=True, record_result=True)
def another_function(data: dict) -> list:
    return list(data.keys())

# Works with async functions too
@traced(tracer)
async def async_operation():
    await some_async_work()
```

## Integration Examples

### With ContextEngine Pipeline

```python
from context_core import ContextGraph
from context_compression import CompressionPipeline
from context_observe import ContextTracer, ContextMetrics, ContextEvents

# Initialize observability
tracer = ContextTracer(service_name="context-engine")
metrics = ContextMetrics(service_name="context-engine")
events = ContextEvents(service_name="context-engine")

# Create pipeline with tracing
pipeline = CompressionPipeline()

# Trace the entire compression flow
with tracer.span("compress_session") as span:
    span.set_attribute("session_id", graph.session_id)

    events.compression_started(strategy="pipeline", target_tokens=50000)

    results = pipeline.compress(graph)

    total_saved = sum(r.tokens_saved for r in results)
    span.set_attribute("tokens_saved", total_saved)

    for result in results:
        metrics.record_compression(
            strategy=result.strategy_name,
            original_tokens=result.original_tokens,
            compressed_tokens=result.compressed_tokens,
            duration_seconds=0.1,
        )

    events.compression_completed(
        strategy="pipeline",
        tokens_saved=total_saved,
        compression_ratio=results[-1].compression_ratio if results else 1.0,
    )
```

### Grafana Dashboard

Export metrics for Grafana visualization:

```yaml
# Prometheus scrape config (prometheus.yml)
scrape_configs:
  - job_name: 'context-engine'
    static_configs:
      - targets: ['localhost:9090']
```

Example Grafana panel queries:
- Compression rate: `rate(context_compression_total[5m])`
- Average compression ratio: `histogram_quantile(0.5, context_compression_ratio)`
- Token savings: `increase(context_compression_tokens_saved[1h])`
- Budget utilization: `context_budget_usage_ratio`

### Log Aggregation (ELK)

```python
# Configure JSON output for Logstash
events = ContextEvents(
    service_name="context-engine",
    json_output=True,
)
```

## Testing

```bash
# Run all observe tests
uv run pytest packages/context-observe/tests/ -v

# Test specific component
uv run pytest packages/context-observe/tests/test_tracer.py
uv run pytest packages/context-observe/tests/test_metrics.py
uv run pytest packages/context-observe/tests/test_events.py
```

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `OTEL_EXPORTER_OTLP_ENDPOINT` | - | OpenTelemetry endpoint |
| `OTEL_SERVICE_NAME` | `context-engine` | Service name for traces |
| `PROMETHEUS_PORT` | `9090` | Prometheus scrape port |
| `LOG_LEVEL` | `INFO` | Logging level |
| `LOG_FORMAT` | `json` | Log format (json/text) |

### Programmatic Configuration

```python
from context_observe import ContextTracer, ContextMetrics, ContextEvents

# Full configuration
tracer = ContextTracer(
    service_name="my-agent",
    endpoint="http://tempo:4317",
    sample_rate=0.1,  # Sample 10% of traces
    propagate_context=True,
)

metrics = ContextMetrics(
    service_name="my-agent",
    port=9090,
    push_gateway="http://pushgateway:9091",  # Optional
)

events = ContextEvents(
    service_name="my-agent",
    log_level="DEBUG",
    json_output=True,
    include_timestamps=True,
    extra_fields={"environment": "production"},
)
```

## License

MIT
