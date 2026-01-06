# Context-Observe: Detailed Implementation Specification

> **Phase 1-4 Observability** | Estimated: 2-3 weeks | Priority: P1
> This document provides implementable specifications for observability.

---

## Table of Contents
1. [Package Overview](#package-overview)
2. [Component 1: ContextTracer](#component-1-contexttracer)
3. [Component 2: ContextMetrics](#component-2-contextmetrics)
4. [Component 3: ContextEvents](#component-3-contextevents)
5. [Component 4: ContextDashboard](#component-4-contextdashboard)
6. [Integration Patterns](#integration-patterns)
7. [Task Breakdown](#task-breakdown)

---

## Package Overview

### Purpose
`context-observe` provides OpenTelemetry-native observability for all context operations, enabling real-time monitoring, debugging, and optimization.

### Observability Philosophy

```
┌─────────────────────────────────────────────────────────────────────────┐
│                      OBSERVABILITY PYRAMID                               │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│                         ┌─────────────┐                                 │
│                         │  Dashboards │ ◄── Business KPIs              │
│                         └──────┬──────┘                                 │
│                                │                                         │
│                    ┌───────────┴───────────┐                            │
│                    │       Alerts          │ ◄── Threshold violations  │
│                    └───────────┬───────────┘                            │
│                                │                                         │
│           ┌────────────────────┴────────────────────┐                   │
│           │              Metrics                     │ ◄── Aggregates   │
│           └────────────────────┬────────────────────┘                   │
│                                │                                         │
│    ┌───────────────────────────┴───────────────────────────┐            │
│    │                      Events                            │ ◄── Logs  │
│    └───────────────────────────┬───────────────────────────┘            │
│                                │                                         │
│  ┌─────────────────────────────┴─────────────────────────────┐          │
│  │                       Traces                               │ ◄── Ops │
│  └───────────────────────────────────────────────────────────┘          │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### Key Metrics to Track

| Category | Metric | Target | Alert Threshold |
|----------|--------|--------|-----------------|
| **Token Usage** | Current utilization | < 65% | > 50% warning, > 65% critical |
| **Compression** | Ratio achieved | > 3x | < 2x warning |
| **Cache** | Hit rate | > 60% | < 40% warning |
| **Retrieval** | p99 latency | < 100ms | > 200ms warning |
| **Quality** | Pre-rot events | 0/session | > 3 warning |

### Dependencies

```toml
[project]
name = "context-observe"
version = "0.1.0"
requires-python = ">=3.10"

dependencies = [
    "pydantic>=2.0",
    "opentelemetry-api>=1.20",
    "opentelemetry-sdk>=1.20",
    "structlog>=23.0",
]

[project.optional-dependencies]
prometheus = ["prometheus-client>=0.17"]
jaeger = ["opentelemetry-exporter-jaeger>=1.20"]
otlp = ["opentelemetry-exporter-otlp>=1.20"]
all = ["context-observe[prometheus,otlp]"]
```

### Module Structure

```
context_observe/
├── __init__.py
├── py.typed
├── config.py                 # ObservabilityConfig
├── tracer.py                 # ContextTracer
├── metrics.py                # ContextMetrics
├── events.py                 # ContextEvents
├── dashboard.py              # ContextDashboard
├── exporters/
│   ├── __init__.py
│   ├── console.py            # Console exporter for dev
│   ├── prometheus.py         # Prometheus exporter
│   └── otlp.py               # OTLP exporter
└── decorators.py             # @traced, @metered decorators
```

---

## Component 1: ContextTracer

```python
# context_observe/tracer.py
from typing import Optional, Any, Callable
from uuid import UUID
from datetime import datetime
from contextlib import contextmanager
from functools import wraps

from opentelemetry import trace
from opentelemetry.trace import Span, Status, StatusCode
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from pydantic import BaseModel


class SpanAttributes(BaseModel):
    """Standard attributes for context spans."""
    # Session info
    session_id: Optional[str] = None
    agent_id: Optional[str] = None

    # Operation info
    operation_type: str
    node_count: Optional[int] = None
    token_count: Optional[int] = None

    # Compression info
    compression_strategy: Optional[str] = None
    compression_ratio: Optional[float] = None
    tokens_saved: Optional[int] = None

    # Memory info
    storage_tier: Optional[str] = None
    cache_hit: Optional[bool] = None

    # Error info
    error_type: Optional[str] = None
    error_message: Optional[str] = None


class ContextTracer:
    """
    OpenTelemetry tracer for context operations.

    Provides span creation for all context operations with
    standardized attributes and error handling.
    """

    TRACER_NAME = "context-engine"

    def __init__(
        self,
        service_name: str = "context-engine",
        exporter: Optional[Any] = None,
    ):
        self._service_name = service_name
        self._provider = TracerProvider()

        if exporter:
            self._provider.add_span_processor(
                BatchSpanProcessor(exporter)
            )

        trace.set_tracer_provider(self._provider)
        self._tracer = trace.get_tracer(self.TRACER_NAME)

    @contextmanager
    def span(
        self,
        name: str,
        attributes: Optional[SpanAttributes] = None,
        parent: Optional[Span] = None,
    ):
        """
        Create a traced span for an operation.

        Usage:
            with tracer.span("compress", SpanAttributes(strategy="dedupe")):
                do_compression()
        """
        context = trace.set_span_in_context(parent) if parent else None

        with self._tracer.start_as_current_span(
            name,
            context=context,
        ) as span:
            try:
                if attributes:
                    self._set_attributes(span, attributes)
                yield span
            except Exception as e:
                span.set_status(Status(StatusCode.ERROR))
                span.record_exception(e)
                raise

    def _set_attributes(self, span: Span, attributes: SpanAttributes) -> None:
        """Set span attributes from model."""
        for key, value in attributes.model_dump(exclude_none=True).items():
            if value is not None:
                span.set_attribute(f"context.{key}", value)

    # Convenience methods for common operations

    @contextmanager
    def trace_graph_operation(
        self,
        operation: str,  # "add_node", "query", "serialize"
        session_id: str,
        node_count: Optional[int] = None,
    ):
        """Trace a graph operation."""
        attrs = SpanAttributes(
            session_id=session_id,
            operation_type=f"graph.{operation}",
            node_count=node_count,
        )
        with self.span(f"graph.{operation}", attrs) as span:
            yield span

    @contextmanager
    def trace_compression(
        self,
        strategy: str,
        session_id: str,
        original_tokens: int,
    ):
        """Trace a compression operation."""
        attrs = SpanAttributes(
            session_id=session_id,
            operation_type="compression",
            compression_strategy=strategy,
            token_count=original_tokens,
        )
        with self.span(f"compress.{strategy}", attrs) as span:
            yield span

    @contextmanager
    def trace_retrieval(
        self,
        strategy: str,
        session_id: str,
    ):
        """Trace a memory retrieval operation."""
        attrs = SpanAttributes(
            session_id=session_id,
            operation_type="retrieval",
        )
        with self.span(f"retrieve.{strategy}", attrs) as span:
            yield span

    @contextmanager
    def trace_tool_call(
        self,
        tool_name: str,
        session_id: str,
        cached: bool = False,
    ):
        """Trace a tool call."""
        attrs = SpanAttributes(
            session_id=session_id,
            operation_type="tool_call",
            cache_hit=cached,
        )
        with self.span(f"tool.{tool_name}", attrs) as span:
            yield span

    def traced(
        self,
        operation_type: str,
    ) -> Callable:
        """
        Decorator for tracing functions.

        Usage:
            @tracer.traced("graph.query")
            def query_nodes(self, ...):
                ...
        """
        def decorator(func: Callable) -> Callable:
            @wraps(func)
            def wrapper(*args, **kwargs):
                attrs = SpanAttributes(operation_type=operation_type)
                with self.span(operation_type, attrs):
                    return func(*args, **kwargs)
            return wrapper
        return decorator

    async def traced_async(
        self,
        operation_type: str,
    ) -> Callable:
        """Async version of traced decorator."""
        def decorator(func: Callable) -> Callable:
            @wraps(func)
            async def wrapper(*args, **kwargs):
                attrs = SpanAttributes(operation_type=operation_type)
                with self.span(operation_type, attrs):
                    return await func(*args, **kwargs)
            return wrapper
        return decorator

    def shutdown(self) -> None:
        """Shutdown tracer and flush pending spans."""
        self._provider.shutdown()
```

---

## Component 2: ContextMetrics

```python
# context_observe/metrics.py
from typing import Optional, Dict, Any
from datetime import datetime
from collections import defaultdict
import threading

from opentelemetry import metrics
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
from pydantic import BaseModel


class MetricSnapshot(BaseModel):
    """Snapshot of current metrics."""
    timestamp: datetime
    token_usage: float
    compression_ratio: float
    cache_hit_rate: float
    retrieval_latency_p99_ms: float
    active_sessions: int
    total_compressions: int
    pre_rot_warnings: int


class ContextMetrics:
    """
    Prometheus/OTEL metrics for context operations.

    Tracks key performance indicators for context management.
    """

    METER_NAME = "context-engine"

    def __init__(
        self,
        exporter: Optional[Any] = None,
        export_interval_ms: int = 10000,
    ):
        readers = []
        if exporter:
            readers.append(
                PeriodicExportingMetricReader(
                    exporter,
                    export_interval_millis=export_interval_ms,
                )
            )

        self._provider = MeterProvider(metric_readers=readers)
        metrics.set_meter_provider(self._provider)
        self._meter = metrics.get_meter(self.METER_NAME)

        self._lock = threading.Lock()

        # Initialize metrics
        self._init_metrics()

        # Local tracking for snapshot
        self._local_stats = defaultdict(float)

    def _init_metrics(self) -> None:
        """Initialize all metric instruments."""
        # Token usage
        self._token_usage = self._meter.create_gauge(
            "context.token_usage_ratio",
            description="Current token usage as ratio of budget",
            unit="1",
        )

        self._token_count = self._meter.create_counter(
            "context.tokens_total",
            description="Total tokens processed",
            unit="tokens",
        )

        # Compression
        self._compression_ratio = self._meter.create_histogram(
            "context.compression_ratio",
            description="Compression ratio achieved",
            unit="1",
        )

        self._compression_tokens_saved = self._meter.create_counter(
            "context.compression_tokens_saved_total",
            description="Total tokens saved by compression",
            unit="tokens",
        )

        self._compression_operations = self._meter.create_counter(
            "context.compression_operations_total",
            description="Total compression operations",
            unit="operations",
        )

        # Cache
        self._cache_hits = self._meter.create_counter(
            "context.cache_hits_total",
            description="Total cache hits",
        )

        self._cache_misses = self._meter.create_counter(
            "context.cache_misses_total",
            description="Total cache misses",
        )

        # Retrieval
        self._retrieval_latency = self._meter.create_histogram(
            "context.retrieval_latency_seconds",
            description="Memory retrieval latency",
            unit="s",
        )

        # Pre-rot
        self._pre_rot_warnings = self._meter.create_counter(
            "context.pre_rot_warnings_total",
            description="Pre-rot warning events",
        )

        self._pre_rot_triggers = self._meter.create_counter(
            "context.pre_rot_triggers_total",
            description="Pre-rot trigger events",
        )

        # Sessions
        self._active_sessions = self._meter.create_up_down_counter(
            "context.active_sessions",
            description="Currently active sessions",
        )

    # Recording methods

    def record_token_usage(
        self,
        used: int,
        total: int,
        session_id: str,
    ) -> None:
        """Record current token usage."""
        ratio = used / total if total > 0 else 0
        self._token_usage.set(ratio, {"session_id": session_id})
        self._local_stats["token_usage"] = ratio

    def record_tokens_processed(
        self,
        count: int,
        operation: str,
        session_id: str,
    ) -> None:
        """Record tokens processed."""
        self._token_count.add(count, {
            "operation": operation,
            "session_id": session_id,
        })

    def record_compression(
        self,
        strategy: str,
        original_tokens: int,
        compressed_tokens: int,
        session_id: str,
    ) -> None:
        """Record a compression operation."""
        ratio = original_tokens / compressed_tokens if compressed_tokens > 0 else 1
        saved = original_tokens - compressed_tokens

        attrs = {"strategy": strategy, "session_id": session_id}

        self._compression_ratio.record(ratio, attrs)
        self._compression_tokens_saved.add(saved, attrs)
        self._compression_operations.add(1, attrs)

        with self._lock:
            self._local_stats["compression_ratio"] = ratio
            self._local_stats["total_compressions"] += 1

    def record_cache_hit(self, tool_name: str, session_id: str) -> None:
        """Record a cache hit."""
        self._cache_hits.add(1, {
            "tool": tool_name,
            "session_id": session_id,
        })
        with self._lock:
            self._local_stats["cache_hits"] += 1
            self._update_hit_rate()

    def record_cache_miss(self, tool_name: str, session_id: str) -> None:
        """Record a cache miss."""
        self._cache_misses.add(1, {
            "tool": tool_name,
            "session_id": session_id,
        })
        with self._lock:
            self._local_stats["cache_misses"] += 1
            self._update_hit_rate()

    def _update_hit_rate(self) -> None:
        """Update cache hit rate."""
        hits = self._local_stats["cache_hits"]
        misses = self._local_stats["cache_misses"]
        total = hits + misses
        self._local_stats["cache_hit_rate"] = hits / total if total > 0 else 0

    def record_retrieval_latency(
        self,
        latency_seconds: float,
        strategy: str,
        session_id: str,
    ) -> None:
        """Record retrieval latency."""
        self._retrieval_latency.record(latency_seconds, {
            "strategy": strategy,
            "session_id": session_id,
        })
        with self._lock:
            # Track p99 approximation
            key = "retrieval_latencies"
            if key not in self._local_stats:
                self._local_stats[key] = []
            self._local_stats[key].append(latency_seconds)
            # Keep last 1000
            if len(self._local_stats[key]) > 1000:
                self._local_stats[key] = self._local_stats[key][-1000:]

    def record_pre_rot_warning(self, session_id: str, usage_ratio: float) -> None:
        """Record a pre-rot warning."""
        self._pre_rot_warnings.add(1, {
            "session_id": session_id,
            "usage_ratio": str(usage_ratio),
        })
        with self._lock:
            self._local_stats["pre_rot_warnings"] += 1

    def record_pre_rot_trigger(self, session_id: str, usage_ratio: float) -> None:
        """Record a pre-rot trigger."""
        self._pre_rot_triggers.add(1, {
            "session_id": session_id,
            "usage_ratio": str(usage_ratio),
        })

    def session_started(self, session_id: str) -> None:
        """Record session start."""
        self._active_sessions.add(1, {"session_id": session_id})
        with self._lock:
            self._local_stats["active_sessions"] += 1

    def session_ended(self, session_id: str) -> None:
        """Record session end."""
        self._active_sessions.add(-1, {"session_id": session_id})
        with self._lock:
            self._local_stats["active_sessions"] -= 1

    def snapshot(self) -> MetricSnapshot:
        """Get current metrics snapshot."""
        with self._lock:
            latencies = self._local_stats.get("retrieval_latencies", [])
            p99 = sorted(latencies)[int(len(latencies) * 0.99)] if latencies else 0

            return MetricSnapshot(
                timestamp=datetime.utcnow(),
                token_usage=self._local_stats.get("token_usage", 0),
                compression_ratio=self._local_stats.get("compression_ratio", 1),
                cache_hit_rate=self._local_stats.get("cache_hit_rate", 0),
                retrieval_latency_p99_ms=p99 * 1000,
                active_sessions=int(self._local_stats.get("active_sessions", 0)),
                total_compressions=int(self._local_stats.get("total_compressions", 0)),
                pre_rot_warnings=int(self._local_stats.get("pre_rot_warnings", 0)),
            )

    def shutdown(self) -> None:
        """Shutdown metrics and flush."""
        self._provider.shutdown()
```

---

## Component 3: ContextEvents

```python
# context_observe/events.py
from typing import Optional, Any, Dict
from datetime import datetime
from enum import Enum
import structlog
from pydantic import BaseModel, Field


class EventLevel(str, Enum):
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class ContextEvent(BaseModel):
    """Structured context event."""
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    level: EventLevel
    event_type: str
    session_id: Optional[str] = None
    message: str
    data: Dict[str, Any] = Field(default_factory=dict)


class ContextEvents:
    """
    Structured event logging for context operations.

    Uses structlog for machine-parseable logs.
    """

    def __init__(
        self,
        service_name: str = "context-engine",
        log_level: str = "INFO",
    ):
        self._service_name = service_name

        structlog.configure(
            processors=[
                structlog.stdlib.filter_by_level,
                structlog.stdlib.add_logger_name,
                structlog.stdlib.add_log_level,
                structlog.stdlib.PositionalArgumentsFormatter(),
                structlog.processors.TimeStamper(fmt="iso"),
                structlog.processors.StackInfoRenderer(),
                structlog.processors.format_exc_info,
                structlog.processors.UnicodeDecoder(),
                structlog.processors.JSONRenderer(),
            ],
            wrapper_class=structlog.stdlib.BoundLogger,
            context_class=dict,
            logger_factory=structlog.stdlib.LoggerFactory(),
            cache_logger_on_first_use=True,
        )

        self._logger = structlog.get_logger(service_name)

    def _log(
        self,
        level: EventLevel,
        event_type: str,
        message: str,
        session_id: Optional[str] = None,
        **data,
    ) -> ContextEvent:
        """Internal logging method."""
        event = ContextEvent(
            level=level,
            event_type=event_type,
            session_id=session_id,
            message=message,
            data=data,
        )

        log_method = getattr(self._logger, level.value)
        log_method(
            message,
            event_type=event_type,
            session_id=session_id,
            **data,
        )

        return event

    # Convenience methods for common events

    def session_started(
        self,
        session_id: str,
        total_tokens: int,
        **kwargs,
    ) -> ContextEvent:
        """Log session start."""
        return self._log(
            EventLevel.INFO,
            "session.started",
            f"Session started with {total_tokens} token budget",
            session_id=session_id,
            total_tokens=total_tokens,
            **kwargs,
        )

    def session_ended(
        self,
        session_id: str,
        duration_seconds: float,
        tokens_used: int,
        compressions: int,
        **kwargs,
    ) -> ContextEvent:
        """Log session end."""
        return self._log(
            EventLevel.INFO,
            "session.ended",
            f"Session ended after {duration_seconds:.1f}s, used {tokens_used} tokens",
            session_id=session_id,
            duration_seconds=duration_seconds,
            tokens_used=tokens_used,
            compressions=compressions,
            **kwargs,
        )

    def pre_rot_warning(
        self,
        session_id: str,
        usage_ratio: float,
        tokens_used: int,
        tokens_total: int,
    ) -> ContextEvent:
        """Log pre-rot warning."""
        return self._log(
            EventLevel.WARNING,
            "budget.pre_rot_warning",
            f"Pre-rot warning: {usage_ratio:.1%} token usage",
            session_id=session_id,
            usage_ratio=usage_ratio,
            tokens_used=tokens_used,
            tokens_total=tokens_total,
        )

    def pre_rot_trigger(
        self,
        session_id: str,
        usage_ratio: float,
        compression_initiated: bool,
    ) -> ContextEvent:
        """Log pre-rot trigger."""
        return self._log(
            EventLevel.WARNING,
            "budget.pre_rot_trigger",
            f"Pre-rot trigger at {usage_ratio:.1%}, compression={'started' if compression_initiated else 'pending'}",
            session_id=session_id,
            usage_ratio=usage_ratio,
            compression_initiated=compression_initiated,
        )

    def compression_completed(
        self,
        session_id: str,
        strategy: str,
        original_tokens: int,
        compressed_tokens: int,
        duration_ms: float,
    ) -> ContextEvent:
        """Log compression completion."""
        ratio = original_tokens / compressed_tokens if compressed_tokens > 0 else 0
        saved = original_tokens - compressed_tokens

        return self._log(
            EventLevel.INFO,
            "compression.completed",
            f"Compression {strategy}: {ratio:.1f}x ratio, saved {saved} tokens in {duration_ms:.1f}ms",
            session_id=session_id,
            strategy=strategy,
            original_tokens=original_tokens,
            compressed_tokens=compressed_tokens,
            compression_ratio=ratio,
            tokens_saved=saved,
            duration_ms=duration_ms,
        )

    def cache_hit(
        self,
        session_id: str,
        tool_name: str,
        tokens_saved: int,
    ) -> ContextEvent:
        """Log cache hit."""
        return self._log(
            EventLevel.DEBUG,
            "cache.hit",
            f"Cache hit for {tool_name}, saved {tokens_saved} tokens",
            session_id=session_id,
            tool_name=tool_name,
            tokens_saved=tokens_saved,
        )

    def retrieval_completed(
        self,
        session_id: str,
        strategy: str,
        results_count: int,
        latency_ms: float,
    ) -> ContextEvent:
        """Log retrieval completion."""
        return self._log(
            EventLevel.DEBUG,
            "retrieval.completed",
            f"Retrieved {results_count} results via {strategy} in {latency_ms:.1f}ms",
            session_id=session_id,
            strategy=strategy,
            results_count=results_count,
            latency_ms=latency_ms,
        )

    def error(
        self,
        session_id: Optional[str],
        error_type: str,
        message: str,
        **kwargs,
    ) -> ContextEvent:
        """Log error."""
        return self._log(
            EventLevel.ERROR,
            f"error.{error_type}",
            message,
            session_id=session_id,
            **kwargs,
        )
```

---

## Component 4: ContextDashboard

```python
# context_observe/dashboard.py
from typing import Optional, Dict, Any, List
from datetime import datetime, timedelta
from pydantic import BaseModel
import json


class SessionSummary(BaseModel):
    """Summary of a session's context health."""
    session_id: str
    start_time: datetime
    duration_seconds: float
    status: str  # "healthy", "warning", "critical"

    # Token stats
    tokens_used: int
    tokens_total: int
    usage_ratio: float

    # Compression stats
    compressions_count: int
    total_tokens_saved: int
    avg_compression_ratio: float

    # Cache stats
    cache_hits: int
    cache_misses: int
    cache_hit_rate: float

    # Health indicators
    pre_rot_warnings: int
    pre_rot_triggers: int


class ContextDashboard:
    """
    Dashboard data provider for visualization.

    Aggregates metrics for Grafana or custom UIs.
    """

    def __init__(
        self,
        metrics: "ContextMetrics",
        events: "ContextEvents",
    ):
        self._metrics = metrics
        self._events = events
        self._sessions: Dict[str, Dict] = {}

    def register_session(self, session_id: str, total_tokens: int) -> None:
        """Register a new session for tracking."""
        self._sessions[session_id] = {
            "start_time": datetime.utcnow(),
            "total_tokens": total_tokens,
            "compressions": [],
            "cache_hits": 0,
            "cache_misses": 0,
            "pre_rot_warnings": 0,
            "pre_rot_triggers": 0,
        }

    def record_compression(
        self,
        session_id: str,
        strategy: str,
        original: int,
        compressed: int,
    ) -> None:
        """Record compression in session."""
        if session_id in self._sessions:
            self._sessions[session_id]["compressions"].append({
                "strategy": strategy,
                "original": original,
                "compressed": compressed,
                "saved": original - compressed,
                "ratio": original / compressed if compressed > 0 else 0,
                "timestamp": datetime.utcnow().isoformat(),
            })

    def get_session_summary(self, session_id: str) -> Optional[SessionSummary]:
        """Get summary for a session."""
        if session_id not in self._sessions:
            return None

        session = self._sessions[session_id]
        snapshot = self._metrics.snapshot()

        compressions = session["compressions"]
        total_saved = sum(c["saved"] for c in compressions)
        avg_ratio = (
            sum(c["ratio"] for c in compressions) / len(compressions)
            if compressions else 1.0
        )

        duration = (datetime.utcnow() - session["start_time"]).total_seconds()

        # Determine status
        if snapshot.token_usage > 0.65 or session["pre_rot_triggers"] > 0:
            status = "critical"
        elif snapshot.token_usage > 0.50 or session["pre_rot_warnings"] > 0:
            status = "warning"
        else:
            status = "healthy"

        return SessionSummary(
            session_id=session_id,
            start_time=session["start_time"],
            duration_seconds=duration,
            status=status,
            tokens_used=int(snapshot.token_usage * session["total_tokens"]),
            tokens_total=session["total_tokens"],
            usage_ratio=snapshot.token_usage,
            compressions_count=len(compressions),
            total_tokens_saved=total_saved,
            avg_compression_ratio=avg_ratio,
            cache_hits=session["cache_hits"],
            cache_misses=session["cache_misses"],
            cache_hit_rate=snapshot.cache_hit_rate,
            pre_rot_warnings=session["pre_rot_warnings"],
            pre_rot_triggers=session["pre_rot_triggers"],
        )

    def get_grafana_dashboard_json(self) -> str:
        """Generate Grafana dashboard JSON."""
        dashboard = {
            "title": "Context Engine Dashboard",
            "panels": [
                {
                    "title": "Token Usage",
                    "type": "gauge",
                    "targets": [
                        {"expr": "context_token_usage_ratio"}
                    ],
                    "fieldConfig": {
                        "defaults": {
                            "thresholds": {
                                "steps": [
                                    {"value": 0, "color": "green"},
                                    {"value": 0.5, "color": "yellow"},
                                    {"value": 0.65, "color": "red"},
                                ]
                            },
                            "max": 1,
                            "unit": "percentunit",
                        }
                    },
                },
                {
                    "title": "Compression Ratio",
                    "type": "stat",
                    "targets": [
                        {"expr": "histogram_quantile(0.5, context_compression_ratio)"}
                    ],
                },
                {
                    "title": "Cache Hit Rate",
                    "type": "gauge",
                    "targets": [
                        {"expr": "rate(context_cache_hits_total[5m]) / (rate(context_cache_hits_total[5m]) + rate(context_cache_misses_total[5m]))"}
                    ],
                },
                {
                    "title": "Retrieval Latency (p99)",
                    "type": "timeseries",
                    "targets": [
                        {"expr": "histogram_quantile(0.99, context_retrieval_latency_seconds)"}
                    ],
                },
                {
                    "title": "Pre-Rot Events",
                    "type": "stat",
                    "targets": [
                        {"expr": "sum(context_pre_rot_triggers_total)"}
                    ],
                },
            ],
        }
        return json.dumps(dashboard, indent=2)

    def export_session_report(
        self,
        session_id: str,
        format: str = "json",
    ) -> str:
        """Export session report."""
        summary = self.get_session_summary(session_id)
        if not summary:
            return "{}"

        if format == "json":
            return summary.model_dump_json(indent=2)

        elif format == "markdown":
            return f"""# Context Session Report

**Session ID**: {summary.session_id}
**Duration**: {summary.duration_seconds:.1f} seconds
**Status**: {summary.status.upper()}

## Token Usage
- Used: {summary.tokens_used:,} / {summary.tokens_total:,} ({summary.usage_ratio:.1%})

## Compression
- Operations: {summary.compressions_count}
- Tokens Saved: {summary.total_tokens_saved:,}
- Average Ratio: {summary.avg_compression_ratio:.1f}x

## Cache Performance
- Hits: {summary.cache_hits}
- Misses: {summary.cache_misses}
- Hit Rate: {summary.cache_hit_rate:.1%}

## Health Indicators
- Pre-Rot Warnings: {summary.pre_rot_warnings}
- Pre-Rot Triggers: {summary.pre_rot_triggers}
"""
        else:
            return summary.model_dump_json()
```

---

## Task Breakdown

### Week 5-6 (during Phase 1): Basic Observability

| Task ID | Task | Est. Hours | Acceptance Criteria |
|---------|------|------------|---------------------|
| OB-001 | Implement `ContextTracer` | 6 | Spans created correctly |
| OB-002 | Implement `ContextMetrics` | 6 | Metrics recorded |
| OB-003 | Implement `ContextEvents` | 4 | Structured logging works |
| OB-004 | Create decorators | 2 | @traced works |
| OB-005 | Write unit tests | 6 | 90%+ coverage |

### Week 23-24 (during Phase 4): Dashboard & Integration

| Task ID | Task | Est. Hours | Acceptance Criteria |
|---------|------|------------|---------------------|
| OB-006 | Implement `ContextDashboard` | 6 | Dashboard data available |
| OB-007 | Create Grafana dashboard | 4 | Dashboard imports |
| OB-008 | Add OTLP exporter | 3 | OTLP export works |
| OB-009 | Integration tests | 4 | End-to-end tracing |
| OB-010 | Documentation | 4 | Runbooks complete |

---

*This specification provides complete observability implementation. OpenTelemetry integration ensures compatibility with existing monitoring infrastructure.*
