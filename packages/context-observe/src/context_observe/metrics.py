"""Prometheus/OTEL metrics for context operations."""

from __future__ import annotations

import threading
from collections import defaultdict
from datetime import UTC, datetime
from typing import Any

from opentelemetry import metrics
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import MetricExporter, PeriodicExportingMetricReader
from pydantic import BaseModel, Field


class MetricSnapshot(BaseModel):
    """Snapshot of current metrics."""

    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    token_usage: float = Field(description="Current token usage ratio (0.0-1.0)")
    compression_ratio: float = Field(description="Last compression ratio achieved")
    cache_hit_rate: float = Field(description="Cache hit rate (0.0-1.0)")
    retrieval_latency_p99_ms: float = Field(description="p99 retrieval latency in milliseconds")
    active_sessions: int = Field(description="Number of active sessions")
    total_compressions: int = Field(description="Total compression operations")
    pre_rot_warnings: int = Field(description="Total pre-rot warning events")


class ContextMetrics:
    """
    Prometheus/OTEL metrics for context operations.

    Tracks key performance indicators for context management including:
    - Token usage and budget utilization
    - Compression ratios and tokens saved
    - Cache hit/miss rates
    - Memory retrieval latencies
    - Pre-rot warning and trigger events

    Example:
        >>> metrics = ContextMetrics()
        >>> metrics.record_token_usage(50000, 100000, "session-123")
        >>> metrics.record_compression("dedupe", 1000, 300, "session-123")
        >>> print(metrics.snapshot())
    """

    METER_NAME = "context-engine"

    def __init__(
        self,
        exporter: MetricExporter | None = None,
        export_interval_ms: int = 10000,
    ) -> None:
        """
        Initialize metrics collection.

        Args:
            exporter: Optional metric exporter (e.g., Prometheus, OTLP)
            export_interval_ms: Export interval in milliseconds
        """
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
        self._local_stats: dict[str, Any] = defaultdict(float)

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

        # Graph operations
        self._graph_operations = self._meter.create_counter(
            "context.graph_operations_total",
            description="Total graph operations",
        )

        self._graph_node_count = self._meter.create_gauge(
            "context.graph_node_count",
            description="Current node count in graph",
        )

    # Recording methods

    def record_token_usage(
        self,
        used: int,
        total: int,
        session_id: str,
    ) -> None:
        """
        Record current token usage.

        Args:
            used: Tokens currently used
            total: Total token budget
            session_id: Session identifier
        """
        ratio = used / total if total > 0 else 0
        self._token_usage.set(ratio, {"session_id": session_id})
        with self._lock:
            self._local_stats["token_usage"] = ratio

    def record_tokens_processed(
        self,
        count: int,
        operation: str,
        session_id: str,
    ) -> None:
        """
        Record tokens processed.

        Args:
            count: Number of tokens processed
            operation: Operation type
            session_id: Session identifier
        """
        self._token_count.add(
            count,
            {
                "operation": operation,
                "session_id": session_id,
            },
        )

    def record_compression(
        self,
        strategy: str,
        original_tokens: int,
        compressed_tokens: int,
        session_id: str,
    ) -> None:
        """
        Record a compression operation.

        Args:
            strategy: Compression strategy name
            original_tokens: Tokens before compression
            compressed_tokens: Tokens after compression
            session_id: Session identifier
        """
        ratio = original_tokens / compressed_tokens if compressed_tokens > 0 else 1
        saved = original_tokens - compressed_tokens

        attrs = {"strategy": strategy, "session_id": session_id}

        self._compression_ratio.record(ratio, attrs)
        self._compression_tokens_saved.add(saved, attrs)
        self._compression_operations.add(1, attrs)

        with self._lock:
            self._local_stats["compression_ratio"] = ratio
            self._local_stats["total_compressions"] = (
                self._local_stats.get("total_compressions", 0) + 1
            )

    def record_cache_hit(self, tool_name: str, session_id: str) -> None:
        """
        Record a cache hit.

        Args:
            tool_name: Name of the tool
            session_id: Session identifier
        """
        self._cache_hits.add(
            1,
            {
                "tool": tool_name,
                "session_id": session_id,
            },
        )
        with self._lock:
            self._local_stats["cache_hits"] = self._local_stats.get("cache_hits", 0) + 1
            self._update_hit_rate()

    def record_cache_miss(self, tool_name: str, session_id: str) -> None:
        """
        Record a cache miss.

        Args:
            tool_name: Name of the tool
            session_id: Session identifier
        """
        self._cache_misses.add(
            1,
            {
                "tool": tool_name,
                "session_id": session_id,
            },
        )
        with self._lock:
            self._local_stats["cache_misses"] = self._local_stats.get("cache_misses", 0) + 1
            self._update_hit_rate()

    def _update_hit_rate(self) -> None:
        """Update cache hit rate calculation."""
        hits = self._local_stats.get("cache_hits", 0)
        misses = self._local_stats.get("cache_misses", 0)
        total = hits + misses
        self._local_stats["cache_hit_rate"] = hits / total if total > 0 else 0

    def record_retrieval_latency(
        self,
        latency_seconds: float,
        strategy: str,
        session_id: str,
    ) -> None:
        """
        Record retrieval latency.

        Args:
            latency_seconds: Latency in seconds
            strategy: Retrieval strategy name
            session_id: Session identifier
        """
        self._retrieval_latency.record(
            latency_seconds,
            {
                "strategy": strategy,
                "session_id": session_id,
            },
        )
        with self._lock:
            # Track latencies for p99 calculation
            key = "retrieval_latencies"
            if key not in self._local_stats:
                self._local_stats[key] = []
            self._local_stats[key].append(latency_seconds)
            # Keep last 1000 samples
            if len(self._local_stats[key]) > 1000:
                self._local_stats[key] = self._local_stats[key][-1000:]

    def record_pre_rot_warning(self, session_id: str, usage_ratio: float) -> None:
        """
        Record a pre-rot warning event.

        Args:
            session_id: Session identifier
            usage_ratio: Current token usage ratio
        """
        self._pre_rot_warnings.add(
            1,
            {
                "session_id": session_id,
                "usage_ratio": f"{usage_ratio:.2f}",
            },
        )
        with self._lock:
            self._local_stats["pre_rot_warnings"] = self._local_stats.get("pre_rot_warnings", 0) + 1

    def record_pre_rot_trigger(self, session_id: str, usage_ratio: float) -> None:
        """
        Record a pre-rot trigger event.

        Args:
            session_id: Session identifier
            usage_ratio: Current token usage ratio
        """
        self._pre_rot_triggers.add(
            1,
            {
                "session_id": session_id,
                "usage_ratio": f"{usage_ratio:.2f}",
            },
        )

    def record_graph_operation(
        self,
        operation: str,
        session_id: str,
        node_count: int | None = None,
    ) -> None:
        """
        Record a graph operation.

        Args:
            operation: Operation type (add_node, query, etc.)
            session_id: Session identifier
            node_count: Current node count (optional)
        """
        self._graph_operations.add(
            1,
            {
                "operation": operation,
                "session_id": session_id,
            },
        )
        if node_count is not None:
            self._graph_node_count.set(node_count, {"session_id": session_id})

    def session_started(self, session_id: str) -> None:
        """
        Record session start.

        Args:
            session_id: Session identifier
        """
        self._active_sessions.add(1, {"session_id": session_id})
        with self._lock:
            self._local_stats["active_sessions"] = self._local_stats.get("active_sessions", 0) + 1

    def session_ended(self, session_id: str) -> None:
        """
        Record session end.

        Args:
            session_id: Session identifier
        """
        self._active_sessions.add(-1, {"session_id": session_id})
        with self._lock:
            self._local_stats["active_sessions"] = max(
                0, self._local_stats.get("active_sessions", 0) - 1
            )

    def snapshot(self) -> MetricSnapshot:
        """
        Get current metrics snapshot.

        Returns:
            MetricSnapshot with current values
        """
        with self._lock:
            latencies = self._local_stats.get("retrieval_latencies", [])
            if latencies:
                sorted_latencies = sorted(latencies)
                p99_index = int(len(sorted_latencies) * 0.99)
                p99 = sorted_latencies[min(p99_index, len(sorted_latencies) - 1)]
            else:
                p99 = 0

            return MetricSnapshot(
                timestamp=datetime.now(UTC),
                token_usage=float(self._local_stats.get("token_usage", 0)),
                compression_ratio=float(self._local_stats.get("compression_ratio", 1)),
                cache_hit_rate=float(self._local_stats.get("cache_hit_rate", 0)),
                retrieval_latency_p99_ms=p99 * 1000,
                active_sessions=int(self._local_stats.get("active_sessions", 0)),
                total_compressions=int(self._local_stats.get("total_compressions", 0)),
                pre_rot_warnings=int(self._local_stats.get("pre_rot_warnings", 0)),
            )

    def reset_local_stats(self) -> None:
        """Reset local statistics (useful for testing)."""
        with self._lock:
            self._local_stats = defaultdict(float)

    def shutdown(self) -> None:
        """Shutdown metrics and flush."""
        self._provider.shutdown()
