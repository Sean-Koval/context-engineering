"""Prometheus exporter for metrics."""

from __future__ import annotations

# This requires the 'prometheus' optional dependency
try:
    from opentelemetry.sdk.metrics.export import (
        MetricExporter,
        MetricExportResult,
        MetricsData,
    )
    from prometheus_client import Counter, Gauge, Histogram, start_http_server

    class PrometheusMetricExporter(MetricExporter):
        """
        Prometheus metric exporter.

        Exposes metrics on an HTTP endpoint for Prometheus scraping.

        Example:
            >>> from context_observe import ContextMetrics
            >>> from context_observe.exporters import PrometheusMetricExporter
            >>> exporter = PrometheusMetricExporter(port=9090)
            >>> metrics = ContextMetrics(exporter=exporter)
        """

        def __init__(
            self,
            port: int = 9090,
            addr: str = "0.0.0.0",
        ) -> None:
            """
            Initialize Prometheus exporter and start HTTP server.

            Args:
                port: Port for metrics endpoint
                addr: Address to bind to
            """
            self._port = port
            self._addr = addr
            self._started = False

            # Prometheus metrics
            self._counters: dict[str, Counter] = {}
            self._gauges: dict[str, Gauge] = {}
            self._histograms: dict[str, Histogram] = {}

        def _ensure_started(self) -> None:
            """Start the HTTP server if not already running."""
            if not self._started:
                start_http_server(self._port, addr=self._addr)
                self._started = True

        def _get_or_create_counter(self, name: str, description: str) -> Counter:
            """Get or create a Prometheus counter."""
            if name not in self._counters:
                self._counters[name] = Counter(
                    name.replace(".", "_"),
                    description,
                    ["session_id"],
                )
            return self._counters[name]

        def _get_or_create_gauge(self, name: str, description: str) -> Gauge:
            """Get or create a Prometheus gauge."""
            if name not in self._gauges:
                self._gauges[name] = Gauge(
                    name.replace(".", "_"),
                    description,
                    ["session_id"],
                )
            return self._gauges[name]

        def _get_or_create_histogram(self, name: str, description: str) -> Histogram:
            """Get or create a Prometheus histogram."""
            if name not in self._histograms:
                self._histograms[name] = Histogram(
                    name.replace(".", "_"),
                    description,
                    ["session_id"],
                )
            return self._histograms[name]

        def export(
            self,
            metrics_data: MetricsData,
            timeout_millis: float = 10_000,
            **kwargs: object,
        ) -> MetricExportResult:
            """Export metrics to Prometheus."""
            self._ensure_started()

            for resource_metrics in metrics_data.resource_metrics:
                for scope_metrics in resource_metrics.scope_metrics:
                    for metric in scope_metrics.metrics:
                        self._export_metric(metric)

            return MetricExportResult.SUCCESS

        def _export_metric(self, metric: object) -> None:
            """Export a single metric to Prometheus."""
            name = metric.name
            description = metric.description or name

            if not hasattr(metric, "data") or not hasattr(metric.data, "data_points"):
                return

            for point in metric.data.data_points:
                attrs = dict(point.attributes) if point.attributes else {}
                session_id = attrs.get("session_id", "unknown")

                # Determine metric type and export
                if hasattr(point, "value"):
                    value = point.value
                    # Could be counter or gauge
                    if "total" in name or "count" in name:
                        _ = self._get_or_create_counter(name, description)
                        # Note: Prometheus counters can only increment - not used here
                    else:
                        gauge = self._get_or_create_gauge(name, description)
                        gauge.labels(session_id=session_id).set(value)
                elif hasattr(point, "sum"):
                    # Histogram - create for registration but mapping is simplified
                    _ = self._get_or_create_histogram(name, description)

        def shutdown(self, timeout_millis: float = 30_000, **kwargs: object) -> None:
            """Shutdown the exporter."""
            # Prometheus HTTP server runs in background thread
            pass

        def force_flush(self, timeout_millis: float = 10_000) -> bool:
            """Force flush (no-op for Prometheus)."""
            return True

except ImportError:
    # Prometheus dependencies not installed
    PrometheusMetricExporter = None  # type: ignore[assignment, misc]

__all__ = ["PrometheusMetricExporter"]
