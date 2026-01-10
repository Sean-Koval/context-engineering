"""Console exporters for development and debugging."""

from __future__ import annotations

import sys
from collections.abc import Sequence
from datetime import UTC, datetime
from typing import TextIO

from opentelemetry.sdk.metrics.export import (
    MetricExporter,
    MetricExportResult,
    MetricsData,
)
from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.sdk.trace.export import SpanExporter, SpanExportResult


class ConsoleSpanExporter(SpanExporter):
    """
    Console exporter for spans (development use).

    Prints span information to stdout/stderr for debugging.

    Example:
        >>> from context_observe import ContextTracer
        >>> from context_observe.exporters import ConsoleSpanExporter
        >>> tracer = ContextTracer(exporter=ConsoleSpanExporter())
    """

    def __init__(
        self,
        out: TextIO = sys.stdout,
        colorize: bool = True,
    ) -> None:
        """
        Initialize console span exporter.

        Args:
            out: Output stream (defaults to stdout)
            colorize: Whether to use ANSI colors
        """
        self._out = out
        self._colorize = colorize

    def export(self, spans: Sequence[ReadableSpan]) -> SpanExportResult:
        """Export spans to console."""
        for span in spans:
            self._print_span(span)
        return SpanExportResult.SUCCESS

    def _print_span(self, span: ReadableSpan) -> None:
        """Print a single span."""
        # Calculate duration
        if span.end_time and span.start_time:
            duration_ns = span.end_time - span.start_time
            duration_ms = duration_ns / 1_000_000
        else:
            duration_ms = 0

        # Format status
        status = span.status.status_code.name if span.status else "UNSET"

        # Colorize if enabled
        if self._colorize:
            if status == "ERROR":
                status_color = "\033[91m"  # Red
            elif status == "OK":
                status_color = "\033[92m"  # Green
            else:
                status_color = "\033[93m"  # Yellow
            reset = "\033[0m"
        else:
            status_color = reset = ""

        # Print span info
        timestamp = datetime.now(UTC).isoformat()
        line = (
            f"[{timestamp}] SPAN {span.name} "
            f"[{status_color}{status}{reset}] "
            f"duration={duration_ms:.2f}ms"
        )

        # Add attributes
        if span.attributes:
            attrs = ", ".join(f"{k}={v}" for k, v in span.attributes.items())
            line += f" | {attrs}"

        print(line, file=self._out)

        # Print events
        for event in span.events:
            print(f"  EVENT: {event.name}", file=self._out)

    def shutdown(self) -> None:
        """Shutdown the exporter."""
        self._out.flush()

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        """Force flush pending spans."""
        self._out.flush()
        return True


class ConsoleMetricExporter(MetricExporter):
    """
    Console exporter for metrics (development use).

    Prints metric data to stdout/stderr for debugging.

    Example:
        >>> from context_observe import ContextMetrics
        >>> from context_observe.exporters import ConsoleMetricExporter
        >>> metrics = ContextMetrics(exporter=ConsoleMetricExporter())
    """

    def __init__(
        self,
        out: TextIO = sys.stdout,
        colorize: bool = True,
    ) -> None:
        """
        Initialize console metric exporter.

        Args:
            out: Output stream (defaults to stdout)
            colorize: Whether to use ANSI colors
        """
        self._out = out
        self._colorize = colorize

    def export(
        self,
        metrics_data: MetricsData,
        timeout_millis: float = 10_000,
        **kwargs: object,
    ) -> MetricExportResult:
        """Export metrics to console."""
        timestamp = datetime.now(UTC).isoformat()

        for resource_metrics in metrics_data.resource_metrics:
            for scope_metrics in resource_metrics.scope_metrics:
                for metric in scope_metrics.metrics:
                    self._print_metric(metric, timestamp)

        return MetricExportResult.SUCCESS

    def _print_metric(self, metric: object, timestamp: str) -> None:
        """Print a single metric."""
        name = metric.name
        description = metric.description or ""

        # Get data points
        data_points = []
        if hasattr(metric, "data") and hasattr(metric.data, "data_points"):
            for point in metric.data.data_points:
                value = getattr(point, "value", None)
                if value is None:
                    # Handle histogram
                    value = getattr(point, "sum", "N/A")
                attrs = dict(point.attributes) if point.attributes else {}
                data_points.append((value, attrs))

        # Print
        if self._colorize:
            name_color = "\033[94m"  # Blue
            reset = "\033[0m"
        else:
            name_color = reset = ""

        for value, attrs in data_points:
            attrs_str = ", ".join(f"{k}={v}" for k, v in attrs.items()) if attrs else ""
            line = f"[{timestamp}] METRIC {name_color}{name}{reset}={value}"
            if attrs_str:
                line += f" | {attrs_str}"
            if description:
                line += f" # {description}"
            print(line, file=self._out)

    def shutdown(self, timeout_millis: float = 30_000, **kwargs: object) -> None:
        """Shutdown the exporter."""
        self._out.flush()

    def force_flush(self, timeout_millis: float = 10_000) -> bool:
        """Force flush pending metrics."""
        self._out.flush()
        return True
