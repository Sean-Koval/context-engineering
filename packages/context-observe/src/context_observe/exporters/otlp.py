"""OTLP exporters for production use."""

from __future__ import annotations

# These require the 'otlp' optional dependency
try:
    from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import (
        OTLPMetricExporter as _OTLPMetricExporter,
    )
    from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (
        OTLPSpanExporter as _OTLPSpanExporter,
    )

    class OTLPSpanExporter(_OTLPSpanExporter):
        """
        OTLP span exporter for production tracing.

        Exports spans to an OTLP-compatible collector (e.g., Jaeger, Tempo).

        Example:
            >>> from context_observe import ContextTracer
            >>> from context_observe.exporters import OTLPSpanExporter
            >>> exporter = OTLPSpanExporter(endpoint="http://localhost:4317")
            >>> tracer = ContextTracer(exporter=exporter)
        """

        def __init__(
            self,
            endpoint: str = "http://localhost:4317",
            insecure: bool = True,
            **kwargs: object,
        ) -> None:
            """
            Initialize OTLP span exporter.

            Args:
                endpoint: OTLP collector endpoint
                insecure: Whether to use insecure connection
                **kwargs: Additional arguments passed to base exporter
            """
            super().__init__(endpoint=endpoint, insecure=insecure, **kwargs)

    class OTLPMetricExporter(_OTLPMetricExporter):
        """
        OTLP metric exporter for production metrics.

        Exports metrics to an OTLP-compatible collector.

        Example:
            >>> from context_observe import ContextMetrics
            >>> from context_observe.exporters import OTLPMetricExporter
            >>> exporter = OTLPMetricExporter(endpoint="http://localhost:4317")
            >>> metrics = ContextMetrics(exporter=exporter)
        """

        def __init__(
            self,
            endpoint: str = "http://localhost:4317",
            insecure: bool = True,
            **kwargs: object,
        ) -> None:
            """
            Initialize OTLP metric exporter.

            Args:
                endpoint: OTLP collector endpoint
                insecure: Whether to use insecure connection
                **kwargs: Additional arguments passed to base exporter
            """
            super().__init__(endpoint=endpoint, insecure=insecure, **kwargs)

except ImportError:
    # OTLP dependencies not installed
    OTLPSpanExporter = None  # type: ignore[assignment, misc]
    OTLPMetricExporter = None  # type: ignore[assignment, misc]

__all__ = ["OTLPSpanExporter", "OTLPMetricExporter"]
