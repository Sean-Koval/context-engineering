"""Exporters for tracing and metrics."""

from __future__ import annotations

from context_observe.exporters.console import ConsoleMetricExporter, ConsoleSpanExporter

__all__ = [
    "ConsoleSpanExporter",
    "ConsoleMetricExporter",
]

# Optional exports based on available dependencies
try:
    from context_observe.exporters.otlp import (  # noqa: F401
        OTLPMetricExporter,
        OTLPSpanExporter,
    )

    __all__.extend(["OTLPSpanExporter", "OTLPMetricExporter"])
except ImportError:
    pass

try:
    from context_observe.exporters.prometheus import (  # noqa: F401
        PrometheusMetricExporter,
    )

    __all__.append("PrometheusMetricExporter")
except ImportError:
    pass
