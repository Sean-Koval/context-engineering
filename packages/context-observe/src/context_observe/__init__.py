"""
context-observe: OpenTelemetry-native observability for ContextEngine.

Provides tracing, metrics, and structured logging for all context operations.
"""

from __future__ import annotations

from context_observe.config import ObservabilityConfig
from context_observe.decorators import metered, traced
from context_observe.events import ContextEvent, ContextEvents, EventLevel
from context_observe.metrics import ContextMetrics, MetricSnapshot
from context_observe.tracer import ContextTracer, SpanAttributes

__version__ = "0.1.0"

__all__ = [
    # Tracer
    "ContextTracer",
    "SpanAttributes",
    # Metrics
    "ContextMetrics",
    "MetricSnapshot",
    # Events
    "ContextEvents",
    "ContextEvent",
    "EventLevel",
    # Config
    "ObservabilityConfig",
    # Decorators
    "traced",
    "metered",
]
