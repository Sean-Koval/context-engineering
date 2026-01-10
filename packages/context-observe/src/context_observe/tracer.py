"""OpenTelemetry tracer for context operations."""

from __future__ import annotations

from collections.abc import Callable, Generator
from contextlib import contextmanager
from functools import wraps
from typing import TYPE_CHECKING, Any, TypeVar

from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, SpanExporter
from opentelemetry.trace import Span, Status, StatusCode
from pydantic import BaseModel, Field

if TYPE_CHECKING:
    pass

F = TypeVar("F", bound=Callable[..., Any])


class SpanAttributes(BaseModel):
    """Standard attributes for context spans."""

    # Session info
    session_id: str | None = None
    agent_id: str | None = None

    # Operation info
    operation_type: str = Field(description="Type of operation being traced")
    node_count: int | None = None
    token_count: int | None = None

    # Compression info
    compression_strategy: str | None = None
    compression_ratio: float | None = None
    tokens_saved: int | None = None

    # Memory info
    storage_tier: str | None = None
    cache_hit: bool | None = None

    # Error info
    error_type: str | None = None
    error_message: str | None = None


class ContextTracer:
    """
    OpenTelemetry tracer for context operations.

    Provides span creation for all context operations with
    standardized attributes and error handling.

    Example:
        >>> tracer = ContextTracer(service_name="my-agent")
        >>> with tracer.span("process", SpanAttributes(operation_type="graph.query")):
        ...     do_work()
    """

    TRACER_NAME = "context-engine"

    def __init__(
        self,
        service_name: str = "context-engine",
        exporter: SpanExporter | None = None,
    ) -> None:
        """
        Initialize the context tracer.

        Args:
            service_name: Name of the service for tracing
            exporter: Optional span exporter (defaults to no-op if None)
        """
        self._service_name = service_name
        self._provider = TracerProvider()

        if exporter:
            self._provider.add_span_processor(BatchSpanProcessor(exporter))

        trace.set_tracer_provider(self._provider)
        self._tracer = trace.get_tracer(self.TRACER_NAME, schema_url=None)

    @contextmanager
    def span(
        self,
        name: str,
        attributes: SpanAttributes | None = None,
        parent: Span | None = None,
    ) -> Generator[Span, None, None]:
        """
        Create a traced span for an operation.

        Args:
            name: Name of the span
            attributes: Standard span attributes
            parent: Optional parent span for nested tracing

        Yields:
            The active span

        Example:
            >>> with tracer.span("compress", SpanAttributes(operation_type="compression")):
            ...     do_compression()
        """
        context = trace.set_span_in_context(parent) if parent else None

        with self._tracer.start_as_current_span(name, context=context) as span:
            try:
                if attributes:
                    self._set_attributes(span, attributes)
                yield span
            except Exception as e:
                span.set_status(Status(StatusCode.ERROR, str(e)))
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
        operation: str,
        session_id: str,
        node_count: int | None = None,
    ) -> Generator[Span, None, None]:
        """
        Trace a graph operation.

        Args:
            operation: Operation name (e.g., "add_node", "query", "serialize")
            session_id: Session identifier
            node_count: Number of nodes involved
        """
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
    ) -> Generator[Span, None, None]:
        """
        Trace a compression operation.

        Args:
            strategy: Compression strategy name
            session_id: Session identifier
            original_tokens: Token count before compression
        """
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
    ) -> Generator[Span, None, None]:
        """
        Trace a memory retrieval operation.

        Args:
            strategy: Retrieval strategy name
            session_id: Session identifier
        """
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
    ) -> Generator[Span, None, None]:
        """
        Trace a tool call.

        Args:
            tool_name: Name of the tool being called
            session_id: Session identifier
            cached: Whether the result was retrieved from cache
        """
        attrs = SpanAttributes(
            session_id=session_id,
            operation_type="tool_call",
            cache_hit=cached,
        )
        with self.span(f"tool.{tool_name}", attrs) as span:
            yield span

    def traced(self, operation_type: str) -> Callable[[F], F]:
        """
        Decorator for tracing synchronous functions.

        Args:
            operation_type: Type of operation for the span

        Example:
            >>> @tracer.traced("graph.query")
            ... def query_nodes(self, filters):
            ...     return self._do_query(filters)
        """

        def decorator(func: F) -> F:
            @wraps(func)
            def wrapper(*args: Any, **kwargs: Any) -> Any:
                attrs = SpanAttributes(operation_type=operation_type)
                with self.span(operation_type, attrs):
                    return func(*args, **kwargs)

            return wrapper  # type: ignore[return-value]

        return decorator

    def traced_async(self, operation_type: str) -> Callable[[F], F]:
        """
        Decorator for tracing async functions.

        Args:
            operation_type: Type of operation for the span

        Example:
            >>> @tracer.traced_async("storage.save")
            ... async def save_context(self, data):
            ...     await self._storage.write(data)
        """

        def decorator(func: F) -> F:
            @wraps(func)
            async def wrapper(*args: Any, **kwargs: Any) -> Any:
                attrs = SpanAttributes(operation_type=operation_type)
                with self.span(operation_type, attrs):
                    return await func(*args, **kwargs)

            return wrapper  # type: ignore[return-value]

        return decorator

    def get_current_span(self) -> Span | None:
        """Get the current active span, if any."""
        span = trace.get_current_span()
        return span if span.is_recording() else None

    def add_event(self, name: str, attributes: dict[str, Any] | None = None) -> None:
        """Add an event to the current span."""
        span = self.get_current_span()
        if span:
            span.add_event(name, attributes=attributes)

    def set_error(self, error: Exception, message: str | None = None) -> None:
        """Record an error on the current span."""
        span = self.get_current_span()
        if span:
            span.set_status(Status(StatusCode.ERROR, message or str(error)))
            span.record_exception(error)

    def shutdown(self) -> None:
        """Shutdown tracer and flush pending spans."""
        self._provider.shutdown()
