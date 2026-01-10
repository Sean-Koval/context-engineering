"""Decorators for automatic tracing and metrics collection."""

from __future__ import annotations

import time
from collections.abc import Callable
from functools import wraps
from typing import Any, TypeVar, overload

from context_observe.metrics import ContextMetrics
from context_observe.tracer import ContextTracer, SpanAttributes

F = TypeVar("F", bound=Callable[..., Any])

# Module-level instances (can be configured)
_tracer: ContextTracer | None = None
_metrics: ContextMetrics | None = None


def configure(
    tracer: ContextTracer | None = None,
    metrics: ContextMetrics | None = None,
) -> None:
    """
    Configure the module-level tracer and metrics instances.

    Args:
        tracer: ContextTracer instance to use for @traced decorator
        metrics: ContextMetrics instance to use for @metered decorator

    Example:
        >>> from context_observe import configure, ContextTracer, ContextMetrics
        >>> configure(tracer=ContextTracer(), metrics=ContextMetrics())
    """
    global _tracer, _metrics
    _tracer = tracer
    _metrics = metrics


def get_tracer() -> ContextTracer:
    """Get the configured tracer, creating a default if needed."""
    global _tracer
    if _tracer is None:
        _tracer = ContextTracer()
    return _tracer


def get_metrics() -> ContextMetrics:
    """Get the configured metrics, creating a default if needed."""
    global _metrics
    if _metrics is None:
        _metrics = ContextMetrics()
    return _metrics


@overload
def traced(func: F) -> F: ...


@overload
def traced(
    *,
    operation_type: str | None = None,
    include_args: bool = False,
) -> Callable[[F], F]: ...


def traced(
    func: F | None = None,
    *,
    operation_type: str | None = None,
    include_args: bool = False,
) -> F | Callable[[F], F]:
    """
    Decorator for automatic function tracing.

    Can be used with or without arguments:
        @traced
        def my_func(): ...

        @traced(operation_type="custom.operation")
        def my_func(): ...

    Args:
        func: The function to decorate (when used without parentheses)
        operation_type: Custom operation type (defaults to function name)
        include_args: If True, include function arguments in span attributes

    Returns:
        Decorated function with automatic tracing

    Example:
        >>> @traced
        ... def process_data(data):
        ...     return transform(data)
        >>>
        >>> @traced(operation_type="graph.query", include_args=True)
        ... def query_nodes(filters):
        ...     return db.query(filters)
    """

    def decorator(fn: F) -> F:
        op_type = operation_type or f"function.{fn.__module__}.{fn.__name__}"

        @wraps(fn)
        def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
            tracer = get_tracer()
            attrs = SpanAttributes(operation_type=op_type)

            if include_args:
                # Add safe string representations of args
                attrs_dict = attrs.model_dump(exclude_none=True)
                for i, arg in enumerate(args[:5]):  # Limit to first 5 args
                    attrs_dict[f"arg_{i}"] = _safe_repr(arg)
                for key, value in list(kwargs.items())[:10]:  # Limit to 10 kwargs
                    attrs_dict[f"kwarg_{key}"] = _safe_repr(value)

            with tracer.span(op_type, attrs):
                return fn(*args, **kwargs)

        @wraps(fn)
        async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
            tracer = get_tracer()
            attrs = SpanAttributes(operation_type=op_type)

            if include_args:
                attrs_dict = attrs.model_dump(exclude_none=True)
                for i, arg in enumerate(args[:5]):
                    attrs_dict[f"arg_{i}"] = _safe_repr(arg)
                for key, value in list(kwargs.items())[:10]:
                    attrs_dict[f"kwarg_{key}"] = _safe_repr(value)

            with tracer.span(op_type, attrs):
                return await fn(*args, **kwargs)

        # Choose wrapper based on whether function is async
        import asyncio

        if asyncio.iscoroutinefunction(fn):
            return async_wrapper  # type: ignore[return-value]
        return sync_wrapper  # type: ignore[return-value]

    if func is not None:
        return decorator(func)
    return decorator


@overload
def metered(func: F) -> F: ...


@overload
def metered(
    *,
    operation: str | None = None,
    record_latency: bool = True,
    record_count: bool = True,
) -> Callable[[F], F]: ...


def metered(
    func: F | None = None,
    *,
    operation: str | None = None,
    record_latency: bool = True,
    record_count: bool = True,
) -> F | Callable[[F], F]:
    """
    Decorator for automatic metrics collection.

    Can be used with or without arguments:
        @metered
        def my_func(): ...

        @metered(operation="custom.operation")
        def my_func(): ...

    Args:
        func: The function to decorate (when used without parentheses)
        operation: Custom operation name (defaults to function name)
        record_latency: If True, record execution latency
        record_count: If True, record invocation count

    Returns:
        Decorated function with automatic metrics

    Example:
        >>> @metered
        ... def process_batch(items):
        ...     for item in items:
        ...         process(item)
        >>>
        >>> @metered(operation="retrieval.semantic", record_latency=True)
        ... def search_memories(query):
        ...     return index.search(query)
    """

    def decorator(fn: F) -> F:
        op_name = operation or f"{fn.__module__}.{fn.__name__}"

        @wraps(fn)
        def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
            metrics_instance = get_metrics()
            session_id = _extract_session_id(args, kwargs)

            start_time = time.perf_counter()
            try:
                result = fn(*args, **kwargs)
                return result
            finally:
                if record_latency:
                    elapsed = time.perf_counter() - start_time
                    metrics_instance.record_retrieval_latency(
                        elapsed, op_name, session_id or "unknown"
                    )
                if record_count:
                    metrics_instance.record_graph_operation(op_name, session_id or "unknown")

        @wraps(fn)
        async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
            metrics_instance = get_metrics()
            session_id = _extract_session_id(args, kwargs)

            start_time = time.perf_counter()
            try:
                result = await fn(*args, **kwargs)
                return result
            finally:
                if record_latency:
                    elapsed = time.perf_counter() - start_time
                    metrics_instance.record_retrieval_latency(
                        elapsed, op_name, session_id or "unknown"
                    )
                if record_count:
                    metrics_instance.record_graph_operation(op_name, session_id or "unknown")

        import asyncio

        if asyncio.iscoroutinefunction(fn):
            return async_wrapper  # type: ignore[return-value]
        return sync_wrapper  # type: ignore[return-value]

    if func is not None:
        return decorator(func)
    return decorator


def traced_and_metered(
    *,
    operation_type: str | None = None,
    include_args: bool = False,
    record_latency: bool = True,
) -> Callable[[F], F]:
    """
    Combined decorator for both tracing and metrics.

    Args:
        operation_type: Custom operation type
        include_args: Include function arguments in trace
        record_latency: Record execution latency

    Returns:
        Decorator that applies both @traced and @metered

    Example:
        >>> @traced_and_metered(operation_type="compression.dedupe")
        ... def deduplicate_nodes(graph):
        ...     return compress(graph)
    """

    def decorator(fn: F) -> F:
        traced_fn = traced(operation_type=operation_type, include_args=include_args)(fn)
        return metered(operation=operation_type, record_latency=record_latency)(traced_fn)

    return decorator


def _safe_repr(value: Any, max_length: int = 100) -> str:
    """Create a safe string representation of a value."""
    try:
        repr_str = repr(value)
        if len(repr_str) > max_length:
            return repr_str[: max_length - 3] + "..."
        return repr_str
    except Exception:
        return f"<{type(value).__name__}>"


def _extract_session_id(args: tuple[Any, ...], kwargs: dict[str, Any]) -> str | None:
    """Try to extract session_id from function arguments."""
    # Check kwargs first
    if "session_id" in kwargs:
        return str(kwargs["session_id"])

    # Check if first arg is self with session_id attribute
    if args and hasattr(args[0], "session_id"):
        return str(args[0].session_id)

    # Check if first arg is self with _session_id attribute
    if args and hasattr(args[0], "_session_id"):
        return str(args[0]._session_id)

    return None
