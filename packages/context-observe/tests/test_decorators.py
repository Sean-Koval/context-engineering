"""Tests for decorators."""

from __future__ import annotations

import asyncio
import time

import pytest

from context_observe.decorators import (
    configure,
    get_metrics,
    get_tracer,
    metered,
    traced,
    traced_and_metered,
)
from context_observe.metrics import ContextMetrics
from context_observe.tracer import ContextTracer


class TestDecoratorConfiguration:
    """Tests for decorator configuration."""

    def test_configure(self) -> None:
        """Test configuring global instances."""
        tracer = ContextTracer()
        metrics = ContextMetrics()

        configure(tracer=tracer, metrics=metrics)

        assert get_tracer() is tracer
        assert get_metrics() is metrics

    def test_get_tracer_creates_default(self) -> None:
        """Test that get_tracer creates default if not configured."""
        configure(tracer=None, metrics=None)  # Reset
        tracer = get_tracer()
        assert isinstance(tracer, ContextTracer)

    def test_get_metrics_creates_default(self) -> None:
        """Test that get_metrics creates default if not configured."""
        configure(tracer=None, metrics=None)  # Reset
        metrics = get_metrics()
        assert isinstance(metrics, ContextMetrics)


class TestTracedDecorator:
    """Tests for @traced decorator."""

    def test_traced_without_args(self) -> None:
        """Test @traced without arguments."""

        @traced
        def simple_function(x: int) -> int:
            return x * 2

        result = simple_function(5)
        assert result == 10

    def test_traced_with_operation_type(self) -> None:
        """Test @traced with custom operation type."""

        @traced(operation_type="custom.operation")
        def custom_function(x: int) -> int:
            return x + 1

        result = custom_function(5)
        assert result == 6

    def test_traced_preserves_function_metadata(self) -> None:
        """Test that @traced preserves function name and docstring."""

        @traced
        def documented_function(x: int) -> int:
            """This is a docstring."""
            return x

        assert documented_function.__name__ == "documented_function"
        assert "docstring" in documented_function.__doc__

    def test_traced_with_include_args(self) -> None:
        """Test @traced with include_args=True."""

        @traced(operation_type="test", include_args=True)
        def args_function(a: int, b: str, c: float = 1.0) -> str:
            return f"{a}-{b}-{c}"

        result = args_function(1, "test", c=2.5)
        assert result == "1-test-2.5"

    def test_traced_async_function(self) -> None:
        """Test @traced on async function."""

        @traced
        async def async_function(x: int) -> int:
            await asyncio.sleep(0.01)
            return x * 2

        result = asyncio.run(async_function(5))
        assert result == 10

    def test_traced_propagates_exceptions(self) -> None:
        """Test that @traced propagates exceptions."""

        @traced
        def failing_function() -> None:
            raise ValueError("Test error")

        with pytest.raises(ValueError, match="Test error"):
            failing_function()


class TestMeteredDecorator:
    """Tests for @metered decorator."""

    def test_metered_without_args(self) -> None:
        """Test @metered without arguments."""
        configure(metrics=ContextMetrics())

        @metered
        def simple_function(x: int) -> int:
            return x * 2

        result = simple_function(5)
        assert result == 10

    def test_metered_with_operation(self) -> None:
        """Test @metered with custom operation."""
        configure(metrics=ContextMetrics())

        @metered(operation="custom.op")
        def custom_function(x: int) -> int:
            return x + 1

        result = custom_function(5)
        assert result == 6

    def test_metered_records_latency(self) -> None:
        """Test @metered records execution latency."""
        metrics = ContextMetrics()
        configure(metrics=metrics)

        @metered(operation="test.latency", record_latency=True)
        def slow_function() -> str:
            time.sleep(0.05)  # 50ms
            return "done"

        result = slow_function()
        assert result == "done"

        snapshot = metrics.snapshot()
        # Should have recorded latency
        assert snapshot.retrieval_latency_p99_ms > 0

    def test_metered_preserves_function_metadata(self) -> None:
        """Test that @metered preserves function metadata."""

        @metered
        def documented_function(x: int) -> int:
            """This is a docstring."""
            return x

        assert documented_function.__name__ == "documented_function"
        assert "docstring" in documented_function.__doc__

    def test_metered_async_function(self) -> None:
        """Test @metered on async function."""
        configure(metrics=ContextMetrics())

        @metered
        async def async_function(x: int) -> int:
            await asyncio.sleep(0.01)
            return x * 2

        result = asyncio.run(async_function(5))
        assert result == 10

    def test_metered_extracts_session_id_from_kwargs(self) -> None:
        """Test @metered extracts session_id from kwargs."""
        metrics = ContextMetrics()
        configure(metrics=metrics)

        @metered
        def function_with_session(data: str, session_id: str) -> str:
            return data.upper()

        result = function_with_session("hello", session_id="session-123")
        assert result == "HELLO"

    def test_metered_extracts_session_id_from_self(self) -> None:
        """Test @metered extracts session_id from self attribute."""
        metrics = ContextMetrics()
        configure(metrics=metrics)

        class MyClass:
            session_id = "class-session"

            @metered
            def method(self, x: int) -> int:
                return x * 2

        obj = MyClass()
        result = obj.method(5)
        assert result == 10


class TestTracedAndMeteredDecorator:
    """Tests for @traced_and_metered combined decorator."""

    def test_combined_decorator(self) -> None:
        """Test combined tracing and metrics."""
        tracer = ContextTracer()
        metrics = ContextMetrics()
        configure(tracer=tracer, metrics=metrics)

        @traced_and_metered(operation_type="combined.operation")
        def combined_function(x: int) -> int:
            return x * 2

        result = combined_function(5)
        assert result == 10

    def test_combined_decorator_records_both(self) -> None:
        """Test that combined decorator records both traces and metrics."""
        tracer = ContextTracer()
        metrics = ContextMetrics()
        configure(tracer=tracer, metrics=metrics)

        @traced_and_metered(operation_type="test.combined", record_latency=True)
        def function_to_observe() -> str:
            time.sleep(0.01)
            return "done"

        result = function_to_observe()
        assert result == "done"

        # Metrics should be recorded
        snapshot = metrics.snapshot()
        assert snapshot.retrieval_latency_p99_ms > 0
