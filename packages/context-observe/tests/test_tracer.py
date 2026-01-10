"""Tests for ContextTracer."""

from __future__ import annotations

from io import StringIO

import pytest

from context_observe.exporters.console import ConsoleSpanExporter
from context_observe.tracer import ContextTracer, SpanAttributes


class TestSpanAttributes:
    """Tests for SpanAttributes model."""

    def test_minimal_attributes(self) -> None:
        """Test creating attributes with just operation_type."""
        attrs = SpanAttributes(operation_type="test.operation")
        assert attrs.operation_type == "test.operation"
        assert attrs.session_id is None

    def test_full_attributes(self) -> None:
        """Test creating attributes with all fields."""
        attrs = SpanAttributes(
            session_id="session-123",
            agent_id="agent-456",
            operation_type="compression",
            node_count=100,
            token_count=5000,
            compression_strategy="dedupe",
            compression_ratio=3.5,
            tokens_saved=3500,
            storage_tier="hot",
            cache_hit=True,
        )
        assert attrs.session_id == "session-123"
        assert attrs.compression_ratio == 3.5
        assert attrs.cache_hit is True

    def test_model_dump_excludes_none(self) -> None:
        """Test that model_dump excludes None values."""
        attrs = SpanAttributes(operation_type="test")
        dumped = attrs.model_dump(exclude_none=True)
        assert "operation_type" in dumped
        assert "session_id" not in dumped


class TestContextTracer:
    """Tests for ContextTracer."""

    def test_init_without_exporter(self) -> None:
        """Test tracer initialization without exporter."""
        tracer = ContextTracer(service_name="test-service")
        assert tracer._service_name == "test-service"

    def test_init_with_exporter(self) -> None:
        """Test tracer initialization with console exporter."""
        output = StringIO()
        exporter = ConsoleSpanExporter(out=output, colorize=False)
        tracer = ContextTracer(service_name="test", exporter=exporter)
        assert tracer._service_name == "test"

    def test_span_context_manager(self) -> None:
        """Test basic span creation."""
        tracer = ContextTracer()
        attrs = SpanAttributes(operation_type="test.span")

        with tracer.span("test-span", attrs) as span:
            assert span is not None
            assert span.is_recording()

    def test_span_records_exception(self) -> None:
        """Test that span records exceptions."""
        tracer = ContextTracer()
        attrs = SpanAttributes(operation_type="test.error")

        with pytest.raises(ValueError), tracer.span("error-span", attrs):
            raise ValueError("Test error")

    def test_trace_graph_operation(self) -> None:
        """Test graph operation tracing."""
        tracer = ContextTracer()

        with tracer.trace_graph_operation("add_node", "session-1", node_count=5) as span:
            assert span is not None

    def test_trace_compression(self) -> None:
        """Test compression tracing."""
        tracer = ContextTracer()

        with tracer.trace_compression("dedupe", "session-1", original_tokens=1000) as span:
            assert span is not None

    def test_trace_retrieval(self) -> None:
        """Test retrieval tracing."""
        tracer = ContextTracer()

        with tracer.trace_retrieval("semantic", "session-1") as span:
            assert span is not None

    def test_trace_tool_call(self) -> None:
        """Test tool call tracing."""
        tracer = ContextTracer()

        with tracer.trace_tool_call("read_file", "session-1", cached=True) as span:
            assert span is not None

    def test_traced_decorator(self) -> None:
        """Test the @traced decorator."""
        tracer = ContextTracer()

        @tracer.traced("decorated.operation")
        def sample_function(x: int) -> int:
            return x * 2

        result = sample_function(5)
        assert result == 10

    def test_traced_async_decorator(self) -> None:
        """Test the @traced_async decorator."""
        import asyncio

        tracer = ContextTracer()

        @tracer.traced_async("async.operation")
        async def async_function(x: int) -> int:
            return x * 2

        result = asyncio.run(async_function(5))
        assert result == 10

    def test_add_event(self) -> None:
        """Test adding events to current span."""
        tracer = ContextTracer()

        with tracer.span("test", SpanAttributes(operation_type="test")):
            tracer.add_event("checkpoint", {"step": 1})

    def test_get_current_span(self) -> None:
        """Test getting current span."""
        tracer = ContextTracer()

        # Outside span context
        assert tracer.get_current_span() is None

        # Inside span context
        with tracer.span("test", SpanAttributes(operation_type="test")):
            current = tracer.get_current_span()
            assert current is not None

    def test_shutdown(self) -> None:
        """Test tracer shutdown."""
        tracer = ContextTracer()
        tracer.shutdown()  # Should not raise
