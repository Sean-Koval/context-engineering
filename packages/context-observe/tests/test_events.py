"""Tests for ContextEvents."""

from __future__ import annotations

from datetime import datetime

from context_observe.events import ContextEvent, ContextEvents, EventLevel


class TestEventLevel:
    """Tests for EventLevel enum."""

    def test_event_levels(self) -> None:
        """Test all event levels are defined."""
        assert EventLevel.DEBUG.value == "debug"
        assert EventLevel.INFO.value == "info"
        assert EventLevel.WARNING.value == "warning"
        assert EventLevel.ERROR.value == "error"
        assert EventLevel.CRITICAL.value == "critical"


class TestContextEvent:
    """Tests for ContextEvent model."""

    def test_event_creation(self) -> None:
        """Test creating an event."""
        event = ContextEvent(
            level=EventLevel.INFO,
            event_type="test.event",
            session_id="session-123",
            message="Test message",
            data={"key": "value"},
        )
        assert event.level == EventLevel.INFO
        assert event.event_type == "test.event"
        assert event.session_id == "session-123"
        assert event.message == "Test message"
        assert event.data == {"key": "value"}

    def test_event_timestamp_default(self) -> None:
        """Test that timestamp is set by default."""
        event = ContextEvent(
            level=EventLevel.INFO,
            event_type="test",
            message="Test",
        )
        assert isinstance(event.timestamp, datetime)

    def test_event_data_default(self) -> None:
        """Test that data defaults to empty dict."""
        event = ContextEvent(
            level=EventLevel.INFO,
            event_type="test",
            message="Test",
        )
        assert event.data == {}


class TestContextEvents:
    """Tests for ContextEvents logger."""

    def test_init(self) -> None:
        """Test events initialization."""
        events = ContextEvents(service_name="test-service", log_level="INFO")
        assert events._service_name == "test-service"

    def test_init_json_format(self) -> None:
        """Test events with JSON format."""
        events = ContextEvents(json_format=True)
        assert events._json_format is True

    def test_init_console_format(self) -> None:
        """Test events with console format."""
        events = ContextEvents(json_format=False)
        assert events._json_format is False

    def test_session_started(self) -> None:
        """Test session started event."""
        events = ContextEvents(log_level="DEBUG")
        event = events.session_started("session-123", total_tokens=100000)

        assert event.level == EventLevel.INFO
        assert event.event_type == "session.started"
        assert event.session_id == "session-123"
        assert "100,000" in event.message
        assert event.data["total_tokens"] == 100000

    def test_session_ended(self) -> None:
        """Test session ended event."""
        events = ContextEvents(log_level="DEBUG")
        event = events.session_ended(
            session_id="session-123",
            duration_seconds=120.5,
            tokens_used=50000,
            compressions=5,
        )

        assert event.level == EventLevel.INFO
        assert event.event_type == "session.ended"
        assert event.data["duration_seconds"] == 120.5
        assert event.data["tokens_used"] == 50000
        assert event.data["compressions"] == 5

    def test_pre_rot_warning(self) -> None:
        """Test pre-rot warning event."""
        events = ContextEvents(log_level="DEBUG")
        event = events.pre_rot_warning(
            session_id="session-123",
            usage_ratio=0.55,
            tokens_used=55000,
            tokens_total=100000,
        )

        assert event.level == EventLevel.WARNING
        assert event.event_type == "budget.pre_rot_warning"
        assert event.data["usage_ratio"] == 0.55

    def test_pre_rot_trigger(self) -> None:
        """Test pre-rot trigger event."""
        events = ContextEvents(log_level="DEBUG")
        event = events.pre_rot_trigger(
            session_id="session-123",
            usage_ratio=0.70,
            compression_initiated=True,
        )

        assert event.level == EventLevel.WARNING
        assert event.event_type == "budget.pre_rot_trigger"
        assert "started" in event.message

    def test_compression_started(self) -> None:
        """Test compression started event."""
        events = ContextEvents(log_level="DEBUG")
        event = events.compression_started(
            session_id="session-123",
            strategy="dedupe",
            target_tokens=5000,
        )

        assert event.level == EventLevel.INFO
        assert event.event_type == "compression.started"
        assert "dedupe" in event.message

    def test_compression_completed(self) -> None:
        """Test compression completed event."""
        events = ContextEvents(log_level="DEBUG")
        event = events.compression_completed(
            session_id="session-123",
            strategy="dedupe",
            original_tokens=1000,
            compressed_tokens=250,
            duration_ms=50.5,
        )

        assert event.level == EventLevel.INFO
        assert event.event_type == "compression.completed"
        assert event.data["compression_ratio"] == 4.0
        assert event.data["tokens_saved"] == 750

    def test_compression_failed(self) -> None:
        """Test compression failed event."""
        events = ContextEvents(log_level="DEBUG")
        event = events.compression_failed(
            session_id="session-123",
            strategy="dedupe",
            error="Out of memory",
        )

        assert event.level == EventLevel.ERROR
        assert event.event_type == "compression.failed"
        assert "Out of memory" in event.message

    def test_cache_hit(self) -> None:
        """Test cache hit event."""
        events = ContextEvents(log_level="DEBUG")
        event = events.cache_hit(
            session_id="session-123",
            tool_name="read_file",
            tokens_saved=500,
        )

        assert event.level == EventLevel.DEBUG
        assert event.event_type == "cache.hit"

    def test_cache_miss(self) -> None:
        """Test cache miss event."""
        events = ContextEvents(log_level="DEBUG")
        event = events.cache_miss(
            session_id="session-123",
            tool_name="read_file",
        )

        assert event.level == EventLevel.DEBUG
        assert event.event_type == "cache.miss"

    def test_retrieval_completed(self) -> None:
        """Test retrieval completed event."""
        events = ContextEvents(log_level="DEBUG")
        event = events.retrieval_completed(
            session_id="session-123",
            strategy="semantic",
            results_count=10,
            latency_ms=25.5,
        )

        assert event.level == EventLevel.DEBUG
        assert event.event_type == "retrieval.completed"

    def test_graph_node_added(self) -> None:
        """Test graph node added event."""
        events = ContextEvents(log_level="DEBUG")
        event = events.graph_node_added(
            session_id="session-123",
            node_type="MESSAGE",
            node_id="node-456",
            token_count=100,
        )

        assert event.level == EventLevel.DEBUG
        assert event.event_type == "graph.node_added"

    def test_graph_node_removed(self) -> None:
        """Test graph node removed event."""
        events = ContextEvents(log_level="DEBUG")
        event = events.graph_node_removed(
            session_id="session-123",
            node_id="node-456",
            reason="compression",
        )

        assert event.level == EventLevel.DEBUG
        assert event.event_type == "graph.node_removed"

    def test_error(self) -> None:
        """Test error event."""
        events = ContextEvents(log_level="DEBUG")
        event = events.error(
            session_id="session-123",
            error_type="storage",
            message="Failed to save context",
            details="Disk full",
        )

        assert event.level == EventLevel.ERROR
        assert event.event_type == "error.storage"

    def test_critical(self) -> None:
        """Test critical event."""
        events = ContextEvents(log_level="DEBUG")
        event = events.critical(
            session_id="session-123",
            error_type="memory",
            message="Out of memory",
        )

        assert event.level == EventLevel.CRITICAL
        assert event.event_type == "critical.memory"

    def test_generic_info(self) -> None:
        """Test generic info logging."""
        events = ContextEvents(log_level="DEBUG")
        event = events.info("custom.event", "Custom message", session_id="s1")

        assert event.level == EventLevel.INFO
        assert event.event_type == "custom.event"

    def test_generic_debug(self) -> None:
        """Test generic debug logging."""
        events = ContextEvents(log_level="DEBUG")
        event = events.debug("debug.event", "Debug info")

        assert event.level == EventLevel.DEBUG

    def test_generic_warning(self) -> None:
        """Test generic warning logging."""
        events = ContextEvents(log_level="DEBUG")
        event = events.warning("warn.event", "Warning message")

        assert event.level == EventLevel.WARNING

    def test_extra_kwargs(self) -> None:
        """Test that extra kwargs are passed to data."""
        events = ContextEvents(log_level="DEBUG")
        event = events.session_started(
            "session-123",
            total_tokens=100000,
            custom_field="custom_value",
            another_field=42,
        )

        assert event.data["custom_field"] == "custom_value"
        assert event.data["another_field"] == 42
