"""Tests for ContextMetrics."""

from __future__ import annotations

from datetime import datetime

from context_observe.metrics import ContextMetrics, MetricSnapshot


class TestMetricSnapshot:
    """Tests for MetricSnapshot model."""

    def test_snapshot_defaults(self) -> None:
        """Test snapshot with default values."""
        snapshot = MetricSnapshot(
            token_usage=0.5,
            compression_ratio=3.0,
            cache_hit_rate=0.6,
            retrieval_latency_p99_ms=50.0,
            active_sessions=2,
            total_compressions=10,
            pre_rot_warnings=0,
        )
        assert snapshot.token_usage == 0.5
        assert snapshot.compression_ratio == 3.0
        assert isinstance(snapshot.timestamp, datetime)

    def test_snapshot_serialization(self) -> None:
        """Test snapshot JSON serialization."""
        snapshot = MetricSnapshot(
            token_usage=0.5,
            compression_ratio=3.0,
            cache_hit_rate=0.6,
            retrieval_latency_p99_ms=50.0,
            active_sessions=2,
            total_compressions=10,
            pre_rot_warnings=0,
        )
        json_str = snapshot.model_dump_json()
        assert "token_usage" in json_str
        assert "0.5" in json_str


class TestContextMetrics:
    """Tests for ContextMetrics."""

    def test_init_without_exporter(self) -> None:
        """Test metrics initialization without exporter."""
        metrics = ContextMetrics()
        assert metrics._meter is not None

    def test_record_token_usage(self) -> None:
        """Test recording token usage."""
        metrics = ContextMetrics()
        metrics.record_token_usage(50000, 100000, "session-1")

        snapshot = metrics.snapshot()
        assert snapshot.token_usage == 0.5

    def test_record_tokens_processed(self) -> None:
        """Test recording tokens processed."""
        metrics = ContextMetrics()
        metrics.record_tokens_processed(1000, "add_message", "session-1")
        # Counter incremented - verify no exception

    def test_record_compression(self) -> None:
        """Test recording compression operation."""
        metrics = ContextMetrics()
        metrics.record_compression("dedupe", 1000, 250, "session-1")

        snapshot = metrics.snapshot()
        assert snapshot.compression_ratio == 4.0  # 1000/250
        assert snapshot.total_compressions == 1

    def test_record_compression_multiple(self) -> None:
        """Test recording multiple compressions."""
        metrics = ContextMetrics()
        metrics.record_compression("dedupe", 1000, 500, "session-1")
        metrics.record_compression("collapse", 500, 100, "session-1")

        snapshot = metrics.snapshot()
        assert snapshot.total_compressions == 2
        assert snapshot.compression_ratio == 5.0  # Last one wins

    def test_record_cache_hit(self) -> None:
        """Test recording cache hit."""
        metrics = ContextMetrics()
        metrics.record_cache_hit("read_file", "session-1")
        metrics.record_cache_hit("read_file", "session-1")
        metrics.record_cache_miss("grep", "session-1")

        snapshot = metrics.snapshot()
        # 2 hits, 1 miss = 2/3 = 0.666...
        assert abs(snapshot.cache_hit_rate - 0.666) < 0.01

    def test_record_cache_miss(self) -> None:
        """Test recording cache miss."""
        metrics = ContextMetrics()
        metrics.record_cache_miss("tool", "session-1")

        snapshot = metrics.snapshot()
        assert snapshot.cache_hit_rate == 0.0

    def test_record_retrieval_latency(self) -> None:
        """Test recording retrieval latency."""
        metrics = ContextMetrics()

        # Record multiple latencies
        for i in range(100):
            latency = 0.01 + (i * 0.001)  # 10ms to 110ms
            metrics.record_retrieval_latency(latency, "semantic", "session-1")

        snapshot = metrics.snapshot()
        # p99 should be close to the 99th percentile
        assert snapshot.retrieval_latency_p99_ms > 90.0

    def test_record_pre_rot_warning(self) -> None:
        """Test recording pre-rot warning."""
        metrics = ContextMetrics()
        metrics.record_pre_rot_warning("session-1", 0.55)
        metrics.record_pre_rot_warning("session-1", 0.58)

        snapshot = metrics.snapshot()
        assert snapshot.pre_rot_warnings == 2

    def test_record_pre_rot_trigger(self) -> None:
        """Test recording pre-rot trigger."""
        metrics = ContextMetrics()
        metrics.record_pre_rot_trigger("session-1", 0.70)
        # No exception raised

    def test_session_started(self) -> None:
        """Test recording session start."""
        metrics = ContextMetrics()
        metrics.session_started("session-1")
        metrics.session_started("session-2")

        snapshot = metrics.snapshot()
        assert snapshot.active_sessions == 2

    def test_session_ended(self) -> None:
        """Test recording session end."""
        metrics = ContextMetrics()
        metrics.session_started("session-1")
        metrics.session_started("session-2")
        metrics.session_ended("session-1")

        snapshot = metrics.snapshot()
        assert snapshot.active_sessions == 1

    def test_session_ended_no_negative(self) -> None:
        """Test that session count doesn't go negative."""
        metrics = ContextMetrics()
        metrics.session_ended("session-1")

        snapshot = metrics.snapshot()
        assert snapshot.active_sessions == 0

    def test_record_graph_operation(self) -> None:
        """Test recording graph operation."""
        metrics = ContextMetrics()
        metrics.record_graph_operation("add_node", "session-1", node_count=50)
        # No exception raised

    def test_snapshot_empty(self) -> None:
        """Test snapshot with no recorded data."""
        metrics = ContextMetrics()
        snapshot = metrics.snapshot()

        assert snapshot.token_usage == 0.0
        assert snapshot.compression_ratio == 1.0
        assert snapshot.cache_hit_rate == 0.0
        assert snapshot.retrieval_latency_p99_ms == 0.0
        assert snapshot.active_sessions == 0
        assert snapshot.total_compressions == 0
        assert snapshot.pre_rot_warnings == 0

    def test_reset_local_stats(self) -> None:
        """Test resetting local statistics."""
        metrics = ContextMetrics()
        metrics.record_token_usage(50000, 100000, "session-1")
        metrics.record_compression("dedupe", 1000, 250, "session-1")

        metrics.reset_local_stats()
        snapshot = metrics.snapshot()

        assert snapshot.token_usage == 0.0
        assert snapshot.total_compressions == 0

    def test_shutdown(self) -> None:
        """Test metrics shutdown."""
        metrics = ContextMetrics()
        metrics.shutdown()  # Should not raise

    def test_thread_safety(self) -> None:
        """Test that metrics are thread-safe."""
        import threading

        metrics = ContextMetrics()
        errors = []

        def record_metrics() -> None:
            try:
                for _ in range(100):
                    metrics.record_cache_hit("tool", "session")
                    metrics.record_cache_miss("tool", "session")
                    metrics.record_compression("test", 100, 50, "session")
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=record_metrics) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        snapshot = metrics.snapshot()
        assert snapshot.total_compressions == 1000  # 10 threads * 100 iterations
