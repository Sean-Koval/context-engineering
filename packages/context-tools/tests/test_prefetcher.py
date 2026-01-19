"""Tests for ToolPrefetcher.

Tests cover:
- PrefetchResult and PrefetchStats models
- ToolPrefetcher initialization and configuration
- Prediction triggering on tool completion
- Argument prediction from results and patterns
- Prefetch execution and caching
- Awaiting prefetch results
- Statistics tracking
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from context_tools import (
    PrefetchCandidate,
    PrefetchResult,
    PrefetchStats,
    ToolCallCache,
    ToolCallSignature,
    ToolUsagePatterns,
)
from context_tools.prefetch import ToolPrefetcher


class TestPrefetchResult:
    """Tests for PrefetchResult model."""

    def test_basic_result(self) -> None:
        """Test basic prefetch result creation."""
        result = PrefetchResult(
            tool_name="read_file",
            arguments={"path": "/test.py"},
            success=True,
            cached=True,
            latency_ms=50.0,
        )
        assert result.tool_name == "read_file"
        assert result.success is True
        assert result.cached is True
        assert result.latency_ms == 50.0

    def test_failed_result(self) -> None:
        """Test prefetch result with error."""
        result = PrefetchResult(
            tool_name="read_file",
            arguments={"path": "/missing.py"},
            success=False,
            error="File not found",
        )
        assert result.success is False
        assert result.error == "File not found"

    def test_default_values(self) -> None:
        """Test default values for prefetch result."""
        result = PrefetchResult(tool_name="test")
        assert result.arguments == {}
        assert result.success is True
        assert result.cached is False
        assert result.latency_ms == 0.0
        assert result.error is None
        assert result.result_tokens == 0


class TestPrefetchStats:
    """Tests for PrefetchStats model."""

    def test_basic_stats(self) -> None:
        """Test basic stats creation."""
        stats = PrefetchStats(
            prefetches_started=10,
            prefetches_completed=8,
            prefetches_failed=2,
            prefetch_hits=6,
            prefetch_misses=4,
        )
        assert stats.prefetches_started == 10
        assert stats.prefetches_completed == 8

    def test_hit_rate(self) -> None:
        """Test hit rate calculation."""
        stats = PrefetchStats(prefetch_hits=6, prefetch_misses=4)
        assert stats.hit_rate == pytest.approx(0.6)

    def test_hit_rate_zero_total(self) -> None:
        """Test hit rate with no prefetches."""
        stats = PrefetchStats()
        assert stats.hit_rate == 0.0

    def test_success_rate(self) -> None:
        """Test success rate calculation."""
        stats = PrefetchStats(prefetches_completed=8, prefetches_failed=2)
        assert stats.success_rate == pytest.approx(0.8)

    def test_avg_latency_saved(self) -> None:
        """Test average latency saved calculation."""
        stats = PrefetchStats(
            prefetch_hits=5,
            total_latency_saved_ms=500.0,
        )
        assert stats.avg_latency_saved_ms == pytest.approx(100.0)

    def test_avg_latency_saved_no_hits(self) -> None:
        """Test average latency saved with no hits."""
        stats = PrefetchStats()
        assert stats.avg_latency_saved_ms == 0.0


class TestToolPrefetcher:
    """Tests for ToolPrefetcher class."""

    @pytest.fixture
    def patterns(self) -> ToolUsagePatterns:
        """Create a ToolUsagePatterns instance."""
        return ToolUsagePatterns(min_pattern_frequency=2)

    @pytest.fixture
    def cache(self) -> ToolCallCache:
        """Create a ToolCallCache instance."""
        return ToolCallCache(max_entries=100)

    @pytest.fixture
    def mock_executor(self) -> Any:
        """Create a mock tool executor."""

        async def executor(name: str, args: dict[str, Any]) -> dict[str, Any]:
            """Mock tool executor that returns predictable results."""
            if name == "read_file":
                return {"content": f"Content of {args.get('path', 'unknown')}"}
            if name == "search":
                return [{"path": "/result1.py"}, {"path": "/result2.py"}]
            if name == "list_directory":
                return [{"name": "file1.py"}, {"name": "file2.py"}]
            return {"result": "ok"}

        return executor

    @pytest.fixture
    def prefetcher(
        self,
        patterns: ToolUsagePatterns,
        cache: ToolCallCache,
        mock_executor: Any,
    ) -> ToolPrefetcher:
        """Create a ToolPrefetcher instance."""
        return ToolPrefetcher(
            patterns=patterns,
            cache=cache,
            tool_executor=mock_executor,
            min_confidence=0.3,
        )

    def test_init_defaults(
        self,
        patterns: ToolUsagePatterns,
        cache: ToolCallCache,
        mock_executor: Any,
    ) -> None:
        """Test default initialization."""
        prefetcher = ToolPrefetcher(
            patterns=patterns,
            cache=cache,
            tool_executor=mock_executor,
        )
        assert prefetcher._max_concurrent == 3
        assert prefetcher._min_confidence == 0.3
        assert prefetcher._default_latency_ms == 100.0

    def test_init_custom(
        self,
        patterns: ToolUsagePatterns,
        cache: ToolCallCache,
        mock_executor: Any,
    ) -> None:
        """Test custom initialization."""
        prefetcher = ToolPrefetcher(
            patterns=patterns,
            cache=cache,
            tool_executor=mock_executor,
            max_concurrent_prefetch=5,
            min_confidence=0.5,
            default_latency_ms=200.0,
        )
        assert prefetcher._max_concurrent == 5
        assert prefetcher._min_confidence == 0.5
        assert prefetcher._default_latency_ms == 200.0

    @pytest.mark.asyncio
    async def test_on_tool_complete_records_pattern(
        self,
        prefetcher: ToolPrefetcher,
        patterns: ToolUsagePatterns,
    ) -> None:
        """Test that on_tool_complete records the pattern."""
        sig = ToolCallSignature(tool_name="search", arguments={"query": "test"})
        await prefetcher.on_tool_complete(sig, [])

        assert patterns.history_size == 1

    @pytest.mark.asyncio
    async def test_on_tool_complete_triggers_prefetch(
        self,
        patterns: ToolUsagePatterns,
        cache: ToolCallCache,
        mock_executor: Any,
    ) -> None:
        """Test that on_tool_complete triggers prefetch for predicted tools."""
        # Build up pattern: search -> read_file
        for _ in range(5):
            patterns.record(
                ToolCallSignature(tool_name="search", arguments={"query": "test"})
            )
            patterns.record(
                ToolCallSignature(tool_name="read_file", arguments={"path": "/a.py"})
            )

        prefetcher = ToolPrefetcher(
            patterns=patterns,
            cache=cache,
            tool_executor=mock_executor,
            min_confidence=0.3,
        )

        # Complete a search - should trigger read_file prefetch
        search_sig = ToolCallSignature(tool_name="search", arguments={"query": "test"})
        search_result = [{"path": "/result.py"}]

        results = await prefetcher.on_tool_complete(search_sig, search_result)

        # Should have started a prefetch
        assert len(results) >= 1
        assert results[0].tool_name == "read_file"

    @pytest.mark.asyncio
    async def test_prefetch_caches_result(
        self,
        patterns: ToolUsagePatterns,
        cache: ToolCallCache,
        mock_executor: Any,
    ) -> None:
        """Test that prefetch stores result in cache."""
        # Build pattern
        for _ in range(5):
            patterns.record(
                ToolCallSignature(tool_name="search", arguments={"query": "test"})
            )
            patterns.record(
                ToolCallSignature(tool_name="read_file", arguments={"path": "/a.py"})
            )

        prefetcher = ToolPrefetcher(
            patterns=patterns,
            cache=cache,
            tool_executor=mock_executor,
            min_confidence=0.3,
        )

        # Trigger prefetch
        search_sig = ToolCallSignature(tool_name="search", arguments={"query": "test"})
        search_result = [{"path": "/result.py"}]

        await prefetcher.on_tool_complete(search_sig, search_result)

        # Wait for prefetch to complete
        await asyncio.sleep(0.1)

        # Check cache
        read_sig = ToolCallSignature(
            tool_name="read_file", arguments={"path": "/result.py"}
        )
        cached = await cache.get(read_sig)
        assert cached is not None
        # Result is a dict from mock executor
        assert "content" in cached.result or "Content" in str(cached.result)

    @pytest.mark.asyncio
    async def test_await_prefetch_returns_cached(
        self,
        patterns: ToolUsagePatterns,
        cache: ToolCallCache,
        mock_executor: Any,
    ) -> None:
        """Test await_prefetch returns cached result."""
        # Build pattern
        for _ in range(5):
            patterns.record(
                ToolCallSignature(tool_name="search", arguments={"query": "test"})
            )
            patterns.record(
                ToolCallSignature(tool_name="read_file", arguments={"path": "/a.py"})
            )

        prefetcher = ToolPrefetcher(
            patterns=patterns,
            cache=cache,
            tool_executor=mock_executor,
            min_confidence=0.3,
        )

        # Trigger prefetch
        search_sig = ToolCallSignature(tool_name="search", arguments={"query": "test"})
        await prefetcher.on_tool_complete(search_sig, [{"path": "/result.py"}])

        # Wait for prefetch and then check
        await asyncio.sleep(0.1)

        read_sig = ToolCallSignature(
            tool_name="read_file", arguments={"path": "/result.py"}
        )
        result = await prefetcher.await_prefetch(read_sig, timeout_ms=500)

        assert result is not None

    @pytest.mark.asyncio
    async def test_await_prefetch_timeout(
        self,
        prefetcher: ToolPrefetcher,
    ) -> None:
        """Test await_prefetch returns None on timeout."""
        sig = ToolCallSignature(tool_name="nonexistent", arguments={})
        result = await prefetcher.await_prefetch(sig, timeout_ms=10)
        assert result is None

    @pytest.mark.asyncio
    async def test_stats_tracking(
        self,
        patterns: ToolUsagePatterns,
        cache: ToolCallCache,
        mock_executor: Any,
    ) -> None:
        """Test that statistics are tracked correctly."""
        # Build pattern
        for _ in range(5):
            patterns.record(
                ToolCallSignature(tool_name="search", arguments={"query": "test"})
            )
            patterns.record(
                ToolCallSignature(tool_name="read_file", arguments={"path": "/a.py"})
            )

        prefetcher = ToolPrefetcher(
            patterns=patterns,
            cache=cache,
            tool_executor=mock_executor,
            min_confidence=0.3,
        )

        # Trigger prefetch
        search_sig = ToolCallSignature(tool_name="search", arguments={"query": "test"})
        await prefetcher.on_tool_complete(search_sig, [{"path": "/result.py"}])

        # Wait for completion
        await asyncio.sleep(0.1)

        stats = prefetcher.stats
        assert stats.prefetches_started >= 1

    @pytest.mark.asyncio
    async def test_cancel_pending(
        self,
        prefetcher: ToolPrefetcher,
    ) -> None:
        """Test canceling pending prefetches."""
        # No pending tasks initially
        cancelled = await prefetcher.cancel_pending()
        assert cancelled == 0

    @pytest.mark.asyncio
    async def test_get_candidates(
        self,
        patterns: ToolUsagePatterns,
        cache: ToolCallCache,
        mock_executor: Any,
    ) -> None:
        """Test getting prefetch candidates."""
        # Build pattern
        for _ in range(5):
            patterns.record(
                ToolCallSignature(tool_name="search", arguments={"query": "test"})
            )
            patterns.record(
                ToolCallSignature(tool_name="read_file", arguments={"path": "/a.py"})
            )

        prefetcher = ToolPrefetcher(
            patterns=patterns,
            cache=cache,
            tool_executor=mock_executor,
        )

        candidates = await prefetcher.get_candidates("search", top_k=3)

        assert len(candidates) >= 1
        assert all(isinstance(c, PrefetchCandidate) for c in candidates)
        # read_file should be predicted after search
        assert any(c.tool_name == "read_file" for c in candidates)

    def test_clear_stats(
        self,
        prefetcher: ToolPrefetcher,
    ) -> None:
        """Test clearing statistics."""
        prefetcher._stats.prefetches_started = 10
        prefetcher.clear_stats()
        assert prefetcher.stats.prefetches_started == 0


class TestArgumentPrediction:
    """Tests for argument prediction logic."""

    @pytest.fixture
    def patterns(self) -> ToolUsagePatterns:
        """Create patterns with some history."""
        p = ToolUsagePatterns()
        # Build up parameter patterns
        for i in range(5):
            p.record(
                ToolCallSignature(
                    tool_name="read_file", arguments={"path": f"/common/file{i}.py"}
                )
            )
        return p

    @pytest.fixture
    def prefetcher(
        self,
        patterns: ToolUsagePatterns,
    ) -> ToolPrefetcher:
        """Create prefetcher with patterns."""
        cache = ToolCallCache()

        async def executor(name: str, args: dict[str, Any]) -> dict[str, Any]:
            return {"result": "ok"}

        return ToolPrefetcher(
            patterns=patterns,
            cache=cache,
            tool_executor=executor,
        )

    def test_predict_from_search_result(
        self,
        prefetcher: ToolPrefetcher,
    ) -> None:
        """Test predicting read_file args from search result."""
        search_sig = ToolCallSignature(tool_name="search", arguments={"query": "test"})
        search_result = [{"path": "/found/file.py", "score": 0.9}]

        args = prefetcher._predict_from_result(
            "read_file",
            search_sig,
            search_result,
        )

        assert args is not None
        assert args.get("path") == "/found/file.py"

    def test_predict_from_list_directory(
        self,
        prefetcher: ToolPrefetcher,
    ) -> None:
        """Test predicting read_file args from directory listing."""
        list_sig = ToolCallSignature(
            tool_name="list_directory", arguments={"path": "/src"}
        )
        list_result = [{"name": "main.py"}, {"name": "utils.py"}]

        args = prefetcher._predict_from_result(
            "read_file",
            list_sig,
            list_result,
        )

        assert args is not None
        assert "path" in args

    def test_predict_from_string_list(
        self,
        prefetcher: ToolPrefetcher,
    ) -> None:
        """Test predicting from list of string paths."""
        glob_sig = ToolCallSignature(tool_name="glob", arguments={"pattern": "*.py"})
        glob_result = ["/src/main.py", "/src/utils.py"]

        args = prefetcher._predict_from_result(
            "read_file",
            glob_sig,
            glob_result,
        )

        assert args is not None
        assert args.get("path") == "/src/main.py"

    def test_predict_no_match(
        self,
        prefetcher: ToolPrefetcher,
    ) -> None:
        """Test prediction returns None when no pattern matches."""
        sig = ToolCallSignature(tool_name="unknown", arguments={})
        result = {"data": "something"}

        args = prefetcher._predict_from_result("read_file", sig, result)

        # Should fall through to None since no pattern matches
        assert args is None

    def test_extract_path_from_dict_result(
        self,
        prefetcher: ToolPrefetcher,
    ) -> None:
        """Test extracting path from dict with results key."""
        result = {"results": [{"path": "/nested/file.py"}]}
        path = prefetcher._extract_path_from_result(result)
        assert path == "/nested/file.py"

    def test_extract_path_from_file_key(
        self,
        prefetcher: ToolPrefetcher,
    ) -> None:
        """Test extracting path from 'file' key."""
        result = [{"file": "/using/file/key.py"}]
        path = prefetcher._extract_path_from_result(result)
        assert path == "/using/file/key.py"


class TestPrefetchIntegration:
    """Integration tests for prefetcher with real patterns and cache."""

    @pytest.mark.asyncio
    async def test_full_workflow(self) -> None:
        """Test complete prefetch workflow."""
        patterns = ToolUsagePatterns(min_pattern_frequency=2)
        cache = ToolCallCache()
        executed_tools: list[str] = []

        async def executor(name: str, args: dict[str, Any]) -> dict[str, Any]:
            executed_tools.append(name)
            await asyncio.sleep(0.01)  # Simulate latency
            return {"content": f"Result for {name}"}

        prefetcher = ToolPrefetcher(
            patterns=patterns,
            cache=cache,
            tool_executor=executor,
            min_confidence=0.3,
        )

        # Phase 1: Build up patterns
        for _ in range(5):
            patterns.record(
                ToolCallSignature(tool_name="search", arguments={"query": "test"})
            )
            patterns.record(
                ToolCallSignature(tool_name="read_file", arguments={"path": "/a.py"})
            )

        # Phase 2: Trigger prefetch
        search_sig = ToolCallSignature(
            tool_name="search", arguments={"query": "new_search"}
        )
        search_result = [{"path": "/predicted.py"}]

        await prefetcher.on_tool_complete(search_sig, search_result)

        # Phase 3: Wait and verify
        await asyncio.sleep(0.1)

        # Prefetch should have executed
        assert "read_file" in executed_tools

        # Result should be cached
        read_sig = ToolCallSignature(
            tool_name="read_file", arguments={"path": "/predicted.py"}
        )
        cached = await cache.get(read_sig)
        assert cached is not None

    @pytest.mark.asyncio
    async def test_no_prefetch_for_low_confidence(self) -> None:
        """Test that low confidence predictions don't trigger prefetch."""
        patterns = ToolUsagePatterns(min_pattern_frequency=2)
        cache = ToolCallCache()
        executed_tools: list[str] = []

        async def executor(name: str, args: dict[str, Any]) -> dict[str, Any]:
            executed_tools.append(name)
            return {"result": "ok"}

        prefetcher = ToolPrefetcher(
            patterns=patterns,
            cache=cache,
            tool_executor=executor,
            min_confidence=0.9,  # Very high threshold
        )

        # Create mixed patterns so each transition has 50% confidence
        # search -> read_file (50%) and search -> edit (50%)
        patterns.record(
            ToolCallSignature(tool_name="search", arguments={"query": "test"})
        )
        patterns.record(
            ToolCallSignature(tool_name="read_file", arguments={"path": "/a.py"})
        )
        patterns.record(
            ToolCallSignature(tool_name="search", arguments={"query": "test"})
        )
        patterns.record(
            ToolCallSignature(tool_name="edit", arguments={"path": "/b.py"})
        )

        # Trigger - both transitions have 50% confidence, below 90% threshold
        sig = ToolCallSignature(tool_name="search", arguments={"query": "test"})
        results = await prefetcher.on_tool_complete(sig, [{"path": "/x.py"}])

        # Wait
        await asyncio.sleep(0.1)

        # No prefetch should have happened (both predictions below 0.9 threshold)
        assert len(results) == 0
        assert "read_file" not in executed_tools

    @pytest.mark.asyncio
    async def test_max_concurrent_limit(self) -> None:
        """Test that max concurrent prefetches is respected."""
        patterns = ToolUsagePatterns(min_pattern_frequency=2)
        cache = ToolCallCache()
        active_count = 0
        max_active = 0

        async def slow_executor(name: str, args: dict[str, Any]) -> dict[str, Any]:
            nonlocal active_count, max_active
            active_count += 1
            max_active = max(max_active, active_count)
            await asyncio.sleep(0.1)
            active_count -= 1
            return {"result": "ok"}

        prefetcher = ToolPrefetcher(
            patterns=patterns,
            cache=cache,
            tool_executor=slow_executor,
            max_concurrent_prefetch=2,  # Limit to 2
            min_confidence=0.1,
        )

        # Build patterns for multiple tools
        for _ in range(5):
            patterns.record(ToolCallSignature(tool_name="search", arguments={}))
            patterns.record(ToolCallSignature(tool_name="read_file", arguments={}))
            patterns.record(ToolCallSignature(tool_name="grep", arguments={}))
            patterns.record(ToolCallSignature(tool_name="edit", arguments={}))

        # Trigger
        sig = ToolCallSignature(tool_name="search", arguments={})
        await prefetcher.on_tool_complete(sig, [])

        # Wait for prefetches to complete
        await asyncio.sleep(0.3)

        # Max concurrent should not exceed limit
        assert max_active <= 2

    @pytest.mark.asyncio
    async def test_error_handling(self) -> None:
        """Test that executor errors don't crash prefetcher."""
        patterns = ToolUsagePatterns(min_pattern_frequency=2)
        cache = ToolCallCache()

        async def failing_executor(name: str, args: dict[str, Any]) -> dict[str, Any]:
            raise ValueError("Simulated error")

        prefetcher = ToolPrefetcher(
            patterns=patterns,
            cache=cache,
            tool_executor=failing_executor,
            min_confidence=0.1,
        )

        # Build pattern
        for _ in range(5):
            patterns.record(
                ToolCallSignature(tool_name="search", arguments={"query": "test"})
            )
            patterns.record(
                ToolCallSignature(tool_name="read_file", arguments={"path": "/a.py"})
            )

        # Trigger - should not raise
        sig = ToolCallSignature(tool_name="search", arguments={})
        await prefetcher.on_tool_complete(sig, [{"path": "/x.py"}])

        # Wait for prefetch to fail
        await asyncio.sleep(0.1)

        # Stats should show failure
        stats = prefetcher.stats
        assert stats.prefetches_failed >= 1 or stats.prefetches_started == 0
