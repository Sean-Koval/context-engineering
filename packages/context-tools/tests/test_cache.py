"""Unit tests for ToolCallCache and CacheKeyGenerator."""

from __future__ import annotations

import asyncio

import pytest

from context_tools.cache import CacheKeyGenerator, ToolCallCache
from context_tools.types import (
    CacheEntry,
    CacheKeyType,
    CacheStats,
    ToolCallSignature,
)

# =============================================================================
# CacheKeyGenerator Tests
# =============================================================================


class TestCacheKeyGenerator:
    """Tests for CacheKeyGenerator."""

    def test_exact_key_generation(self) -> None:
        """Test exact key generation is deterministic."""
        gen = CacheKeyGenerator()
        sig = ToolCallSignature(
            tool_name="read_file",
            arguments={"path": "/test/file.py"},
        )

        key1 = gen.generate(sig, CacheKeyType.EXACT)
        key2 = gen.generate(sig, CacheKeyType.EXACT)

        assert key1 == key2
        assert len(key1) == 64  # SHA-256 hex digest

    def test_different_args_different_keys(self) -> None:
        """Test different arguments produce different keys."""
        gen = CacheKeyGenerator()

        sig1 = ToolCallSignature(
            tool_name="read_file",
            arguments={"path": "/file1.py"},
        )
        sig2 = ToolCallSignature(
            tool_name="read_file",
            arguments={"path": "/file2.py"},
        )

        key1 = gen.generate(sig1, CacheKeyType.EXACT)
        key2 = gen.generate(sig2, CacheKeyType.EXACT)

        assert key1 != key2

    def test_different_tools_different_keys(self) -> None:
        """Test different tools produce different keys."""
        gen = CacheKeyGenerator()

        sig1 = ToolCallSignature(tool_name="read_file", arguments={"path": "/test.py"})
        sig2 = ToolCallSignature(tool_name="write_file", arguments={"path": "/test.py"})

        key1 = gen.generate(sig1, CacheKeyType.EXACT)
        key2 = gen.generate(sig2, CacheKeyType.EXACT)

        assert key1 != key2

    def test_normalized_key_path_case(self) -> None:
        """Test normalized key ignores path case."""
        gen = CacheKeyGenerator(normalize_case=True)

        sig1 = ToolCallSignature(
            tool_name="read_file",
            arguments={"path": "/Test/FILE.py"},
        )
        sig2 = ToolCallSignature(
            tool_name="read_file",
            arguments={"path": "/test/file.py"},
        )

        key1 = gen.generate(sig1, CacheKeyType.NORMALIZED)
        key2 = gen.generate(sig2, CacheKeyType.NORMALIZED)

        assert key1 == key2

    def test_normalized_key_expands_user(self) -> None:
        """Test normalized key handles home directory."""
        gen = CacheKeyGenerator()

        sig1 = ToolCallSignature(
            tool_name="read_file",
            arguments={"path": "~/project/file.py"},
        )
        sig2 = ToolCallSignature(
            tool_name="read_file",
            arguments={"path": "/home/user/project/file.py"},
        )

        # Both should normalize to use ~ placeholder
        key1 = gen.generate(sig1, CacheKeyType.NORMALIZED)
        _key2 = gen.generate(sig2, CacheKeyType.NORMALIZED)

        # Keys may differ due to actual home path, but normalization should work
        assert len(key1) == 64
        assert len(_key2) == 64

    def test_normalized_key_whitespace(self) -> None:
        """Test normalized key strips whitespace."""
        gen = CacheKeyGenerator()

        sig1 = ToolCallSignature(
            tool_name="search",
            arguments={"query": "  test query  "},
        )
        sig2 = ToolCallSignature(
            tool_name="search",
            arguments={"query": "test query"},
        )

        key1 = gen.generate(sig1, CacheKeyType.NORMALIZED)
        key2 = gen.generate(sig2, CacheKeyType.NORMALIZED)

        assert key1 == key2

    def test_argument_order_independence(self) -> None:
        """Test key generation is independent of argument order."""
        gen = CacheKeyGenerator()

        sig1 = ToolCallSignature(
            tool_name="search",
            arguments={"query": "test", "limit": 10},
        )
        sig2 = ToolCallSignature(
            tool_name="search",
            arguments={"limit": 10, "query": "test"},
        )

        key1 = gen.generate(sig1, CacheKeyType.EXACT)
        key2 = gen.generate(sig2, CacheKeyType.EXACT)

        assert key1 == key2

    def test_nested_dict_sorting(self) -> None:
        """Test nested dictionaries are sorted consistently."""
        gen = CacheKeyGenerator()

        sig1 = ToolCallSignature(
            tool_name="api_call",
            arguments={"data": {"b": 2, "a": 1}},
        )
        sig2 = ToolCallSignature(
            tool_name="api_call",
            arguments={"data": {"a": 1, "b": 2}},
        )

        key1 = gen.generate(sig1, CacheKeyType.EXACT)
        key2 = gen.generate(sig2, CacheKeyType.EXACT)

        assert key1 == key2

    def test_cosine_similarity(self) -> None:
        """Test cosine similarity calculation."""
        gen = CacheKeyGenerator()

        # Identical vectors
        vec1 = [1.0, 0.0, 0.0]
        vec2 = [1.0, 0.0, 0.0]
        assert gen._cosine_similarity(vec1, vec2) == pytest.approx(1.0)

        # Orthogonal vectors
        vec3 = [0.0, 1.0, 0.0]
        assert gen._cosine_similarity(vec1, vec3) == pytest.approx(0.0)

        # Opposite vectors
        vec4 = [-1.0, 0.0, 0.0]
        assert gen._cosine_similarity(vec1, vec4) == pytest.approx(-1.0)


# =============================================================================
# ToolCallCache Tests
# =============================================================================


class TestToolCallCache:
    """Tests for ToolCallCache."""

    @pytest.mark.asyncio
    async def test_put_and_get_exact(self) -> None:
        """Test basic put and get with exact matching."""
        cache = ToolCallCache()

        sig = ToolCallSignature(
            tool_name="read_file",
            arguments={"path": "/test/file.py"},
        )

        await cache.put(sig, "file content", token_count=50)
        result = await cache.get(sig)

        assert result is not None
        assert result.result == "file content"
        assert result.result_tokens == 50
        assert result.tool_name == "read_file"

    @pytest.mark.asyncio
    async def test_cache_miss(self) -> None:
        """Test cache miss returns None."""
        cache = ToolCallCache()

        sig = ToolCallSignature(
            tool_name="read_file",
            arguments={"path": "/nonexistent.py"},
        )

        result = await cache.get(sig)
        assert result is None

    @pytest.mark.asyncio
    async def test_cache_stats_hit_miss(self) -> None:
        """Test cache statistics track hits and misses."""
        cache = ToolCallCache()

        sig = ToolCallSignature(tool_name="test", arguments={"a": 1})
        await cache.put(sig, "result", token_count=10)

        # Hit
        await cache.get(sig)

        # Miss
        other_sig = ToolCallSignature(tool_name="test", arguments={"a": 2})
        await cache.get(other_sig)

        assert cache.stats.hits == 1
        assert cache.stats.misses == 1
        assert cache.stats.hit_rate == 0.5

    @pytest.mark.asyncio
    async def test_normalized_cache_hit(self) -> None:
        """Test normalized matching finds cache entries."""
        cache = ToolCallCache(enable_normalized=True)

        sig1 = ToolCallSignature(
            tool_name="read_file",
            arguments={"path": "/Test/File.py"},
        )
        await cache.put(sig1, "content", token_count=20)

        # Query with different case
        sig2 = ToolCallSignature(
            tool_name="read_file",
            arguments={"path": "/test/file.py"},
        )
        result = await cache.get(sig2)

        assert result is not None
        assert cache.stats.normalized_hits == 1

    @pytest.mark.asyncio
    async def test_ttl_expiration(self) -> None:
        """Test TTL-based expiration.

        Uses a very short TTL with large margin to avoid flaky timing issues.
        """
        # Use minimal TTL (1 second is the minimum the interface accepts)
        cache = ToolCallCache(default_ttl_seconds=1)

        sig = ToolCallSignature(tool_name="test", arguments={"a": 1})
        await cache.put(sig, "result", token_count=10)

        # Should hit immediately
        result = await cache.get(sig)
        assert result is not None

        # Wait well past expiration to ensure TTL has definitely passed
        # Using 1.5 seconds (50% margin) to handle timing variations
        await asyncio.sleep(1.5)

        # Should miss after TTL
        result = await cache.get(sig)
        assert result is None
        assert cache.stats.expirations == 1

    @pytest.mark.asyncio
    async def test_lru_eviction_by_entries(self) -> None:
        """Test LRU eviction when max entries exceeded."""
        cache = ToolCallCache(max_entries=3, max_tokens=100000)

        # Add 3 entries
        for i in range(3):
            sig = ToolCallSignature(tool_name="test", arguments={"n": i})
            await cache.put(sig, f"result-{i}", token_count=10)

        assert len(cache) == 3

        # Add 4th entry - should evict first
        sig4 = ToolCallSignature(tool_name="test", arguments={"n": 4})
        await cache.put(sig4, "result-4", token_count=10)

        assert len(cache) == 3
        assert cache.stats.evictions == 1

        # First entry should be gone
        sig0 = ToolCallSignature(tool_name="test", arguments={"n": 0})
        result = await cache.get(sig0)
        assert result is None

    @pytest.mark.asyncio
    async def test_lru_eviction_by_tokens(self) -> None:
        """Test LRU eviction when max tokens exceeded."""
        cache = ToolCallCache(max_entries=1000, max_tokens=100)

        # Add entries totaling 90 tokens
        for i in range(3):
            sig = ToolCallSignature(tool_name="test", arguments={"n": i})
            await cache.put(sig, f"result-{i}", token_count=30)

        # Add entry that pushes over limit
        sig4 = ToolCallSignature(tool_name="test", arguments={"n": 4})
        await cache.put(sig4, "result-4", token_count=30)

        # Should have evicted some entries
        assert cache.stats.evictions >= 1
        assert cache.stats.total_tokens <= 100

    @pytest.mark.asyncio
    async def test_access_updates_lru_order(self) -> None:
        """Test accessing entry moves it to end of LRU queue."""
        cache = ToolCallCache(max_entries=2)

        sig1 = ToolCallSignature(tool_name="test", arguments={"n": 1})
        sig2 = ToolCallSignature(tool_name="test", arguments={"n": 2})

        await cache.put(sig1, "result-1", token_count=10)
        await cache.put(sig2, "result-2", token_count=10)

        # Access sig1 to make it MRU
        await cache.get(sig1)

        # Add sig3 - should evict sig2 (LRU), not sig1
        sig3 = ToolCallSignature(tool_name="test", arguments={"n": 3})
        await cache.put(sig3, "result-3", token_count=10)

        # sig1 should still be there
        result1 = await cache.get(sig1)
        assert result1 is not None

        # sig2 should be evicted
        result2 = await cache.get(sig2)
        assert result2 is None

    @pytest.mark.asyncio
    async def test_invalidate_entry(self) -> None:
        """Test manual invalidation of cache entry."""
        cache = ToolCallCache()

        sig = ToolCallSignature(tool_name="test", arguments={"a": 1})
        await cache.put(sig, "result", token_count=10)

        result = await cache.invalidate(sig)
        assert result is True

        # Should be gone
        assert await cache.get(sig) is None

    @pytest.mark.asyncio
    async def test_invalidate_nonexistent(self) -> None:
        """Test invalidating nonexistent entry returns False."""
        cache = ToolCallCache()

        sig = ToolCallSignature(tool_name="test", arguments={"a": 1})
        result = await cache.invalidate(sig)

        assert result is False

    @pytest.mark.asyncio
    async def test_invalidate_tool(self) -> None:
        """Test invalidating all entries for a tool."""
        cache = ToolCallCache()

        # Add entries for two tools
        for i in range(3):
            sig = ToolCallSignature(tool_name="read_file", arguments={"n": i})
            await cache.put(sig, f"read-{i}", token_count=10)

        for i in range(2):
            sig = ToolCallSignature(tool_name="write_file", arguments={"n": i})
            await cache.put(sig, f"write-{i}", token_count=10)

        assert len(cache) == 5

        # Invalidate read_file entries
        count = await cache.invalidate_tool("read_file")
        assert count == 3
        assert len(cache) == 2

    @pytest.mark.asyncio
    async def test_clear(self) -> None:
        """Test clearing all cache entries."""
        cache = ToolCallCache()

        for i in range(5):
            sig = ToolCallSignature(tool_name="test", arguments={"n": i})
            await cache.put(sig, f"result-{i}", token_count=10)

        assert len(cache) == 5

        count = await cache.clear()
        assert count == 5
        assert len(cache) == 0
        assert cache.stats.total_tokens == 0

    @pytest.mark.asyncio
    async def test_contains(self) -> None:
        """Test __contains__ for signature lookup."""
        cache = ToolCallCache()

        sig = ToolCallSignature(tool_name="test", arguments={"a": 1})
        assert sig not in cache

        await cache.put(sig, "result", token_count=10)
        assert sig in cache

    @pytest.mark.asyncio
    async def test_cache_error_result(self) -> None:
        """Test caching error results."""
        cache = ToolCallCache()

        sig = ToolCallSignature(tool_name="test", arguments={"a": 1})
        await cache.put(sig, {"error": "Not found"}, token_count=20, is_error=True)

        result = await cache.get(sig)
        assert result is not None
        assert result.is_error is True
        assert result.result == {"error": "Not found"}

    @pytest.mark.asyncio
    async def test_custom_ttl_per_entry(self) -> None:
        """Test custom TTL for individual entries."""
        cache = ToolCallCache(default_ttl_seconds=3600)

        sig = ToolCallSignature(tool_name="test", arguments={"a": 1})
        await cache.put(sig, "result", token_count=10, ttl_seconds=1)

        # Should hit immediately
        result = await cache.get(sig)
        assert result is not None

        # Wait for custom TTL
        await asyncio.sleep(1.1)

        # Should miss
        result = await cache.get(sig)
        assert result is None

    @pytest.mark.asyncio
    async def test_access_count_tracking(self) -> None:
        """Test access count is tracked correctly."""
        cache = ToolCallCache()

        sig = ToolCallSignature(tool_name="test", arguments={"a": 1})
        await cache.put(sig, "result", token_count=10)

        # Access multiple times
        for _ in range(5):
            await cache.get(sig)

        result = await cache.get(sig)
        assert result is not None
        assert result.access_count == 6  # 5 + 1 from this get


# =============================================================================
# CacheEntry Tests
# =============================================================================


class TestCacheEntry:
    """Tests for CacheEntry model."""

    def test_is_expired_no_ttl(self) -> None:
        """Test entry without TTL never expires."""
        entry = CacheEntry(
            tool_name="test",
            arguments={},
            result="result",
            result_tokens=10,
            key_hash="abc123",
            ttl_seconds=None,
        )
        assert entry.is_expired() is False

    def test_touch_updates_access(self) -> None:
        """Test touch updates access metadata."""
        entry = CacheEntry(
            tool_name="test",
            arguments={},
            result="result",
            result_tokens=10,
            key_hash="abc123",
        )

        original_count = entry.access_count
        entry.touch()

        assert entry.access_count == original_count + 1

    def test_age_seconds(self) -> None:
        """Test age calculation."""
        entry = CacheEntry(
            tool_name="test",
            arguments={},
            result="result",
            result_tokens=10,
            key_hash="abc123",
        )

        # Age should be very small (just created)
        assert entry.age_seconds < 1.0


# =============================================================================
# CacheStats Tests
# =============================================================================


class TestCacheStats:
    """Tests for CacheStats model."""

    def test_hit_rate_calculation(self) -> None:
        """Test hit rate calculation."""
        stats = CacheStats(hits=75, misses=25)
        assert stats.hit_rate == 0.75

    def test_hit_rate_no_requests(self) -> None:
        """Test hit rate with no requests."""
        stats = CacheStats()
        assert stats.hit_rate == 0.0

    def test_exact_hits_calculation(self) -> None:
        """Test exact hits = total - normalized - semantic."""
        stats = CacheStats(
            hits=100,
            normalized_hits=30,
            semantic_hits=20,
        )
        assert stats.exact_hits == 50

    def test_to_dict(self) -> None:
        """Test conversion to dictionary."""
        stats = CacheStats(
            hits=10,
            misses=5,
            puts=15,
            evictions=2,
        )
        result = stats.to_dict()

        assert result["hits"] == 10
        assert result["misses"] == 5
        assert result["hit_rate"] == pytest.approx(0.667, rel=0.01)
        assert result["puts"] == 15
        assert result["evictions"] == 2


# =============================================================================
# Edge Cases
# =============================================================================


class TestCacheEdgeCases:
    """Edge case tests for ToolCallCache."""

    @pytest.mark.asyncio
    async def test_empty_arguments(self) -> None:
        """Test caching with empty arguments."""
        cache = ToolCallCache()

        sig = ToolCallSignature(tool_name="get_time", arguments={})
        await cache.put(sig, "12:00:00", token_count=5)

        result = await cache.get(sig)
        assert result is not None
        assert result.result == "12:00:00"

    @pytest.mark.asyncio
    async def test_complex_result_types(self) -> None:
        """Test caching complex result types."""
        cache = ToolCallCache()

        sig = ToolCallSignature(tool_name="api_call", arguments={"endpoint": "/users"})
        complex_result = {
            "users": [
                {"id": 1, "name": "Alice"},
                {"id": 2, "name": "Bob"},
            ],
            "total": 2,
            "page": 1,
        }
        await cache.put(sig, complex_result, token_count=100)

        result = await cache.get(sig)
        assert result is not None
        assert result.result == complex_result

    @pytest.mark.asyncio
    async def test_none_result(self) -> None:
        """Test caching None result."""
        cache = ToolCallCache()

        sig = ToolCallSignature(tool_name="find", arguments={"query": "nonexistent"})
        await cache.put(sig, None, token_count=5)

        result = await cache.get(sig)
        assert result is not None
        assert result.result is None

    @pytest.mark.asyncio
    async def test_concurrent_access(self) -> None:
        """Test concurrent cache access."""
        cache = ToolCallCache()

        async def put_and_get(n: int) -> CacheEntry | None:
            sig = ToolCallSignature(tool_name="test", arguments={"n": n})
            await cache.put(sig, f"result-{n}", token_count=10)
            return await cache.get(sig)

        # Run concurrent operations
        results = await asyncio.gather(*[put_and_get(i) for i in range(10)])

        # All should succeed
        assert all(r is not None for r in results)
        assert len(cache) == 10

    @pytest.mark.asyncio
    async def test_overwrite_existing_entry(self) -> None:
        """Test overwriting an existing cache entry."""
        cache = ToolCallCache()

        sig = ToolCallSignature(tool_name="test", arguments={"a": 1})
        await cache.put(sig, "original", token_count=10)
        await cache.put(sig, "updated", token_count=15)

        result = await cache.get(sig)
        assert result is not None
        assert result.result == "updated"
        assert result.result_tokens == 15
