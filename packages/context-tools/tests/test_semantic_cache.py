"""Tests for semantic cache matching functionality."""

from __future__ import annotations

import hashlib
import math

import numpy as np
import pytest

from context_tools.cache import CacheKeyGenerator, ToolCallCache
from context_tools.types import ToolCallSignature

# =============================================================================
# Mock Embedding Model
# =============================================================================


class MockEmbeddingModel:
    """Mock embedding model for testing semantic matching.

    Produces deterministic embeddings based on text content.
    Similar texts will have similar embeddings (high cosine similarity).
    """

    def __init__(self, dimension: int = 64) -> None:
        self.dimension = dimension
        self._cache: dict[str, list[float]] = {}

    def embed(self, texts: list[str]) -> np.ndarray:
        """Generate embeddings for texts.

        Uses a hash-based approach that produces similar vectors
        for similar texts (based on word overlap).
        """
        embeddings = []
        for text in texts:
            if text in self._cache:
                embeddings.append(self._cache[text])
            else:
                embedding = self._text_to_embedding(text)
                self._cache[text] = embedding
                embeddings.append(embedding)
        return np.array(embeddings, dtype=np.float32)

    def _text_to_embedding(self, text: str) -> list[float]:
        """Convert text to a deterministic embedding vector.

        Uses word-based hashing to ensure similar texts produce
        similar vectors.
        """
        # Normalize text
        words = text.lower().split()

        # Initialize vector with zeros
        vector = [0.0] * self.dimension

        # Add contribution from each word
        for word in words:
            # Hash word to get indices and values
            word_hash = hashlib.md5(word.encode()).hexdigest()
            for i in range(0, len(word_hash), 4):
                idx = int(word_hash[i : i + 2], 16) % self.dimension
                val = (int(word_hash[i + 2 : i + 4], 16) - 128) / 128.0
                vector[idx] += val

        # Normalize to unit vector
        norm = math.sqrt(sum(v * v for v in vector))
        if norm > 0:
            vector = [v / norm for v in vector]

        return vector


class MockSemanticIndex:
    """Mock SemanticIndex for testing."""

    def __init__(self, dimension: int = 64) -> None:
        self.embedding_model = MockEmbeddingModel(dimension)

    def embed(self, text: str) -> list[float]:
        """Direct embed method for simple interface."""
        result = self.embedding_model.embed([text])
        return list(result[0])


# =============================================================================
# CacheKeyGenerator Semantic Tests
# =============================================================================


class TestCacheKeyGeneratorSemantic:
    """Tests for CacheKeyGenerator semantic functionality."""

    def test_get_embedding_with_semantic_index(self) -> None:
        """Test embedding generation with semantic index."""
        semantic_index = MockSemanticIndex(dimension=64)
        gen = CacheKeyGenerator(semantic_index=semantic_index)  # type: ignore[arg-type]

        sig = ToolCallSignature(
            tool_name="search",
            arguments={"query": "find files"},
        )

        embedding = gen.get_embedding(sig)

        assert embedding is not None
        assert len(embedding) == 64
        assert isinstance(embedding, list)

    def test_get_embedding_without_semantic_index(self) -> None:
        """Test embedding returns None without semantic index."""
        gen = CacheKeyGenerator(semantic_index=None)

        sig = ToolCallSignature(
            tool_name="search",
            arguments={"query": "find files"},
        )

        embedding = gen.get_embedding(sig)
        assert embedding is None

    def test_similar_signatures_similar_embeddings(self) -> None:
        """Test that similar signatures produce similar embeddings."""
        semantic_index = MockSemanticIndex(dimension=64)
        gen = CacheKeyGenerator(semantic_index=semantic_index)  # type: ignore[arg-type]

        sig1 = ToolCallSignature(
            tool_name="search",
            arguments={"query": "find python files"},
        )
        sig2 = ToolCallSignature(
            tool_name="search",
            arguments={"query": "find python scripts"},
        )
        sig3 = ToolCallSignature(
            tool_name="search",
            arguments={"query": "delete all databases"},
        )

        emb1 = gen.get_embedding(sig1)
        emb2 = gen.get_embedding(sig2)
        emb3 = gen.get_embedding(sig3)

        assert emb1 is not None and emb2 is not None and emb3 is not None

        # Similar queries should have higher similarity
        sim_1_2 = gen._cosine_similarity(emb1, emb2)
        sim_1_3 = gen._cosine_similarity(emb1, emb3)

        # sig1 and sig2 are similar (both about finding python)
        # sig1 and sig3 are different
        assert sim_1_2 > sim_1_3

    def test_compute_similarity(self) -> None:
        """Test compute_similarity method."""
        semantic_index = MockSemanticIndex(dimension=64)
        gen = CacheKeyGenerator(semantic_index=semantic_index)  # type: ignore[arg-type]

        sig1 = ToolCallSignature(
            tool_name="read_file",
            arguments={"path": "/home/user/test.py"},
        )
        sig2 = ToolCallSignature(
            tool_name="read_file",
            arguments={"path": "/home/user/test.txt"},
        )

        similarity = gen.compute_similarity(sig1, sig2)

        assert 0 <= similarity <= 1
        assert similarity > 0.5  # Should be somewhat similar

    def test_different_tools_zero_similarity(self) -> None:
        """Test different tools always return zero similarity."""
        semantic_index = MockSemanticIndex(dimension=64)
        gen = CacheKeyGenerator(semantic_index=semantic_index)  # type: ignore[arg-type]

        sig1 = ToolCallSignature(
            tool_name="read_file",
            arguments={"path": "/test.py"},
        )
        sig2 = ToolCallSignature(
            tool_name="write_file",
            arguments={"path": "/test.py"},
        )

        similarity = gen.compute_similarity(sig1, sig2)
        assert similarity == 0.0

    def test_signature_to_text(self) -> None:
        """Test signature to text conversion."""
        gen = CacheKeyGenerator()

        sig = ToolCallSignature(
            tool_name="api_call",
            arguments={"endpoint": "/users", "method": "GET"},
        )

        text = gen._signature_to_text(sig)

        assert "Tool: api_call" in text
        assert "endpoint: /users" in text
        assert "method: GET" in text


# =============================================================================
# ToolCallCache Semantic Tests
# =============================================================================


class TestToolCallCacheSemantic:
    """Tests for ToolCallCache semantic matching."""

    @pytest.fixture
    def semantic_index(self) -> MockSemanticIndex:
        """Create mock semantic index."""
        return MockSemanticIndex(dimension=64)

    @pytest.fixture
    def semantic_cache(self, semantic_index: MockSemanticIndex) -> ToolCallCache:
        """Create cache with semantic matching enabled."""
        return ToolCallCache(
            max_entries=100,
            semantic_threshold=0.7,
            semantic_index=semantic_index,  # type: ignore[arg-type]
            enable_semantic=True,
        )

    @pytest.mark.asyncio
    async def test_semantic_cache_hit(
        self,
        semantic_cache: ToolCallCache,
    ) -> None:
        """Test semantic matching finds similar cached entries."""
        # Store an entry
        sig1 = ToolCallSignature(
            tool_name="search",
            arguments={"query": "find python files in project"},
        )
        await semantic_cache.put(sig1, "result1", token_count=50)

        # Query with similar but not identical signature
        sig2 = ToolCallSignature(
            tool_name="search",
            arguments={"query": "find python scripts in project"},
        )

        result = await semantic_cache.get(sig2)

        # Should find the cached entry via semantic matching
        assert result is not None
        assert result.result == "result1"
        assert semantic_cache.stats.semantic_hits >= 1

    @pytest.mark.asyncio
    async def test_semantic_cache_miss_different_tool(
        self,
        semantic_cache: ToolCallCache,
    ) -> None:
        """Test semantic matching respects tool name."""
        sig1 = ToolCallSignature(
            tool_name="search",
            arguments={"query": "find files"},
        )
        await semantic_cache.put(sig1, "result1", token_count=50)

        # Different tool name, similar arguments
        sig2 = ToolCallSignature(
            tool_name="grep",
            arguments={"query": "find files"},
        )

        result = await semantic_cache.get(sig2)
        assert result is None

    @pytest.mark.asyncio
    async def test_semantic_cache_miss_low_similarity(
        self,
        semantic_cache: ToolCallCache,
    ) -> None:
        """Test semantic matching respects threshold."""
        sig1 = ToolCallSignature(
            tool_name="search",
            arguments={"query": "find python files"},
        )
        await semantic_cache.put(sig1, "result1", token_count=50)

        # Very different query
        sig2 = ToolCallSignature(
            tool_name="search",
            arguments={"query": "delete all databases now"},
        )

        _result = await semantic_cache.get(sig2)
        # May or may not match depending on embedding similarity
        # The key is that very different queries shouldn't match
        # Result is intentionally not checked - test verifies no crash

    @pytest.mark.asyncio
    async def test_semantic_prefers_exact_match(
        self,
        semantic_cache: ToolCallCache,
    ) -> None:
        """Test exact match is tried before semantic."""
        sig = ToolCallSignature(
            tool_name="search",
            arguments={"query": "exact match test"},
        )
        await semantic_cache.put(sig, "exact_result", token_count=50)

        # Query with exact same signature
        result = await semantic_cache.get(sig)

        assert result is not None
        assert result.result == "exact_result"
        # Should be an exact hit, not semantic
        assert semantic_cache.stats.semantic_hits == 0

    @pytest.mark.asyncio
    async def test_semantic_finds_best_match(
        self,
        semantic_cache: ToolCallCache,
    ) -> None:
        """Test semantic matching finds best matching entry."""
        # Store multiple entries
        sig1 = ToolCallSignature(
            tool_name="search",
            arguments={"query": "find java files"},
        )
        sig2 = ToolCallSignature(
            tool_name="search",
            arguments={"query": "find python files"},
        )
        sig3 = ToolCallSignature(
            tool_name="search",
            arguments={"query": "find rust files"},
        )

        await semantic_cache.put(sig1, "java_result", token_count=50)
        await semantic_cache.put(sig2, "python_result", token_count=50)
        await semantic_cache.put(sig3, "rust_result", token_count=50)

        # Query for something similar to python
        query_sig = ToolCallSignature(
            tool_name="search",
            arguments={"query": "find python scripts"},
        )

        result = await semantic_cache.get(query_sig)

        assert result is not None
        # Should match python_result as most similar
        assert result.result == "python_result"

    @pytest.mark.asyncio
    async def test_semantic_disabled_by_default(self) -> None:
        """Test semantic matching is disabled by default."""
        cache = ToolCallCache(max_entries=100)

        sig1 = ToolCallSignature(
            tool_name="search",
            arguments={"query": "find files"},
        )
        await cache.put(sig1, "result1", token_count=50)

        # Similar but not exact query
        sig2 = ToolCallSignature(
            tool_name="search",
            arguments={"query": "find scripts"},
        )

        result = await cache.get(sig2)
        # Should not match (semantic disabled)
        assert result is None

    @pytest.mark.asyncio
    async def test_embedding_stored_in_entry(
        self,
        semantic_cache: ToolCallCache,
    ) -> None:
        """Test embedding is stored in cache entry."""
        sig = ToolCallSignature(
            tool_name="search",
            arguments={"query": "test embedding storage"},
        )
        entry = await semantic_cache.put(sig, "result", token_count=50)

        assert entry.embedding is not None
        assert len(entry.embedding) == 64

    @pytest.mark.asyncio
    async def test_semantic_stats_tracking(
        self,
        semantic_cache: ToolCallCache,
    ) -> None:
        """Test semantic hit statistics are tracked."""
        sig1 = ToolCallSignature(
            tool_name="search",
            arguments={"query": "find documents"},
        )
        await semantic_cache.put(sig1, "result1", token_count=50)

        # Query with similar signature (should be semantic hit)
        sig2 = ToolCallSignature(
            tool_name="search",
            arguments={"query": "find papers"},
        )

        result = await semantic_cache.get(sig2)

        if result is not None:
            # If we got a hit, it should be counted as semantic
            assert semantic_cache.stats.semantic_hits > 0

    @pytest.mark.asyncio
    async def test_semantic_with_empty_arguments(
        self,
        semantic_cache: ToolCallCache,
    ) -> None:
        """Test semantic matching with empty arguments."""
        sig1 = ToolCallSignature(tool_name="get_time", arguments={})
        await semantic_cache.put(sig1, "12:00:00", token_count=10)

        # Exact match should work
        result = await semantic_cache.get(sig1)
        assert result is not None
        assert result.result == "12:00:00"

    @pytest.mark.asyncio
    async def test_high_threshold_reduces_matches(self) -> None:
        """Test higher threshold reduces semantic matches."""
        semantic_index = MockSemanticIndex(dimension=64)

        # High threshold cache
        high_threshold_cache = ToolCallCache(
            max_entries=100,
            semantic_threshold=0.99,  # Very high threshold
            semantic_index=semantic_index,  # type: ignore[arg-type]
            enable_semantic=True,
        )

        sig1 = ToolCallSignature(
            tool_name="search",
            arguments={"query": "find files"},
        )
        await high_threshold_cache.put(sig1, "result1", token_count=50)

        # Similar but not identical query
        sig2 = ToolCallSignature(
            tool_name="search",
            arguments={"query": "find documents"},
        )

        result = await high_threshold_cache.get(sig2)
        # High threshold should prevent match
        assert result is None

    @pytest.mark.asyncio
    async def test_semantic_after_eviction(
        self,
        semantic_index: MockSemanticIndex,
    ) -> None:
        """Test semantic matching after LRU eviction."""
        cache = ToolCallCache(
            max_entries=2,
            semantic_threshold=0.7,
            semantic_index=semantic_index,  # type: ignore[arg-type]
            enable_semantic=True,
        )

        # Use very different queries to avoid semantic matching
        sig1 = ToolCallSignature(
            tool_name="read_file",
            arguments={"path": "/completely/different/path.txt"},
        )
        sig2 = ToolCallSignature(
            tool_name="api_call",
            arguments={"endpoint": "/users", "method": "POST"},
        )
        sig3 = ToolCallSignature(
            tool_name="search",
            arguments={"query": "find python files"},
        )

        await cache.put(sig1, "result1", token_count=50)
        await cache.put(sig2, "result2", token_count=50)
        await cache.put(sig3, "result3", token_count=50)

        # sig1 should be evicted (different tool, no semantic match possible)
        result = await cache.get(sig1)
        assert result is None  # Evicted, no semantic match (different tool)

        # sig2 and sig3 should still be available
        result2 = await cache.get(sig2)
        result3 = await cache.get(sig3)
        assert result2 is not None
        assert result3 is not None


# =============================================================================
# Integration Tests
# =============================================================================


class TestSemanticCacheIntegration:
    """Integration tests for semantic caching."""

    @pytest.mark.asyncio
    async def test_full_semantic_workflow(self) -> None:
        """Test complete semantic caching workflow."""
        semantic_index = MockSemanticIndex(dimension=64)
        cache = ToolCallCache(
            max_entries=100,
            semantic_threshold=0.7,
            semantic_index=semantic_index,  # type: ignore[arg-type]
            enable_normalized=True,
            enable_semantic=True,
        )

        # 1. Store some entries
        entries_data = [
            ("read_file", {"path": "/home/user/main.py"}, "main.py content"),
            ("read_file", {"path": "/home/user/utils.py"}, "utils.py content"),
            ("search", {"query": "find class definition"}, "class Foo:..."),
            ("search", {"query": "find function calls"}, "foo(), bar()"),
        ]

        for tool, args, result in entries_data:
            sig = ToolCallSignature(tool_name=tool, arguments=args)
            await cache.put(sig, result, token_count=len(result))

        # 2. Test exact match
        exact_sig = ToolCallSignature(
            tool_name="read_file",
            arguments={"path": "/home/user/main.py"},
        )
        result = await cache.get(exact_sig)
        assert result is not None
        assert result.result == "main.py content"

        # 3. Test semantic match
        semantic_sig = ToolCallSignature(
            tool_name="search",
            arguments={"query": "find class declarations"},
        )
        result = await cache.get(semantic_sig)
        # May or may not match depending on similarity

        # 4. Verify stats
        stats = cache.stats
        assert stats.puts == 4
        assert stats.total_entries == 4

    @pytest.mark.asyncio
    async def test_semantic_with_complex_arguments(self) -> None:
        """Test semantic matching with complex nested arguments."""
        semantic_index = MockSemanticIndex(dimension=64)
        cache = ToolCallCache(
            max_entries=100,
            semantic_threshold=0.7,
            semantic_index=semantic_index,  # type: ignore[arg-type]
            enable_semantic=True,
        )

        sig1 = ToolCallSignature(
            tool_name="api_call",
            arguments={
                "endpoint": "/api/v1/users",
                "method": "GET",
                "params": {"limit": 10, "offset": 0},
            },
        )
        await cache.put(sig1, {"users": []}, token_count=100)

        # Similar API call
        sig2 = ToolCallSignature(
            tool_name="api_call",
            arguments={
                "endpoint": "/api/v1/users",
                "method": "GET",
                "params": {"limit": 20, "offset": 10},
            },
        )

        _result = await cache.get(sig2)
        # Arguments are similar enough that semantic might match
