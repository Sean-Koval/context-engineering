"""ToolCallCache - Caching for tool call results.

Provides multi-strategy caching for tool results with:
- Exact matching for identical calls
- Normalized matching for path/case variations
- Semantic matching for similar queries
- LRU eviction and TTL-based expiration
"""

from __future__ import annotations

import asyncio
from collections import OrderedDict
from typing import TYPE_CHECKING, Any

from context_tools.cache.keys import CacheKeyGenerator
from context_tools.types import (
    CacheEntry,
    CacheKeyType,
    CacheStats,
    InvalidationReason,
    ToolCallSignature,
)

if TYPE_CHECKING:
    from context_core.semantic import SemanticIndex


class ToolCallCache:
    """Cache for tool call results with multiple matching strategies.

    Provides efficient caching of tool results with support for:
    - **Exact matching**: Identical tool name and arguments
    - **Normalized matching**: Path normalization, case insensitivity
    - **Semantic matching**: Embedding-based similarity for fuzzy matches
    - **TTL expiration**: Automatic expiration of stale entries
    - **LRU eviction**: Capacity-based eviction of least-used entries

    Example:
        >>> cache = ToolCallCache(max_entries=1000, default_ttl_seconds=3600)
        >>> sig = ToolCallSignature(tool_name="read_file", arguments={"path": "x"})
        >>> await cache.put(sig, "file content", token_count=50)
        >>> result = await cache.get(sig)
        >>> if result:
        ...     print(f"Cache hit! Result: {result.result}")
        >>> print(f"Hit rate: {cache.stats.hit_rate:.2%}")

    Attributes:
        stats: Cache statistics including hit rate and eviction counts
    """

    def __init__(
        self,
        max_entries: int = 1000,
        max_tokens: int = 100000,
        default_ttl_seconds: int | None = None,
        semantic_threshold: float = 0.85,
        semantic_index: SemanticIndex | None = None,
        enable_normalized: bool = True,
        enable_semantic: bool = False,
    ) -> None:
        """Initialize ToolCallCache.

        Args:
            max_entries: Maximum number of cache entries
            max_tokens: Maximum total tokens to cache
            default_ttl_seconds: Default TTL for entries (None = no expiry)
            semantic_threshold: Minimum similarity for semantic matches
            semantic_index: Optional semantic index for embedding-based matching
            enable_normalized: Enable normalized key matching
            enable_semantic: Enable semantic similarity matching
        """
        self._max_entries = max_entries
        self._max_tokens = max_tokens
        self._default_ttl = default_ttl_seconds
        self._semantic_threshold = semantic_threshold
        self._enable_normalized = enable_normalized
        self._enable_semantic = enable_semantic and semantic_index is not None

        # Key generator
        self._key_gen = CacheKeyGenerator(semantic_index=semantic_index)

        # Primary cache: exact key -> entry
        self._exact_cache: OrderedDict[str, CacheEntry] = OrderedDict()

        # Secondary indices for normalized and semantic matching
        self._normalized_index: dict[str, set[str]] = {}  # normalized_key -> exact_keys
        self._tool_index: dict[str, set[str]] = {}  # tool_name -> exact_keys

        # Embeddings for semantic matching
        self._embeddings: dict[str, list[float]] = {}  # exact_key -> embedding

        # Statistics
        self._stats = CacheStats()
        self._current_tokens = 0

        # Lock for thread safety
        self._lock = asyncio.Lock()

    @property
    def stats(self) -> CacheStats:
        """Get cache statistics."""
        self._stats.total_entries = len(self._exact_cache)
        self._stats.total_tokens = self._current_tokens
        return self._stats

    async def get(
        self,
        signature: ToolCallSignature,
    ) -> CacheEntry | None:
        """Look up a cached result for a tool call.

        Tries matching strategies in order:
        1. Exact match (fastest)
        2. Normalized match (if enabled)
        3. Semantic match (if enabled, slowest)

        Args:
            signature: Tool call signature to look up

        Returns:
            CacheEntry if found and valid, None otherwise
        """
        async with self._lock:
            # Try exact match first
            exact_key = self._key_gen.generate(signature, CacheKeyType.EXACT)
            entry = self._exact_cache.get(exact_key)

            if entry is not None:
                if entry.is_expired():
                    await self._remove_entry(exact_key, InvalidationReason.EXPIRED)
                    self._stats.misses += 1
                    return None
                entry.touch()
                self._exact_cache.move_to_end(exact_key)
                self._stats.hits += 1
                return entry

            # Try normalized match
            if self._enable_normalized:
                entry = await self._get_normalized(signature)
                if entry is not None:
                    self._stats.hits += 1
                    self._stats.normalized_hits += 1
                    return entry

            # Try semantic match
            if self._enable_semantic:
                entry = await self._get_semantic(signature)
                if entry is not None:
                    self._stats.hits += 1
                    self._stats.semantic_hits += 1
                    return entry

            self._stats.misses += 1
            return None

    async def put(
        self,
        signature: ToolCallSignature,
        result: Any,
        token_count: int,
        ttl_seconds: int | None = None,
        is_error: bool = False,
    ) -> CacheEntry:
        """Store a tool call result in the cache.

        Args:
            signature: Tool call signature
            result: Result to cache
            token_count: Token count of result
            ttl_seconds: TTL for this entry (None = use default)
            is_error: Whether this caches an error result

        Returns:
            The created CacheEntry
        """
        async with self._lock:
            # Generate keys
            exact_key = self._key_gen.generate(signature, CacheKeyType.EXACT)
            normalized_key = self._key_gen.generate(signature, CacheKeyType.NORMALIZED)

            # Evict if necessary
            await self._ensure_capacity(token_count)

            # Get embedding for semantic matching (if enabled)
            embedding: list[float] | None = None
            if self._enable_semantic:
                embedding = self._key_gen.get_embedding(signature)

            # Create entry
            entry = CacheEntry(
                tool_name=signature.tool_name,
                arguments=signature.arguments,
                result=result,
                result_tokens=token_count,
                key_type=CacheKeyType.EXACT,
                key_hash=exact_key,
                embedding=embedding,
                ttl_seconds=ttl_seconds
                if ttl_seconds is not None
                else self._default_ttl,
                is_error=is_error,
            )

            # Store in primary cache
            self._exact_cache[exact_key] = entry
            self._exact_cache.move_to_end(exact_key)
            self._current_tokens += token_count
            self._stats.puts += 1

            # Update indices
            if self._enable_normalized:
                if normalized_key not in self._normalized_index:
                    self._normalized_index[normalized_key] = set()
                self._normalized_index[normalized_key].add(exact_key)

            if signature.tool_name not in self._tool_index:
                self._tool_index[signature.tool_name] = set()
            self._tool_index[signature.tool_name].add(exact_key)

            # Store embedding in lookup index for semantic matching
            if embedding is not None:
                self._embeddings[exact_key] = embedding

            return entry

    async def invalidate(
        self,
        signature: ToolCallSignature,
        reason: InvalidationReason = InvalidationReason.MANUAL,
    ) -> bool:
        """Invalidate a cached entry.

        Args:
            signature: Signature of entry to invalidate
            reason: Reason for invalidation

        Returns:
            True if entry was found and removed
        """
        async with self._lock:
            exact_key = self._key_gen.generate(signature, CacheKeyType.EXACT)
            if exact_key in self._exact_cache:
                await self._remove_entry(exact_key, reason)
                return True
            return False

    async def invalidate_tool(
        self,
        tool_name: str,
        reason: InvalidationReason = InvalidationReason.MANUAL,
    ) -> int:
        """Invalidate all cached entries for a tool.

        Args:
            tool_name: Name of tool to invalidate
            reason: Reason for invalidation

        Returns:
            Number of entries invalidated
        """
        async with self._lock:
            keys = list(self._tool_index.get(tool_name, set()))
            for key in keys:
                await self._remove_entry(key, reason)
            return len(keys)

    async def clear(self) -> int:
        """Clear all cache entries.

        Returns:
            Number of entries cleared
        """
        async with self._lock:
            count = len(self._exact_cache)
            self._exact_cache.clear()
            self._normalized_index.clear()
            self._tool_index.clear()
            self._embeddings.clear()
            self._current_tokens = 0
            return count

    async def _get_normalized(
        self,
        signature: ToolCallSignature,
    ) -> CacheEntry | None:
        """Look up entry using normalized key."""
        normalized_key = self._key_gen.generate(signature, CacheKeyType.NORMALIZED)
        exact_keys = self._normalized_index.get(normalized_key, set())

        for exact_key in exact_keys:
            entry = self._exact_cache.get(exact_key)
            if entry is not None:
                if entry.is_expired():
                    await self._remove_entry(exact_key, InvalidationReason.EXPIRED)
                    continue
                entry.touch()
                self._exact_cache.move_to_end(exact_key)
                return entry

        return None

    async def _get_semantic(
        self,
        signature: ToolCallSignature,
    ) -> CacheEntry | None:
        """Look up entry using semantic similarity."""
        query_embedding = self._key_gen.get_embedding(signature)
        if query_embedding is None:
            return None

        # Only search entries for the same tool
        candidate_keys = self._tool_index.get(signature.tool_name, set())
        if not candidate_keys:
            return None

        best_entry: CacheEntry | None = None
        best_similarity = 0.0

        for exact_key in candidate_keys:
            entry_embedding = self._embeddings.get(exact_key)
            if entry_embedding is None:
                continue

            similarity = self._key_gen._cosine_similarity(
                query_embedding, entry_embedding
            )
            if similarity > best_similarity and similarity >= self._semantic_threshold:
                entry = self._exact_cache.get(exact_key)
                if entry is not None and not entry.is_expired():
                    best_entry = entry
                    best_similarity = similarity

        if best_entry is not None:
            best_entry.touch()
            self._exact_cache.move_to_end(best_entry.key_hash)

        return best_entry

    async def _ensure_capacity(self, new_tokens: int) -> None:
        """Ensure cache has capacity for new entry."""
        # Evict until we have room
        while (
            len(self._exact_cache) >= self._max_entries
            or self._current_tokens + new_tokens > self._max_tokens
        ):
            if not self._exact_cache:
                break
            await self._evict_one()

    async def _evict_one(self) -> None:
        """Evict the least recently used entry."""
        if not self._exact_cache:
            return

        # Get LRU entry (first in OrderedDict)
        exact_key = next(iter(self._exact_cache))
        await self._remove_entry(exact_key, InvalidationReason.EVICTED)

    async def _remove_entry(
        self,
        exact_key: str,
        reason: InvalidationReason,
    ) -> None:
        """Remove an entry from all indices."""
        entry = self._exact_cache.pop(exact_key, None)
        if entry is None:
            return

        # Update token count
        self._current_tokens -= entry.result_tokens

        # Remove from normalized index
        normalized_key = self._key_gen.generate(
            ToolCallSignature(
                tool_name=entry.tool_name,
                arguments=entry.arguments,
            ),
            CacheKeyType.NORMALIZED,
        )
        if normalized_key in self._normalized_index:
            self._normalized_index[normalized_key].discard(exact_key)
            if not self._normalized_index[normalized_key]:
                del self._normalized_index[normalized_key]

        # Remove from tool index
        if entry.tool_name in self._tool_index:
            self._tool_index[entry.tool_name].discard(exact_key)
            if not self._tool_index[entry.tool_name]:
                del self._tool_index[entry.tool_name]

        # Remove embedding
        self._embeddings.pop(exact_key, None)

        # Update stats
        if reason == InvalidationReason.EVICTED:
            self._stats.evictions += 1
        elif reason == InvalidationReason.EXPIRED:
            self._stats.expirations += 1

    def __len__(self) -> int:
        """Return number of cached entries."""
        return len(self._exact_cache)

    def __contains__(self, signature: ToolCallSignature) -> bool:
        """Check if signature is in cache (exact match only)."""
        exact_key = self._key_gen.generate(signature, CacheKeyType.EXACT)
        entry = self._exact_cache.get(exact_key)
        return entry is not None and not entry.is_expired()
