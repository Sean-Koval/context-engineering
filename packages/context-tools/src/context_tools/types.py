"""Shared types for context-tools package.

Provides type definitions used across caching, patterns, compression,
and prefetching modules.
"""

from __future__ import annotations

from datetime import UTC, datetime
from enum import Enum
from typing import Any
from uuid import UUID, uuid4

from pydantic import BaseModel, Field


class CacheKeyType(str, Enum):
    """Type of cache key matching strategy."""

    EXACT = "exact"  # Exact argument match
    NORMALIZED = "normalized"  # Normalized argument match (path normalization, etc.)
    SEMANTIC = "semantic"  # Semantic similarity match via embeddings


class InvalidationReason(str, Enum):
    """Reason for cache entry invalidation."""

    EXPIRED = "expired"  # TTL exceeded
    EVICTED = "evicted"  # LRU eviction due to capacity
    MANUAL = "manual"  # Explicit invalidation
    DEPENDENCY = "dependency"  # Dependent resource changed
    STALE = "stale"  # Staleness score exceeded threshold


class ToolCallSignature(BaseModel):
    """Signature of a tool call for caching.

    Represents the unique identity of a tool call, used for
    cache key generation and matching.

    Attributes:
        tool_name: Name of the tool being called
        arguments: Arguments passed to the tool
        context_hash: Optional hash of relevant context (for context-aware caching)
    """

    tool_name: str = Field(description="Name of the tool")
    arguments: dict[str, Any] = Field(
        default_factory=dict,
        description="Arguments passed to the tool",
    )
    context_hash: str | None = Field(
        default=None,
        description="Hash of relevant context for context-aware caching",
    )

    def __hash__(self) -> int:
        """Hash based on tool name and sorted arguments."""
        args_str = str(sorted(self.arguments.items()))
        return hash((self.tool_name, args_str, self.context_hash))


class CacheEntry(BaseModel):
    """An entry in the tool call cache.

    Stores the cached result along with metadata for cache management
    including TTL, access patterns, and quality metrics.

    Attributes:
        id: Unique entry identifier
        tool_name: Name of the cached tool
        arguments: Original arguments
        result: Cached result value
        result_tokens: Token count of the result
        key_type: Type of key matching used
        key_hash: Hash of the cache key
        embedding: Optional embedding vector for semantic matching
        created_at: When entry was created
        accessed_at: Last access timestamp
        access_count: Number of cache hits
        ttl_seconds: Time-to-live in seconds (None = no expiry)
        is_error: Whether this caches an error result
        staleness_score: Staleness estimate (0 = fresh, 1 = stale)
    """

    id: UUID = Field(default_factory=uuid4)
    tool_name: str
    arguments: dict[str, Any]
    result: Any
    result_tokens: int = Field(ge=0)

    # Key information
    key_type: CacheKeyType = CacheKeyType.EXACT
    key_hash: str
    embedding: list[float] | None = None

    # Timestamps and access tracking
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    accessed_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    access_count: int = Field(default=0, ge=0)
    ttl_seconds: int | None = None

    # Quality metrics
    is_error: bool = False
    staleness_score: float = Field(default=0.0, ge=0.0, le=1.0)

    def is_expired(self) -> bool:
        """Check if entry has expired based on TTL."""
        if self.ttl_seconds is None:
            return False
        age = (datetime.now(UTC) - self.created_at).total_seconds()
        return age > self.ttl_seconds

    def touch(self) -> None:
        """Update access metadata on cache hit."""
        self.accessed_at = datetime.now(UTC)
        self.access_count += 1

    @property
    def age_seconds(self) -> float:
        """Get age of entry in seconds."""
        return (datetime.now(UTC) - self.created_at).total_seconds()


class CacheStats(BaseModel):
    """Statistics for cache monitoring.

    Tracks cache performance metrics including hit rates,
    storage usage, and eviction counts.

    Attributes:
        hits: Number of cache hits
        misses: Number of cache misses
        puts: Number of items stored
        evictions: Number of items evicted
        expirations: Number of items expired
        total_entries: Current number of entries
        total_tokens: Total tokens stored
        semantic_hits: Hits from semantic matching
        normalized_hits: Hits from normalized matching
    """

    hits: int = Field(default=0, ge=0)
    misses: int = Field(default=0, ge=0)
    puts: int = Field(default=0, ge=0)
    evictions: int = Field(default=0, ge=0)
    expirations: int = Field(default=0, ge=0)
    total_entries: int = Field(default=0, ge=0)
    total_tokens: int = Field(default=0, ge=0)
    semantic_hits: int = Field(default=0, ge=0)
    normalized_hits: int = Field(default=0, ge=0)

    @property
    def hit_rate(self) -> float:
        """Calculate cache hit rate."""
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0

    @property
    def exact_hits(self) -> int:
        """Calculate exact match hits."""
        return self.hits - self.semantic_hits - self.normalized_hits

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary with computed metrics."""
        return {
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": self.hit_rate,
            "puts": self.puts,
            "evictions": self.evictions,
            "expirations": self.expirations,
            "total_entries": self.total_entries,
            "total_tokens": self.total_tokens,
            "exact_hits": self.exact_hits,
            "normalized_hits": self.normalized_hits,
            "semantic_hits": self.semantic_hits,
        }


class ToolPattern(BaseModel):
    """A detected tool usage pattern.

    Represents a recurring sequence or pattern in tool usage
    that can be used for prediction and optimization.

    Attributes:
        id: Unique pattern identifier
        sequence: Ordered list of tool names in the pattern
        frequency: Number of times pattern observed
        confidence: Confidence score for pattern (0-1)
        avg_gap_ms: Average time between tools in sequence
        last_seen: When pattern was last observed
        context_tags: Tags describing contexts where pattern appears
    """

    id: UUID = Field(default_factory=uuid4)
    sequence: list[str] = Field(min_length=2, description="Tool names in order")
    frequency: int = Field(default=1, ge=1)
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    avg_gap_ms: float = Field(default=0.0, ge=0.0)
    last_seen: datetime = Field(default_factory=lambda: datetime.now(UTC))
    context_tags: set[str] = Field(default_factory=set)

    def update(self, gap_ms: float) -> None:
        """Update pattern with new observation."""
        # Update running average
        self.avg_gap_ms = (self.avg_gap_ms * self.frequency + gap_ms) / (
            self.frequency + 1
        )
        self.frequency += 1
        self.confidence = min(1.0, self.frequency / 10)  # Saturates at 10 occurrences
        self.last_seen = datetime.now(UTC)


class AntipatternType(str, Enum):
    """Types of inefficient tool usage patterns."""

    REDUNDANT_CALL = "redundant_call"  # Same call repeated unnecessarily
    INEFFICIENT_SEQUENCE = "inefficient_sequence"  # Better ordering exists
    EXCESSIVE_RETRY = "excessive_retry"  # Too many retries
    UNUSED_RESULT = "unused_result"  # Result not used
    OVERSIZED_REQUEST = "oversized_request"  # Request larger than needed


class Antipattern(BaseModel):
    """A detected antipattern in tool usage.

    Represents an inefficient or wasteful pattern that should
    be flagged for optimization.

    Attributes:
        type: Type of antipattern detected
        tool_name: Primary tool involved
        description: Human-readable description
        occurrences: Number of times detected
        tokens_wasted: Estimated tokens wasted
        suggestion: Recommended fix
    """

    type: AntipatternType
    tool_name: str
    description: str
    occurrences: int = Field(default=1, ge=1)
    tokens_wasted: int = Field(default=0, ge=0)
    suggestion: str | None = None


class CompressionResult(BaseModel):
    """Result of compressing a tool result.

    Attributes:
        original_tokens: Token count before compression
        compressed_tokens: Token count after compression
        compression_ratio: Ratio of compression achieved
        strategy_used: Name of compression strategy applied
        is_reversible: Whether original can be recovered
        metadata: Additional compression metadata
    """

    original_tokens: int = Field(ge=0)
    compressed_tokens: int = Field(ge=0)
    compressed_content: Any
    strategy_used: str
    is_reversible: bool = True
    metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def compression_ratio(self) -> float:
        """Calculate compression ratio."""
        if self.original_tokens == 0:
            return 1.0
        return self.original_tokens / self.compressed_tokens

    @property
    def tokens_saved(self) -> int:
        """Calculate tokens saved by compression."""
        return self.original_tokens - self.compressed_tokens


class PrefetchCandidate(BaseModel):
    """A candidate tool call for prefetching.

    Represents a predicted future tool call that may benefit
    from speculative execution.

    Attributes:
        tool_name: Name of tool to prefetch
        predicted_arguments: Predicted arguments
        confidence: Prediction confidence (0-1)
        expected_latency_ms: Expected execution time
        pattern_id: Pattern that suggested this prefetch
    """

    tool_name: str
    predicted_arguments: dict[str, Any] = Field(default_factory=dict)
    confidence: float = Field(ge=0.0, le=1.0)
    expected_latency_ms: float = Field(default=100.0, ge=0.0)
    pattern_id: UUID | None = None

    @property
    def should_prefetch(self) -> bool:
        """Determine if prefetch is worthwhile."""
        # Prefetch if confidence is high enough and latency justifies it
        return self.confidence > 0.7 and self.expected_latency_ms > 50
