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


class PrefetchResult(BaseModel):
    """Result of a prefetch operation.

    Tracks the outcome of speculatively executing a predicted tool call,
    including whether it was successful and cache status.

    Attributes:
        tool_name: Name of the prefetched tool
        arguments: Arguments used for prefetch
        success: Whether prefetch executed successfully
        cached: Whether result was stored in cache
        latency_ms: Execution time in milliseconds
        error: Error message if prefetch failed
        result_tokens: Token count of prefetched result
    """

    tool_name: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    success: bool = True
    cached: bool = False
    latency_ms: float = Field(default=0.0, ge=0.0)
    error: str | None = None
    result_tokens: int = Field(default=0, ge=0)


class PrefetchStats(BaseModel):
    """Statistics for prefetch performance monitoring.

    Tracks metrics to evaluate the effectiveness of predictive
    tool prefetching, including hit rates and latency savings.

    Attributes:
        prefetches_started: Total prefetch operations initiated
        prefetches_completed: Prefetches that completed successfully
        prefetches_failed: Prefetches that failed
        prefetch_hits: Times a prefetched result was used
        prefetch_misses: Times prefetch was not available when needed
        total_latency_saved_ms: Estimated latency saved from hits
        pending_count: Currently pending prefetch operations
    """

    prefetches_started: int = Field(default=0, ge=0)
    prefetches_completed: int = Field(default=0, ge=0)
    prefetches_failed: int = Field(default=0, ge=0)
    prefetch_hits: int = Field(default=0, ge=0)
    prefetch_misses: int = Field(default=0, ge=0)
    total_latency_saved_ms: float = Field(default=0.0, ge=0.0)
    pending_count: int = Field(default=0, ge=0)

    @property
    def hit_rate(self) -> float:
        """Calculate prefetch hit rate."""
        total = self.prefetch_hits + self.prefetch_misses
        return self.prefetch_hits / total if total > 0 else 0.0

    @property
    def success_rate(self) -> float:
        """Calculate prefetch success rate."""
        total = self.prefetches_completed + self.prefetches_failed
        return self.prefetches_completed / total if total > 0 else 0.0

    @property
    def avg_latency_saved_ms(self) -> float:
        """Calculate average latency saved per hit."""
        return (
            self.total_latency_saved_ms / self.prefetch_hits
            if self.prefetch_hits > 0
            else 0.0
        )


# Schema extraction types


class SchemaFieldType(str, Enum):
    """JSON Schema field types for schema extraction."""

    STRING = "string"
    NUMBER = "number"
    INTEGER = "integer"
    BOOLEAN = "boolean"
    ARRAY = "array"
    OBJECT = "object"
    NULL = "null"
    MIXED = "mixed"  # Multiple types detected


class SchemaField(BaseModel):
    """A field in an extracted schema.

    Represents a single field with its type information,
    including support for nested structures and optional fields.

    Attributes:
        name: Field name/key
        field_type: Primary type of the field
        optional: Whether field is optional (not in all items)
        nullable: Whether field can be null
        nested_schema: For object types, the nested field definitions
        item_type: For array types, the type of array items
        sample_values: Sample values for context (limited to 3)
    """

    name: str
    field_type: SchemaFieldType
    optional: bool = False
    nullable: bool = False
    nested_schema: list[SchemaField] | None = None
    item_type: SchemaFieldType | None = None
    sample_values: list[Any] = Field(default_factory=list, max_length=3)

    def __hash__(self) -> int:
        """Hash for schema comparison."""
        nested_hash = (
            tuple(hash(f) for f in self.nested_schema) if self.nested_schema else None
        )
        return hash(
            (
                self.name,
                self.field_type,
                self.optional,
                self.nullable,
                nested_hash,
                self.item_type,
            )
        )


class ExtractedSchema(BaseModel):
    """An extracted and cached schema.

    Represents a schema extracted from structured data,
    with content-addressable hashing for deduplication.

    Attributes:
        schema_hash: Content-addressable hash of the schema
        fields: List of field definitions
        source_tool: Tool that produced data with this schema
        ref_count: Number of results using this schema
        created_at: When schema was first extracted
        last_used: When schema was last referenced
        sample_size: Number of items used to infer schema
    """

    schema_hash: str = Field(description="Content-addressable hash")
    fields: list[SchemaField]
    source_tool: str | None = None
    ref_count: int = Field(default=1, ge=0)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    last_used: datetime = Field(default_factory=lambda: datetime.now(UTC))
    sample_size: int = Field(default=0, ge=0)

    def touch(self) -> None:
        """Update usage tracking on schema reference."""
        self.ref_count += 1
        self.last_used = datetime.now(UTC)

    @property
    def field_names(self) -> list[str]:
        """Get ordered list of field names."""
        return [f.name for f in self.fields]

    @property
    def field_count(self) -> int:
        """Get number of fields in schema."""
        return len(self.fields)


class SchemaCompressedData(BaseModel):
    """Data compressed using schema extraction.

    Stores data in a columnar format with a schema reference,
    significantly reducing token usage for repeated structures.

    Attributes:
        schema_ref: Reference to the cached schema hash
        keys: Ordered field names (for inline schema)
        values: List of value tuples matching key order
        inline_schema: Whether schema is embedded or referenced
    """

    schema_ref: str | None = Field(
        default=None, description="Hash reference to cached schema"
    )
    keys: list[str] = Field(default_factory=list, description="Field names in order")
    values: list[list[Any]] = Field(default_factory=list, description="Values matrix")
    inline_schema: bool = Field(
        default=True, description="Whether schema is inline or cached"
    )

    @property
    def item_count(self) -> int:
        """Number of items in compressed data."""
        return len(self.values)


class SchemaCacheStats(BaseModel):
    """Statistics for schema cache monitoring.

    Tracks cache performance including hit rates,
    deduplication effectiveness, and memory savings.

    Attributes:
        total_schemas: Number of unique schemas cached
        total_references: Total times schemas were referenced
        cache_hits: Times an existing schema was reused
        cache_misses: Times a new schema was created
        bytes_saved: Estimated bytes saved from deduplication
        tokens_saved: Estimated tokens saved from deduplication
    """

    total_schemas: int = Field(default=0, ge=0)
    total_references: int = Field(default=0, ge=0)
    cache_hits: int = Field(default=0, ge=0)
    cache_misses: int = Field(default=0, ge=0)
    bytes_saved: int = Field(default=0, ge=0)
    tokens_saved: int = Field(default=0, ge=0)

    @property
    def hit_rate(self) -> float:
        """Calculate cache hit rate."""
        total = self.cache_hits + self.cache_misses
        return self.cache_hits / total if total > 0 else 0.0

    @property
    def dedup_ratio(self) -> float:
        """Calculate deduplication ratio (refs per schema)."""
        if self.total_schemas == 0:
            return 0.0
        return self.total_references / self.total_schemas


# List truncation types


class TruncationStrategy(str, Enum):
    """Strategy for truncating long lists."""

    HEAD_TAIL = "head_tail"  # Keep first N and last N items
    UNIFORM = "uniform"  # Evenly spaced samples across the list
    RESERVOIR = "reservoir"  # Random reservoir sampling
    DIVERSE = "diverse"  # Maximize diversity in selection
    STRATIFIED = "stratified"  # Stratified sampling by type/value


class StatisticalSummary(BaseModel):
    """Statistical summary for numeric lists.

    Provides key statistics when truncating numeric arrays,
    preserving essential information about the distribution.

    Attributes:
        count: Total number of items
        min_value: Minimum value
        max_value: Maximum value
        mean: Arithmetic mean
        std_dev: Standard deviation (None if < 2 items)
        median: Median value
        sum_value: Sum of all values
        percentiles: Key percentiles (25th, 75th)
    """

    count: int = Field(ge=0)
    min_value: float
    max_value: float
    mean: float
    std_dev: float | None = None
    median: float | None = None
    sum_value: float | None = None
    percentiles: dict[int, float] = Field(default_factory=dict)

    @property
    def range(self) -> float:
        """Calculate value range."""
        return self.max_value - self.min_value


class TypeDistribution(BaseModel):
    """Distribution of types in a heterogeneous list.

    Tracks what types of items appear in a list and their counts,
    useful for understanding list composition.

    Attributes:
        type_counts: Count of each type (string, number, object, etc.)
        total_items: Total items analyzed
        is_homogeneous: Whether all items are the same type
        dominant_type: Most common type
    """

    type_counts: dict[str, int] = Field(default_factory=dict)
    total_items: int = Field(default=0, ge=0)

    @property
    def is_homogeneous(self) -> bool:
        """Check if all items are the same type."""
        return len(self.type_counts) <= 1

    @property
    def dominant_type(self) -> str | None:
        """Get the most common type."""
        if not self.type_counts:
            return None
        return max(self.type_counts, key=lambda k: self.type_counts[k])

    @property
    def type_percentages(self) -> dict[str, float]:
        """Get percentage distribution of types."""
        if self.total_items == 0:
            return {}
        return {
            t: count / self.total_items * 100 for t, count in self.type_counts.items()
        }


class TruncationResult(BaseModel):
    """Result of list truncation operation.

    Contains the truncated items along with metadata about
    the truncation including statistics and type information.

    Attributes:
        items: The kept items after truncation
        original_count: Number of items before truncation
        kept_count: Number of items kept
        omitted_count: Number of items omitted
        strategy: Strategy used for truncation
        statistical_summary: Statistics for numeric lists
        type_distribution: Type breakdown for heterogeneous lists
        sample_indices: Original indices of kept items
        is_truncated: Whether truncation actually occurred
    """

    items: list[Any]
    original_count: int = Field(ge=0)
    kept_count: int = Field(ge=0)
    omitted_count: int = Field(ge=0)
    strategy: TruncationStrategy = TruncationStrategy.HEAD_TAIL
    statistical_summary: StatisticalSummary | None = None
    type_distribution: TypeDistribution | None = None
    sample_indices: list[int] = Field(default_factory=list)
    is_truncated: bool = False

    @property
    def compression_ratio(self) -> float:
        """Calculate compression ratio from truncation."""
        if self.kept_count == 0:
            return float("inf") if self.original_count > 0 else 1.0
        return self.original_count / self.kept_count

    def to_compressed_format(self) -> dict[str, Any]:
        """Convert to dictionary format for JSON output."""
        result: dict[str, Any] = {
            "_truncated": self.is_truncated,
            "_total_items": self.original_count,
            "_kept_items": self.kept_count,
            "_strategy": self.strategy.value,
            "items": self.items,
        }

        if self.statistical_summary:
            result["_statistics"] = {
                "min": self.statistical_summary.min_value,
                "max": self.statistical_summary.max_value,
                "mean": self.statistical_summary.mean,
                "std_dev": self.statistical_summary.std_dev,
                "median": self.statistical_summary.median,
            }

        if self.type_distribution and not self.type_distribution.is_homogeneous:
            result["_type_distribution"] = self.type_distribution.type_counts

        return result
