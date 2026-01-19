"""Context-Tools - Tool call optimization for ContextEngine.

Provides caching, pattern detection, compression, and prefetching
to optimize tool call handling in LLM-powered agents.

Modules:
    cache: Tool call caching with exact, normalized, and semantic matching
    patterns: Tool usage pattern detection and antipattern identification
    compression: Tool result compression strategies
    prefetch: Predictive tool prefetching

Example:
    >>> from context_tools import ToolCallCache, ToolCallSignature
    >>> cache = ToolCallCache(max_entries=1000)
    >>> sig = ToolCallSignature(tool_name="read_file", arguments={"path": "/test.py"})
    >>> await cache.put(sig, "file content", token_count=50)
    >>> result = await cache.get(sig)
"""

from __future__ import annotations

from context_tools.cache import (
    DEFAULT_POLICIES,
    CacheKeyGenerator,
    DependencyTracker,
    InvalidationPolicy,
    InvalidationTrigger,
    PolicyRegistry,
    StalenessCalculator,
    ToolCallCache,
)
from context_tools.compression import SchemaCache, SchemaExtractor, ToolResultCompressor
from context_tools.patterns import (
    ParameterStats,
    ToolSequence,
    ToolUsagePatterns,
    UsageStats,
)
from context_tools.types import (
    Antipattern,
    AntipatternType,
    CacheEntry,
    CacheKeyType,
    CacheStats,
    CompressionResult,
    ExtractedSchema,
    InvalidationReason,
    PrefetchCandidate,
    SchemaCacheStats,
    SchemaCompressedData,
    SchemaField,
    SchemaFieldType,
    ToolCallSignature,
    ToolPattern,
)

__version__ = "0.1.0"

__all__ = [
    "__version__",
    # Cache
    "CacheKeyGenerator",
    "DEFAULT_POLICIES",
    "DependencyTracker",
    "InvalidationPolicy",
    "InvalidationTrigger",
    "PolicyRegistry",
    "StalenessCalculator",
    "ToolCallCache",
    # Compression
    "SchemaCache",
    "SchemaExtractor",
    "ToolResultCompressor",
    # Patterns
    "ParameterStats",
    "ToolSequence",
    "ToolUsagePatterns",
    "UsageStats",
    # Types
    "Antipattern",
    "AntipatternType",
    "CacheEntry",
    "CacheKeyType",
    "CacheStats",
    "CompressionResult",
    "ExtractedSchema",
    "InvalidationReason",
    "PrefetchCandidate",
    "SchemaCompressedData",
    "SchemaCacheStats",
    "SchemaField",
    "SchemaFieldType",
    "ToolCallSignature",
    "ToolPattern",
]
