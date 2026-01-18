"""Tool call caching module.

Provides caching for tool call results with multiple matching strategies:
- Exact matching (identical arguments)
- Normalized matching (path normalization, case insensitivity)
- Semantic matching (embedding-based similarity)

Includes invalidation policies for:
- Time-based expiration (TTL)
- Write-through invalidation (when underlying data changes)
- Version-based invalidation
- Dependency tracking

Example:
    >>> from context_tools.cache import ToolCallCache, CacheKeyGenerator
    >>> cache = ToolCallCache(max_entries=1000)
    >>> sig = ToolCallSignature(tool_name="read_file", arguments={"path": "/test.py"})
    >>> await cache.put(sig, "file content", token_count=50)
    >>> result = await cache.get(sig)
"""

from context_tools.cache.cache import ToolCallCache
from context_tools.cache.keys import CacheKeyGenerator
from context_tools.cache.policies import (
    DEFAULT_POLICIES,
    DependencyTracker,
    InvalidationPolicy,
    InvalidationTrigger,
    PolicyRegistry,
    StalenessCalculator,
)

__all__ = [
    "CacheKeyGenerator",
    "DEFAULT_POLICIES",
    "DependencyTracker",
    "InvalidationPolicy",
    "InvalidationTrigger",
    "PolicyRegistry",
    "StalenessCalculator",
    "ToolCallCache",
]
