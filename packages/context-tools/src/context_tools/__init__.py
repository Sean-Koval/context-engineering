"""Context Tools: Tool call caching, usage patterns, and result compression.

This package provides optimization strategies for tool calls in LLM-powered agents:
- ToolCallCache: Exact and semantic caching of tool results
- CacheKeyGenerator: Deterministic and normalized key generation
- ToolUsagePatterns: Pattern detection and next-tool prediction (planned)
- ToolResultCompressor: Schema extraction and list truncation (planned)
- ToolPrefetcher: Predictive tool execution (planned)
"""

from __future__ import annotations

__version__ = "0.1.0"

__all__: list[str] = [
    "__version__",
]
