"""Tool prefetching module.

Provides predictive tool execution based on learned usage patterns
to reduce latency through speculative execution and caching.
"""

from __future__ import annotations

from context_tools.prefetch.prefetcher import ToolPrefetcher

__all__ = ["ToolPrefetcher"]
