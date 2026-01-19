"""Tool result compression module.

Provides compression strategies for reducing token usage in tool results,
including schema extraction for deduplication of repeated structures.
"""

from __future__ import annotations

from context_tools.compression.compressor import ToolResultCompressor
from context_tools.compression.schema import SchemaCache, SchemaExtractor

__all__ = ["SchemaCache", "SchemaExtractor", "ToolResultCompressor"]
