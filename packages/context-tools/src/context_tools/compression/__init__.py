"""Tool result compression module.

Provides compression strategies for reducing token usage in tool results,
including schema extraction for deduplication and smart list truncation.
"""

from __future__ import annotations

from context_tools.compression.compressor import ToolResultCompressor
from context_tools.compression.schema import SchemaCache, SchemaExtractor
from context_tools.compression.truncation import ListTruncator

__all__ = ["ListTruncator", "SchemaCache", "SchemaExtractor", "ToolResultCompressor"]
