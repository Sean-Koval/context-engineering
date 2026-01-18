"""Tool result compression for reducing token usage.

Provides compression strategies including:
- Low-value field filtering
- Long list truncation with summaries
- Schema extraction for repeated structures
- Tool-specific compression rules
"""

from __future__ import annotations

import json
import re
from typing import TYPE_CHECKING, Any

from context_tools.types import CompressionResult

if TYPE_CHECKING:
    from collections.abc import Callable


class ToolResultCompressor:
    """Compress tool results to reduce token usage.

    Applies multiple compression strategies:
    - Field filtering: Remove low-value metadata fields
    - List truncation: Keep first/last N items with summary
    - Schema extraction: Replace repeated structures with schema reference
    - Tool-specific: Apply custom rules per tool type

    Example:
        >>> compressor = ToolResultCompressor()
        >>> result = compressor.compress("search", [{"path": "/a.py"}, ...])
        >>> print(result.compression_ratio)  # e.g., 3.5x compression

    Attributes:
        max_tokens: Target maximum tokens for compressed output
        list_truncate_threshold: Lists longer than this get truncated
        list_keep_items: Number of items to keep when truncating
    """

    def __init__(
        self,
        max_tokens: int = 2000,
        list_truncate_threshold: int = 10,
        list_keep_items: int = 5,
    ) -> None:
        """Initialize compressor with configuration.

        Args:
            max_tokens: Target maximum tokens for output
            list_truncate_threshold: Truncate lists longer than this
            list_keep_items: Number of items to keep from truncated lists
        """
        self._max_tokens = max_tokens
        self._list_threshold = list_truncate_threshold
        self._list_keep = list_keep_items

        # Fields to remove (low value, high token cost)
        self._low_value_fields = {
            "_links",
            "_metadata",
            "_raw",
            "debug",
            "trace",
            "request_id",
            "correlation_id",
            "timestamp_ns",
        }

    def compress(
        self,
        tool_name: str,
        result: Any,
        token_counter: Callable[[str], int] | None = None,
    ) -> CompressionResult:
        """Compress a tool result.

        Args:
            tool_name: Name of the tool that produced the result
            result: The tool result to compress
            token_counter: Optional function to count tokens in a string

        Returns:
            CompressionResult with compressed data and metrics
        """
        original_str = json.dumps(result, default=str)
        original_tokens = self._estimate_tokens(original_str, token_counter)

        # Track what strategies were applied
        strategies_applied: list[str] = []
        metadata: dict[str, Any] = {}

        # Apply compression strategies
        compressed = result

        # 1. Filter low-value fields
        if isinstance(compressed, dict):
            compressed = self._filter_fields(compressed)
            strategies_applied.append("field_filter")

        # 2. Truncate long lists
        compressed, list_truncated = self._truncate_lists(compressed)
        if list_truncated:
            strategies_applied.append("list_truncation")
            metadata["list_truncated"] = True

        # 3. Extract schemas for repeated structures
        if self._should_extract_schema(compressed):
            compressed, schema_extracted = self._extract_schema(compressed)
            if schema_extracted:
                strategies_applied.append("schema_extraction")
                metadata["schema_extracted"] = True

        # 4. Apply tool-specific compression
        tool_compressed = self._tool_specific_compression(tool_name, compressed)
        if tool_compressed is not compressed:
            strategies_applied.append(f"tool:{tool_name}")
            compressed = tool_compressed

        # Calculate compressed size
        compressed_str = json.dumps(compressed, default=str)
        compressed_tokens = self._estimate_tokens(compressed_str, token_counter)

        strategy_name = "+".join(strategies_applied) if strategies_applied else "none"

        return CompressionResult(
            original_tokens=original_tokens,
            compressed_tokens=compressed_tokens,
            compressed_content=compressed,
            strategy_used=strategy_name,
            is_reversible="schema_extraction" not in strategies_applied,
            metadata=metadata,
        )

    def _estimate_tokens(
        self,
        text: str,
        token_counter: Callable[[str], int] | None = None,
    ) -> int:
        """Estimate token count for text."""
        if token_counter:
            return token_counter(text)
        # Rough estimate: ~4 chars per token for JSON
        return len(text) // 4

    def _filter_fields(self, data: dict[str, Any], depth: int = 0) -> dict[str, Any]:
        """Remove low-value fields recursively.

        Args:
            data: Dictionary to filter
            depth: Current recursion depth (max 10)

        Returns:
            Filtered dictionary
        """
        if depth > 10:
            return data

        result: dict[str, Any] = {}
        for key, value in data.items():
            # Skip low-value fields
            if key.lower() in self._low_value_fields:
                continue
            if key.startswith("_") and key not in ("_type", "_id"):
                continue

            # Recurse into nested dicts
            if isinstance(value, dict):
                result[key] = self._filter_fields(value, depth + 1)
            elif isinstance(value, list) and value and isinstance(value[0], dict):
                result[key] = [self._filter_fields(v, depth + 1) for v in value]
            else:
                result[key] = value

        return result

    def _truncate_lists(self, data: Any) -> tuple[Any, bool]:
        """Truncate long lists with summary.

        Args:
            data: Data potentially containing lists

        Returns:
            Tuple of (processed data, whether any truncation occurred)
        """
        truncated = False

        if isinstance(data, list):
            if len(data) > self._list_threshold:
                truncated = True
                keep = self._list_keep // 2
                head = data[:keep]
                tail = data[-keep:]
                omitted = len(data) - 2 * keep
                summary = {
                    "_truncated": True,
                    "_total_items": len(data),
                    "_showing": f"first {keep} and last {keep}",
                    "items": head + [{"...": f"{omitted} more items"}] + tail,
                }
                return summary, truncated

            # Recurse into list items
            result = []
            for item in data:
                compressed, item_truncated = self._truncate_lists(item)
                result.append(compressed)
                truncated = truncated or item_truncated
            return result, truncated

        elif isinstance(data, dict):
            result = {}
            for key, value in data.items():
                compressed, item_truncated = self._truncate_lists(value)
                result[key] = compressed
                truncated = truncated or item_truncated
            return result, truncated

        return data, truncated

    def _should_extract_schema(self, data: Any) -> bool:
        """Check if data would benefit from schema extraction.

        Schema extraction is beneficial for lists of objects with
        identical keys (like database rows or API results).
        """
        if not isinstance(data, list):
            return False
        if len(data) < 3:
            return False
        if not all(isinstance(item, dict) for item in data):
            return False

        # Check if all items have the same keys
        first_keys = set(data[0].keys())
        return all(set(item.keys()) == first_keys for item in data)

    def _extract_schema(
        self, data: list[dict[str, Any]]
    ) -> tuple[dict[str, Any], bool]:
        """Extract schema from list of similar objects.

        Replaces list of dicts with schema + values matrix for
        significant token savings on repeated structures.

        Args:
            data: List of dictionaries with identical keys

        Returns:
            Tuple of (schema-compressed data, success flag)
        """
        if not data:
            return {"_schema": {}, "_keys": [], "_values": []}, True

        first = data[0]
        schema = {key: type(value).__name__ for key, value in first.items()}
        keys = list(first.keys())
        values = [list(item.values()) for item in data]

        return {
            "_schema": schema,
            "_keys": keys,
            "_values": values,
        }, True

    def _tool_specific_compression(self, tool_name: str, data: Any) -> Any:
        """Apply tool-specific compression rules.

        Args:
            tool_name: Name of the tool
            data: Data to compress

        Returns:
            Compressed data
        """
        tool_lower = tool_name.lower()

        if "read" in tool_lower or "file" in tool_lower:
            if isinstance(data, str):
                return self._compress_code(data)

        elif "search" in tool_lower or "grep" in tool_lower or "find" in tool_lower:
            if isinstance(data, list):
                return [self._compress_search_result(r) for r in data]

        elif ("web" in tool_lower or "fetch" in tool_lower) and isinstance(data, str):
            return self._compress_web_content(data)

        return data

    def _compress_code(self, code: str) -> str:
        """Compress code content by removing excessive whitespace."""
        lines = code.split("\n")

        # Remove consecutive blank lines
        compressed_lines: list[str] = []
        prev_blank = False
        for line in lines:
            is_blank = not line.strip()
            if is_blank and prev_blank:
                continue
            compressed_lines.append(line)
            prev_blank = is_blank

        return "\n".join(compressed_lines)

    def _compress_search_result(self, result: Any) -> Any:
        """Compress a search result to essential fields."""
        if not isinstance(result, dict):
            return result

        essential_keys = {"title", "path", "line", "match", "score", "url", "file"}
        essential = {k: v for k, v in result.items() if k in essential_keys}
        return essential if essential else result

    def _compress_web_content(self, content: str) -> str:
        """Compress web content by stripping HTML and collapsing whitespace."""
        # Remove HTML tags
        content = re.sub(r"<[^>]+>", " ", content)
        # Collapse whitespace
        content = re.sub(r"\s+", " ", content)
        # Truncate if too long
        max_len = 5000
        if len(content) > max_len:
            half = max_len // 2
            content = content[:half] + "\n... [truncated] ...\n" + content[-half:]
        return content.strip()
