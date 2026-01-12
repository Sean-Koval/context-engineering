"""Schema compression strategy.

This strategy extracts common JSON schemas from tool results and replaces
repeated structures with schema references, reducing token usage while
preserving the essential data structure.
"""

from __future__ import annotations

import hashlib
import json
import time
from typing import TYPE_CHECKING, Any
from uuid import UUID

from context_compression.recovery import RecoveryManifest
from context_compression.recovery.operations import CompactOperation
from context_compression.strategies.base import BaseCompressionStrategy
from context_compression.types import CompressionResult, CompressionTier
from context_core.graph.types import CompressionLevel, NodeType

if TYPE_CHECKING:
    from context_core.graph import ContextGraph, ContextNode


class SchemaCompression(BaseCompressionStrategy):
    """Compresses JSON data by extracting common schemas.

    Identifies repeated JSON structures in tool results and replaces
    them with schema references. For example, if multiple tool results
    contain arrays of objects with the same keys, extract the schema
    once and reference it.

    This is COMPACTION tier - structure is preserved but some data
    context may be simplified.

    Configuration:
        min_occurrences: Minimum times a schema must appear to extract (default 3)
        min_array_length: Minimum array length to consider for compression (default 2)

    The strategy is partially reversible - the schema structure is known
    but the original formatting/ordering may be simplified.
    """

    def __init__(
        self,
        min_occurrences: int = 3,
        min_array_length: int = 2,
    ) -> None:
        """Initialize the strategy.

        Args:
            min_occurrences: Minimum times a schema must appear to extract
            min_array_length: Minimum array length to consider for compression
        """
        self._min_occurrences = min_occurrences
        self._min_array_length = min_array_length
        # Store extracted schemas for reference during compression
        self._schemas: dict[str, dict[str, Any]] = {}

    @property
    def _name(self) -> str:
        return "schema_compression"

    @property
    def _tier(self) -> CompressionTier:
        return CompressionTier.COMPACTION

    @property
    def _priority(self) -> int:
        return 10  # Run early in compaction tier

    def _extract_schema(self, data: dict[str, Any] | list[Any]) -> dict[str, Any]:
        """Extract JSON schema from data.

        Generates a simplified schema that describes the structure of the data.

        Args:
            data: The JSON data to extract schema from

        Returns:
            A simplified JSON schema describing the structure
        """
        if isinstance(data, dict):
            properties: dict[str, Any] = {}
            for key, value in data.items():
                properties[key] = self._get_type_schema(value)
            return {"type": "object", "properties": properties}
        elif isinstance(data, list):
            if not data:
                return {"type": "array", "items": {}}
            # Get schema from first item as representative
            item_schema = self._get_type_schema(data[0])
            return {"type": "array", "items": item_schema}
        else:
            return self._get_type_schema(data)

    def _get_type_schema(self, value: Any) -> dict[str, Any]:
        """Get schema for a single value.

        Args:
            value: The value to get schema for

        Returns:
            Schema describing the value type
        """
        if value is None:
            return {"type": "null"}
        elif isinstance(value, bool):
            return {"type": "boolean"}
        elif isinstance(value, int):
            return {"type": "integer"}
        elif isinstance(value, float):
            return {"type": "number"}
        elif isinstance(value, str):
            return {"type": "string"}
        elif isinstance(value, (list, dict)):
            return self._extract_schema(value)
        else:
            return {"type": "unknown"}

    def _compute_schema_hash(self, schema: dict[str, Any]) -> str:
        """Compute a hash for a schema to identify duplicates.

        Args:
            schema: The schema to hash

        Returns:
            A hex string hash of the schema
        """
        # Sort keys for consistent hashing
        schema_str = json.dumps(schema, sort_keys=True)
        return hashlib.sha256(schema_str.encode()).hexdigest()[:16]

    def _get_json_content(self, node: ContextNode) -> dict[str, Any] | list[Any] | None:
        """Extract JSON content from a node.

        Args:
            node: The node to extract content from

        Returns:
            The JSON content if present, None otherwise
        """
        if node.content.tool_output is not None:
            output = node.content.tool_output
            if isinstance(output, (dict, list)):
                return output
            if isinstance(output, str):
                try:
                    parsed = json.loads(output)
                    if isinstance(parsed, (dict, list)):
                        return parsed
                except (json.JSONDecodeError, TypeError):
                    pass
        return None

    def _is_eligible(self, node: ContextNode) -> bool:
        """Check if a node is eligible for schema compression.

        Args:
            node: The node to check

        Returns:
            True if the node can be schema-compressed
        """
        # Must be a tool result
        if node.type != NodeType.TOOL_RESULT:
            return False

        # Must not be already compressed
        if node.compression_level != CompressionLevel.FULL:
            return False

        # Must not be pinned
        if node.metadata.pinned:
            return False

        # Must have JSON content
        content = self._get_json_content(node)
        return content is not None

    def _find_compressible_arrays(
        self, data: dict[str, Any] | list[Any], path: str = ""
    ) -> list[tuple[str, list[Any], dict[str, Any]]]:
        """Find arrays in the data that can be compressed.

        Args:
            data: The data to search
            path: Current path in the data structure

        Returns:
            List of (path, array, item_schema) tuples for compressible arrays
        """
        results: list[tuple[str, list[Any], dict[str, Any]]] = []

        if isinstance(data, list) and len(data) >= self._min_array_length:
            # Check if all items have the same schema (for objects)
            if all(isinstance(item, dict) for item in data):
                # Get schema from first item
                first_schema = self._extract_schema(data[0])
                # Verify all items match
                if all(
                    self._compute_schema_hash(self._extract_schema(item))
                    == self._compute_schema_hash(first_schema)
                    for item in data
                ):
                    results.append((path, data, first_schema))

        elif isinstance(data, dict):
            for key, value in data.items():
                new_path = f"{path}.{key}" if path else key
                if isinstance(value, (dict, list)):
                    results.extend(self._find_compressible_arrays(value, new_path))

        return results

    def _find_schema_occurrences(
        self,
        graph: ContextGraph,
        target_node_ids: set[UUID] | None,
    ) -> dict[str, list[tuple[ContextNode, str, list[Any]]]]:
        """Find nodes with matching schemas across the graph.

        Args:
            graph: The context graph to search
            target_node_ids: Optional set of node IDs to target

        Returns:
            Dict mapping schema_hash to list of (node, path, array) tuples
        """
        schema_occurrences: dict[str, list[tuple[ContextNode, str, list[Any]]]] = {}

        for node in graph:
            if target_node_ids and node.id not in target_node_ids:
                continue

            if not self._is_eligible(node):
                continue

            content = self._get_json_content(node)
            if content is None:
                continue

            # Find compressible arrays in this node's content
            arrays = self._find_compressible_arrays(content)

            for path, array, item_schema in arrays:
                schema_hash = self._compute_schema_hash(item_schema)

                if schema_hash not in schema_occurrences:
                    schema_occurrences[schema_hash] = []
                    self._schemas[schema_hash] = item_schema

                schema_occurrences[schema_hash].append((node, path, array))

        return schema_occurrences

    def _extract_values(
        self, array: list[dict[str, Any]], schema: dict[str, Any]
    ) -> list[list[Any]]:
        """Extract values from array items according to schema.

        Converts array of objects to array of value tuples for compact storage.

        Args:
            array: The array of objects to extract from
            schema: The schema describing the objects

        Returns:
            List of value lists (one per object)
        """
        if schema.get("type") != "object":
            return [[item] for item in array]

        properties = schema.get("properties", {})
        keys = sorted(properties.keys())

        values: list[list[Any]] = []
        for item in array:
            if isinstance(item, dict):
                row = [item.get(key) for key in keys]
                values.append(row)
            else:
                values.append([item])

        return values

    def _compress_node_content(
        self,
        content: dict[str, Any] | list[Any],
        path: str,
        schema_id: str,
        schema: dict[str, Any],
    ) -> dict[str, Any] | list[Any]:
        """Compress content at a specific path using schema reference.

        Args:
            content: The full content to modify
            path: Path to the array to compress
            schema_id: ID to reference the schema
            schema: The schema for the array items

        Returns:
            Modified content with schema reference
        """
        if not path:
            # Content itself is the array
            if isinstance(content, list):
                values = self._extract_values(content, schema)
                keys = sorted(schema.get("properties", {}).keys())
                return {
                    "$schema_ref": schema_id,
                    "keys": keys,
                    "values": values,
                }
            return content

        # Navigate to the path and replace
        parts = path.split(".")
        result = json.loads(json.dumps(content))  # Deep copy

        # Navigate to parent
        current = result
        for part in parts[:-1]:
            if isinstance(current, dict):
                current = current[part]
            elif isinstance(current, list) and part.isdigit():
                current = current[int(part)]

        # Replace the target array
        last_part = parts[-1]
        if isinstance(current, dict) and last_part in current:
            array = current[last_part]
            if isinstance(array, list):
                values = self._extract_values(array, schema)
                keys = sorted(schema.get("properties", {}).keys())
                current[last_part] = {
                    "$schema_ref": schema_id,
                    "keys": keys,
                    "values": values,
                }

        return result

    def _estimate_savings_impl(
        self,
        graph: ContextGraph,
        target_node_ids: list[UUID] | None,
    ) -> int:
        """Estimate tokens that would be saved.

        Args:
            graph: The context graph to analyze
            target_node_ids: Optional list of node IDs to target

        Returns:
            Estimated number of tokens that would be saved
        """
        target_ids = set(target_node_ids) if target_node_ids else None
        schema_occurrences = self._find_schema_occurrences(graph, target_ids)

        total_savings = 0

        for schema_hash, occurrences in schema_occurrences.items():
            if len(occurrences) < self._min_occurrences:
                continue

            schema = self._schemas.get(schema_hash, {})
            properties = schema.get("properties", {})

            for _node, _path, array in occurrences:
                if not array:
                    continue

                # Estimate original tokens (rough: 4 chars per token)
                original_json = json.dumps(array)
                original_tokens = len(original_json) // 4

                # Estimate compressed tokens (schema ref + keys + values)
                values = self._extract_values(array, schema)
                compressed_json = json.dumps(
                    {
                        "$schema_ref": schema_hash,
                        "keys": sorted(properties.keys()),
                        "values": values,
                    }
                )
                compressed_tokens = len(compressed_json) // 4

                savings = max(0, original_tokens - compressed_tokens)
                total_savings += savings

        return total_savings

    def _compress_impl(
        self,
        graph: ContextGraph,
        manifest: RecoveryManifest,
        target_node_ids: list[UUID] | None,
        target_tokens: int | None,
    ) -> CompressionResult:
        """Execute schema compression.

        Args:
            graph: The context graph to compress
            manifest: Recovery manifest to log operations
            target_node_ids: Optional list of node IDs to target
            target_tokens: Stop when this many tokens saved

        Returns:
            CompressionResult with metrics about the operation
        """
        start_time = time.perf_counter()
        target_ids = set(target_node_ids) if target_node_ids else None

        schema_occurrences = self._find_schema_occurrences(graph, target_ids)

        original_tokens = 0
        compressed_tokens = 0
        nodes_compressed = 0
        tokens_saved = 0

        # Track which nodes have been compressed to avoid double-processing
        compressed_nodes: set[UUID] = set()

        for schema_hash, occurrences in schema_occurrences.items():
            if len(occurrences) < self._min_occurrences:
                continue

            # Check target limit
            if target_tokens and tokens_saved >= target_tokens:
                break

            schema = self._schemas.get(schema_hash, {})
            properties = schema.get("properties", {})
            schema_id = f"schema_{schema_hash[:8]}"

            for node, path, array in occurrences:
                # Skip if already compressed
                if node.id in compressed_nodes:
                    continue

                # Check target limit
                if target_tokens and tokens_saved >= target_tokens:
                    break

                if not array:
                    continue

                # Get current content
                content = self._get_json_content(node)
                if content is None:
                    continue

                # Calculate original tokens
                original_json = json.dumps(array)
                node_original_tokens = len(original_json) // 4

                # Compress the content
                compressed_content = self._compress_node_content(
                    content, path, schema_id, schema
                )

                # Calculate compressed tokens
                compressed_json = json.dumps(compressed_content)
                node_compressed_tokens = len(compressed_json) // 4

                # Skip if compression doesn't save tokens
                node_savings = node_original_tokens - node_compressed_tokens
                if node_savings <= 0:
                    continue

                original_tokens += node_original_tokens
                compressed_tokens += node_compressed_tokens

                # Update the node
                node.content.tool_output = compressed_content
                node.compression_level = CompressionLevel.COMPACTED
                node.content.original_tokens = node.token_count
                node.content.compressed_tokens = node_compressed_tokens
                node.token_count = node_compressed_tokens

                # Log to manifest
                preserved_fields = sorted(properties.keys())
                manifest.log_operation(
                    _create_compact_operation(
                        node_id=node.id,
                        original_tokens=node_original_tokens,
                        compressed_tokens=node_compressed_tokens,
                        compaction_method="schema_compression",
                        preserved_fields=preserved_fields,
                    )
                )

                tokens_saved += node_savings
                nodes_compressed += 1
                compressed_nodes.add(node.id)

        return CompressionResult(
            success=True,
            strategy_name=self.name,
            tier=self.tier,
            original_tokens=original_tokens,
            compressed_tokens=compressed_tokens,
            tokens_saved=tokens_saved,
            nodes_processed=len(list(graph)),
            nodes_compressed=nodes_compressed,
            nodes_removed=0,
            nodes_created=0,
            duration_ms=(time.perf_counter() - start_time) * 1000,
            is_recoverable=False,  # Compaction is partially recoverable
        )

    def _can_apply_impl(self, graph: ContextGraph) -> bool:
        """Check if schema compression can be applied.

        Args:
            graph: The context graph to check

        Returns:
            True if there are enough matching schemas to compress
        """
        schema_occurrences = self._find_schema_occurrences(graph, None)

        # Check if any schema has enough occurrences
        for occurrences in schema_occurrences.values():
            if len(occurrences) >= self._min_occurrences:
                return True

        return False

    def get_schema(self, schema_id: str) -> dict[str, Any] | None:
        """Get a stored schema by ID.

        Args:
            schema_id: The schema ID (format: schema_<hash>)

        Returns:
            The schema if found, None otherwise
        """
        # Extract hash from schema_id
        if schema_id.startswith("schema_"):
            short_hash = schema_id[7:]
            # Find matching schema
            for full_hash, schema in self._schemas.items():
                if full_hash.startswith(short_hash):
                    return schema
        return None


def _create_compact_operation(
    node_id: UUID,
    original_tokens: int,
    compressed_tokens: int,
    compaction_method: str,
    preserved_fields: list[str],
) -> CompactOperation:
    """Create a CompactOperation for logging.

    Args:
        node_id: ID of the compressed node
        original_tokens: Token count before compression
        compressed_tokens: Token count after compression
        compaction_method: Method used for compaction
        preserved_fields: Fields that were preserved

    Returns:
        A CompactOperation instance
    """
    return CompactOperation(
        node_id=node_id,
        original_tokens=original_tokens,
        compressed_tokens=compressed_tokens,
        compaction_method=compaction_method,
        preserved_fields=preserved_fields,
        removed_fields=[],
    )
