"""Schema extraction and caching for tool result compression.

Provides schema extraction from structured data and caching with
content-addressable deduplication for significant token savings.

Example:
    >>> extractor = SchemaExtractor()
    >>> cache = SchemaCache()
    >>> data = [{"name": "a", "value": 1}, {"name": "b", "value": 2}]
    >>> schema = extractor.extract(data)
    >>> schema_hash = cache.put(schema)
    >>> compressed = extractor.compress_with_schema(data, schema)
"""

from __future__ import annotations

import hashlib
import json
from typing import TYPE_CHECKING, Any

from context_tools.types import (
    ExtractedSchema,
    SchemaCacheStats,
    SchemaCompressedData,
    SchemaField,
    SchemaFieldType,
)

if TYPE_CHECKING:
    from collections.abc import Sequence


class SchemaExtractor:
    """Extract schemas from structured data.

    Analyzes data structures to extract reusable schemas,
    supporting nested objects, arrays, and optional fields.

    Attributes:
        min_items_for_schema: Minimum items needed to extract schema
        max_sample_values: Maximum sample values to store per field
        max_nesting_depth: Maximum depth for nested schema extraction
    """

    def __init__(
        self,
        min_items_for_schema: int = 2,
        max_sample_values: int = 3,
        max_nesting_depth: int = 5,
    ) -> None:
        """Initialize schema extractor.

        Args:
            min_items_for_schema: Minimum list items to extract schema
            max_sample_values: Max samples to store per field
            max_nesting_depth: Max depth for nested extraction
        """
        self._min_items = min_items_for_schema
        self._max_samples = max_sample_values
        self._max_depth = max_nesting_depth

    def extract(
        self,
        data: Any,
        source_tool: str | None = None,
    ) -> ExtractedSchema | None:
        """Extract schema from structured data.

        Supports lists of objects (most common for API results)
        and single objects.

        Args:
            data: Data to extract schema from
            source_tool: Name of tool that produced this data

        Returns:
            ExtractedSchema if extraction successful, None otherwise
        """
        if isinstance(data, list):
            return self._extract_from_list(data, source_tool)
        elif isinstance(data, dict):
            return self._extract_from_object(data, source_tool)
        return None

    def can_extract(self, data: Any) -> bool:
        """Check if data is suitable for schema extraction.

        Args:
            data: Data to check

        Returns:
            True if schema extraction would be beneficial
        """
        if isinstance(data, list):
            if len(data) < self._min_items:
                return False
            if not all(isinstance(item, dict) for item in data):
                return False
            # Check key consistency (at least 80% overlap)
            if not data:
                return False
            first_keys = set(data[0].keys())
            for item in data[1:]:
                overlap = len(first_keys & set(item.keys()))
                if overlap < len(first_keys) * 0.8:
                    return False
            return True
        elif isinstance(data, dict):
            # Single objects with many keys benefit from schema
            return len(data) >= 3
        return False

    def _extract_from_list(
        self,
        data: list[dict[str, Any]],
        source_tool: str | None,
    ) -> ExtractedSchema | None:
        """Extract schema from list of objects."""
        if not data or not all(isinstance(item, dict) for item in data):
            return None

        # Collect all keys and their types across items
        field_info: dict[str, dict[str, Any]] = {}
        total_items = len(data)

        for item in data:
            for key, value in item.items():
                if key not in field_info:
                    field_info[key] = {
                        "types": set(),
                        "count": 0,
                        "nullable": False,
                        "samples": [],
                        "nested_items": [],
                    }

                field_info[key]["count"] += 1
                field_type = self._infer_type(value)
                field_info[key]["types"].add(field_type)

                if value is None:
                    field_info[key]["nullable"] = True

                # Collect samples and nested items
                non_complex = field_type not in (
                    SchemaFieldType.OBJECT,
                    SchemaFieldType.ARRAY,
                )
                if len(field_info[key]["samples"]) < self._max_samples and non_complex:
                    field_info[key]["samples"].append(value)

                if isinstance(value, dict):
                    field_info[key]["nested_items"].append(value)

        # Build field list
        fields = []
        for name, info in field_info.items():
            field = self._build_field(
                name=name,
                types=info["types"],
                optional=info["count"] < total_items,
                nullable=info["nullable"],
                samples=info["samples"],
                nested_items=info["nested_items"],
                depth=0,
            )
            fields.append(field)

        # Sort fields for consistent hashing
        fields.sort(key=lambda f: f.name)

        schema_hash = self._compute_hash(fields)

        return ExtractedSchema(
            schema_hash=schema_hash,
            fields=fields,
            source_tool=source_tool,
            sample_size=total_items,
        )

    def _extract_from_object(
        self,
        data: dict[str, Any],
        source_tool: str | None,
    ) -> ExtractedSchema | None:
        """Extract schema from single object."""
        if not data:
            return None

        fields = []
        for name, value in data.items():
            field_type = self._infer_type(value)
            samples = (
                [value]
                if field_type not in (SchemaFieldType.OBJECT, SchemaFieldType.ARRAY)
                else []
            )

            nested_items = [value] if isinstance(value, dict) else []

            field = self._build_field(
                name=name,
                types={field_type},
                optional=False,
                nullable=value is None,
                samples=samples[: self._max_samples],
                nested_items=nested_items,
                depth=0,
            )
            fields.append(field)

        fields.sort(key=lambda f: f.name)
        schema_hash = self._compute_hash(fields)

        return ExtractedSchema(
            schema_hash=schema_hash,
            fields=fields,
            source_tool=source_tool,
            sample_size=1,
        )

    def _build_field(
        self,
        name: str,
        types: set[SchemaFieldType],
        optional: bool,
        nullable: bool,
        samples: list[Any],
        nested_items: list[dict[str, Any]],
        depth: int,
    ) -> SchemaField:
        """Build a SchemaField from collected information."""
        # Determine primary type
        types_without_null = types - {SchemaFieldType.NULL}
        if len(types_without_null) == 0:
            field_type = SchemaFieldType.NULL
        elif len(types_without_null) == 1:
            field_type = next(iter(types_without_null))
        else:
            field_type = SchemaFieldType.MIXED

        # Handle nested schemas for objects
        nested_schema = None
        if (
            field_type == SchemaFieldType.OBJECT
            and nested_items
            and depth < self._max_depth
        ):
            nested_schema = self._extract_nested_schema(nested_items, depth + 1)

        # Handle array item types
        item_type = None
        if field_type == SchemaFieldType.ARRAY and samples:
            # Try to infer array item types
            first_array = samples[0] if samples else []
            if isinstance(first_array, list) and first_array:
                item_type = self._infer_type(first_array[0])

        return SchemaField(
            name=name,
            field_type=field_type,
            optional=optional,
            nullable=nullable,
            nested_schema=nested_schema,
            item_type=item_type,
            sample_values=samples[: self._max_samples],
        )

    def _extract_nested_schema(
        self,
        items: list[dict[str, Any]],
        depth: int,
    ) -> list[SchemaField]:
        """Extract schema for nested objects."""
        if not items:
            return []

        # Merge all nested items
        field_info: dict[str, dict[str, Any]] = {}
        total_items = len(items)

        for item in items:
            if not isinstance(item, dict):
                continue
            for key, value in item.items():
                if key not in field_info:
                    field_info[key] = {
                        "types": set(),
                        "count": 0,
                        "nullable": False,
                        "samples": [],
                        "nested_items": [],
                    }

                field_info[key]["count"] += 1
                field_info[key]["types"].add(self._infer_type(value))

                if value is None:
                    field_info[key]["nullable"] = True

                if (
                    len(field_info[key]["samples"]) < self._max_samples
                    and not isinstance(value, (dict, list))
                ):
                    field_info[key]["samples"].append(value)

                if isinstance(value, dict):
                    field_info[key]["nested_items"].append(value)

        fields = []
        for name, info in field_info.items():
            field = self._build_field(
                name=name,
                types=info["types"],
                optional=info["count"] < total_items,
                nullable=info["nullable"],
                samples=info["samples"],
                nested_items=info["nested_items"],
                depth=depth,
            )
            fields.append(field)

        fields.sort(key=lambda f: f.name)
        return fields

    def _infer_type(self, value: Any) -> SchemaFieldType:
        """Infer schema type from a Python value."""
        if value is None:
            return SchemaFieldType.NULL
        elif isinstance(value, bool):
            return SchemaFieldType.BOOLEAN
        elif isinstance(value, int):
            return SchemaFieldType.INTEGER
        elif isinstance(value, float):
            return SchemaFieldType.NUMBER
        elif isinstance(value, str):
            return SchemaFieldType.STRING
        elif isinstance(value, list):
            return SchemaFieldType.ARRAY
        elif isinstance(value, dict):
            return SchemaFieldType.OBJECT
        else:
            return SchemaFieldType.STRING  # Fallback for unknown types

    def _compute_hash(self, fields: Sequence[SchemaField]) -> str:
        """Compute content-addressable hash for fields."""
        # Create a canonical representation
        canonical = []
        for field in fields:
            field_repr = {
                "name": field.name,
                "type": field.field_type.value,
                "optional": field.optional,
                "nullable": field.nullable,
            }
            if field.nested_schema:
                field_repr["nested"] = self._compute_hash(field.nested_schema)
            if field.item_type:
                field_repr["item_type"] = field.item_type.value
            canonical.append(field_repr)

        canonical_str = json.dumps(canonical, sort_keys=True)
        return hashlib.sha256(canonical_str.encode()).hexdigest()[:16]

    def compress_with_schema(
        self,
        data: list[dict[str, Any]],
        schema: ExtractedSchema,
    ) -> SchemaCompressedData:
        """Compress data using extracted schema.

        Converts list of objects to columnar format with schema reference.

        Args:
            data: List of objects matching the schema
            schema: Schema to use for compression

        Returns:
            SchemaCompressedData with values matrix
        """
        keys = schema.field_names
        values = []

        for item in data:
            row = [item.get(key) for key in keys]
            values.append(row)

        return SchemaCompressedData(
            schema_ref=schema.schema_hash,
            keys=keys,
            values=values,
            inline_schema=False,
        )

    def decompress_with_schema(
        self,
        compressed: SchemaCompressedData,
        schema: ExtractedSchema | None = None,
    ) -> list[dict[str, Any]]:
        """Decompress data back to list of objects.

        Args:
            compressed: Compressed data
            schema: Schema for field names (optional if inline)

        Returns:
            Original list of objects
        """
        keys = (
            compressed.keys
            if compressed.keys
            else (schema.field_names if schema else [])
        )

        result = []
        for row in compressed.values:
            item = dict(zip(keys, row, strict=False))
            result.append(item)

        return result


class SchemaCache:
    """Cache for extracted schemas with deduplication.

    Stores schemas by content-addressable hash, enabling
    reuse across tool results and significant token savings.

    Attributes:
        max_schemas: Maximum schemas to cache
    """

    def __init__(self, max_schemas: int = 1000) -> None:
        """Initialize schema cache.

        Args:
            max_schemas: Maximum schemas to store
        """
        self._max_schemas = max_schemas
        self._schemas: dict[str, ExtractedSchema] = {}
        self._stats = SchemaCacheStats()

    def put(self, schema: ExtractedSchema) -> str:
        """Store schema in cache.

        If schema already exists (by hash), increments reference count.
        Returns the schema hash for future lookup.

        Args:
            schema: Schema to store

        Returns:
            Schema hash for reference
        """
        schema_hash = schema.schema_hash

        if schema_hash in self._schemas:
            # Existing schema - update reference count
            self._schemas[schema_hash].touch()
            self._stats.cache_hits += 1
        else:
            # New schema
            if len(self._schemas) >= self._max_schemas:
                self._evict_lru()

            self._schemas[schema_hash] = schema
            self._stats.cache_misses += 1
            self._stats.total_schemas += 1

        self._stats.total_references += 1
        return schema_hash

    def get(self, schema_hash: str) -> ExtractedSchema | None:
        """Retrieve schema by hash.

        Args:
            schema_hash: Hash of schema to retrieve

        Returns:
            Schema if found, None otherwise
        """
        schema = self._schemas.get(schema_hash)
        if schema:
            schema.touch()
        return schema

    def find_matching(
        self,
        data: list[dict[str, Any]],
        extractor: SchemaExtractor,
    ) -> tuple[ExtractedSchema | None, bool]:
        """Find matching cached schema for data.

        Extracts schema from data and checks if it exists in cache.

        Args:
            data: Data to find schema for
            extractor: SchemaExtractor instance

        Returns:
            Tuple of (schema, is_cached) - schema may be new or cached
        """
        schema = extractor.extract(data)
        if schema is None:
            return None, False

        cached = self._schemas.get(schema.schema_hash)
        if cached:
            cached.touch()
            self._stats.cache_hits += 1
            return cached, True
        else:
            self._stats.cache_misses += 1
            return schema, False

    def contains(self, schema_hash: str) -> bool:
        """Check if schema exists in cache."""
        return schema_hash in self._schemas

    def remove(self, schema_hash: str) -> bool:
        """Remove schema from cache.

        Args:
            schema_hash: Hash of schema to remove

        Returns:
            True if schema was removed
        """
        if schema_hash in self._schemas:
            del self._schemas[schema_hash]
            self._stats.total_schemas -= 1
            return True
        return False

    def clear(self) -> None:
        """Clear all cached schemas."""
        self._schemas.clear()
        self._stats = SchemaCacheStats()

    def stats(self) -> SchemaCacheStats:
        """Get cache statistics."""
        return self._stats.model_copy()

    def _evict_lru(self) -> None:
        """Evict least recently used schema."""
        if not self._schemas:
            return

        # Find LRU by last_used timestamp
        lru_hash = min(
            self._schemas.keys(),
            key=lambda h: self._schemas[h].last_used,
        )
        del self._schemas[lru_hash]
        self._stats.total_schemas -= 1

    @property
    def size(self) -> int:
        """Number of schemas in cache."""
        return len(self._schemas)

    def get_all_schemas(self) -> list[ExtractedSchema]:
        """Get all cached schemas."""
        return list(self._schemas.values())

    def get_schemas_for_tool(self, tool_name: str) -> list[ExtractedSchema]:
        """Get all schemas associated with a tool.

        Args:
            tool_name: Name of tool

        Returns:
            List of schemas from that tool
        """
        return [s for s in self._schemas.values() if s.source_tool == tool_name]
