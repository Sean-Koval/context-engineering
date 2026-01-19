"""Tests for schema extraction and caching.

Tests cover:
- SchemaExtractor: Type inference, extraction from lists/objects, nested schemas
- SchemaCache: Deduplication, LRU eviction, statistics
- Integration with ToolResultCompressor
"""

from __future__ import annotations

import pytest

from context_tools import (
    ExtractedSchema,
    SchemaCache,
    SchemaCacheStats,
    SchemaCompressedData,
    SchemaExtractor,
    SchemaField,
    SchemaFieldType,
    ToolResultCompressor,
)


class TestSchemaFieldType:
    """Tests for SchemaFieldType enum."""

    def test_enum_values(self) -> None:
        """Test all enum values exist."""
        assert SchemaFieldType.STRING == "string"
        assert SchemaFieldType.NUMBER == "number"
        assert SchemaFieldType.INTEGER == "integer"
        assert SchemaFieldType.BOOLEAN == "boolean"
        assert SchemaFieldType.ARRAY == "array"
        assert SchemaFieldType.OBJECT == "object"
        assert SchemaFieldType.NULL == "null"
        assert SchemaFieldType.MIXED == "mixed"


class TestSchemaField:
    """Tests for SchemaField model."""

    def test_basic_field(self) -> None:
        """Test basic field creation."""
        field = SchemaField(name="test", field_type=SchemaFieldType.STRING)
        assert field.name == "test"
        assert field.field_type == SchemaFieldType.STRING
        assert field.optional is False
        assert field.nullable is False

    def test_optional_nullable_field(self) -> None:
        """Test optional and nullable flags."""
        field = SchemaField(
            name="value",
            field_type=SchemaFieldType.INTEGER,
            optional=True,
            nullable=True,
        )
        assert field.optional is True
        assert field.nullable is True

    def test_nested_schema(self) -> None:
        """Test field with nested schema."""
        nested = [
            SchemaField(name="inner", field_type=SchemaFieldType.STRING),
        ]
        field = SchemaField(
            name="outer",
            field_type=SchemaFieldType.OBJECT,
            nested_schema=nested,
        )
        assert field.nested_schema is not None
        assert len(field.nested_schema) == 1
        assert field.nested_schema[0].name == "inner"

    def test_array_item_type(self) -> None:
        """Test array field with item type."""
        field = SchemaField(
            name="items",
            field_type=SchemaFieldType.ARRAY,
            item_type=SchemaFieldType.STRING,
        )
        assert field.item_type == SchemaFieldType.STRING

    def test_sample_values(self) -> None:
        """Test sample values storage."""
        field = SchemaField(
            name="score",
            field_type=SchemaFieldType.NUMBER,
            sample_values=[1.5, 2.5, 3.5],
        )
        assert len(field.sample_values) == 3

    def test_field_hash(self) -> None:
        """Test field hashing for comparison."""
        field1 = SchemaField(name="test", field_type=SchemaFieldType.STRING)
        field2 = SchemaField(name="test", field_type=SchemaFieldType.STRING)
        field3 = SchemaField(name="test", field_type=SchemaFieldType.INTEGER)

        assert hash(field1) == hash(field2)
        assert hash(field1) != hash(field3)


class TestExtractedSchema:
    """Tests for ExtractedSchema model."""

    def test_basic_schema(self) -> None:
        """Test basic schema creation."""
        fields = [SchemaField(name="id", field_type=SchemaFieldType.INTEGER)]
        schema = ExtractedSchema(schema_hash="abc123", fields=fields)

        assert schema.schema_hash == "abc123"
        assert len(schema.fields) == 1
        assert schema.ref_count == 1

    def test_touch_updates_tracking(self) -> None:
        """Test touch updates reference count."""
        fields = [SchemaField(name="id", field_type=SchemaFieldType.INTEGER)]
        schema = ExtractedSchema(schema_hash="abc123", fields=fields)

        original_last_used = schema.last_used
        schema.touch()

        assert schema.ref_count == 2
        assert schema.last_used >= original_last_used

    def test_field_names_property(self) -> None:
        """Test field_names property."""
        fields = [
            SchemaField(name="a", field_type=SchemaFieldType.STRING),
            SchemaField(name="b", field_type=SchemaFieldType.INTEGER),
        ]
        schema = ExtractedSchema(schema_hash="abc", fields=fields)

        assert schema.field_names == ["a", "b"]
        assert schema.field_count == 2


class TestSchemaExtractor:
    """Tests for SchemaExtractor class."""

    def test_init_defaults(self) -> None:
        """Test default initialization."""
        extractor = SchemaExtractor()
        assert extractor._min_items == 2
        assert extractor._max_samples == 3
        assert extractor._max_depth == 5

    def test_init_custom(self) -> None:
        """Test custom initialization."""
        extractor = SchemaExtractor(
            min_items_for_schema=5,
            max_sample_values=10,
            max_nesting_depth=3,
        )
        assert extractor._min_items == 5
        assert extractor._max_samples == 10
        assert extractor._max_depth == 3

    def test_infer_type_string(self) -> None:
        """Test string type inference."""
        extractor = SchemaExtractor()
        assert extractor._infer_type("hello") == SchemaFieldType.STRING

    def test_infer_type_integer(self) -> None:
        """Test integer type inference."""
        extractor = SchemaExtractor()
        assert extractor._infer_type(42) == SchemaFieldType.INTEGER

    def test_infer_type_number(self) -> None:
        """Test float type inference."""
        extractor = SchemaExtractor()
        assert extractor._infer_type(3.14) == SchemaFieldType.NUMBER

    def test_infer_type_boolean(self) -> None:
        """Test boolean type inference."""
        extractor = SchemaExtractor()
        assert extractor._infer_type(True) == SchemaFieldType.BOOLEAN
        assert extractor._infer_type(False) == SchemaFieldType.BOOLEAN

    def test_infer_type_null(self) -> None:
        """Test null type inference."""
        extractor = SchemaExtractor()
        assert extractor._infer_type(None) == SchemaFieldType.NULL

    def test_infer_type_array(self) -> None:
        """Test array type inference."""
        extractor = SchemaExtractor()
        assert extractor._infer_type([1, 2, 3]) == SchemaFieldType.ARRAY

    def test_infer_type_object(self) -> None:
        """Test object type inference."""
        extractor = SchemaExtractor()
        assert extractor._infer_type({"key": "value"}) == SchemaFieldType.OBJECT

    def test_can_extract_valid_list(self) -> None:
        """Test can_extract with valid list of objects."""
        extractor = SchemaExtractor()
        data = [
            {"name": "a", "value": 1},
            {"name": "b", "value": 2},
            {"name": "c", "value": 3},
        ]
        assert extractor.can_extract(data) is True

    def test_can_extract_too_short(self) -> None:
        """Test can_extract rejects short lists."""
        extractor = SchemaExtractor(min_items_for_schema=3)
        data = [{"name": "a"}, {"name": "b"}]
        assert extractor.can_extract(data) is False

    def test_can_extract_non_dict_items(self) -> None:
        """Test can_extract rejects non-dict items."""
        extractor = SchemaExtractor()
        data = [1, 2, 3]
        assert extractor.can_extract(data) is False

    def test_can_extract_inconsistent_keys(self) -> None:
        """Test can_extract rejects inconsistent keys."""
        extractor = SchemaExtractor()
        data = [
            {"name": "a", "value": 1},
            {"name": "b", "other": 2},  # Different key
            {"name": "c", "value": 3},
        ]
        assert extractor.can_extract(data) is False

    def test_can_extract_single_object(self) -> None:
        """Test can_extract with single object."""
        extractor = SchemaExtractor()
        data = {"a": 1, "b": 2, "c": 3}
        assert extractor.can_extract(data) is True

    def test_can_extract_small_object(self) -> None:
        """Test can_extract rejects small objects."""
        extractor = SchemaExtractor()
        data = {"a": 1}
        assert extractor.can_extract(data) is False

    def test_extract_from_list(self) -> None:
        """Test extract from list of objects."""
        extractor = SchemaExtractor()
        data = [
            {"name": "Alice", "age": 30},
            {"name": "Bob", "age": 25},
        ]

        schema = extractor.extract(data)

        assert schema is not None
        assert schema.sample_size == 2
        assert schema.field_count == 2

        # Check fields (sorted alphabetically)
        field_names = schema.field_names
        assert "age" in field_names
        assert "name" in field_names

    def test_extract_with_source_tool(self) -> None:
        """Test extract with source tool tracking."""
        extractor = SchemaExtractor()
        data = [{"x": 1}, {"x": 2}]

        schema = extractor.extract(data, source_tool="test_tool")

        assert schema is not None
        assert schema.source_tool == "test_tool"

    def test_extract_optional_fields(self) -> None:
        """Test extraction detects optional fields."""
        extractor = SchemaExtractor()
        data = [
            {"name": "a", "value": 1},
            {"name": "b"},  # Missing 'value'
            {"name": "c", "value": 3},
        ]

        schema = extractor.extract(data)

        assert schema is not None
        value_field = next(f for f in schema.fields if f.name == "value")
        assert value_field.optional is True

    def test_extract_nullable_fields(self) -> None:
        """Test extraction detects nullable fields."""
        extractor = SchemaExtractor()
        data = [
            {"name": "a", "value": 1},
            {"name": "b", "value": None},
            {"name": "c", "value": 3},
        ]

        schema = extractor.extract(data)

        assert schema is not None
        value_field = next(f for f in schema.fields if f.name == "value")
        assert value_field.nullable is True

    def test_extract_mixed_types(self) -> None:
        """Test extraction handles mixed types."""
        extractor = SchemaExtractor()
        data = [
            {"value": 1},
            {"value": "string"},
            {"value": 3.14},
        ]

        schema = extractor.extract(data)

        assert schema is not None
        value_field = next(f for f in schema.fields if f.name == "value")
        assert value_field.field_type == SchemaFieldType.MIXED

    def test_extract_from_single_object(self) -> None:
        """Test extract from single object."""
        extractor = SchemaExtractor()
        data = {"id": 1, "name": "test", "active": True}

        schema = extractor.extract(data)

        assert schema is not None
        assert schema.sample_size == 1
        assert schema.field_count == 3

    def test_extract_nested_objects(self) -> None:
        """Test extract with nested objects."""
        extractor = SchemaExtractor()
        data = [
            {"user": {"name": "Alice", "id": 1}},
            {"user": {"name": "Bob", "id": 2}},
        ]

        schema = extractor.extract(data)

        assert schema is not None
        user_field = next(f for f in schema.fields if f.name == "user")
        assert user_field.field_type == SchemaFieldType.OBJECT
        assert user_field.nested_schema is not None
        assert len(user_field.nested_schema) == 2

    def test_extract_returns_none_for_primitives(self) -> None:
        """Test extract returns None for primitives."""
        extractor = SchemaExtractor()

        assert extractor.extract("string") is None
        assert extractor.extract(42) is None
        assert extractor.extract(None) is None

    def test_schema_hash_deterministic(self) -> None:
        """Test schema hash is deterministic."""
        extractor = SchemaExtractor()
        data1 = [{"a": 1, "b": 2}]
        data2 = [{"a": 1, "b": 2}]

        schema1 = extractor.extract(data1)
        schema2 = extractor.extract(data2)

        assert schema1 is not None
        assert schema2 is not None
        assert schema1.schema_hash == schema2.schema_hash

    def test_schema_hash_different_for_different_schemas(self) -> None:
        """Test different schemas have different hashes."""
        extractor = SchemaExtractor()
        data1 = [{"a": 1}]
        data2 = [{"b": 2}]

        schema1 = extractor.extract(data1)
        schema2 = extractor.extract(data2)

        assert schema1 is not None
        assert schema2 is not None
        assert schema1.schema_hash != schema2.schema_hash

    def test_compress_with_schema(self) -> None:
        """Test compression with extracted schema."""
        extractor = SchemaExtractor()
        data = [
            {"name": "Alice", "age": 30},
            {"name": "Bob", "age": 25},
        ]

        schema = extractor.extract(data)
        assert schema is not None

        compressed = extractor.compress_with_schema(data, schema)

        assert isinstance(compressed, SchemaCompressedData)
        assert compressed.schema_ref == schema.schema_hash
        assert len(compressed.keys) == 2
        assert len(compressed.values) == 2

    def test_decompress_with_schema(self) -> None:
        """Test decompression roundtrip."""
        extractor = SchemaExtractor()
        data = [
            {"name": "Alice", "age": 30},
            {"name": "Bob", "age": 25},
        ]

        schema = extractor.extract(data)
        assert schema is not None

        compressed = extractor.compress_with_schema(data, schema)
        decompressed = extractor.decompress_with_schema(compressed, schema)

        assert len(decompressed) == 2
        assert decompressed[0]["name"] == "Alice"
        assert decompressed[1]["age"] == 25


class TestSchemaCache:
    """Tests for SchemaCache class."""

    def test_init_defaults(self) -> None:
        """Test default initialization."""
        cache = SchemaCache()
        assert cache._max_schemas == 1000
        assert cache.size == 0

    def test_init_custom(self) -> None:
        """Test custom max size."""
        cache = SchemaCache(max_schemas=100)
        assert cache._max_schemas == 100

    def test_put_new_schema(self) -> None:
        """Test putting new schema."""
        cache = SchemaCache()
        fields = [SchemaField(name="test", field_type=SchemaFieldType.STRING)]
        schema = ExtractedSchema(schema_hash="hash1", fields=fields)

        result = cache.put(schema)

        assert result == "hash1"
        assert cache.size == 1
        stats = cache.stats()
        assert stats.cache_misses == 1
        assert stats.total_schemas == 1

    def test_put_existing_schema(self) -> None:
        """Test putting existing schema increments ref count."""
        cache = SchemaCache()
        fields = [SchemaField(name="test", field_type=SchemaFieldType.STRING)]
        schema = ExtractedSchema(schema_hash="hash1", fields=fields)

        cache.put(schema)
        cache.put(schema)

        assert cache.size == 1
        stats = cache.stats()
        assert stats.cache_hits == 1
        assert stats.total_references == 2

    def test_get_existing(self) -> None:
        """Test getting existing schema."""
        cache = SchemaCache()
        fields = [SchemaField(name="test", field_type=SchemaFieldType.STRING)]
        schema = ExtractedSchema(schema_hash="hash1", fields=fields)
        cache.put(schema)

        result = cache.get("hash1")

        assert result is not None
        assert result.schema_hash == "hash1"

    def test_get_missing(self) -> None:
        """Test getting missing schema returns None."""
        cache = SchemaCache()
        result = cache.get("nonexistent")
        assert result is None

    def test_contains(self) -> None:
        """Test contains check."""
        cache = SchemaCache()
        fields = [SchemaField(name="test", field_type=SchemaFieldType.STRING)]
        schema = ExtractedSchema(schema_hash="hash1", fields=fields)
        cache.put(schema)

        assert cache.contains("hash1") is True
        assert cache.contains("hash2") is False

    def test_remove(self) -> None:
        """Test removing schema."""
        cache = SchemaCache()
        fields = [SchemaField(name="test", field_type=SchemaFieldType.STRING)]
        schema = ExtractedSchema(schema_hash="hash1", fields=fields)
        cache.put(schema)

        result = cache.remove("hash1")

        assert result is True
        assert cache.size == 0
        assert cache.contains("hash1") is False

    def test_remove_missing(self) -> None:
        """Test removing missing schema."""
        cache = SchemaCache()
        result = cache.remove("nonexistent")
        assert result is False

    def test_clear(self) -> None:
        """Test clearing cache."""
        cache = SchemaCache()
        fields = [SchemaField(name="test", field_type=SchemaFieldType.STRING)]
        cache.put(ExtractedSchema(schema_hash="hash1", fields=fields))
        cache.put(ExtractedSchema(schema_hash="hash2", fields=fields))

        cache.clear()

        assert cache.size == 0

    def test_lru_eviction(self) -> None:
        """Test LRU eviction when cache full."""
        cache = SchemaCache(max_schemas=2)
        fields = [SchemaField(name="test", field_type=SchemaFieldType.STRING)]

        cache.put(ExtractedSchema(schema_hash="hash1", fields=fields))
        cache.put(ExtractedSchema(schema_hash="hash2", fields=fields))
        # Access hash1 to make it more recently used
        cache.get("hash1")
        # Add third schema - should evict hash2 (LRU)
        cache.put(ExtractedSchema(schema_hash="hash3", fields=fields))

        assert cache.size == 2
        assert cache.contains("hash1") is True
        assert cache.contains("hash2") is False
        assert cache.contains("hash3") is True

    def test_find_matching_cached(self) -> None:
        """Test finding matching cached schema."""
        cache = SchemaCache()
        extractor = SchemaExtractor()
        data = [{"name": "test", "value": 1}, {"name": "test2", "value": 2}]

        # First, put a schema for this data
        schema = extractor.extract(data)
        assert schema is not None
        cache.put(schema)

        # Now find matching
        found, is_cached = cache.find_matching(data, extractor)

        assert found is not None
        assert is_cached is True
        assert found.schema_hash == schema.schema_hash

    def test_find_matching_new(self) -> None:
        """Test finding schema for new data."""
        cache = SchemaCache()
        extractor = SchemaExtractor()
        data = [{"name": "test", "value": 1}, {"name": "test2", "value": 2}]

        found, is_cached = cache.find_matching(data, extractor)

        assert found is not None
        assert is_cached is False

    def test_find_matching_invalid_data(self) -> None:
        """Test finding schema for invalid data."""
        cache = SchemaCache()
        extractor = SchemaExtractor()

        found, is_cached = cache.find_matching("not a list", extractor)

        assert found is None
        assert is_cached is False

    def test_stats(self) -> None:
        """Test statistics tracking."""
        cache = SchemaCache()
        fields = [SchemaField(name="test", field_type=SchemaFieldType.STRING)]

        cache.put(ExtractedSchema(schema_hash="hash1", fields=fields))
        cache.put(ExtractedSchema(schema_hash="hash1", fields=fields))  # Hit
        cache.put(ExtractedSchema(schema_hash="hash2", fields=fields))

        stats = cache.stats()

        assert isinstance(stats, SchemaCacheStats)
        assert stats.total_schemas == 2
        assert stats.total_references == 3
        assert stats.cache_hits == 1
        assert stats.cache_misses == 2
        assert stats.hit_rate == pytest.approx(1 / 3, abs=0.01)

    def test_get_all_schemas(self) -> None:
        """Test getting all schemas."""
        cache = SchemaCache()
        fields = [SchemaField(name="test", field_type=SchemaFieldType.STRING)]
        cache.put(ExtractedSchema(schema_hash="hash1", fields=fields))
        cache.put(ExtractedSchema(schema_hash="hash2", fields=fields))

        all_schemas = cache.get_all_schemas()

        assert len(all_schemas) == 2

    def test_get_schemas_for_tool(self) -> None:
        """Test getting schemas by tool."""
        cache = SchemaCache()
        fields = [SchemaField(name="test", field_type=SchemaFieldType.STRING)]
        cache.put(
            ExtractedSchema(schema_hash="h1", fields=fields, source_tool="tool_a")
        )
        cache.put(
            ExtractedSchema(schema_hash="h2", fields=fields, source_tool="tool_a")
        )
        cache.put(
            ExtractedSchema(schema_hash="h3", fields=fields, source_tool="tool_b")
        )

        tool_a_schemas = cache.get_schemas_for_tool("tool_a")

        assert len(tool_a_schemas) == 2


class TestToolResultCompressorSchemaIntegration:
    """Tests for ToolResultCompressor with schema extraction."""

    def test_compressor_with_schema_cache(self) -> None:
        """Test compressor uses schema cache."""
        cache = SchemaCache()
        compressor = ToolResultCompressor(schema_cache=cache)

        data = [
            {"name": "a", "value": 1},
            {"name": "b", "value": 2},
            {"name": "c", "value": 3},
        ]

        result = compressor.compress("test_tool", data)

        assert result.metadata.get("schema_extracted") is True
        assert cache.size == 1

    def test_compressor_schema_reuse(self) -> None:
        """Test compressor reuses schemas from cache."""
        cache = SchemaCache()
        compressor = ToolResultCompressor(schema_cache=cache)

        data1 = [
            {"name": "a", "value": 1},
            {"name": "b", "value": 2},
            {"name": "c", "value": 3},
        ]
        data2 = [
            {"name": "x", "value": 10},
            {"name": "y", "value": 20},
        ]

        # Same schema structure
        compressor.compress("test_tool", data1)
        result2 = compressor.compress("test_tool", data2)

        # Schema should have been cached and reused
        assert cache.size == 1
        assert result2.metadata.get("schema_cached") is True

    def test_compressor_without_cache(self) -> None:
        """Test compressor works without cache."""
        compressor = ToolResultCompressor()

        data = [
            {"name": "a", "value": 1},
            {"name": "b", "value": 2},
            {"name": "c", "value": 3},
        ]

        result = compressor.compress("test_tool", data)

        assert result.metadata.get("schema_extracted") is True
        assert (
            "_schema" in result.compressed_content
            or "_keys" in result.compressed_content
        )

    def test_compressor_disable_cache(self) -> None:
        """Test disabling cache usage."""
        cache = SchemaCache()
        compressor = ToolResultCompressor(
            schema_cache=cache,
            use_cached_schemas=False,
        )

        data = [
            {"name": "a", "value": 1},
            {"name": "b", "value": 2},
            {"name": "c", "value": 3},
        ]

        result = compressor.compress("test_tool", data)

        # Should still extract schema but not use cache
        assert result.metadata.get("schema_extracted") is True
        assert cache.size == 0


class TestSchemaCompressedData:
    """Tests for SchemaCompressedData model."""

    def test_basic_compressed_data(self) -> None:
        """Test basic compressed data creation."""
        data = SchemaCompressedData(
            schema_ref="abc123",
            keys=["name", "value"],
            values=[["a", 1], ["b", 2]],
        )

        assert data.schema_ref == "abc123"
        assert data.keys == ["name", "value"]
        assert data.item_count == 2

    def test_inline_schema(self) -> None:
        """Test inline vs cached schema flag."""
        inline = SchemaCompressedData(
            keys=["a", "b"],
            values=[[1, 2]],
            inline_schema=True,
        )
        cached = SchemaCompressedData(
            schema_ref="hash",
            keys=["a", "b"],
            values=[[1, 2]],
            inline_schema=False,
        )

        assert inline.inline_schema is True
        assert cached.inline_schema is False


class TestSchemaCacheStats:
    """Tests for SchemaCacheStats model."""

    def test_hit_rate_calculation(self) -> None:
        """Test hit rate calculation."""
        stats = SchemaCacheStats(cache_hits=3, cache_misses=7)
        assert stats.hit_rate == pytest.approx(0.3)

    def test_hit_rate_zero_total(self) -> None:
        """Test hit rate with no operations."""
        stats = SchemaCacheStats()
        assert stats.hit_rate == 0.0

    def test_dedup_ratio(self) -> None:
        """Test deduplication ratio."""
        stats = SchemaCacheStats(total_schemas=5, total_references=20)
        assert stats.dedup_ratio == pytest.approx(4.0)

    def test_dedup_ratio_zero_schemas(self) -> None:
        """Test dedup ratio with no schemas."""
        stats = SchemaCacheStats()
        assert stats.dedup_ratio == 0.0
