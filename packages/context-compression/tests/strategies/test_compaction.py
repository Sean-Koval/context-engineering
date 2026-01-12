"""Tests for compaction compression strategies."""

from __future__ import annotations

import json

import pytest

from context_compression.recovery import RecoveryManifest
from context_compression.strategies.compaction import SchemaCompression
from context_compression.types import CompressionTier
from context_core.graph import ContextGraph
from context_core.graph.types import CompressionLevel, NodeType


class TestSchemaCompression:
    """Tests for SchemaCompression strategy."""

    @pytest.fixture
    def strategy(self):
        """Create strategy with default settings."""
        return SchemaCompression(min_occurrences=3, min_array_length=2)

    @pytest.fixture
    def graph_with_repeated_schemas(self):
        """Create graph with tool results having repeated JSON schemas.

        Uses arrays large enough to benefit from schema compression.
        The schema reference overhead is offset by repeated key removal.
        """
        graph = ContextGraph()

        # Add 5 tool results with same schema (users array with many items)
        # Larger arrays benefit more from schema compression since keys are
        # removed from each item
        for i in range(5):
            call = graph.add_tool_call("get_users", {"page": i})
            # Use longer field names and more items to ensure savings
            users = [
                {
                    "user_id": i * 100 + j,
                    "username": f"user_{i * 100 + j}_account",
                    "email_address": f"user{i * 100 + j}@example.com",
                    "is_active": j % 2 == 0,
                    "registration_date": f"2024-01-{(j % 28) + 1:02d}",
                }
                for j in range(10)  # 10 items per array
            ]
            result = graph.add_tool_result(call.id, {"users": users, "page": i})
            result.token_count = 500

        return graph

    @pytest.fixture
    def graph_with_different_schemas(self):
        """Create graph with tool results having different JSON schemas."""
        graph = ContextGraph()

        # Add tool results with different schemas
        call1 = graph.add_tool_call("get_users", {})
        graph.add_tool_result(call1.id, {"users": [{"id": 1, "name": "Alice"}]})

        call2 = graph.add_tool_call("get_files", {})
        graph.add_tool_result(call2.id, {"files": [{"path": "/a.txt", "size": 100}]})

        call3 = graph.add_tool_call("get_events", {})
        graph.add_tool_result(call3.id, {"events": [{"type": "click", "count": 5}]})

        return graph

    def test_strategy_properties(self, strategy):
        """Test strategy properties."""
        assert strategy.name == "schema_compression"
        assert strategy.tier == CompressionTier.COMPACTION
        assert strategy.priority == 10

    def test_extract_schema_from_dict(self, strategy):
        """Test schema extraction from dictionary."""
        data = {"id": 1, "name": "Alice", "active": True}
        schema = strategy._extract_schema(data)

        assert schema["type"] == "object"
        assert "properties" in schema
        assert schema["properties"]["id"]["type"] == "integer"
        assert schema["properties"]["name"]["type"] == "string"
        assert schema["properties"]["active"]["type"] == "boolean"

    def test_extract_schema_from_nested_dict(self, strategy):
        """Test schema extraction from nested dictionary."""
        data = {
            "user": {"id": 1, "profile": {"bio": "Hello", "verified": False}},
            "count": 5,
        }
        schema = strategy._extract_schema(data)

        assert schema["type"] == "object"
        user_schema = schema["properties"]["user"]
        assert user_schema["type"] == "object"
        assert user_schema["properties"]["profile"]["type"] == "object"

    def test_extract_schema_from_array(self, strategy):
        """Test schema extraction from array."""
        data = [{"id": 1, "name": "Alice"}, {"id": 2, "name": "Bob"}]
        schema = strategy._extract_schema(data)

        assert schema["type"] == "array"
        assert "items" in schema
        assert schema["items"]["type"] == "object"
        assert schema["items"]["properties"]["id"]["type"] == "integer"
        assert schema["items"]["properties"]["name"]["type"] == "string"

    def test_extract_schema_with_null(self, strategy):
        """Test schema extraction handles null values."""
        data = {"id": 1, "optional": None}
        schema = strategy._extract_schema(data)

        assert schema["properties"]["optional"]["type"] == "null"

    def test_extract_schema_with_float(self, strategy):
        """Test schema extraction handles float values."""
        data = {"score": 3.14}
        schema = strategy._extract_schema(data)

        assert schema["properties"]["score"]["type"] == "number"

    def test_find_schema_occurrences(self, strategy, graph_with_repeated_schemas):
        """Test finding schema occurrences across nodes."""
        occurrences = strategy._find_schema_occurrences(
            graph_with_repeated_schemas, None
        )

        # Should find at least one schema with multiple occurrences
        assert len(occurrences) > 0

        # Check that we found occurrences
        for _schema_hash, schema_occurrences in occurrences.items():
            if len(schema_occurrences) >= 3:
                # Each occurrence should be a (node, path, array) tuple
                for node, _path, array in schema_occurrences:
                    assert node.type == NodeType.TOOL_RESULT
                    assert isinstance(array, list)

    def test_find_schema_occurrences_respects_targets(
        self, strategy, graph_with_repeated_schemas
    ):
        """Test that find_schema_occurrences respects target node IDs."""
        # Get first two tool result node IDs
        tool_results = [
            node
            for node in graph_with_repeated_schemas
            if node.type == NodeType.TOOL_RESULT
        ]
        target_ids = {tool_results[0].id, tool_results[1].id}

        occurrences = strategy._find_schema_occurrences(
            graph_with_repeated_schemas, target_ids
        )

        # Count total occurrences
        total = sum(len(occ) for occ in occurrences.values())
        assert total <= 2

    def test_extract_values_from_array(self, strategy):
        """Test extracting values from array according to schema."""
        array = [{"id": 1, "name": "Alice"}, {"id": 2, "name": "Bob"}]
        schema = {"type": "object", "properties": {"id": {}, "name": {}}}

        values = strategy._extract_values(array, schema)

        # Keys are sorted, so order is: id, name
        assert values == [[1, "Alice"], [2, "Bob"]]

    def test_extract_values_preserves_order(self, strategy):
        """Test that value extraction uses consistent key ordering."""
        array = [
            {"z_field": "z", "a_field": "a", "m_field": "m"},
            {"z_field": "zz", "a_field": "aa", "m_field": "mm"},
        ]
        schema = {
            "type": "object",
            "properties": {"z_field": {}, "a_field": {}, "m_field": {}},
        }

        values = strategy._extract_values(array, schema)

        # Keys should be sorted alphabetically: a_field, m_field, z_field
        assert values == [["a", "m", "z"], ["aa", "mm", "zz"]]

    def test_compress_with_repeated_structures(
        self, strategy, graph_with_repeated_schemas
    ):
        """Test compression with repeated structures."""
        manifest = RecoveryManifest()

        result = strategy.compress(graph_with_repeated_schemas, manifest)

        assert result.success is True
        assert result.strategy_name == "schema_compression"
        assert result.tier == CompressionTier.COMPACTION
        # Should have compressed some nodes
        assert result.nodes_compressed >= 3
        assert result.tokens_saved >= 0

    def test_compress_logs_compact_operation(
        self, strategy, graph_with_repeated_schemas
    ):
        """Test that compression logs CompactOperation to manifest."""
        manifest = RecoveryManifest()

        strategy.compress(graph_with_repeated_schemas, manifest)

        # Should have logged operations
        assert len(manifest.operations) >= 3

        # Check operation type
        for op in manifest.operations:
            assert op.op_type == "compact"
            assert op.compaction_method == "schema_compression"

    def test_compress_updates_node_compression_level(
        self, strategy, graph_with_repeated_schemas
    ):
        """Test that compression updates node compression level."""
        manifest = RecoveryManifest()

        strategy.compress(graph_with_repeated_schemas, manifest)

        # Check that compressed nodes have updated compression level
        compressed_count = 0
        for node in graph_with_repeated_schemas:
            if (
                node.type == NodeType.TOOL_RESULT
                and node.compression_level == CompressionLevel.COMPACTED
            ):
                compressed_count += 1

        assert compressed_count >= 3

    def test_compress_creates_schema_reference(
        self, strategy, graph_with_repeated_schemas
    ):
        """Test that compression creates schema reference in content."""
        manifest = RecoveryManifest()

        strategy.compress(graph_with_repeated_schemas, manifest)

        # Find a compressed node and check its content
        for node in graph_with_repeated_schemas:
            if (
                node.type == NodeType.TOOL_RESULT
                and node.compression_level == CompressionLevel.COMPACTED
            ):
                content = node.content.tool_output
                # Content should have schema reference somewhere
                content_str = json.dumps(content)
                assert "$schema_ref" in content_str
                break

    def test_respects_minimum_occurrences(self):
        """Test compression respects minimum occurrences threshold."""
        # Create strategy with high threshold
        strategy = SchemaCompression(min_occurrences=10, min_array_length=2)

        graph = ContextGraph()
        # Add only 3 tool results with same schema
        for i in range(3):
            call = graph.add_tool_call("get_data", {"i": i})
            data = [{"id": j, "value": f"val{j}"} for j in range(3)]
            result = graph.add_tool_result(call.id, {"items": data})
            result.token_count = 100

        manifest = RecoveryManifest()
        result = strategy.compress(graph, manifest)

        # Should not compress because occurrences < min_occurrences
        assert result.nodes_compressed == 0

    def test_respects_minimum_array_length(self):
        """Test compression respects minimum array length threshold."""
        strategy = SchemaCompression(min_occurrences=3, min_array_length=5)

        graph = ContextGraph()
        # Add tool results with small arrays
        for i in range(5):
            call = graph.add_tool_call("get_data", {"i": i})
            # Arrays with only 2 items (below threshold)
            data = [{"id": 1, "name": "a"}, {"id": 2, "name": "b"}]
            result = graph.add_tool_result(call.id, {"items": data})
            result.token_count = 100

        manifest = RecoveryManifest()
        result = strategy.compress(graph, manifest)

        # Should not compress because arrays are too small
        assert result.nodes_compressed == 0

    def test_skips_non_json_content(self, strategy):
        """Test compression skips non-JSON content."""
        graph = ContextGraph()

        # Add tool results with string content
        for i in range(5):
            call = graph.add_tool_call("run_command", {"cmd": f"echo {i}"})
            result = graph.add_tool_result(call.id, f"Output line {i}")
            result.token_count = 50

        manifest = RecoveryManifest()
        result = strategy.compress(graph, manifest)

        assert result.nodes_compressed == 0

    def test_skips_pinned_nodes(self, strategy):
        """Test compression skips pinned nodes."""
        graph = ContextGraph()

        # Add pinned tool results
        for i in range(5):
            call = graph.add_tool_call("get_data", {"i": i})
            data = [{"id": j, "value": f"val{j}"} for j in range(3)]
            result = graph.add_tool_result(call.id, {"items": data})
            result.token_count = 100
            result.metadata.pinned = True  # Pin the node

        manifest = RecoveryManifest()
        result = strategy.compress(graph, manifest)

        assert result.nodes_compressed == 0

    def test_skips_already_compressed_nodes(self, strategy):
        """Test compression skips already compressed nodes."""
        graph = ContextGraph()

        # Add pre-compressed tool results
        for i in range(5):
            call = graph.add_tool_call("get_data", {"i": i})
            data = [{"id": j, "value": f"val{j}"} for j in range(3)]
            result = graph.add_tool_result(call.id, {"items": data})
            result.token_count = 100
            result.compression_level = CompressionLevel.COMPACTED

        manifest = RecoveryManifest()
        result = strategy.compress(graph, manifest)

        assert result.nodes_compressed == 0

    def test_estimate_savings(self, strategy, graph_with_repeated_schemas):
        """Test savings estimation."""
        savings = strategy.estimate_savings(graph_with_repeated_schemas)

        # Should estimate some savings
        assert savings >= 0

    def test_estimate_savings_empty_graph(self, strategy):
        """Test savings estimation on empty graph."""
        graph = ContextGraph()
        savings = strategy.estimate_savings(graph)
        assert savings == 0

    def test_can_apply_with_eligible_nodes(self, strategy, graph_with_repeated_schemas):
        """Test can_apply returns True for eligible graph."""
        assert strategy.can_apply(graph_with_repeated_schemas) is True

    def test_can_apply_without_eligible_nodes(
        self, strategy, graph_with_different_schemas
    ):
        """Test can_apply returns False when not enough matching schemas."""
        assert strategy.can_apply(graph_with_different_schemas) is False

    def test_can_apply_empty_graph(self, strategy):
        """Test can_apply returns False for empty graph."""
        graph = ContextGraph()
        assert strategy.can_apply(graph) is False

    def test_compress_respects_target_tokens(self, strategy):
        """Test compression respects target tokens limit."""
        graph = ContextGraph()

        # Add many tool results with same schema
        for i in range(10):
            call = graph.add_tool_call("get_data", {"i": i})
            data = [{"id": j, "name": f"Name{j}", "value": j * 100} for j in range(5)]
            result = graph.add_tool_result(call.id, {"items": data})
            result.token_count = 200

        manifest = RecoveryManifest()
        # Set a low target to stop early
        result = strategy.compress(graph, manifest, target_tokens=50)

        # Should have stopped before processing all nodes
        assert result.nodes_compressed < 10

    def test_compress_with_target_node_ids(self, strategy, graph_with_repeated_schemas):
        """Test compression targeting specific nodes."""
        # Get first 2 tool result node IDs
        tool_results = [
            node
            for node in graph_with_repeated_schemas
            if node.type == NodeType.TOOL_RESULT
        ]
        target_ids = [tool_results[0].id, tool_results[1].id]

        manifest = RecoveryManifest()
        result = strategy.compress(
            graph_with_repeated_schemas, manifest, target_node_ids=target_ids
        )

        # With only 2 targets and min_occurrences=3, should not compress
        assert result.nodes_compressed == 0

    def test_get_schema_retrieves_stored_schema(self, strategy):
        """Test that get_schema retrieves stored schemas."""
        # First, trigger schema storage by finding occurrences
        graph = ContextGraph()
        for i in range(5):
            call = graph.add_tool_call("get_data", {"i": i})
            data = [{"id": j, "name": f"Name{j}"} for j in range(3)]
            result = graph.add_tool_result(call.id, data)
            result.token_count = 100

        strategy._find_schema_occurrences(graph, None)

        # Get a schema ID from stored schemas
        for schema_hash in strategy._schemas:
            schema_id = f"schema_{schema_hash[:8]}"
            retrieved = strategy.get_schema(schema_id)
            assert retrieved is not None
            assert "type" in retrieved
            break

    def test_get_schema_returns_none_for_unknown(self, strategy):
        """Test that get_schema returns None for unknown schema ID."""
        result = strategy.get_schema("schema_unknown123")
        assert result is None

    def test_compress_handles_json_string_content(self, strategy):
        """Test compression handles JSON stored as string."""
        graph = ContextGraph()

        # Add tool results with JSON string content
        # Use larger arrays with longer keys to ensure savings
        for i in range(5):
            call = graph.add_tool_call("get_json", {"i": i})
            data = [
                {
                    "item_id": j,
                    "item_value": f"value_{j}_content",
                    "description": f"Description for item {j}",
                }
                for j in range(10)
            ]
            # Store as JSON string instead of dict
            result = graph.add_tool_result(call.id, json.dumps({"items": data}))
            result.token_count = 300

        manifest = RecoveryManifest()
        result = strategy.compress(graph, manifest)

        # Should be able to handle JSON strings
        assert result.success is True

    def test_compress_handles_root_level_array(self, strategy):
        """Test compression handles root-level arrays."""
        graph = ContextGraph()

        # Add tool results that are arrays at root level
        # Use larger arrays with longer keys to ensure savings
        for i in range(5):
            call = graph.add_tool_call("list_items", {"i": i})
            data = [
                {
                    "item_id": j,
                    "item_name": f"Item_{j}_name",
                    "item_description": f"Description for item number {j}",
                }
                for j in range(10)
            ]
            result = graph.add_tool_result(call.id, data)
            result.token_count = 300

        manifest = RecoveryManifest()
        result = strategy.compress(graph, manifest)

        assert result.success is True
        # Should compress root-level arrays
        assert result.nodes_compressed >= 3

    def test_result_is_not_recoverable(self, strategy, graph_with_repeated_schemas):
        """Test that compression result marks as not recoverable (compaction)."""
        manifest = RecoveryManifest()
        result = strategy.compress(graph_with_repeated_schemas, manifest)

        # Compaction is not fully recoverable
        assert result.is_recoverable is False

    def test_compress_updates_token_counts(self, strategy, graph_with_repeated_schemas):
        """Test that compression updates token counts on nodes."""
        manifest = RecoveryManifest()

        # Store original token counts
        original_counts = {}
        for node in graph_with_repeated_schemas:
            if node.type == NodeType.TOOL_RESULT:
                original_counts[node.id] = node.token_count

        strategy.compress(graph_with_repeated_schemas, manifest)

        # Check that compressed nodes have updated token counts
        for node in graph_with_repeated_schemas:
            if (
                node.type == NodeType.TOOL_RESULT
                and node.compression_level == CompressionLevel.COMPACTED
            ):
                # Compressed token count should be set
                assert node.content.compressed_tokens is not None
                # Original tokens should be stored
                assert node.content.original_tokens is not None
