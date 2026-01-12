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


class TestEntityCentricCompression:
    """Tests for EntityCentricCompression strategy."""

    @pytest.fixture
    def entity_tracker(self):
        """Create an EntityTracker with some entities."""
        from uuid import uuid4

        from context_core.entities.tracker import EntityTracker
        from context_core.entities.types import Entity, EntityType

        tracker = EntityTracker()

        # Add some entities directly to the tracker
        entity1 = Entity(
            type=EntityType.PERSON,
            canonical_name="Alice",
            importance=0.8,
            aliases={"alice"},
        )
        entity1.node_ids.add(uuid4())
        tracker._entities[entity1.id] = entity1
        tracker._name_index["alice"] = entity1.id
        tracker._type_index[EntityType.PERSON] = {entity1.id}

        entity2 = Entity(
            type=EntityType.FILE_PATH,
            canonical_name="/src/main.py",
            importance=0.7,
            aliases={"main.py"},
        )
        entity2.node_ids.add(uuid4())
        tracker._entities[entity2.id] = entity2
        tracker._name_index["/src/main.py"] = entity2.id
        if EntityType.FILE_PATH not in tracker._type_index:
            tracker._type_index[EntityType.FILE_PATH] = set()
        tracker._type_index[EntityType.FILE_PATH].add(entity2.id)

        entity3 = Entity(
            type=EntityType.ORGANIZATION,
            canonical_name="Anthropic",
            importance=0.6,
            aliases={"anthropic"},
        )
        entity3.node_ids.add(uuid4())
        tracker._entities[entity3.id] = entity3
        tracker._name_index["anthropic"] = entity3.id
        if EntityType.ORGANIZATION not in tracker._type_index:
            tracker._type_index[EntityType.ORGANIZATION] = set()
        tracker._type_index[EntityType.ORGANIZATION].add(entity3.id)

        return tracker

    @pytest.fixture
    def strategy(self, entity_tracker):
        """Create strategy with default settings."""
        from context_compression.strategies.compaction import EntityCentricCompression

        return EntityCentricCompression(entity_tracker=entity_tracker)

    @pytest.fixture
    def graph_with_messages(self):
        """Create graph with MESSAGE nodes containing text."""
        graph = ContextGraph()

        # Add messages with various content
        msg1 = graph.add_message(
            "user",
            "Hello! I met Alice yesterday. She works at Anthropic. "
            "The weather was nice. I enjoyed our conversation.",
        )
        msg1.token_count = 50

        msg2 = graph.add_message(
            "assistant",
            "That's interesting! I heard about their work. "
            "How did your meeting go? Did you discuss the project?",
        )
        msg2.token_count = 40

        msg3 = graph.add_message(
            "user",
            "Alice mentioned /src/main.py needs refactoring. "
            "The coffee shop was crowded. Anthropic has great engineers.",
        )
        msg3.token_count = 60

        return graph

    @pytest.fixture
    def graph_with_tool_results(self):
        """Create graph with TOOL_RESULT nodes containing text."""
        graph = ContextGraph()

        call1 = graph.add_tool_call("read_file", {"path": "/src/main.py"})
        result1 = graph.add_tool_result(
            call1.id,
            "File contents show Alice wrote this module. "
            "It has good documentation. The tests are passing.",
        )
        result1.token_count = 40

        call2 = graph.add_tool_call("search", {"query": "Anthropic"})
        result2 = graph.add_tool_result(
            call2.id,
            "Found 3 results. The first result mentions Anthropic directly. "
            "Other results are unrelated. Some noise in the data.",
        )
        result2.token_count = 45

        return graph

    def test_strategy_properties(self, strategy):
        """Test strategy properties."""
        assert strategy.name == "entity_centric"
        assert strategy.tier == CompressionTier.COMPACTION
        assert strategy.priority == 20

    def test_split_sentences_basic(self, strategy):
        """Test sentence splitting on basic text."""
        text = "Hello world. This is a test. Another sentence!"
        sentences = strategy._split_sentences(text)

        assert len(sentences) == 3
        assert "Hello world" in sentences[0]
        assert "This is a test" in sentences[1]
        assert "Another sentence" in sentences[2]

    def test_split_sentences_empty(self, strategy):
        """Test sentence splitting on empty text."""
        assert strategy._split_sentences("") == []
        assert strategy._split_sentences("   ") == []
        assert strategy._split_sentences(None) == []

    def test_split_sentences_no_punctuation(self, strategy):
        """Test sentence splitting on text without clear boundaries."""
        text = "This is a long sentence without clear boundaries"
        sentences = strategy._split_sentences(text)

        assert len(sentences) == 1
        assert sentences[0] == text

    def test_split_sentences_question(self, strategy):
        """Test sentence splitting with questions."""
        text = "What is this? It is a test. Are you sure?"
        sentences = strategy._split_sentences(text)

        assert len(sentences) == 3

    def test_split_sentences_exclamation(self, strategy):
        """Test sentence splitting with exclamations."""
        text = "Wow! That's amazing! I can't believe it."
        sentences = strategy._split_sentences(text)

        assert len(sentences) == 3

    def test_sentence_has_entity_match(self, strategy):
        """Test entity detection in sentence - match case."""
        entities = {"Alice", "Bob", "main.py"}

        assert strategy._sentence_has_entity("Alice went home.", entities) is True
        assert strategy._sentence_has_entity("I met Bob.", entities) is True
        assert strategy._sentence_has_entity("Check main.py please.", entities) is True

    def test_sentence_has_entity_no_match(self, strategy):
        """Test entity detection in sentence - no match case."""
        entities = {"Alice", "Bob", "main.py"}

        assert strategy._sentence_has_entity("The weather is nice.", entities) is False
        assert strategy._sentence_has_entity("", entities) is False

    def test_sentence_has_entity_case_insensitive(self, strategy):
        """Test entity detection is case-insensitive."""
        entities = {"Alice", "main.py"}

        assert strategy._sentence_has_entity("ALICE said hello.", entities) is True
        assert strategy._sentence_has_entity("alice is here.", entities) is True
        assert strategy._sentence_has_entity("Modify MAIN.PY now.", entities) is True

    def test_sentence_has_entity_empty(self, strategy):
        """Test entity detection with empty inputs."""
        assert strategy._sentence_has_entity("Hello.", set()) is False
        assert strategy._sentence_has_entity("", {"Alice"}) is False

    def test_get_important_entity_names(self, strategy):
        """Test getting important entity names."""
        names = strategy._get_important_entity_names()

        assert "Alice" in names
        assert "/src/main.py" in names
        assert "Anthropic" in names
        # Should include aliases
        assert "alice" in names or "main.py" in names

    def test_get_important_entity_names_min_threshold(self, entity_tracker):
        """Test important entity filtering by threshold."""
        from context_compression.strategies.compaction import EntityCentricCompression

        # Create strategy with high threshold
        strategy = EntityCentricCompression(
            entity_tracker=entity_tracker, min_importance=0.75
        )

        names = strategy._get_important_entity_names()

        # Only Alice (0.8) should pass 0.75 threshold
        assert "Alice" in names
        # Anthropic (0.6) should not pass
        assert "Anthropic" not in names

    def test_compress_with_messages(self, strategy, graph_with_messages):
        """Test compression with MESSAGE nodes."""
        from context_compression.recovery import RecoveryManifest

        manifest = RecoveryManifest()
        result = strategy.compress(graph_with_messages, manifest)

        assert result.success is True
        assert result.strategy_name == "entity_centric"
        assert result.tier == CompressionTier.COMPACTION
        # Should have compressed at least one node
        assert result.nodes_compressed >= 1

    def test_compress_logs_compact_operation(self, strategy, graph_with_messages):
        """Test that compression logs CompactOperation to manifest."""
        from context_compression.recovery import RecoveryManifest

        manifest = RecoveryManifest()
        strategy.compress(graph_with_messages, manifest)

        # Should have logged operations
        assert len(manifest.operations) >= 1

        # Check operation type
        for op in manifest.operations:
            assert op.op_type == "compact"
            assert op.compaction_method == "entity_centric"

    def test_compress_updates_node_compression_level(
        self, strategy, graph_with_messages
    ):
        """Test that compression updates node compression level."""
        from context_compression.recovery import RecoveryManifest

        manifest = RecoveryManifest()
        strategy.compress(graph_with_messages, manifest)

        # Check that at least one message was compressed
        compressed_count = 0
        for node in graph_with_messages:
            if (
                node.type == NodeType.MESSAGE
                and node.compression_level == CompressionLevel.COMPACTED
            ):
                compressed_count += 1

        assert compressed_count >= 1

    def test_compress_preserves_entity_sentences(self, strategy, graph_with_messages):
        """Test that compression preserves entity-containing sentences."""
        from context_compression.recovery import RecoveryManifest

        manifest = RecoveryManifest()
        strategy.compress(graph_with_messages, manifest)

        # Find compressed message nodes and check content
        for node in graph_with_messages:
            if (
                node.type == NodeType.MESSAGE
                and node.compression_level == CompressionLevel.COMPACTED
            ):
                text = node.content.text or ""
                # Entity mentions should be preserved in compressed text
                # At least one entity should be present
                has_alice = "alice" in text.lower()
                has_anthropic = "anthropic" in text.lower()
                has_main_py = "main.py" in text.lower()
                assert has_alice or has_anthropic or has_main_py

    def test_compress_tool_results(self, strategy, graph_with_tool_results):
        """Test compression of TOOL_RESULT nodes with text content."""
        from context_compression.recovery import RecoveryManifest

        manifest = RecoveryManifest()
        result = strategy.compress(graph_with_tool_results, manifest)

        assert result.success is True
        # Should handle tool results with string output
        assert result.nodes_processed > 0

    def test_skips_pinned_nodes(self, strategy, graph_with_messages):
        """Test compression skips pinned nodes."""
        from context_compression.recovery import RecoveryManifest

        # Pin all message nodes
        for node in graph_with_messages:
            if node.type == NodeType.MESSAGE:
                node.metadata.pinned = True

        manifest = RecoveryManifest()
        result = strategy.compress(graph_with_messages, manifest)

        assert result.nodes_compressed == 0

    def test_skips_already_compressed_nodes(self, strategy, graph_with_messages):
        """Test compression skips already compressed nodes."""
        from context_compression.recovery import RecoveryManifest

        # Mark all message nodes as already compressed
        for node in graph_with_messages:
            if node.type == NodeType.MESSAGE:
                node.compression_level = CompressionLevel.COMPACTED

        manifest = RecoveryManifest()
        result = strategy.compress(graph_with_messages, manifest)

        assert result.nodes_compressed == 0

    def test_skips_json_tool_results(self, strategy):
        """Test compression skips JSON/dict tool results."""
        from context_compression.recovery import RecoveryManifest

        graph = ContextGraph()

        # Add tool results with JSON content (should be handled by SchemaCompression)
        call = graph.add_tool_call("get_data", {})
        result = graph.add_tool_result(call.id, {"data": [1, 2, 3]})
        result.token_count = 30

        manifest = RecoveryManifest()
        compression_result = strategy.compress(graph, manifest)

        # JSON content should be skipped
        assert compression_result.nodes_compressed == 0

    def test_estimate_savings(self, strategy, graph_with_messages):
        """Test savings estimation."""
        savings = strategy.estimate_savings(graph_with_messages)

        # Should estimate some savings (may be 0 if no content can be removed)
        assert savings >= 0

    def test_estimate_savings_empty_graph(self, strategy):
        """Test savings estimation on empty graph."""
        graph = ContextGraph()
        savings = strategy.estimate_savings(graph)
        assert savings == 0

    def test_can_apply_with_eligible_nodes(self, strategy, graph_with_messages):
        """Test can_apply returns True for eligible graph."""
        # Can apply should return True since we have entities and text content
        assert strategy.can_apply(graph_with_messages) is True

    def test_can_apply_empty_graph(self, strategy):
        """Test can_apply returns False for empty graph."""
        graph = ContextGraph()
        assert strategy.can_apply(graph) is False

    def test_can_apply_no_entities(self, graph_with_messages):
        """Test can_apply returns False when no important entities."""
        from context_compression.strategies.compaction import EntityCentricCompression
        from context_core.entities.tracker import EntityTracker

        # Create tracker with no entities
        empty_tracker = EntityTracker()
        strategy = EntityCentricCompression(entity_tracker=empty_tracker)

        assert strategy.can_apply(graph_with_messages) is False

    def test_compress_respects_target_tokens(self, strategy):
        """Test compression respects target tokens limit."""
        from context_compression.recovery import RecoveryManifest

        graph = ContextGraph()

        # Add many messages with content
        for i in range(10):
            msg = graph.add_message(
                "user",
                f"Message {i} mentions Alice and her work. "
                "The weather is nice today. It's a beautiful day. "
                "Other random content here.",
            )
            msg.token_count = 50

        manifest = RecoveryManifest()
        # Set a low target to stop early
        result = strategy.compress(graph, manifest, target_tokens=20)

        # Should have stopped before processing all nodes
        assert result.nodes_compressed < 10

    def test_compress_with_target_node_ids(self, strategy, graph_with_messages):
        """Test compression targeting specific nodes."""
        from context_compression.recovery import RecoveryManifest

        # Get first message node ID
        message_nodes = [
            node for node in graph_with_messages if node.type == NodeType.MESSAGE
        ]
        target_ids = [message_nodes[0].id]

        manifest = RecoveryManifest()
        result = strategy.compress(
            graph_with_messages, manifest, target_node_ids=target_ids
        )

        # Should only process targeted nodes
        assert result.nodes_compressed <= 1

    def test_result_is_not_recoverable(self, strategy, graph_with_messages):
        """Test that compression result marks as not recoverable (compaction)."""
        from context_compression.recovery import RecoveryManifest

        manifest = RecoveryManifest()
        result = strategy.compress(graph_with_messages, manifest)

        # Compaction is not fully recoverable
        assert result.is_recoverable is False

    def test_compress_updates_token_counts(self, strategy, graph_with_messages):
        """Test that compression updates token counts on nodes."""
        from context_compression.recovery import RecoveryManifest

        manifest = RecoveryManifest()
        strategy.compress(graph_with_messages, manifest)

        # Check that compressed nodes have updated token counts
        for node in graph_with_messages:
            if (
                node.type == NodeType.MESSAGE
                and node.compression_level == CompressionLevel.COMPACTED
            ):
                # Compressed token count should be set
                assert node.content.compressed_tokens is not None
                # Original tokens should be stored
                assert node.content.original_tokens is not None

    def test_include_context_sentences(self, entity_tracker):
        """Test including adjacent sentences for context."""
        from context_compression.recovery import RecoveryManifest
        from context_compression.strategies.compaction import EntityCentricCompression

        strategy = EntityCentricCompression(
            entity_tracker=entity_tracker, include_context_sentences=True
        )

        graph = ContextGraph()
        msg = graph.add_message(
            "user", "First sentence. Alice said hello. Third sentence."
        )
        msg.token_count = 30

        manifest = RecoveryManifest()
        strategy.compress(graph, manifest)

        # With context sentences enabled, adjacent sentences should be kept
        for node in graph:
            if node.compression_level == CompressionLevel.COMPACTED:
                text = node.content.text or ""
                # Should keep "Alice said hello" and potentially adjacent
                assert "alice" in text.lower()

    def test_compress_text_removes_irrelevant(self, strategy):
        """Test that _compress_text removes irrelevant sentences."""
        entity_names = {"Alice", "/src/main.py"}

        text = (
            "Alice is working on the project. "
            "The weather is nice today. "
            "Check /src/main.py for details. "
            "Random unrelated content here."
        )

        compressed, preserved, removed = strategy._compress_text(text, entity_names)

        # Should keep Alice and main.py sentences
        assert len(preserved) == 2
        # Should remove weather and random sentences
        assert len(removed) == 2
        # Compressed text should be shorter
        assert len(compressed) < len(text)
