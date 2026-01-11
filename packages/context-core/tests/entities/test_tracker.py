"""Tests for EntityTracker."""

from __future__ import annotations

from uuid import uuid4

from context_core.entities import (
    EntityMention,
    EntityPattern,
    EntityTracker,
    EntityType,
    NERBackend,
    NoOpNERBackend,
)


class MockNERBackend:
    """Mock NER backend for testing."""

    def __init__(self, mentions: list[EntityMention] | None = None) -> None:
        self._mentions = mentions or []

    def extract(self, text: str) -> list[EntityMention]:
        return self._mentions

    def supported_types(self) -> list[EntityType]:
        return [EntityType.PERSON, EntityType.ORGANIZATION]


class TestEntityTracker:
    """Tests for EntityTracker class."""

    def test_tracker_creation(self) -> None:
        """Test creating an entity tracker."""
        tracker = EntityTracker()
        assert tracker.entity_count() == 0

    def test_tracker_with_ner_backend(self) -> None:
        """Test creating tracker with NER backend."""
        backend = MockNERBackend()
        tracker = EntityTracker(ner_backend=backend)
        assert tracker.entity_count() == 0

    def test_extract_url_pattern(self) -> None:
        """Test URL pattern extraction."""
        tracker = EntityTracker()
        node_id = uuid4()

        entities = tracker.extract_from_text(
            "Check out https://github.com/anthropics/claude",
            node_id=node_id,
        )

        assert len(entities) >= 1
        url_entities = [e for e in entities if e.type == EntityType.URL]
        assert len(url_entities) == 1
        assert "github.com" in url_entities[0].canonical_name

    def test_extract_file_path_pattern(self) -> None:
        """Test file path pattern extraction."""
        tracker = EntityTracker()
        node_id = uuid4()

        entities = tracker.extract_from_text(
            "Edit the file /src/main.py to fix the bug",
            node_id=node_id,
        )

        assert len(entities) >= 1
        path_entities = [e for e in entities if e.type == EntityType.FILE_PATH]
        assert len(path_entities) == 1
        assert "/src/main.py" in path_entities[0].canonical_name

    def test_extract_code_symbol_import(self) -> None:
        """Test code symbol extraction from imports."""
        tracker = EntityTracker()
        node_id = uuid4()

        entities = tracker.extract_from_text(
            "from context_core.entities import EntityTracker",
            node_id=node_id,
        )

        code_entities = [e for e in entities if e.type == EntityType.CODE_SYMBOL]
        assert len(code_entities) >= 1

    def test_extract_with_ner_backend(self) -> None:
        """Test extraction with NER backend."""
        mentions = [
            EntityMention(
                text="John Smith",
                entity_type=EntityType.PERSON,
                start=0,
                end=10,
                confidence=0.95,
            ),
            EntityMention(
                text="Google",
                entity_type=EntityType.ORGANIZATION,
                start=20,
                end=26,
                confidence=0.9,
            ),
        ]
        backend = MockNERBackend(mentions=mentions)
        tracker = EntityTracker(ner_backend=backend)
        node_id = uuid4()

        entities = tracker.extract_from_text(
            "John Smith works at Google.",
            node_id=node_id,
        )

        assert len(entities) >= 2
        names = [e.canonical_name for e in entities]
        assert "John Smith" in names
        assert "Google" in names

    def test_entity_resolution(self) -> None:
        """Test that same entity is resolved across mentions."""
        tracker = EntityTracker()
        node1 = uuid4()
        node2 = uuid4()

        # First mention
        tracker.extract_from_text(
            "Check https://example.com for info",
            node_id=node1,
        )

        # Same URL mentioned again
        tracker.extract_from_text(
            "Visit https://example.com today",
            node_id=node2,
        )

        # Should be same entity
        assert tracker.entity_count() >= 1
        url_entities = tracker.get_entities_by_type(EntityType.URL)
        example_entities = [
            e for e in url_entities if "example.com" in e.canonical_name
        ]
        assert len(example_entities) == 1
        assert example_entities[0].mention_count == 2
        assert node1 in example_entities[0].node_ids
        assert node2 in example_entities[0].node_ids

    def test_register_custom_pattern(self) -> None:
        """Test registering a custom extraction pattern."""
        tracker = EntityTracker()

        tracker.register_pattern(
            EntityPattern(
                name="issue_number",
                entity_type=EntityType.CUSTOM,
                pattern=r"#(\d+)",
                importance=0.7,
                group=1,
            )
        )

        node_id = uuid4()
        entities = tracker.extract_from_text(
            "Fixed in #123 and #456",
            node_id=node_id,
        )

        custom_entities = [e for e in entities if e.type == EntityType.CUSTOM]
        assert len(custom_entities) == 2
        names = {e.canonical_name for e in custom_entities}
        assert "123" in names
        assert "456" in names

    def test_get_entity_by_id(self) -> None:
        """Test getting entity by ID."""
        tracker = EntityTracker()
        node_id = uuid4()

        entities = tracker.extract_from_text(
            "Visit https://test.com",
            node_id=node_id,
        )

        if entities:
            entity = tracker.get_entity(entities[0].id)
            assert entity is not None
            assert entity.id == entities[0].id

    def test_get_entity_by_name(self) -> None:
        """Test getting entity by name."""
        tracker = EntityTracker()
        node_id = uuid4()

        tracker.extract_from_text(
            "Edit /src/app.py",
            node_id=node_id,
        )

        entity = tracker.get_entity_by_name("/src/app.py")
        assert entity is not None
        assert "/src/app.py" in entity.canonical_name

    def test_get_entity_by_name_with_type(self) -> None:
        """Test getting entity by name with type filter."""
        tracker = EntityTracker()
        node_id = uuid4()

        tracker.extract_from_text(
            "Edit /src/app.py",
            node_id=node_id,
        )

        # Should find with correct type
        entity = tracker.get_entity_by_name(
            "/src/app.py",
            entity_type=EntityType.FILE_PATH,
        )
        assert entity is not None

        # Should not find with wrong type
        entity = tracker.get_entity_by_name(
            "/src/app.py",
            entity_type=EntityType.PERSON,
        )
        assert entity is None

    def test_get_entities_by_type(self) -> None:
        """Test getting all entities of a type."""
        tracker = EntityTracker()
        node_id = uuid4()

        tracker.extract_from_text(
            "Check https://a.com and https://b.com",
            node_id=node_id,
        )

        url_entities = tracker.get_entities_by_type(EntityType.URL)
        assert len(url_entities) == 2

    def test_get_entities_by_node(self) -> None:
        """Test getting entities by node ID."""
        tracker = EntityTracker()
        node1 = uuid4()
        node2 = uuid4()

        tracker.extract_from_text("Visit https://a.com", node_id=node1)
        tracker.extract_from_text("Visit https://b.com", node_id=node2)

        node1_entities = tracker.get_entities_by_node(node1)
        assert len(node1_entities) >= 1
        assert all(node1 in e.node_ids for e in node1_entities)

    def test_get_top_entities(self) -> None:
        """Test getting top entities by importance."""
        tracker = EntityTracker()

        # Create multiple entities with different mention counts
        for i in range(5):
            for _ in range(i + 1):
                node_id = uuid4()
                tracker.extract_from_text(
                    f"Visit https://site{i}.com",
                    node_id=node_id,
                )

        top = tracker.get_top_entities(limit=3)
        assert len(top) == 3

        # Higher mention counts should have higher importance
        assert top[0].mention_count >= top[1].mention_count

    def test_merge_entities(self) -> None:
        """Test merging two entities."""
        tracker = EntityTracker()
        node1 = uuid4()
        node2 = uuid4()

        # Create two separate entities
        tracker.extract_from_text("Visit https://a.com", node_id=node1)
        tracker.extract_from_text("Visit https://b.com", node_id=node2)

        entities = tracker.get_entities_by_type(EntityType.URL)
        assert len(entities) == 2

        # Merge them
        primary = entities[0]
        secondary = entities[1]
        merged = tracker.merge_entities(primary.id, secondary.id)

        assert merged is not None
        assert tracker.entity_count() >= 1
        assert secondary.canonical_name in merged.aliases

    def test_all_entities(self) -> None:
        """Test getting all entities."""
        tracker = EntityTracker()
        node_id = uuid4()

        tracker.extract_from_text(
            "Visit https://a.com and /path/to/file.py",
            node_id=node_id,
        )

        all_entities = tracker.all_entities()
        assert len(all_entities) >= 2

    def test_clear(self) -> None:
        """Test clearing all entities."""
        tracker = EntityTracker()
        node_id = uuid4()

        tracker.extract_from_text("Visit https://test.com", node_id=node_id)
        assert tracker.entity_count() >= 1

        tracker.clear()
        assert tracker.entity_count() == 0

    def test_empty_text_extraction(self) -> None:
        """Test extraction from empty text."""
        tracker = EntityTracker()
        node_id = uuid4()

        entities = tracker.extract_from_text("", node_id=node_id)
        assert entities == []

        entities = tracker.extract_from_text("   ", node_id=node_id)
        assert entities == []

    def test_to_dict_and_from_dict(self) -> None:
        """Test serialization and deserialization."""
        tracker = EntityTracker()
        node_id = uuid4()

        tracker.register_pattern(
            EntityPattern(
                name="custom",
                entity_type=EntityType.CUSTOM,
                pattern=r"TEST\d+",
            )
        )

        tracker.extract_from_text(
            "Visit https://example.com TEST123",
            node_id=node_id,
        )

        # Serialize
        data = tracker.to_dict()
        assert "entities" in data
        assert "patterns" in data
        assert len(data["entities"]) >= 1

        # Deserialize
        new_tracker = EntityTracker.from_dict(data)
        assert new_tracker.entity_count() == tracker.entity_count()

        # Check entities match
        for entity in tracker.all_entities():
            restored = new_tracker.get_entity(entity.id)
            assert restored is not None
            assert restored.canonical_name == entity.canonical_name
            assert restored.type == entity.type


class TestNoOpNERBackend:
    """Tests for NoOpNERBackend."""

    def test_extract_returns_empty(self) -> None:
        """Test that NoOpNERBackend returns no entities."""
        backend = NoOpNERBackend()
        mentions = backend.extract("John works at Google")
        assert mentions == []

    def test_supported_types_empty(self) -> None:
        """Test that NoOpNERBackend supports no types."""
        backend = NoOpNERBackend()
        types = backend.supported_types()
        assert types == []

    def test_is_ner_backend(self) -> None:
        """Test that NoOpNERBackend implements NERBackend protocol."""
        backend = NoOpNERBackend()
        assert isinstance(backend, NERBackend)
