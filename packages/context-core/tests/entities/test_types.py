"""Tests for entity type definitions."""

from __future__ import annotations

from datetime import UTC, datetime
from uuid import uuid4

import pytest

from context_core.entities.types import Entity, EntityPattern, EntityType


class TestEntityType:
    """Tests for EntityType enum."""

    def test_entity_type_values(self) -> None:
        """Test that all expected entity types exist."""
        assert EntityType.PERSON == "person"
        assert EntityType.ORGANIZATION == "organization"
        assert EntityType.LOCATION == "location"
        assert EntityType.FILE_PATH == "file_path"
        assert EntityType.URL == "url"
        assert EntityType.CODE_SYMBOL == "code_symbol"
        assert EntityType.TECHNICAL_TERM == "technical_term"
        assert EntityType.CUSTOM == "custom"

    def test_entity_type_is_string(self) -> None:
        """Test that EntityType values are strings."""
        for etype in EntityType:
            assert isinstance(etype.value, str)


class TestEntity:
    """Tests for Entity model."""

    def test_entity_creation_minimal(self) -> None:
        """Test creating entity with minimal fields."""
        entity = Entity(
            type=EntityType.PERSON,
            canonical_name="John Doe",
        )
        assert entity.type == EntityType.PERSON
        assert entity.canonical_name == "John Doe"
        assert entity.mention_count == 1
        assert entity.importance == 0.5
        assert len(entity.aliases) == 0
        assert len(entity.node_ids) == 0

    def test_entity_creation_full(self) -> None:
        """Test creating entity with all fields."""
        node_id = uuid4()
        now = datetime.now(UTC)

        entity = Entity(
            type=EntityType.FILE_PATH,
            canonical_name="/src/main.py",
            aliases={"main.py", "src/main.py"},
            first_seen=now,
            last_seen=now,
            mention_count=5,
            node_ids={node_id},
            importance=0.8,
            properties={"language": "python"},
        )

        assert entity.type == EntityType.FILE_PATH
        assert entity.canonical_name == "/src/main.py"
        assert "main.py" in entity.aliases
        assert entity.mention_count == 5
        assert node_id in entity.node_ids
        assert entity.importance == 0.8
        assert entity.properties["language"] == "python"

    def test_entity_add_mention(self) -> None:
        """Test adding a mention to an entity."""
        entity = Entity(
            type=EntityType.PERSON,
            canonical_name="John Doe",
        )
        node_id = uuid4()

        entity.add_mention(node_id, alias="J. Doe")

        assert entity.mention_count == 2
        assert node_id in entity.node_ids
        assert "J. Doe" in entity.aliases

    def test_entity_add_mention_same_name(self) -> None:
        """Test that canonical name is not added as alias."""
        entity = Entity(
            type=EntityType.PERSON,
            canonical_name="John Doe",
        )
        node_id = uuid4()

        entity.add_mention(node_id, alias="John Doe")

        assert "John Doe" not in entity.aliases

    def test_entity_merge_from(self) -> None:
        """Test merging two entities."""
        node1 = uuid4()
        node2 = uuid4()

        primary = Entity(
            type=EntityType.PERSON,
            canonical_name="John Doe",
            aliases={"John"},
            node_ids={node1},
            mention_count=3,
        )

        secondary = Entity(
            type=EntityType.PERSON,
            canonical_name="J. Doe",
            aliases={"Johnny"},
            node_ids={node2},
            mention_count=2,
            properties={"role": "developer"},
        )

        primary.merge_from(secondary)

        assert primary.mention_count == 5
        assert "John" in primary.aliases
        assert "Johnny" in primary.aliases
        assert "J. Doe" in primary.aliases
        assert node1 in primary.node_ids
        assert node2 in primary.node_ids
        assert primary.properties["role"] == "developer"

    def test_entity_importance_bounds(self) -> None:
        """Test that importance must be between 0 and 1."""
        with pytest.raises(ValueError):
            Entity(
                type=EntityType.PERSON,
                canonical_name="Test",
                importance=1.5,
            )

        with pytest.raises(ValueError):
            Entity(
                type=EntityType.PERSON,
                canonical_name="Test",
                importance=-0.1,
            )


class TestEntityPattern:
    """Tests for EntityPattern model."""

    def test_pattern_creation(self) -> None:
        """Test creating an entity pattern."""
        pattern = EntityPattern(
            name="python_import",
            entity_type=EntityType.CODE_SYMBOL,
            pattern=r"(?:from|import)\s+([\w.]+)",
            importance=0.6,
            group=1,
        )

        assert pattern.name == "python_import"
        assert pattern.entity_type == EntityType.CODE_SYMBOL
        assert pattern.importance == 0.6
        assert pattern.group == 1

    def test_pattern_defaults(self) -> None:
        """Test pattern default values."""
        pattern = EntityPattern(
            name="test",
            entity_type=EntityType.CUSTOM,
            pattern=r"\w+",
        )

        assert pattern.importance == 0.5
        assert pattern.group == 0
