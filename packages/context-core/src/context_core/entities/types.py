"""Type definitions for entity tracking.

This module defines core types for entity extraction and tracking:
- EntityType: Classification of entities (person, file, URL, etc.)
- Entity: A tracked entity with metadata and mentions
- EntityPattern: Custom regex patterns for entity extraction
"""

from __future__ import annotations

from datetime import UTC, datetime
from enum import Enum
from typing import Any
from uuid import UUID, uuid4

from pydantic import BaseModel, Field


class EntityType(str, Enum):
    """Classification of entity types.

    Built-in entity types for common extraction targets:
    - PERSON: Human names and references
    - ORGANIZATION: Company, team, or group names
    - LOCATION: Physical or virtual locations
    - FILE_PATH: File system paths
    - URL: Web URLs and URIs
    - CODE_SYMBOL: Function, class, variable names
    - TECHNICAL_TERM: Domain-specific terminology
    - CUSTOM: User-defined entity types
    """

    PERSON = "person"
    ORGANIZATION = "organization"
    LOCATION = "location"
    FILE_PATH = "file_path"
    URL = "url"
    CODE_SYMBOL = "code_symbol"
    TECHNICAL_TERM = "technical_term"
    CUSTOM = "custom"


class Entity(BaseModel):
    """A tracked entity with metadata.

    Entities represent unique concepts, people, files, or other named items
    that appear across the context. The tracker resolves multiple mentions
    to canonical entities.

    Attributes:
        id: Unique identifier for this entity
        type: Classification of the entity
        canonical_name: Normalized primary name
        aliases: Alternative names/mentions for this entity
        first_seen: When this entity was first encountered
        last_seen: Most recent mention timestamp
        mention_count: Total number of times mentioned
        node_ids: Context graph nodes that mention this entity
        importance: Calculated importance score (0.0-1.0)
        properties: Additional metadata key-value pairs

    Example:
        >>> entity = Entity(
        ...     type=EntityType.FILE_PATH,
        ...     canonical_name="/src/main.py",
        ...     importance=0.8
        ... )
        >>> entity.add_mention(node_id, alias="main.py")
    """

    id: UUID = Field(default_factory=uuid4)
    type: EntityType
    canonical_name: str
    aliases: set[str] = Field(default_factory=set)

    # Occurrence tracking
    first_seen: datetime = Field(default_factory=lambda: datetime.now(UTC))
    last_seen: datetime = Field(default_factory=lambda: datetime.now(UTC))
    mention_count: int = Field(default=1, ge=1)
    node_ids: set[UUID] = Field(default_factory=set)

    # Importance score
    importance: float = Field(default=0.5, ge=0.0, le=1.0)

    # Custom properties
    properties: dict[str, Any] = Field(default_factory=dict)

    model_config = {"arbitrary_types_allowed": True}

    def add_mention(
        self,
        node_id: UUID,
        alias: str | None = None,
        timestamp: datetime | None = None,
    ) -> None:
        """Record a new mention of this entity.

        Args:
            node_id: ID of the context node containing this mention
            alias: Alternative name used in this mention (if different from canonical)
            timestamp: When the mention occurred (defaults to now)
        """
        self.mention_count += 1
        self.last_seen = timestamp or datetime.now(UTC)
        self.node_ids.add(node_id)
        if alias and alias != self.canonical_name:
            self.aliases.add(alias)

    def merge_from(self, other: Entity) -> None:
        """Merge another entity into this one.

        Used when resolving duplicate entities. Combines aliases,
        node references, and updates timestamps.

        Args:
            other: Entity to merge into this one
        """
        self.aliases.update(other.aliases)
        self.aliases.add(other.canonical_name)
        self.node_ids.update(other.node_ids)
        self.mention_count += other.mention_count

        # Update timestamps
        if other.first_seen < self.first_seen:
            self.first_seen = other.first_seen
        if other.last_seen > self.last_seen:
            self.last_seen = other.last_seen

        # Merge properties (other's values take precedence for conflicts)
        merged_props = {**self.properties, **other.properties}
        self.properties = merged_props


class EntityPattern(BaseModel):
    """Custom pattern for entity extraction.

    Defines a regex pattern to extract entities of a specific type.
    Patterns are applied during entity extraction alongside NER backends.

    Attributes:
        name: Identifier for this pattern
        entity_type: Type to assign to matched entities
        pattern: Regex pattern string
        importance: Default importance for entities matched by this pattern
        group: Regex group number to extract (0 for full match)

    Example:
        >>> pattern = EntityPattern(
        ...     name="python_import",
        ...     entity_type=EntityType.CODE_SYMBOL,
        ...     pattern=r"(?:from|import)\\s+([\\w.]+)",
        ...     group=1,
        ...     importance=0.5
        ... )
    """

    name: str
    entity_type: EntityType
    pattern: str
    importance: float = Field(default=0.5, ge=0.0, le=1.0)
    group: int = Field(default=0, ge=0)
