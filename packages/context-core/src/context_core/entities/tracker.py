"""EntityTracker for tracking entities across the context graph.

The EntityTracker extracts, resolves, and tracks entities mentioned
in context. It combines NER-based extraction with pattern matching
for comprehensive entity detection.
"""

from __future__ import annotations

import logging
import re
from datetime import UTC, datetime
from typing import Any
from uuid import UUID

from context_core.entities.backends.base import EntityMention, NERBackend
from context_core.entities.types import Entity, EntityPattern, EntityType

logger = logging.getLogger(__name__)


class EntityTracker:
    """Tracks entities across the context graph.

    The EntityTracker provides:
    - NER-based extraction (via pluggable backends)
    - Pattern-based extraction (regex)
    - Entity resolution (merging duplicate mentions)
    - Importance scoring based on frequency and recency

    Entities are identified by their canonical name and type. Multiple
    mentions (including aliases) resolve to the same entity.

    Attributes:
        similarity_threshold: Threshold for fuzzy name matching (0.0-1.0)

    Example:
        >>> tracker = EntityTracker()
        >>> tracker.register_pattern(EntityPattern(
        ...     name="github_repo",
        ...     entity_type=EntityType.URL,
        ...     pattern=r"github\\.com/[\\w-]+/[\\w-]+",
        ... ))
        >>> entities = tracker.extract_from_text(
        ...     "Check out github.com/anthropics/claude",
        ...     node_id=some_uuid
        ... )
    """

    def __init__(
        self,
        ner_backend: NERBackend | None = None,
        similarity_threshold: float = 0.85,
    ) -> None:
        """Initialize the EntityTracker.

        Args:
            ner_backend: NER backend for entity extraction (optional)
            similarity_threshold: Threshold for fuzzy name matching
        """
        self._entities: dict[UUID, Entity] = {}
        self._name_index: dict[str, UUID] = {}  # normalized_name -> entity_id
        self._type_index: dict[EntityType, set[UUID]] = {}  # type -> entity_ids
        self._ner_backend = ner_backend
        self._patterns: list[EntityPattern] = []
        self._similarity_threshold = similarity_threshold

        # Register default patterns
        self._register_default_patterns()

    def _register_default_patterns(self) -> None:
        """Register patterns for common technical entities."""
        self._patterns.extend(
            [
                EntityPattern(
                    name="file_path_unix",
                    entity_type=EntityType.FILE_PATH,
                    pattern=r"(?:/[\w\-. ]+)+(?:\.\w+)?",
                    importance=0.7,
                ),
                EntityPattern(
                    name="file_path_windows",
                    entity_type=EntityType.FILE_PATH,
                    pattern=r"[A-Za-z]:\\(?:[\w\-. ]+\\)*[\w\-. ]+(?:\.\w+)?",
                    importance=0.7,
                ),
                EntityPattern(
                    name="url",
                    entity_type=EntityType.URL,
                    pattern=r'https?://[^\s<>"{}|\\^`\[\]]+',
                    importance=0.6,
                ),
                EntityPattern(
                    name="python_import",
                    entity_type=EntityType.CODE_SYMBOL,
                    pattern=r"(?:from|import)\s+([\w.]+)",
                    importance=0.5,
                    group=1,
                ),
                EntityPattern(
                    name="python_class",
                    entity_type=EntityType.CODE_SYMBOL,
                    pattern=r"class\s+(\w+)",
                    importance=0.6,
                    group=1,
                ),
                EntityPattern(
                    name="python_function",
                    entity_type=EntityType.CODE_SYMBOL,
                    pattern=r"def\s+(\w+)",
                    importance=0.5,
                    group=1,
                ),
            ]
        )

    def register_pattern(self, pattern: EntityPattern) -> None:
        """Register a custom extraction pattern.

        Args:
            pattern: Pattern configuration for entity extraction
        """
        self._patterns.append(pattern)
        logger.debug(f"Registered entity pattern: {pattern.name}")

    def extract_from_text(
        self,
        text: str,
        node_id: UUID,
        timestamp: datetime | None = None,
    ) -> list[Entity]:
        """Extract entities from text and register them.

        Combines NER extraction with pattern matching. Resolves
        mentions to existing entities or creates new ones.

        Args:
            text: Text content to extract entities from
            node_id: ID of the context node containing this text
            timestamp: When this text was added (defaults to now)

        Returns:
            List of entities found (new or existing)
        """
        if not text or not text.strip():
            return []

        timestamp = timestamp or datetime.now(UTC)
        mentions: list[EntityMention] = []

        # NER extraction
        if self._ner_backend:
            try:
                ner_mentions = self._ner_backend.extract(text)
                mentions.extend(ner_mentions)
            except Exception as e:
                logger.warning(f"NER extraction failed: {e}")

        # Pattern extraction
        for pattern in self._patterns:
            try:
                for match in re.finditer(pattern.pattern, text):
                    # Get the matched text (use group if specified)
                    if pattern.group > 0 and pattern.group <= len(match.groups()):
                        matched_text = match.group(pattern.group)
                        start = match.start(pattern.group)
                        end = match.end(pattern.group)
                    else:
                        matched_text = match.group(0)
                        start = match.start()
                        end = match.end()

                    if matched_text:
                        mentions.append(
                            EntityMention(
                                text=matched_text,
                                entity_type=pattern.entity_type,
                                start=start,
                                end=end,
                                confidence=0.9,
                            )
                        )
            except re.error as e:
                logger.warning(f"Pattern '{pattern.name}' regex error: {e}")

        # Deduplicate mentions by position
        seen_positions: set[tuple[int, int]] = set()
        unique_mentions: list[EntityMention] = []
        for mention in mentions:
            pos = (mention.start, mention.end)
            if pos not in seen_positions:
                seen_positions.add(pos)
                unique_mentions.append(mention)

        # Process mentions and resolve to entities
        entities: list[Entity] = []
        for mention in unique_mentions:
            entity = self._resolve_or_create(mention, node_id, timestamp)
            if entity not in entities:
                entities.append(entity)

        return entities

    def _normalize_name(self, name: str) -> str:
        """Normalize an entity name for matching.

        Args:
            name: Raw entity name

        Returns:
            Normalized lowercase name with extra whitespace removed
        """
        return " ".join(name.lower().split())

    def _resolve_or_create(
        self,
        mention: EntityMention,
        node_id: UUID,
        timestamp: datetime,
    ) -> Entity:
        """Find existing entity or create new one.

        Args:
            mention: The entity mention to resolve
            node_id: ID of the containing context node
            timestamp: When the mention occurred

        Returns:
            Resolved or newly created Entity
        """
        # Normalize the name
        canonical = self._normalize_name(mention.text)

        # Check exact match in name index
        if canonical in self._name_index:
            entity_id = self._name_index[canonical]
            entity = self._entities[entity_id]
            entity.add_mention(node_id, mention.text, timestamp)
            self._update_importance(entity)
            return entity

        # Check aliases of existing entities
        for entity in self._entities.values():
            if entity.type != mention.entity_type:
                continue
            normalized_aliases = {self._normalize_name(a) for a in entity.aliases}
            if canonical in normalized_aliases:
                entity.add_mention(node_id, mention.text, timestamp)
                self._update_importance(entity)
                return entity

        # Create new entity
        entity = Entity(
            type=mention.entity_type,
            canonical_name=mention.text,
            first_seen=timestamp,
            last_seen=timestamp,
            mention_count=1,
            node_ids={node_id},
            importance=0.5,
        )

        # Register in indices
        self._entities[entity.id] = entity
        self._name_index[canonical] = entity.id

        if mention.entity_type not in self._type_index:
            self._type_index[mention.entity_type] = set()
        self._type_index[mention.entity_type].add(entity.id)

        logger.debug(
            f"Created new entity: {entity.canonical_name} ({entity.type.value})"
        )
        return entity

    def _update_importance(self, entity: Entity) -> None:
        """Update entity importance based on mentions and recency.

        Importance formula combines:
        - Mention frequency (log scale)
        - Recency (exponential decay)

        Args:
            entity: Entity to update importance for
        """
        import math

        # Frequency component (log scale, capped)
        freq_score = min(1.0, math.log(entity.mention_count + 1) / math.log(20))

        # Recency component (higher is more recent)
        now = datetime.now(UTC)
        hours_since_last = (now - entity.last_seen).total_seconds() / 3600
        recency_score = math.exp(-hours_since_last / 24)  # 24-hour half-life

        # Combined score (weighted average)
        entity.importance = 0.6 * freq_score + 0.4 * recency_score

    def get_entity(self, entity_id: UUID) -> Entity | None:
        """Get an entity by ID.

        Args:
            entity_id: UUID of the entity

        Returns:
            Entity if found, None otherwise
        """
        return self._entities.get(entity_id)

    def get_entity_by_name(
        self,
        name: str,
        entity_type: EntityType | None = None,
    ) -> Entity | None:
        """Find an entity by name.

        Args:
            name: Entity name to search for
            entity_type: Optional type filter

        Returns:
            Entity if found, None otherwise
        """
        normalized = self._normalize_name(name)

        if normalized in self._name_index:
            entity = self._entities[self._name_index[normalized]]
            if entity_type is None or entity.type == entity_type:
                return entity

        # Check aliases
        for entity in self._entities.values():
            if entity_type and entity.type != entity_type:
                continue
            normalized_aliases = {self._normalize_name(a) for a in entity.aliases}
            if normalized in normalized_aliases:
                return entity

        return None

    def get_entities_by_type(self, entity_type: EntityType) -> list[Entity]:
        """Get all entities of a specific type.

        Args:
            entity_type: Type to filter by

        Returns:
            List of entities of the specified type
        """
        entity_ids = self._type_index.get(entity_type, set())
        return [self._entities[eid] for eid in entity_ids]

    def get_entities_by_node(self, node_id: UUID) -> list[Entity]:
        """Get all entities mentioned in a specific node.

        Args:
            node_id: Context node ID

        Returns:
            List of entities mentioned in that node
        """
        return [e for e in self._entities.values() if node_id in e.node_ids]

    def get_top_entities(
        self,
        limit: int = 10,
        entity_type: EntityType | None = None,
    ) -> list[Entity]:
        """Get the most important entities.

        Args:
            limit: Maximum number of entities to return
            entity_type: Optional type filter

        Returns:
            List of entities sorted by importance (descending)
        """
        entities = list(self._entities.values())
        if entity_type:
            entities = [e for e in entities if e.type == entity_type]

        entities.sort(key=lambda e: e.importance, reverse=True)
        return entities[:limit]

    def merge_entities(self, primary_id: UUID, secondary_id: UUID) -> Entity | None:
        """Merge two entities into one.

        The primary entity absorbs the secondary entity's data.
        The secondary entity is removed.

        Args:
            primary_id: ID of entity to keep
            secondary_id: ID of entity to merge and remove

        Returns:
            Merged entity, or None if IDs not found
        """
        primary = self._entities.get(primary_id)
        secondary = self._entities.get(secondary_id)

        if not primary or not secondary:
            return None

        # Merge data
        primary.merge_from(secondary)

        # Update indices
        secondary_canonical = self._normalize_name(secondary.canonical_name)
        self._name_index[secondary_canonical] = primary_id

        for alias in secondary.aliases:
            self._name_index[self._normalize_name(alias)] = primary_id

        # Remove secondary from type index
        if secondary.type in self._type_index:
            self._type_index[secondary.type].discard(secondary_id)

        # Remove secondary entity
        del self._entities[secondary_id]

        logger.debug(
            f"Merged entity '{secondary.canonical_name}' into "
            f"'{primary.canonical_name}'"
        )
        return primary

    def all_entities(self) -> list[Entity]:
        """Get all tracked entities.

        Returns:
            List of all entities
        """
        return list(self._entities.values())

    def entity_count(self) -> int:
        """Get total number of tracked entities.

        Returns:
            Count of entities
        """
        return len(self._entities)

    def clear(self) -> None:
        """Remove all tracked entities."""
        self._entities.clear()
        self._name_index.clear()
        self._type_index.clear()
        logger.debug("Cleared all entities")

    def to_dict(self) -> dict[str, Any]:
        """Serialize tracker state to dictionary.

        Returns:
            Dictionary representation of tracker state
        """
        return {
            "entities": [
                {
                    "id": str(e.id),
                    "type": e.type.value,
                    "canonical_name": e.canonical_name,
                    "aliases": list(e.aliases),
                    "mention_count": e.mention_count,
                    "importance": e.importance,
                    "node_ids": [str(nid) for nid in e.node_ids],
                    "first_seen": e.first_seen.isoformat(),
                    "last_seen": e.last_seen.isoformat(),
                    "properties": e.properties,
                }
                for e in self._entities.values()
            ],
            "patterns": [
                {
                    "name": p.name,
                    "entity_type": p.entity_type.value,
                    "pattern": p.pattern,
                    "importance": p.importance,
                    "group": p.group,
                }
                for p in self._patterns
            ],
        }

    @classmethod
    def from_dict(
        cls,
        data: dict[str, Any],
        ner_backend: NERBackend | None = None,
    ) -> EntityTracker:
        """Deserialize tracker from dictionary.

        Args:
            data: Dictionary from to_dict()
            ner_backend: Optional NER backend to use

        Returns:
            Reconstructed EntityTracker
        """
        from datetime import datetime
        from uuid import UUID as _UUID

        tracker = cls(ner_backend=ner_backend)
        tracker._patterns.clear()  # Remove default patterns

        # Restore patterns
        for p_data in data.get("patterns", []):
            tracker._patterns.append(
                EntityPattern(
                    name=p_data["name"],
                    entity_type=EntityType(p_data["entity_type"]),
                    pattern=p_data["pattern"],
                    importance=p_data.get("importance", 0.5),
                    group=p_data.get("group", 0),
                )
            )

        # Restore entities
        for e_data in data.get("entities", []):
            entity = Entity(
                id=_UUID(e_data["id"]),
                type=EntityType(e_data["type"]),
                canonical_name=e_data["canonical_name"],
                aliases=set(e_data.get("aliases", [])),
                mention_count=e_data.get("mention_count", 1),
                importance=e_data.get("importance", 0.5),
                node_ids={_UUID(nid) for nid in e_data.get("node_ids", [])},
                first_seen=datetime.fromisoformat(e_data["first_seen"]),
                last_seen=datetime.fromisoformat(e_data["last_seen"]),
                properties=e_data.get("properties", {}),
            )
            tracker._entities[entity.id] = entity
            tracker._name_index[tracker._normalize_name(entity.canonical_name)] = (
                entity.id
            )

            if entity.type not in tracker._type_index:
                tracker._type_index[entity.type] = set()
            tracker._type_index[entity.type].add(entity.id)

        return tracker
