"""Entity tracking module for named entity recognition and linking.

This module provides:
- EntityTracker: Tracks entities across the context graph
- Entity: Represents a unique tracked entity
- EntityType: Classification of entity types
- EntityPattern: Custom regex patterns for extraction
- NERBackend: Protocol for NER implementations
- SpaCyNERBackend: spaCy-based NER (requires spacy package)

Example:
    >>> from context_core.entities import EntityTracker, EntityType
    >>> tracker = EntityTracker()
    >>> entities = tracker.extract_from_text(
    ...     "John works at Google in NYC.",
    ...     node_id=some_uuid
    ... )
    >>> for e in entities:
    ...     print(f"{e.canonical_name}: {e.type.value}")
"""

from __future__ import annotations

from context_core.entities.backends import (
    EntityMention,
    NERBackend,
    NoOpNERBackend,
    SpaCyNERBackend,
    get_ner_backend,
)
from context_core.entities.tracker import EntityTracker
from context_core.entities.types import Entity, EntityPattern, EntityType

__all__ = [
    # Types
    "Entity",
    "EntityMention",
    "EntityPattern",
    "EntityType",
    # Tracker
    "EntityTracker",
    # Backends
    "NERBackend",
    "NoOpNERBackend",
    "SpaCyNERBackend",
    "get_ner_backend",
]
