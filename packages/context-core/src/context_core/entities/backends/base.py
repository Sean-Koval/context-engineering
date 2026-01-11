"""Base protocol for NER (Named Entity Recognition) backends.

This module defines the interface that all NER implementations must follow,
allowing pluggable entity extraction backends (spaCy, custom, etc.).
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from pydantic import BaseModel, Field

from context_core.entities.types import EntityType


class EntityMention(BaseModel):
    """A single mention of an entity in text.

    Represents one occurrence of an entity found by NER or pattern matching.
    Multiple mentions may resolve to the same canonical Entity.

    Attributes:
        text: The actual text matched
        entity_type: Classification of the entity
        start: Character offset where mention starts
        end: Character offset where mention ends
        confidence: Confidence score from extractor (0.0-1.0)

    Example:
        >>> mention = EntityMention(
        ...     text="John Smith",
        ...     entity_type=EntityType.PERSON,
        ...     start=0,
        ...     end=10,
        ...     confidence=0.95
        ... )
    """

    text: str
    entity_type: EntityType
    start: int = Field(ge=0)
    end: int = Field(ge=0)
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)

    def __hash__(self) -> int:
        """Allow mentions to be used in sets."""
        return hash((self.text, self.entity_type, self.start, self.end))


@runtime_checkable
class NERBackend(Protocol):
    """Protocol for NER (Named Entity Recognition) implementations.

    NER backends extract entity mentions from text. Implementations
    can use various approaches (spaCy, transformers, custom rules).

    The EntityTracker uses NER backends alongside pattern-based extraction
    to identify entities in context.

    Example implementation:
        >>> class CustomNERBackend:
        ...     def extract(self, text: str) -> list[EntityMention]:
        ...         # Custom extraction logic
        ...         return []
        ...
        ...     def supported_types(self) -> list[EntityType]:
        ...         return [EntityType.PERSON, EntityType.ORGANIZATION]
    """

    def extract(self, text: str) -> list[EntityMention]:
        """Extract entity mentions from text.

        Args:
            text: Input text to analyze

        Returns:
            List of entity mentions found in the text
        """
        ...

    def supported_types(self) -> list[EntityType]:
        """Return entity types this backend can extract.

        Returns:
            List of EntityType values this backend supports
        """
        ...
