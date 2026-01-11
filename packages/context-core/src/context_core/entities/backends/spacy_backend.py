"""spaCy-based NER backend for entity extraction.

This module provides a spaCy implementation of the NERBackend protocol.
spaCy is optional - if not installed, SpaCyNERBackend will raise an error
when instantiated.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from context_core.entities.backends.base import EntityMention
from context_core.entities.types import EntityType

if TYPE_CHECKING:
    import spacy
    from spacy.language import Language

logger = logging.getLogger(__name__)

# Mapping from spaCy entity labels to our EntityType
SPACY_LABEL_MAP: dict[str, EntityType] = {
    # Person names
    "PERSON": EntityType.PERSON,
    "PER": EntityType.PERSON,
    # Organizations
    "ORG": EntityType.ORGANIZATION,
    "NORP": EntityType.ORGANIZATION,  # Nationalities, religious/political groups
    "FAC": EntityType.ORGANIZATION,  # Facilities
    # Locations
    "GPE": EntityType.LOCATION,  # Countries, cities, states
    "LOC": EntityType.LOCATION,  # Non-GPE locations
    # Technical (mapped to TECHNICAL_TERM as fallback)
    "PRODUCT": EntityType.TECHNICAL_TERM,
    "EVENT": EntityType.TECHNICAL_TERM,
    "WORK_OF_ART": EntityType.TECHNICAL_TERM,
    "LAW": EntityType.TECHNICAL_TERM,
    "LANGUAGE": EntityType.TECHNICAL_TERM,
}


def _get_spacy() -> spacy:
    """Import spaCy with helpful error message if not available."""
    try:
        import spacy

        return spacy
    except ImportError as e:
        raise ImportError(
            "spaCy is required for SpaCyNERBackend. "
            "Install it with: uv add spacy\n"
            "Then download a model: python -m spacy download en_core_web_sm"
        ) from e


class SpaCyNERBackend:
    """spaCy-based Named Entity Recognition backend.

    Uses spaCy's NER pipeline to extract entities from text.
    Supports English models by default but can be configured
    for other languages.

    Attributes:
        model_name: Name of the spaCy model to use
        nlp: Loaded spaCy language model

    Example:
        >>> backend = SpaCyNERBackend()  # Uses en_core_web_sm
        >>> mentions = backend.extract("John works at Google in NYC.")
        >>> for m in mentions:
        ...     print(f"{m.text}: {m.entity_type}")
        John: person
        Google: organization
        NYC: location
    """

    def __init__(
        self,
        model_name: str = "en_core_web_sm",
        nlp: Language | None = None,
    ) -> None:
        """Initialize the spaCy NER backend.

        Args:
            model_name: Name of spaCy model to load (default: en_core_web_sm)
            nlp: Pre-loaded spaCy Language object (optional, for testing)

        Raises:
            ImportError: If spaCy is not installed
            OSError: If the specified model is not available
        """
        self.model_name = model_name

        if nlp is not None:
            self._nlp = nlp
        else:
            spacy = _get_spacy()
            try:
                self._nlp = spacy.load(model_name)
            except OSError as e:
                raise OSError(
                    f"spaCy model '{model_name}' not found. "
                    f"Download it with: python -m spacy download {model_name}"
                ) from e

        logger.debug(f"Initialized SpaCyNERBackend with model: {model_name}")

    def extract(self, text: str) -> list[EntityMention]:
        """Extract entity mentions from text using spaCy NER.

        Args:
            text: Input text to analyze

        Returns:
            List of EntityMention objects for recognized entities
        """
        if not text or not text.strip():
            return []

        doc = self._nlp(text)
        mentions: list[EntityMention] = []

        for ent in doc.ents:
            # Map spaCy label to our EntityType
            entity_type = SPACY_LABEL_MAP.get(ent.label_)
            if entity_type is None:
                # Skip entity types we don't recognize
                logger.debug(
                    f"Skipping unknown spaCy entity type: {ent.label_} "
                    f"for text: {ent.text}"
                )
                continue

            mention = EntityMention(
                text=ent.text,
                entity_type=entity_type,
                start=ent.start_char,
                end=ent.end_char,
                confidence=0.9,  # spaCy doesn't provide per-entity confidence
            )
            mentions.append(mention)

        return mentions

    def supported_types(self) -> list[EntityType]:
        """Return entity types this backend can extract.

        Returns:
            List of EntityType values supported by the loaded model
        """
        return list(set(SPACY_LABEL_MAP.values()))


class NoOpNERBackend:
    """A no-operation NER backend for when spaCy is unavailable.

    This backend extracts no entities and is used as a fallback
    when spaCy is not installed. Pattern-based extraction in
    EntityTracker will still work.

    Example:
        >>> backend = NoOpNERBackend()
        >>> backend.extract("Any text")
        []
    """

    def extract(self, text: str) -> list[EntityMention]:
        """Return empty list - no NER capability.

        Args:
            text: Input text (ignored)

        Returns:
            Empty list
        """
        return []

    def supported_types(self) -> list[EntityType]:
        """Return empty list - no types supported.

        Returns:
            Empty list
        """
        return []


def get_ner_backend(
    model_name: str = "en_core_web_sm",
    fallback_to_noop: bool = True,
) -> SpaCyNERBackend | NoOpNERBackend:
    """Factory function to get an NER backend.

    Attempts to create a SpaCyNERBackend. If spaCy is not available
    and fallback_to_noop is True, returns a NoOpNERBackend instead.

    Args:
        model_name: spaCy model name to load
        fallback_to_noop: If True, return NoOpNERBackend on import error

    Returns:
        NER backend instance

    Raises:
        ImportError: If spaCy not available and fallback_to_noop is False
    """
    try:
        return SpaCyNERBackend(model_name=model_name)
    except ImportError:
        if fallback_to_noop:
            logger.warning(
                "spaCy not available, using NoOpNERBackend. "
                "Install spaCy for NER: uv add spacy"
            )
            return NoOpNERBackend()
        raise
    except OSError:
        if fallback_to_noop:
            logger.warning(
                f"spaCy model '{model_name}' not found, using NoOpNERBackend. "
                f"Download with: python -m spacy download {model_name}"
            )
            return NoOpNERBackend()
        raise
