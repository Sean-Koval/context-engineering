"""NER backends for entity extraction.

Provides pluggable Named Entity Recognition implementations:
- SpaCyNERBackend: Uses spaCy for NER (requires spacy package)
- NoOpNERBackend: Fallback when spaCy is unavailable
- get_ner_backend: Factory function with automatic fallback
"""

from __future__ import annotations

from context_core.entities.backends.base import EntityMention, NERBackend
from context_core.entities.backends.spacy_backend import (
    NoOpNERBackend,
    SpaCyNERBackend,
    get_ner_backend,
)

__all__ = [
    "EntityMention",
    "NERBackend",
    "NoOpNERBackend",
    "SpaCyNERBackend",
    "get_ner_backend",
]
