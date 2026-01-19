"""Memory retrieval strategies for context-memory.

This module provides multiple retrieval strategies that can be used
independently or combined via EnsembleRetriever:

- SemanticRetrieval: Embedding-based similarity search
- EntityRetrieval: Entity-mention based retrieval
- TemporalRetrieval: Time-window based retrieval
- EnsembleRetriever: Combines strategies with RRF ranking
- MemoryRetriever: High-level convenience class

Example:
    >>> from context_memory.retrieval import (
    ...     EnsembleRetriever,
    ...     SemanticRetrieval,
    ...     TemporalRetrieval,
    ... )
    >>> ensemble = EnsembleRetriever([
    ...     (SemanticRetrieval(index), 1.0),
    ...     (TemporalRetrieval(), 0.5),
    ... ])
"""

from context_memory.retrieval.base import RetrievalQuery, RetrievalStrategy
from context_memory.retrieval.ensemble import (
    EnsembleResult,
    EnsembleRetriever,
    MemoryRetriever,
    StrategyConfig,
)
from context_memory.retrieval.entity import EntityRetrieval
from context_memory.retrieval.semantic import SemanticRetrieval
from context_memory.retrieval.temporal import TemporalRetrieval

__all__ = [
    # Base types
    "RetrievalQuery",
    "RetrievalStrategy",
    # Strategies
    "SemanticRetrieval",
    "EntityRetrieval",
    "TemporalRetrieval",
    # Ensemble
    "EnsembleRetriever",
    "EnsembleResult",
    "StrategyConfig",
    # High-level
    "MemoryRetriever",
]
