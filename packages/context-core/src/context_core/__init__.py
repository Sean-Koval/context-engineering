"""Context Core - Foundation layer for ContextEngine.

Provides data structures and abstractions for context management:
- ContextGraph: Graph-based context representation
- EntityTracker: Named entity recognition and linking
- SemanticIndex: Embedding-based search and similarity
- TokenBudget: Token allocation and monitoring
- Tokenizer: Model-agnostic token counting
"""

from __future__ import annotations

# Budget
from context_core.budget import TokenBudget

# Entities
from context_core.entities import (
    Entity,
    EntityMention,
    EntityPattern,
    EntityTracker,
    EntityType,
    NERBackend,
    get_ner_backend,
)

# Graph
from context_core.graph import (
    CompressionLevel,
    Content,
    ContextGraph,
    ContextNode,
    EdgeType,
    NodeMetadata,
    NodeType,
    Role,
)

# Tokenizer
from context_core.tokenizer import (
    EstimateTokenizer,
    TiktokenTokenizer,
    Tokenizer,
    get_tokenizer,
)

__version__ = "0.1.0"

__all__ = [
    # Graph types
    "CompressionLevel",
    "Content",
    "ContextGraph",
    "ContextNode",
    "EdgeType",
    "NodeMetadata",
    "NodeType",
    "Role",
    # Budget
    "TokenBudget",
    # Tokenizer
    "EstimateTokenizer",
    "TiktokenTokenizer",
    "Tokenizer",
    "get_tokenizer",
    # Entities
    "Entity",
    "EntityMention",
    "EntityPattern",
    "EntityTracker",
    "EntityType",
    "NERBackend",
    "get_ner_backend",
]
