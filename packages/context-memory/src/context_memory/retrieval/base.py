"""Base types and protocols for retrieval strategies.

This module defines the core abstractions for memory retrieval:
- RetrievalQuery: Query parameters for retrieval operations
- RetrievalStrategy: Protocol for pluggable retrieval strategies
"""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from context_memory.store import MemoryStore
    from context_memory.types import RetrievalResult


class RetrievalQuery(BaseModel):
    """Query parameters for retrieval operations.

    Supports filtering by text, entities, time range, node types,
    and importance thresholds.

    Example:
        >>> query = RetrievalQuery(
        ...     session_id="sess-123",
        ...     query_text="authentication flow",
        ...     max_results=10,
        ...     min_importance=0.5,
        ... )
    """

    session_id: str = Field(description="Session to retrieve from")
    query_text: str | None = Field(
        default=None,
        description="Text query for semantic search",
    )
    entity_ids: list[str] | None = Field(
        default=None,
        description="Entity IDs to filter by",
    )
    since: datetime | None = Field(
        default=None,
        description="Only retrieve items created after this time",
    )
    until: datetime | None = Field(
        default=None,
        description="Only retrieve items created before this time",
    )
    node_types: list[str] | None = Field(
        default=None,
        description="Filter by node types (MESSAGE, TOOL_CALL, etc.)",
    )
    min_importance: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Minimum importance score",
    )
    max_results: int = Field(
        default=10,
        ge=1,
        le=100,
        description="Maximum results to return",
    )


@runtime_checkable
class RetrievalStrategy(Protocol):
    """Protocol for retrieval strategies.

    Implementations provide different methods for finding relevant
    context from memory stores: semantic similarity, entity matching,
    temporal proximity, etc.

    Example:
        >>> class MyStrategy:
        ...     @property
        ...     def name(self) -> str:
        ...         return "my_strategy"
        ...
        ...     async def retrieve(
        ...         self,
        ...         query: RetrievalQuery,
        ...         store: MemoryStore,
        ...         current_context: Any = None,
        ...     ) -> list[RetrievalResult]:
        ...         # Implementation
        ...         pass
    """

    @property
    def name(self) -> str:
        """Strategy identifier for logging and metrics."""
        ...

    async def retrieve(
        self,
        query: RetrievalQuery,
        store: MemoryStore,
        current_context: Any | None = None,
    ) -> list[RetrievalResult]:
        """Retrieve relevant context from memory.

        Args:
            query: Retrieval query parameters
            store: Memory store to query
            current_context: Optional current context for relevance scoring

        Returns:
            List of retrieval results sorted by relevance
        """
        ...
