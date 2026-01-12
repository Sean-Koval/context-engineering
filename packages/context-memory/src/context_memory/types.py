"""Core type definitions for context-memory package.

This module defines the foundational data structures used across all storage
backends and retrieval operations.
"""

from __future__ import annotations

from datetime import UTC, datetime
from enum import Enum
from typing import TYPE_CHECKING, Any
from uuid import UUID

from pydantic import BaseModel, Field, field_validator

if TYPE_CHECKING:
    pass


class StorageTier(str, Enum):
    """Storage tier classification for tiered storage architecture.

    Each tier represents a different trade-off between latency, capacity, and cost:
    - HOT: Fastest access, limited capacity, for recently accessed data
    - WARM: Moderate speed, larger capacity, for active session data
    - COLD: Slowest access, unlimited capacity, for archived data
    """

    HOT = "hot"  # < 10ms latency, Redis/Memory, < 1 hour old
    WARM = "warm"  # < 50ms latency, PostgreSQL/SQLite, < 24 hours old
    COLD = "cold"  # < 500ms latency, S3/Filesystem, archived


class StorageKey(BaseModel):
    """Unique identifier for stored items.

    Combines session namespace, node ID, and version to create a composite key
    that supports multi-tenancy and optimistic concurrency control.

    Example:
        >>> key = StorageKey(session_id="sess-123", node_id=UUID("..."), version=1)
        >>> str(key)
        'sess-123/550e8400-e29b-41d4-a716-446655440000/1'
        >>> StorageKey.from_string(str(key))
        StorageKey(session_id='sess-123', node_id=UUID('...'), version=1)
    """

    session_id: str = Field(description="Session namespace identifier")
    node_id: UUID = Field(description="Unique node identifier")
    version: int = Field(
        default=1, ge=1, description="Version for optimistic concurrency"
    )

    def __str__(self) -> str:
        """Return string representation: {session_id}/{node_id}/{version}."""
        return f"{self.session_id}/{self.node_id}/{self.version}"

    def __hash__(self) -> int:
        """Make StorageKey hashable for use in sets and dict keys."""
        return hash((self.session_id, self.node_id, self.version))

    @classmethod
    def from_string(cls, s: str) -> StorageKey:
        """Parse StorageKey from string representation.

        Args:
            s: String in format "{session_id}/{node_id}/{version}" or
               "{session_id}/{node_id}" (defaults version to 1)

        Returns:
            Parsed StorageKey instance

        Raises:
            ValueError: If string format is invalid
        """
        parts = s.split("/")
        if len(parts) < 2:
            raise ValueError(f"Invalid StorageKey format: {s}")

        return cls(
            session_id=parts[0],
            node_id=UUID(parts[1]),
            version=int(parts[2]) if len(parts) > 2 else 1,
        )


class StorageMetadata(BaseModel):
    """Rich metadata for stored items.

    Supports tier management, access tracking, and compression state tracking.
    This metadata is stored alongside node content to enable efficient queries
    without loading full node data.
    """

    # Identity
    key: StorageKey = Field(description="Reference to the stored item")
    tier: StorageTier = Field(
        default=StorageTier.HOT, description="Current storage tier"
    )

    # Size tracking
    size_bytes: int = Field(ge=0, description="Serialized size in bytes")
    token_count: int = Field(ge=0, description="Token count for budget tracking")

    # Temporal fields
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        description="Creation timestamp",
    )
    accessed_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        description="Last access timestamp",
    )
    access_count: int = Field(default=0, ge=0, description="Access frequency counter")

    # Classification
    node_type: str = Field(description="Type of node (MESSAGE, TOOL_CALL, etc.)")
    importance: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Importance score [0, 1]",
    )
    tags: set[str] = Field(default_factory=set, description="Classification tags")

    # Compression state
    is_compressed: bool = Field(
        default=False, description="Whether content is compressed"
    )
    original_size_bytes: int | None = Field(
        default=None,
        ge=0,
        description="Size before compression (None if not compressed)",
    )

    model_config = {"frozen": False}

    def touch(self) -> None:
        """Update access tracking.

        Updates accessed_at to current time and increments access_count.
        Call this method whenever the associated node is retrieved.
        """
        self.accessed_at = datetime.now(UTC)
        self.access_count += 1

    @field_validator("tags", mode="before")
    @classmethod
    def convert_tags_to_set(cls, v: Any) -> set[str]:
        """Convert list to set for tags field (handles JSON deserialization)."""
        if isinstance(v, list):
            return set(v)
        return v


class StorageStats(BaseModel):
    """Aggregate statistics for storage monitoring and capacity planning.

    Provides a snapshot of storage state including counts, sizes, and
    temporal bounds. Used for dashboards, alerting, and capacity planning.
    """

    # Totals
    total_items: int = Field(ge=0, description="Total stored items")
    total_size_bytes: int = Field(ge=0, description="Total storage used in bytes")
    total_tokens: int = Field(ge=0, description="Total tokens stored")

    # Per-tier breakdown
    items_by_tier: dict[str, int] = Field(
        default_factory=dict,
        description="Item count per tier (tier name -> count)",
    )
    size_by_tier: dict[str, int] = Field(
        default_factory=dict,
        description="Bytes per tier (tier name -> bytes)",
    )

    # Access patterns
    avg_access_count: float = Field(
        default=0.0,
        ge=0.0,
        description="Average access frequency across all items",
    )

    # Temporal bounds
    oldest_item: datetime | None = Field(
        default=None,
        description="Creation timestamp of oldest item",
    )
    newest_item: datetime | None = Field(
        default=None,
        description="Creation timestamp of newest item",
    )

    @property
    def compression_ratio(self) -> float | None:
        """Calculate overall compression ratio if applicable.

        Returns:
            Ratio of original to compressed size, or None if no compression data.
        """
        # This would need to be calculated from individual items
        # Placeholder for future implementation
        return None


class RetrievalResult(BaseModel):
    """Result container for memory retrieval operations.

    Wraps a retrieved node with scoring and provenance information
    to support ranking and debugging of retrieval operations.
    """

    node: Any = Field(
        description="Retrieved ContextNode"
    )  # Any to avoid circular import
    score: float = Field(
        ge=0.0,
        le=1.0,
        description="Relevance/similarity score [0, 1]",
    )
    source_tier: StorageTier = Field(description="Tier the node was retrieved from")
    retrieval_method: str = Field(
        description="Strategy used (e.g., 'semantic', 'entity', 'ensemble')",
    )
    latency_ms: float = Field(ge=0.0, description="Retrieval latency in milliseconds")

    model_config = {"arbitrary_types_allowed": True}
