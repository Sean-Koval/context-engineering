"""Core types for the compression pipeline.

This module defines the fundamental types used throughout the compression system:
- CompressionTier: Categorizes strategies by reversibility
- CompressionResult: Captures outcomes of compression operations
- CompressionPlan: Represents a planned compression before execution
- PreservationRule: Defines criteria for protecting nodes from compression
"""

from __future__ import annotations

from collections.abc import Callable
from enum import Enum
from typing import Any
from uuid import UUID

from pydantic import BaseModel, Field, field_validator


class CompressionTier(str, Enum):
    """Tier of compression strategy.

    Strategies are organized into tiers based on reversibility:
    - LOSSLESS: Fully reversible, no information loss
    - COMPACTION: Mostly reversible, minimal information loss
    - SUMMARIZATION: Irreversible, significant information reduction

    The pipeline executes strategies in tier order: LOSSLESS first,
    then COMPACTION, then SUMMARIZATION.
    """

    LOSSLESS = "lossless"
    COMPACTION = "compaction"
    SUMMARIZATION = "summarization"


class CompressionResult(BaseModel):
    """Result of a compression operation.

    Captures comprehensive metrics about what happened during compression,
    including token savings, node changes, timing, and recoverability.

    Attributes:
        success: Whether the compression completed without errors
        strategy_name: Name of the strategy that produced this result
        tier: The compression tier of the strategy
        original_tokens: Token count before compression
        compressed_tokens: Token count after compression
        tokens_saved: Number of tokens saved (original - compressed)
        nodes_processed: Number of nodes examined
        nodes_compressed: Number of nodes that were compressed
        nodes_removed: Number of nodes removed from graph
        nodes_created: Number of new nodes created (e.g., summaries)
        duration_ms: Time taken in milliseconds
        is_recoverable: Whether the compression can be reversed
        manifest_id: ID of the recovery manifest entry, if any
        error_message: Error details if success is False
    """

    success: bool
    strategy_name: str
    tier: CompressionTier

    # Token metrics
    original_tokens: int = Field(ge=0)
    compressed_tokens: int = Field(ge=0)
    tokens_saved: int = Field(ge=0)

    # Node metrics
    nodes_processed: int = Field(default=0, ge=0)
    nodes_compressed: int = Field(default=0, ge=0)
    nodes_removed: int = Field(default=0, ge=0)
    nodes_created: int = Field(default=0, ge=0)

    # Timing
    duration_ms: float = Field(default=0.0, ge=0.0)

    # Recovery
    is_recoverable: bool = True
    manifest_id: UUID | None = None

    # Error handling
    error_message: str | None = None

    @property
    def compression_ratio(self) -> float:
        """Ratio of original to compressed tokens.

        Higher values indicate more compression.
        Returns infinity if compressed_tokens is 0.
        """
        if self.compressed_tokens == 0:
            return float("inf")
        return self.original_tokens / self.compressed_tokens

    @property
    def savings_percent(self) -> float:
        """Percentage of tokens saved.

        Returns a value between 0.0 and 100.0.
        """
        if self.original_tokens == 0:
            return 0.0
        return (self.tokens_saved / self.original_tokens) * 100


class CompressionPlan(BaseModel):
    """A plan for compression before execution.

    Used for dry-run previews to understand what compression would do
    without actually modifying the graph.

    Attributes:
        strategies: Names of strategies that would be applied
        target_tokens: Target token savings, if specified
        estimated_savings: Estimated total tokens that would be saved
        nodes_affected: IDs of nodes that would be compressed
        preserved_nodes: IDs of nodes that would be preserved
        preservations: Human-readable reasons for preservation
    """

    strategies: list[str] = Field(default_factory=list)
    target_tokens: int | None = None
    estimated_savings: int = Field(default=0, ge=0)
    nodes_affected: list[UUID] = Field(default_factory=list)
    preserved_nodes: list[UUID] = Field(default_factory=list)
    preservations: list[str] = Field(default_factory=list)


class PreservationRule(BaseModel):
    """Rule for preserving nodes from compression.

    Preservation rules define criteria for protecting nodes from being
    compressed. Multiple rules can be combined; a node is preserved if
    it matches ANY rule.

    Attributes:
        name: Unique identifier for the rule
        description: Human-readable explanation
        priority: Higher priority rules are checked first (default 0)
        node_types: If set, only preserve nodes of these types
        min_importance: Minimum importance score to preserve
        max_age_seconds: Maximum age in seconds to preserve
        required_tags: Node must have at least one of these tags
        entity_ids: Node must reference one of these entities
        pinned: If True, preserve nodes with metadata.pinned=True
        custom_predicate: Optional custom function for complex rules
    """

    name: str = Field(min_length=1)
    description: str = ""
    priority: int = Field(default=0, ge=0)

    # Criteria (any match within a rule = preserve)
    node_types: list[str] | None = None
    min_importance: float | None = Field(default=None, ge=0.0, le=1.0)
    max_age_seconds: int | None = Field(default=None, ge=0)
    required_tags: set[str] | None = None
    entity_ids: list[str] | None = None
    pinned: bool = False

    # Custom predicate (not serializable)
    # Type is Callable[[ContextNode], bool] but we use Any to avoid import issues
    custom_predicate: Callable[..., bool] | None = Field(default=None, exclude=True)

    model_config = {"arbitrary_types_allowed": True}

    @field_validator("node_types", mode="before")
    @classmethod
    def validate_node_types(cls, v: Any) -> list[str] | None:
        """Ensure node_types is a list or None."""
        if v is None:
            return None
        if isinstance(v, str):
            return [v]
        return list(v)


class PipelineConfig(BaseModel):
    """Configuration for the compression pipeline.

    Attributes:
        preserve_recent_n: Number of most recent nodes to always preserve
        enable_recovery: Whether to track operations for recovery
        max_iterations: Maximum compression iterations per run
        stop_on_error: Whether to stop pipeline on first error
        min_savings_threshold: Minimum tokens to save to continue
    """

    preserve_recent_n: int = Field(default=10, ge=0)
    enable_recovery: bool = True
    max_iterations: int = Field(default=10, ge=1)
    stop_on_error: bool = False
    min_savings_threshold: int = Field(default=100, ge=0)
