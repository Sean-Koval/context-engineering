"""Base classes and protocol for compression strategies.

This module defines:
- CompressionStrategy: Protocol that all strategies must implement
- BaseCompressionStrategy: Abstract base class with shared functionality
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Protocol, runtime_checkable
from uuid import UUID

from context_compression.types import CompressionResult, CompressionTier

if TYPE_CHECKING:
    from context_compression.recovery import RecoveryManifest
    from context_core.graph import ContextGraph


@runtime_checkable
class CompressionStrategy(Protocol):
    """Protocol for compression strategies.

    All compression strategies must implement this interface. The protocol
    approach allows for flexibility - implementations don't need to inherit
    from a base class.

    Strategies are characterized by:
    - name: Unique identifier
    - tier: LOSSLESS, COMPACTION, or SUMMARIZATION
    - priority: Execution order within tier (lower = first)

    The compress() method modifies the graph in place and logs operations
    to the recovery manifest for potential rollback.
    """

    @property
    def name(self) -> str:
        """Unique strategy identifier (e.g., 'externalize_payloads')."""
        ...

    @property
    def tier(self) -> CompressionTier:
        """Compression tier: LOSSLESS, COMPACTION, or SUMMARIZATION."""
        ...

    @property
    def priority(self) -> int:
        """Execution priority within tier (lower = run first)."""
        ...

    def estimate_savings(
        self,
        graph: ContextGraph,
        target_node_ids: list[UUID] | None = None,
    ) -> int:
        """Estimate tokens that would be saved without modifying graph.

        Args:
            graph: The context graph to analyze
            target_node_ids: Specific nodes to target (None = all eligible)

        Returns:
            Estimated number of tokens that would be saved
        """
        ...

    def compress(
        self,
        graph: ContextGraph,
        manifest: RecoveryManifest,
        target_node_ids: list[UUID] | None = None,
        target_tokens: int | None = None,
    ) -> CompressionResult:
        """Execute compression on the graph.

        Args:
            graph: The context graph to compress (modified in place)
            manifest: Recovery manifest to log operations
            target_node_ids: Specific nodes to target (None = all eligible)
            target_tokens: Stop when this many tokens saved

        Returns:
            CompressionResult with metrics about the operation
        """
        ...

    def can_apply(self, graph: ContextGraph) -> bool:
        """Check if this strategy can be applied to the graph.

        Returns False if the strategy has no work to do or prerequisites
        are not met.

        Args:
            graph: The context graph to check

        Returns:
            True if the strategy can be applied
        """
        ...


class BaseCompressionStrategy(ABC):
    """Abstract base class for compression strategies.

    Provides shared functionality for common operations. Strategies can
    inherit from this class for convenience, but it's not required -
    they only need to implement the CompressionStrategy protocol.

    Subclasses must implement:
    - _name property
    - _tier property
    - _priority property
    - _compress_impl method
    - _estimate_savings_impl method
    - _can_apply_impl method
    """

    @property
    @abstractmethod
    def _name(self) -> str:
        """Internal name property."""
        ...

    @property
    @abstractmethod
    def _tier(self) -> CompressionTier:
        """Internal tier property."""
        ...

    @property
    @abstractmethod
    def _priority(self) -> int:
        """Internal priority property."""
        ...

    @property
    def name(self) -> str:
        """Unique strategy identifier."""
        return self._name

    @property
    def tier(self) -> CompressionTier:
        """Compression tier."""
        return self._tier

    @property
    def priority(self) -> int:
        """Execution priority within tier."""
        return self._priority

    @abstractmethod
    def _estimate_savings_impl(
        self,
        graph: ContextGraph,
        target_node_ids: list[UUID] | None,
    ) -> int:
        """Implementation of savings estimation."""
        ...

    @abstractmethod
    def _compress_impl(
        self,
        graph: ContextGraph,
        manifest: RecoveryManifest,
        target_node_ids: list[UUID] | None,
        target_tokens: int | None,
    ) -> CompressionResult:
        """Implementation of compression logic."""
        ...

    @abstractmethod
    def _can_apply_impl(self, graph: ContextGraph) -> bool:
        """Implementation of applicability check."""
        ...

    def estimate_savings(
        self,
        graph: ContextGraph,
        target_node_ids: list[UUID] | None = None,
    ) -> int:
        """Estimate tokens that would be saved.

        Delegates to _estimate_savings_impl after basic validation.
        """
        if len(graph) == 0:
            return 0
        return self._estimate_savings_impl(graph, target_node_ids)

    def compress(
        self,
        graph: ContextGraph,
        manifest: RecoveryManifest,
        target_node_ids: list[UUID] | None = None,
        target_tokens: int | None = None,
    ) -> CompressionResult:
        """Execute compression on the graph.

        Delegates to _compress_impl after basic validation.
        """
        if len(graph) == 0:
            return CompressionResult(
                success=True,
                strategy_name=self.name,
                tier=self.tier,
                original_tokens=0,
                compressed_tokens=0,
                tokens_saved=0,
                nodes_processed=0,
                is_recoverable=True,
            )
        return self._compress_impl(graph, manifest, target_node_ids, target_tokens)

    def can_apply(self, graph: ContextGraph) -> bool:
        """Check if this strategy can be applied.

        Returns False for empty graphs, then delegates to _can_apply_impl.
        """
        if len(graph) == 0:
            return False
        return self._can_apply_impl(graph)

    def _get_target_nodes(
        self,
        graph: ContextGraph,
        target_node_ids: list[UUID] | None,
    ) -> list[UUID]:
        """Helper to get list of node IDs to process.

        Args:
            graph: The context graph
            target_node_ids: Specific nodes, or None for all

        Returns:
            List of node IDs to target
        """
        if target_node_ids is not None:
            return target_node_ids
        return [node.id for node in graph]
