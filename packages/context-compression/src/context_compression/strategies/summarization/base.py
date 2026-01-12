"""Base class for summarization strategies.

This module provides shared functionality for all summarization strategies:
- Text extraction from nodes
- Chunking logic
- Summary node creation
"""

from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING
from uuid import UUID, uuid4

from context_compression.strategies.base import BaseCompressionStrategy
from context_compression.types import CompressionResult, CompressionTier
from context_core.graph.nodes import Content, ContextNode, NodeMetadata
from context_core.graph.types import CompressionLevel, NodeType

if TYPE_CHECKING:
    from context_compression.recovery import RecoveryManifest
    from context_core.graph import ContextGraph


class BaseSummarizationStrategy(BaseCompressionStrategy):
    """Base class for summarization strategies.

    Provides shared functionality for creating summaries:
    - Text extraction from various node types
    - Chunking nodes by count or token threshold
    - Creating SUMMARY nodes with proper metadata

    Summarization strategies are IRREVERSIBLE - original content is lost.
    They should be used as a last resort when other compression methods
    are insufficient.

    Subclasses must implement:
    - _name property
    - _priority property
    - _compress_impl method
    - _estimate_savings_impl method
    - _can_apply_impl method
    """

    @property
    def _tier(self) -> CompressionTier:
        """All summarization strategies are in the SUMMARIZATION tier."""
        return CompressionTier.SUMMARIZATION

    @property
    @abstractmethod
    def _name(self) -> str:
        """Unique strategy identifier."""
        ...

    @property
    @abstractmethod
    def _priority(self) -> int:
        """Execution priority within tier."""
        ...

    def _extract_text_from_node(self, node: ContextNode) -> str:
        """Extract text content from a node.

        Handles different node types appropriately:
        - MESSAGE: Returns the text content
        - TOOL_CALL: Returns tool name and args summary
        - TOOL_RESULT: Returns string output or JSON summary
        - SUMMARY: Returns the summary text
        - Others: Returns available text or empty string

        Args:
            node: The node to extract text from

        Returns:
            The text content of the node
        """
        if node.type == NodeType.MESSAGE:
            return node.content.text or ""

        if node.type == NodeType.TOOL_CALL:
            name = node.content.tool_name or "unknown"
            args = node.content.tool_args or {}
            args_str = ", ".join(f"{k}={v!r}" for k, v in list(args.items())[:5])
            return f"Tool call: {name}({args_str})"

        if node.type == NodeType.TOOL_RESULT:
            output = node.content.tool_output
            if isinstance(output, str):
                return output
            if output is not None:
                # Summarize structured data
                import json

                try:
                    json_str = json.dumps(output)
                    if len(json_str) > 500:
                        return f"Tool output: {json_str[:500]}..."
                    return f"Tool output: {json_str}"
                except (TypeError, ValueError):
                    return f"Tool output: {output!r}"[:500]
            return ""

        if node.type == NodeType.SUMMARY:
            return node.content.text or ""

        if node.type == NodeType.SYSTEM:
            return node.content.text or ""

        if node.type == NodeType.ARTIFACT:
            data = node.content.artifact_data
            if isinstance(data, str):
                return data[:500] if len(data) > 500 else data
            return f"Artifact: {node.content.artifact_type or 'unknown'}"

        # Default: try to get text
        return node.content.text or ""

    def _chunk_nodes(
        self,
        nodes: list[ContextNode],
        chunk_size: int = 10,
        token_threshold: int = 2000,
    ) -> list[list[ContextNode]]:
        """Split nodes into chunks for summarization.

        Creates chunks that don't exceed either:
        - chunk_size nodes per chunk, OR
        - token_threshold tokens per chunk

        Args:
            nodes: List of nodes to chunk
            chunk_size: Maximum nodes per chunk
            token_threshold: Maximum tokens per chunk

        Returns:
            List of node chunks
        """
        if not nodes:
            return []

        chunks: list[list[ContextNode]] = []
        current_chunk: list[ContextNode] = []
        current_tokens = 0

        for node in nodes:
            node_tokens = (
                node.token_count or len(self._extract_text_from_node(node)) // 4
            )

            # Check if adding this node would exceed limits
            would_exceed_count = len(current_chunk) >= chunk_size
            would_exceed_tokens = (
                current_tokens + node_tokens > token_threshold and current_chunk
            )

            if would_exceed_count or would_exceed_tokens:
                # Start a new chunk
                chunks.append(current_chunk)
                current_chunk = []
                current_tokens = 0

            current_chunk.append(node)
            current_tokens += node_tokens

        # Don't forget the last chunk
        if current_chunk:
            chunks.append(current_chunk)

        return chunks

    def _create_summary_node(
        self,
        summary_text: str,
        original_nodes: list[ContextNode],
        summary_method: str,
    ) -> ContextNode:
        """Create a SUMMARY node from original nodes.

        The summary node:
        - Has type SUMMARY
        - Contains the summary text
        - References original node IDs
        - Has compression_level SUMMARIZED
        - Inherits max importance from originals

        Args:
            summary_text: The generated summary text
            original_nodes: The nodes being summarized
            summary_method: Name of the summarization method

        Returns:
            A new ContextNode of type SUMMARY
        """
        # Calculate token count (rough estimate)
        summary_tokens = len(summary_text) // 4

        # Calculate original token count
        original_tokens = sum(
            n.token_count or len(self._extract_text_from_node(n)) // 4
            for n in original_nodes
        )

        # Get max importance from original nodes
        max_importance = max(
            (n.compute_importance() for n in original_nodes),
            default=0.5,
        )

        # Collect original node IDs
        original_ids = [n.id for n in original_nodes]

        return ContextNode(
            id=uuid4(),
            type=NodeType.SUMMARY,
            content=Content(
                text=summary_text,
                summarized_node_ids=original_ids,
                summary_method=summary_method,
                original_tokens=original_tokens,
                compressed_tokens=summary_tokens,
            ),
            metadata=NodeMetadata(
                importance=max_importance,
                tags={"summary", summary_method},
            ),
            compression_level=CompressionLevel.SUMMARIZED,
            token_count=summary_tokens,
        )

    def _get_message_nodes(
        self,
        graph: ContextGraph,
        target_node_ids: list[UUID] | None = None,
    ) -> list[ContextNode]:
        """Get MESSAGE nodes from the graph in sequence order.

        Filters to target_node_ids if specified, excludes pinned and
        already summarized nodes.

        Args:
            graph: The context graph
            target_node_ids: Optional filter to specific nodes

        Returns:
            List of MESSAGE nodes in sequence order
        """
        target_set = set(target_node_ids) if target_node_ids else None

        nodes = []
        for node in graph:
            # Filter by target IDs if specified
            if target_set and node.id not in target_set:
                continue

            # Only MESSAGE nodes
            if node.type != NodeType.MESSAGE:
                continue

            # Skip pinned nodes
            if node.metadata.pinned:
                continue

            # Skip already summarized nodes
            if node.compression_level >= CompressionLevel.SUMMARIZED:
                continue

            nodes.append(node)

        # Sort by sequence number
        nodes.sort(key=lambda n: n.sequence_number or 0)

        return nodes

    def _estimate_savings_impl(
        self,
        graph: ContextGraph,
        target_node_ids: list[UUID] | None,
    ) -> int:
        """Estimate tokens that would be saved by summarization.

        Default implementation estimates based on compression ratio.

        Args:
            graph: The context graph to analyze
            target_node_ids: Optional filter to specific nodes

        Returns:
            Estimated tokens saved
        """
        nodes = self._get_message_nodes(graph, target_node_ids)
        if not nodes:
            return 0

        total_tokens = sum(
            n.token_count or len(self._extract_text_from_node(n)) // 4 for n in nodes
        )

        # Assume ~80% compression (5x ratio)
        return int(total_tokens * 0.8)

    def _can_apply_impl(self, graph: ContextGraph) -> bool:
        """Check if summarization can be applied.

        Default implementation checks for eligible MESSAGE nodes.

        Args:
            graph: The context graph to check

        Returns:
            True if there are nodes to summarize
        """
        nodes = self._get_message_nodes(graph, None)
        return len(nodes) >= 2  # Need at least 2 messages to summarize

    @abstractmethod
    def _compress_impl(
        self,
        graph: ContextGraph,
        manifest: RecoveryManifest,
        target_node_ids: list[UUID] | None,
        target_tokens: int | None,
    ) -> CompressionResult:
        """Implementation of summarization logic.

        Subclasses must implement this method to perform their specific
        summarization approach.

        Args:
            graph: The context graph to compress (modified in place)
            manifest: Recovery manifest to log operations
            target_node_ids: Optional filter to specific nodes
            target_tokens: Stop when this many tokens saved

        Returns:
            CompressionResult with metrics about the operation
        """
        ...
