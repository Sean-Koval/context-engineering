"""Collapse tool chains strategy.

This strategy identifies sequences of related tool calls and collapses
them into a summary, preserving the essential information while reducing
token usage.
"""

from __future__ import annotations

import json
import time
from typing import TYPE_CHECKING
from uuid import UUID, uuid4

from context_compression.recovery import RecoveryManifest
from context_compression.strategies.base import BaseCompressionStrategy
from context_compression.types import CompressionResult, CompressionTier
from context_core.graph.types import CompressionLevel, EdgeType, NodeType

if TYPE_CHECKING:
    from context_core.graph import ContextGraph, ContextNode


class CollapseToolChains(BaseCompressionStrategy):
    """Collapse sequences of tool calls into summaries.

    This strategy identifies chains of related tool calls (e.g., multiple
    file reads, sequential API calls) and collapses them into a single
    summary node that captures the essential information.

    Configuration:
        min_chain_length: Minimum number of calls to form a chain (default 3)
        max_chain_gap: Maximum sequence gap between calls (default 2)

    The strategy is fully reversible - the original sequence is stored
    in the recovery manifest.
    """

    def __init__(
        self,
        min_chain_length: int = 3,
        max_chain_gap: int = 2,
    ):
        """Initialize the strategy.

        Args:
            min_chain_length: Minimum calls to trigger collapse
            max_chain_gap: Maximum gap between chain members
        """
        self._min_chain_length = min_chain_length
        self._max_chain_gap = max_chain_gap

    @property
    def _name(self) -> str:
        return "collapse_tool_chains"

    @property
    def _tier(self) -> CompressionTier:
        return CompressionTier.LOSSLESS

    @property
    def _priority(self) -> int:
        return 20  # Run after externalize

    def _find_tool_chains(
        self,
        graph: ContextGraph,
        target_node_ids: set[UUID] | None,
    ) -> list[list[ContextNode]]:
        """Find sequences of related tool calls.

        A chain is a sequence of tool calls to the same tool or related tools
        that occur close together in the sequence.

        Args:
            graph: The context graph
            target_node_ids: Nodes to consider

        Returns:
            List of chains, each chain is a list of nodes
        """
        # Get all tool call nodes in sequence order
        tool_calls = []
        for node in graph:
            if target_node_ids and node.id not in target_node_ids:
                continue
            is_tool_call = node.type == NodeType.TOOL_CALL
            is_full = node.compression_level == CompressionLevel.FULL
            if is_tool_call and is_full:
                tool_calls.append(node)

        # Sort by sequence number
        tool_calls.sort(key=lambda n: n.sequence_number or 0)

        if len(tool_calls) < self._min_chain_length:
            return []

        # Find chains of same tool calls
        chains: list[list[ContextNode]] = []
        current_chain: list[ContextNode] = []
        current_tool: str | None = None
        last_seq: int = -999

        for node in tool_calls:
            tool_name = node.content.tool_name
            seq = node.sequence_number or 0

            # Check if this continues the current chain
            if tool_name == current_tool and seq - last_seq <= self._max_chain_gap + 1:
                current_chain.append(node)
            else:
                # Save current chain if long enough
                if len(current_chain) >= self._min_chain_length:
                    chains.append(current_chain)
                # Start new chain
                current_chain = [node]
                current_tool = tool_name

            last_seq = seq

        # Don't forget the last chain
        if len(current_chain) >= self._min_chain_length:
            chains.append(current_chain)

        return chains

    def _get_tool_results(
        self,
        graph: ContextGraph,
        tool_calls: list[ContextNode],
    ) -> dict[UUID, ContextNode | None]:
        """Get tool results for each tool call.

        Args:
            graph: The context graph
            tool_calls: List of tool call nodes

        Returns:
            Mapping of tool call ID to result node
        """
        results: dict[UUID, ContextNode | None] = {}

        for call in tool_calls:
            results[call.id] = None
            # Find result by looking for TOOL_IO edge
            edges = graph.get_edges(call.id)
            for edge in edges:
                if edge.type == EdgeType.TOOL_IO:
                    result_node = graph.get_node(edge.target_id)
                    if result_node and result_node.type == NodeType.TOOL_RESULT:
                        results[call.id] = result_node
                        break

        return results

    def _create_chain_summary(
        self,
        tool_calls: list[ContextNode],
        results: dict[UUID, ContextNode | None],
    ) -> str:
        """Create a summary of a tool chain.

        Args:
            tool_calls: The tool calls in the chain
            results: Tool call ID to result mapping

        Returns:
            Human-readable summary of the chain
        """
        tool_name = tool_calls[0].content.tool_name or "unknown"
        count = len(tool_calls)

        # Collect key info
        args_summary: list[str] = []
        for call in tool_calls[:3]:  # Show first 3 args
            args = call.content.tool_args or {}
            if args:
                # Get first meaningful arg
                for key, value in args.items():
                    if isinstance(value, str) and len(value) < 100:
                        args_summary.append(f"{key}={value[:50]}")
                        break
                    elif not isinstance(value, (dict, list)):
                        args_summary.append(f"{key}={value}")
                        break

        # Check for errors
        error_count = sum(
            1
            for call in tool_calls
            if results.get(call.id) and results[call.id].content.is_error
        )

        # Build summary
        summary_parts = [
            f"[Tool Chain: {count} calls to `{tool_name}`]",
        ]

        if args_summary:
            summary_parts.append(f"Args: {', '.join(args_summary[:3])}")
            if len(tool_calls) > 3:
                summary_parts.append(f"... and {len(tool_calls) - 3} more")

        if error_count > 0:
            summary_parts.append(f"Errors: {error_count}/{count}")

        return "\n".join(summary_parts)

    def _serialize_chain(
        self,
        tool_calls: list[ContextNode],
        results: dict[UUID, ContextNode | None],
    ) -> list[dict]:
        """Serialize a chain for recovery.

        Args:
            tool_calls: The tool calls
            results: Tool results

        Returns:
            Serialized chain data
        """
        chain_data = []
        for call in tool_calls:
            entry = {
                "call_id": str(call.id),
                "tool_name": call.content.tool_name,
                "tool_args": call.content.tool_args,
                "sequence_number": call.sequence_number,
            }
            result = results.get(call.id)
            if result:
                entry["result_id"] = str(result.id)
                entry["is_error"] = result.content.is_error
                # Store truncated output
                output = result.content.tool_output
                if isinstance(output, str):
                    entry["output_preview"] = output[:500]
                elif output is not None:
                    entry["output_preview"] = json.dumps(output)[:500]
            chain_data.append(entry)
        return chain_data

    def _estimate_savings_impl(
        self,
        graph: ContextGraph,
        target_node_ids: list[UUID] | None,
    ) -> int:
        """Estimate tokens that would be saved."""
        target_ids = set(target_node_ids) if target_node_ids else None
        chains = self._find_tool_chains(graph, target_ids)

        total_savings = 0
        for chain in chains:
            # Get results
            results = self._get_tool_results(graph, chain)

            # Estimate original tokens
            chain_tokens = sum(n.token_count or 0 for n in chain)
            result_tokens = sum((r.token_count or 0) for r in results.values() if r)
            original = chain_tokens + result_tokens

            # Estimate summary tokens (rough: ~50 tokens per chain)
            summary_tokens = 50 + len(chain) * 5

            savings = max(0, original - summary_tokens)
            total_savings += savings

        return total_savings

    def _compress_impl(
        self,
        graph: ContextGraph,
        manifest: RecoveryManifest,
        target_node_ids: list[UUID] | None,
        target_tokens: int | None,
    ) -> CompressionResult:
        """Execute chain collapse compression."""
        start_time = time.perf_counter()
        target_ids = set(target_node_ids) if target_node_ids else None

        chains = self._find_tool_chains(graph, target_ids)

        original_tokens = 0
        compressed_tokens = 0
        nodes_removed = 0
        nodes_created = 0
        tokens_saved = 0

        for chain in chains:
            # Check target limit
            if target_tokens and tokens_saved >= target_tokens:
                break

            # Get results
            results = self._get_tool_results(graph, chain)

            # Calculate original tokens
            chain_tokens = sum(n.token_count or 0 for n in chain)
            result_tokens = sum((r.token_count or 0) for r in results.values() if r)
            chain_original_tokens = chain_tokens + result_tokens
            original_tokens += chain_original_tokens

            # Create summary
            summary_text = self._create_chain_summary(chain, results)
            summary_tokens = len(summary_text) // 4  # Rough estimate

            # Serialize for recovery
            chain_data = self._serialize_chain(chain, results)

            # Create summary node
            from context_core.graph.nodes import Content, ContextNode

            summary_node = ContextNode(
                id=uuid4(),
                type=NodeType.SUMMARY,
                content=Content(
                    text=summary_text,
                    summarized_node_ids=[n.id for n in chain],
                    summary_method="collapse_tool_chains",
                    original_tokens=chain_original_tokens,
                    compressed_tokens=summary_tokens,
                ),
                compression_level=CompressionLevel.COMPACTED,
                token_count=summary_tokens,
                sequence_number=chain[0].sequence_number,
            )

            # Log to manifest
            manifest.log_collapse(
                node_id=chain[0].id,
                original_node_ids=[n.id for n in chain],
                collapsed_node_id=summary_node.id,
                original_tokens=chain_original_tokens,
                chain_description=f"{len(chain)} calls to {chain[0].content.tool_name}",
                original_sequence=chain_data,
            )

            # Add summary node to graph
            graph.add_node(summary_node)
            nodes_created += 1

            # Remove original nodes
            for node in chain:
                graph.remove_node(node.id)
                nodes_removed += 1
                # Also remove result if exists
                result = results.get(node.id)
                if result:
                    graph.remove_node(result.id)
                    nodes_removed += 1

            compressed_tokens += summary_tokens
            tokens_saved += chain_original_tokens - summary_tokens

        return CompressionResult(
            success=True,
            strategy_name=self.name,
            tier=self.tier,
            original_tokens=original_tokens,
            compressed_tokens=compressed_tokens,
            tokens_saved=tokens_saved,
            nodes_processed=len(graph) + nodes_removed,
            nodes_compressed=0,
            nodes_removed=nodes_removed,
            nodes_created=nodes_created,
            duration_ms=(time.perf_counter() - start_time) * 1000,
            is_recoverable=True,
        )

    def _can_apply_impl(self, graph: ContextGraph) -> bool:
        """Check if there are collapsible chains."""
        return bool(self._find_tool_chains(graph, None))
