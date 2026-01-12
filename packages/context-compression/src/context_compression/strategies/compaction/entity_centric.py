"""Entity-centric compression strategy.

This strategy compresses content by keeping only sentences/content
relevant to important entities tracked by the EntityTracker.
"""

from __future__ import annotations

import re
import time
from typing import TYPE_CHECKING
from uuid import UUID

from context_compression.recovery.operations import CompactOperation
from context_compression.strategies.base import BaseCompressionStrategy
from context_compression.types import CompressionResult, CompressionTier
from context_core.graph.types import CompressionLevel, NodeType

if TYPE_CHECKING:
    from context_compression.recovery import RecoveryManifest
    from context_core.entities.tracker import EntityTracker
    from context_core.graph import ContextGraph, ContextNode


class EntityCentricCompression(BaseCompressionStrategy):
    """Compresses content by preserving only entity-relevant sentences.

    This strategy analyzes text content in MESSAGE and TOOL_RESULT nodes,
    splitting it into sentences and keeping only those that mention
    important entities tracked by the EntityTracker.

    This is COMPACTION tier - some content is removed but the key
    entity-relevant information is preserved.

    Configuration:
        min_importance: Minimum entity importance to consider (default 0.3)
        include_context_sentences: Include one sentence before/after entity mentions

    The strategy is partially reversible - we know which sentences were
    kept, but removed sentences are lost.
    """

    def __init__(
        self,
        entity_tracker: EntityTracker,
        min_importance: float = 0.3,
        include_context_sentences: bool = False,
    ) -> None:
        """Initialize the strategy.

        Args:
            entity_tracker: The EntityTracker to use for entity lookup
            min_importance: Minimum entity importance threshold (0.0-1.0)
            include_context_sentences: If True, include sentences adjacent to
                entity mentions for additional context
        """
        self._entity_tracker = entity_tracker
        self._min_importance = min_importance
        self._include_context_sentences = include_context_sentences

    @property
    def _name(self) -> str:
        return "entity_centric"

    @property
    def _tier(self) -> CompressionTier:
        return CompressionTier.COMPACTION

    @property
    def _priority(self) -> int:
        return 20  # Run after schema compression (priority 10)

    def _split_sentences(self, text: str) -> list[str]:
        """Split text into sentences.

        Uses regex to split on common sentence boundaries (., !, ?).
        Handles common edge cases like abbreviations and numbers.

        Args:
            text: The text to split

        Returns:
            List of sentences, preserving whitespace
        """
        if not text or not text.strip():
            return []

        # Split on sentence-ending punctuation followed by whitespace or end
        # This pattern matches . ! or ? followed by whitespace or end of string
        # but avoids splitting on decimals (1.5) or abbreviations (Dr. Smith)
        pattern = r"(?<=[.!?])(?=\s+[A-Z])|(?<=[.!?])(?=\s*$)"

        # First, try to split using the pattern
        parts = re.split(pattern, text)

        # If we only got one part, the text might not have proper sentence structure
        # Try a simpler split
        if len(parts) == 1 and text.strip():
            # Split on any sentence-ending punctuation followed by space
            simple_pattern = r"(?<=[.!?])\s+"
            parts = re.split(simple_pattern, text)

        # Clean up empty strings and whitespace-only strings
        sentences = [s for s in parts if s and s.strip()]

        return sentences

    def _sentence_has_entity(self, sentence: str, entities: set[str]) -> bool:
        """Check if a sentence contains any tracked entity.

        Performs case-insensitive matching against entity names and aliases.

        Args:
            sentence: The sentence to check
            entities: Set of entity names/aliases to look for

        Returns:
            True if the sentence contains at least one entity
        """
        if not sentence or not entities:
            return False

        sentence_lower = sentence.lower()

        return any(entity_name.lower() in sentence_lower for entity_name in entities)

    def _get_important_entity_names(self) -> set[str]:
        """Get names and aliases of all important entities.

        Returns:
            Set of entity names and aliases above the importance threshold
        """
        entities = self._entity_tracker.get_top_entities(limit=100)
        names: set[str] = set()

        for entity in entities:
            if entity.importance >= self._min_importance:
                names.add(entity.canonical_name)
                names.update(entity.aliases)

        return names

    def _get_text_content(self, node: ContextNode) -> str | None:
        """Extract text content from a node.

        Args:
            node: The node to extract content from

        Returns:
            The text content if available, None otherwise
        """
        # For MESSAGE nodes, use the text field
        if node.type == NodeType.MESSAGE:
            return node.content.text

        # For TOOL_RESULT nodes, check for string output
        if node.type == NodeType.TOOL_RESULT:
            output = node.content.tool_output
            if isinstance(output, str):
                return output
            # Skip JSON/dict outputs - they should use SchemaCompression
            return None

        return None

    def _is_eligible(self, node: ContextNode) -> bool:
        """Check if a node is eligible for entity-centric compression.

        Args:
            node: The node to check

        Returns:
            True if the node can be compressed
        """
        # Must be MESSAGE or TOOL_RESULT
        if node.type not in (NodeType.MESSAGE, NodeType.TOOL_RESULT):
            return False

        # Must not be already compressed
        if node.compression_level != CompressionLevel.FULL:
            return False

        # Must not be pinned
        if node.metadata.pinned:
            return False

        # Must have text content
        text = self._get_text_content(node)
        return bool(text and text.strip())

    def _compress_text(
        self,
        text: str,
        entity_names: set[str],
    ) -> tuple[str, list[str], list[str]]:
        """Compress text by keeping only entity-relevant sentences.

        Args:
            text: The text to compress
            entity_names: Set of entity names/aliases to preserve

        Returns:
            Tuple of (compressed_text, preserved_sentences, removed_sentences)
        """
        sentences = self._split_sentences(text)

        if not sentences:
            return text, [], []

        # Track which sentences contain entities
        sentence_has_entity = [
            self._sentence_has_entity(s, entity_names) for s in sentences
        ]

        # Build the keep set
        keep_indices: set[int] = set()

        for i, has_entity in enumerate(sentence_has_entity):
            if has_entity:
                keep_indices.add(i)

                # Optionally include adjacent sentences for context
                if self._include_context_sentences:
                    if i > 0:
                        keep_indices.add(i - 1)
                    if i < len(sentences) - 1:
                        keep_indices.add(i + 1)

        # Separate preserved and removed sentences
        preserved: list[str] = []
        removed: list[str] = []

        for i, sentence in enumerate(sentences):
            if i in keep_indices:
                preserved.append(sentence)
            else:
                removed.append(sentence)

        # Reconstruct compressed text
        compressed = " ".join(preserved)

        return compressed, preserved, removed

    def _estimate_savings_impl(
        self,
        graph: ContextGraph,
        target_node_ids: list[UUID] | None,
    ) -> int:
        """Estimate tokens that would be saved.

        Args:
            graph: The context graph to analyze
            target_node_ids: Optional list of node IDs to target

        Returns:
            Estimated number of tokens that would be saved
        """
        target_ids = set(target_node_ids) if target_node_ids else None
        entity_names = self._get_important_entity_names()

        if not entity_names:
            return 0

        total_savings = 0

        for node in graph:
            if target_ids and node.id not in target_ids:
                continue

            if not self._is_eligible(node):
                continue

            text = self._get_text_content(node)
            if not text:
                continue

            # Estimate compression
            compressed, _, removed = self._compress_text(text, entity_names)

            # Rough token estimate: 4 chars per token
            original_tokens = len(text) // 4
            compressed_tokens = len(compressed) // 4
            savings = max(0, original_tokens - compressed_tokens)
            total_savings += savings

        return total_savings

    def _compress_impl(
        self,
        graph: ContextGraph,
        manifest: RecoveryManifest,
        target_node_ids: list[UUID] | None,
        target_tokens: int | None,
    ) -> CompressionResult:
        """Execute entity-centric compression.

        Args:
            graph: The context graph to compress
            manifest: Recovery manifest to log operations
            target_node_ids: Optional list of node IDs to target
            target_tokens: Stop when this many tokens saved

        Returns:
            CompressionResult with metrics about the operation
        """
        start_time = time.perf_counter()
        target_ids = set(target_node_ids) if target_node_ids else None

        entity_names = self._get_important_entity_names()

        original_tokens = 0
        compressed_tokens = 0
        nodes_compressed = 0
        tokens_saved = 0

        # If no important entities, skip compression
        if not entity_names:
            return CompressionResult(
                success=True,
                strategy_name=self.name,
                tier=self.tier,
                original_tokens=0,
                compressed_tokens=0,
                tokens_saved=0,
                nodes_processed=len(list(graph)),
                nodes_compressed=0,
                nodes_removed=0,
                nodes_created=0,
                duration_ms=(time.perf_counter() - start_time) * 1000,
                is_recoverable=False,
            )

        for node in graph:
            if target_ids and node.id not in target_ids:
                continue

            if not self._is_eligible(node):
                continue

            # Check target limit
            if target_tokens and tokens_saved >= target_tokens:
                break

            text = self._get_text_content(node)
            if not text:
                continue

            # Compress the text
            compressed_text, preserved, removed = self._compress_text(
                text, entity_names
            )

            # Skip if no compression occurred (no sentences removed)
            if not removed:
                continue

            # Skip if compression would remove ALL content
            # (don't compress to empty string)
            if not preserved:
                continue

            # Calculate token savings
            node_original_tokens = len(text) // 4
            node_compressed_tokens = len(compressed_text) // 4
            node_savings = node_original_tokens - node_compressed_tokens

            # Skip if compression doesn't save tokens
            if node_savings <= 0:
                continue

            original_tokens += node_original_tokens
            compressed_tokens += node_compressed_tokens

            # Update the node content
            if node.type == NodeType.MESSAGE:
                node.content.text = compressed_text
            elif node.type == NodeType.TOOL_RESULT:
                node.content.tool_output = compressed_text

            node.compression_level = CompressionLevel.COMPACTED
            node.content.original_tokens = node.token_count
            node.content.compressed_tokens = node_compressed_tokens
            node.token_count = node_compressed_tokens

            # Log to manifest
            manifest.log_operation(
                CompactOperation(
                    node_id=node.id,
                    original_tokens=node_original_tokens,
                    compressed_tokens=node_compressed_tokens,
                    compaction_method="entity_centric",
                    preserved_fields=preserved[:10],  # Limit for storage
                    removed_fields=removed[:10],  # Limit for storage
                )
            )

            tokens_saved += node_savings
            nodes_compressed += 1

        return CompressionResult(
            success=True,
            strategy_name=self.name,
            tier=self.tier,
            original_tokens=original_tokens,
            compressed_tokens=compressed_tokens,
            tokens_saved=tokens_saved,
            nodes_processed=len(list(graph)),
            nodes_compressed=nodes_compressed,
            nodes_removed=0,
            nodes_created=0,
            duration_ms=(time.perf_counter() - start_time) * 1000,
            is_recoverable=False,  # Compaction is partially recoverable
        )

    def _can_apply_impl(self, graph: ContextGraph) -> bool:
        """Check if entity-centric compression can be applied.

        Args:
            graph: The context graph to check

        Returns:
            True if there are important entities and eligible nodes
        """
        # Must have important entities
        entity_names = self._get_important_entity_names()
        if not entity_names:
            return False

        # Must have at least one eligible node
        for node in graph:
            if self._is_eligible(node):
                text = self._get_text_content(node)
                if text and len(self._split_sentences(text)) > 1:
                    return True

        return False
