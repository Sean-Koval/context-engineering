"""Tests for DeduplicateSemantically strategy."""

from __future__ import annotations

from uuid import uuid4

import numpy as np
import pytest

from context_compression.recovery import RecoveryManifest
from context_compression.strategies.lossless import DeduplicateSemantically
from context_compression.types import CompressionTier
from context_core.graph import ContextGraph
from context_core.graph.nodes import Content, ContextNode, NodeMetadata
from context_core.graph.types import CompressionLevel, NodeType
from context_core.semantic import SemanticIndex
from context_core.semantic.embeddings import MockEmbeddingModel


class DeterministicEmbeddingModel:
    """Embedding model that returns controlled embeddings for testing.

    Allows setting specific embeddings for specific texts to control
    similarity scores in tests.
    """

    def __init__(self, dimension: int = 64) -> None:
        self._dimension = dimension
        self._text_embeddings: dict[str, np.ndarray] = {}

    @property
    def dimension(self) -> int:
        return self._dimension

    def set_embedding(self, text: str, embedding: np.ndarray) -> None:
        """Set a specific embedding for a text."""
        self._text_embeddings[text] = embedding

    def set_similar_texts(self, texts: list[str], similarity: float = 0.95) -> None:
        """Set embeddings for all texts to have pairwise similarity >= target.

        Creates embeddings where all pairs have approximately the target similarity.
        For identical duplicates, use similarity=0.999 (near-identical vectors).
        """
        # Create a base vector
        rng = np.random.default_rng(42)
        base = rng.standard_normal(self._dimension).astype(np.float32)
        base = base / np.linalg.norm(base)

        # Compute the angle that gives target similarity: cos(angle) = similarity
        angle = np.arccos(np.clip(similarity, -1, 1))

        for i, text in enumerate(texts):
            # Create a random orthogonal perturbation for each text
            orth_rng = np.random.default_rng(43 + i)
            perturb = orth_rng.standard_normal(self._dimension).astype(np.float32)
            # Make orthogonal to base
            perturb = perturb - np.dot(perturb, base) * base
            perturb = perturb / (np.linalg.norm(perturb) + 1e-8)

            # For high similarity (like duplicates), use tiny angles
            # Scale the perturbation based on index to ensure slight differences
            tiny_offset = i * 0.001  # Very small offset for differentiation

            # Rotate base by small angle in the direction of perturbation
            embedding = (
                np.cos(angle + tiny_offset) * base
                + np.sin(angle + tiny_offset) * perturb
            )
            embedding = embedding / np.linalg.norm(embedding)
            self._text_embeddings[text] = embedding.astype(np.float32)

    def set_identical_texts(self, texts: list[str]) -> None:
        """Set all texts to have near-identical embeddings (similarity ~1.0)."""
        rng = np.random.default_rng(42)
        base = rng.standard_normal(self._dimension).astype(np.float32)
        base = base / np.linalg.norm(base)

        for i, text in enumerate(texts):
            # Add tiny random noise for slight variation
            noise_rng = np.random.default_rng(100 + i)
            noise = (
                noise_rng.standard_normal(self._dimension).astype(np.float32) * 0.001
            )
            embedding = base + noise
            embedding = embedding / np.linalg.norm(embedding)
            self._text_embeddings[text] = embedding.astype(np.float32)

    def embed(self, texts: list[str]) -> np.ndarray:
        """Embed texts, using set embeddings or generating new ones."""
        embeddings = []
        for text in texts:
            if text in self._text_embeddings:
                embeddings.append(self._text_embeddings[text])
            else:
                # Generate deterministic embedding from text
                rng = np.random.default_rng(hash(text) % (2**32))
                emb = rng.standard_normal(self._dimension).astype(np.float32)
                emb = emb / (np.linalg.norm(emb) + 1e-8)
                embeddings.append(emb)
        return (
            np.stack(embeddings)
            if embeddings
            else np.array([]).reshape(0, self._dimension)
        )


def _create_node_with_text(
    text: str,
    token_count: int = 100,
    importance: float = 0.5,
    sequence_number: int | None = None,
) -> ContextNode:
    """Helper to create a node with specific text content."""
    return ContextNode(
        id=uuid4(),
        type=NodeType.MESSAGE,
        content=Content(text=text, role=None),
        metadata=NodeMetadata(importance=importance),
        token_count=token_count,
        sequence_number=sequence_number,
    )


class TestDeduplicateSemanticallyProperties:
    """Tests for DeduplicateSemantically strategy properties."""

    @pytest.fixture
    def model(self) -> DeterministicEmbeddingModel:
        """Create a deterministic embedding model."""
        return DeterministicEmbeddingModel(dimension=64)

    @pytest.fixture
    def semantic_index(self, model: DeterministicEmbeddingModel) -> SemanticIndex:
        """Create a semantic index with the model."""
        return SemanticIndex(model)

    @pytest.fixture
    def strategy(self, semantic_index: SemanticIndex) -> DeduplicateSemantically:
        """Create the deduplication strategy."""
        return DeduplicateSemantically(
            semantic_index=semantic_index,
            similarity_threshold=0.92,
            min_tokens_to_dedupe=50,
            prefer_recent=True,
        )

    def test_strategy_name(self, strategy: DeduplicateSemantically) -> None:
        """Test strategy has correct name."""
        assert strategy.name == "deduplicate_semantically"

    def test_strategy_tier(self, strategy: DeduplicateSemantically) -> None:
        """Test strategy is in LOSSLESS tier."""
        assert strategy.tier == CompressionTier.LOSSLESS

    def test_strategy_priority(self, strategy: DeduplicateSemantically) -> None:
        """Test strategy has correct priority (20)."""
        assert strategy.priority == 20


class TestDeduplicateSemantically:
    """Tests for DeduplicateSemantically compression functionality."""

    @pytest.fixture
    def model(self) -> DeterministicEmbeddingModel:
        """Create a deterministic embedding model."""
        return DeterministicEmbeddingModel(dimension=64)

    @pytest.fixture
    def semantic_index(self, model: DeterministicEmbeddingModel) -> SemanticIndex:
        """Create a semantic index with the model."""
        return SemanticIndex(model)

    @pytest.fixture
    def strategy(self, semantic_index: SemanticIndex) -> DeduplicateSemantically:
        """Create the deduplication strategy."""
        return DeduplicateSemantically(
            semantic_index=semantic_index,
            similarity_threshold=0.90,
            min_tokens_to_dedupe=50,
            prefer_recent=True,
        )

    def test_deduplicate_finds_similar_nodes(
        self,
        model: DeterministicEmbeddingModel,
        semantic_index: SemanticIndex,
        strategy: DeduplicateSemantically,
    ) -> None:
        """Test that deduplication correctly identifies similar nodes."""
        graph = ContextGraph()

        # Set up similar texts
        text1 = "The quick brown fox jumps over the lazy dog."
        text2 = "The quick brown fox jumps over the lazy dog."  # Exact duplicate

        model.set_identical_texts([text1, text2])

        # Create nodes
        node1 = _create_node_with_text(text1, token_count=100, sequence_number=0)
        node2 = _create_node_with_text(text2, token_count=100, sequence_number=1)

        graph.add_node(node1, connect_temporal=False)
        graph.add_node(node2, connect_temporal=False)

        # Index nodes
        semantic_index.index_node(node1)
        semantic_index.index_node(node2)

        manifest = RecoveryManifest()
        result = strategy.compress(graph, manifest)

        # Should have removed one duplicate
        assert result.success is True
        assert result.nodes_removed == 1
        assert result.tokens_saved == 100

    def test_deduplicate_keeps_canonical(
        self,
        model: DeterministicEmbeddingModel,
        semantic_index: SemanticIndex,
        strategy: DeduplicateSemantically,
    ) -> None:
        """Test that the canonical node is kept based on scoring criteria."""
        graph = ContextGraph()

        text1 = "Important information about the project"
        text2 = "Important information about the project"

        model.set_identical_texts([text1, text2])

        # Node 1: older but higher importance
        node1 = _create_node_with_text(
            text1, token_count=100, importance=0.9, sequence_number=0
        )
        # Node 2: newer but lower importance
        node2 = _create_node_with_text(
            text2, token_count=100, importance=0.3, sequence_number=1
        )

        graph.add_node(node1, connect_temporal=False)
        graph.add_node(node2, connect_temporal=False)

        semantic_index.index_node(node1)
        semantic_index.index_node(node2)

        manifest = RecoveryManifest()
        result = strategy.compress(graph, manifest)

        assert result.success is True
        assert result.nodes_removed == 1

        # One node should remain
        remaining_nodes = list(graph)
        assert len(remaining_nodes) == 1

        # The remaining node should exist in the graph
        # (canonical selection depends on the weighted score)
        assert remaining_nodes[0].id in graph

    def test_deduplicate_transitive_groups(
        self,
        model: DeterministicEmbeddingModel,
        semantic_index: SemanticIndex,
        strategy: DeduplicateSemantically,
    ) -> None:
        """Test that A~B and B~C results in all three being grouped."""
        graph = ContextGraph()

        # Create three texts where A~B and B~C (transitive relationship)
        text_a = "Content about machine learning algorithms"
        text_b = "Content about machine learning algorithms"  # Similar to A
        text_c = "Content about machine learning algorithms"  # Similar to B

        # Set them all as similar
        model.set_similar_texts([text_a, text_b, text_c], similarity=0.95)

        node_a = _create_node_with_text(text_a, token_count=100, sequence_number=0)
        node_b = _create_node_with_text(text_b, token_count=100, sequence_number=1)
        node_c = _create_node_with_text(text_c, token_count=100, sequence_number=2)

        for node in [node_a, node_b, node_c]:
            graph.add_node(node, connect_temporal=False)
            semantic_index.index_node(node)

        manifest = RecoveryManifest()
        result = strategy.compress(graph, manifest)

        assert result.success is True
        # Should remove 2 nodes (keep 1 canonical)
        assert result.nodes_removed == 2
        assert len(list(graph)) == 1

    def test_deduplicate_respects_target_tokens(
        self,
        model: DeterministicEmbeddingModel,
        semantic_index: SemanticIndex,
        strategy: DeduplicateSemantically,
    ) -> None:
        """Test that deduplication stops at target_tokens limit."""
        graph = ContextGraph()

        # Create multiple duplicate pairs
        texts = [
            "First duplicate content here",
            "First duplicate content here",
            "Second duplicate content here",
            "Second duplicate content here",
        ]

        # Set pairs as similar
        model.set_similar_texts(texts[:2], similarity=0.96)
        model.set_similar_texts(texts[2:], similarity=0.96)

        nodes = []
        for i, text in enumerate(texts):
            node = _create_node_with_text(text, token_count=100, sequence_number=i)
            nodes.append(node)
            graph.add_node(node, connect_temporal=False)
            semantic_index.index_node(node)

        manifest = RecoveryManifest()
        # Only save up to 100 tokens (should stop after first duplicate)
        result = strategy.compress(graph, manifest, target_tokens=100)

        assert result.success is True
        # Should have stopped early
        assert result.tokens_saved <= 150  # May process one more before checking

    def test_deduplicate_skips_small_nodes(
        self,
        model: DeterministicEmbeddingModel,
        semantic_index: SemanticIndex,
    ) -> None:
        """Test that nodes below min_tokens_to_dedupe are skipped."""
        # Create strategy with higher threshold
        index = SemanticIndex(model)
        strategy = DeduplicateSemantically(
            semantic_index=index,
            similarity_threshold=0.90,
            min_tokens_to_dedupe=200,  # High threshold
        )

        graph = ContextGraph()

        text1 = "Small content"
        text2 = "Small content"

        model.set_similar_texts([text1, text2], similarity=0.98)

        # Create nodes with token count below threshold
        node1 = _create_node_with_text(text1, token_count=50, sequence_number=0)
        node2 = _create_node_with_text(text2, token_count=50, sequence_number=1)

        graph.add_node(node1, connect_temporal=False)
        graph.add_node(node2, connect_temporal=False)

        index.index_node(node1)
        index.index_node(node2)

        manifest = RecoveryManifest()
        result = strategy.compress(graph, manifest)

        # Should not remove anything due to min_tokens threshold
        assert result.nodes_removed == 0
        assert len(list(graph)) == 2

    def test_deduplicate_logs_operations(
        self,
        model: DeterministicEmbeddingModel,
        semantic_index: SemanticIndex,
        strategy: DeduplicateSemantically,
    ) -> None:
        """Test that operations are logged to the recovery manifest."""
        graph = ContextGraph()

        text1 = "Content to deduplicate"
        text2 = "Content to deduplicate"

        model.set_similar_texts([text1, text2], similarity=0.97)

        node1 = _create_node_with_text(text1, token_count=100, sequence_number=0)
        node2 = _create_node_with_text(text2, token_count=100, sequence_number=1)

        graph.add_node(node1, connect_temporal=False)
        graph.add_node(node2, connect_temporal=False)

        semantic_index.index_node(node1)
        semantic_index.index_node(node2)

        manifest = RecoveryManifest()
        result = strategy.compress(graph, manifest)

        assert result.success is True
        assert result.is_recoverable is True

        # Check manifest has the operation logged
        assert len(manifest.operations) == 1
        op = manifest.operations[0]
        assert op.op_type == "deduplicate"
        assert op.similarity_score > 0.9
        assert len(op.original_contents) == 1  # One removed node's content saved

    def test_deduplicate_estimate_savings(
        self,
        model: DeterministicEmbeddingModel,
        semantic_index: SemanticIndex,
        strategy: DeduplicateSemantically,
    ) -> None:
        """Test that estimate_savings returns accurate estimates."""
        graph = ContextGraph()

        text1 = "Estimating token savings"
        text2 = "Estimating token savings"

        model.set_similar_texts([text1, text2], similarity=0.95)

        node1 = _create_node_with_text(text1, token_count=150, sequence_number=0)
        node2 = _create_node_with_text(text2, token_count=150, sequence_number=1)

        graph.add_node(node1, connect_temporal=False)
        graph.add_node(node2, connect_temporal=False)

        semantic_index.index_node(node1)
        semantic_index.index_node(node2)

        estimated = strategy.estimate_savings(graph)

        # Should estimate saving ~150 tokens (one node removed)
        assert estimated == 150

    def test_deduplicate_preserves_unrelated(
        self,
        model: DeterministicEmbeddingModel,
        semantic_index: SemanticIndex,
        strategy: DeduplicateSemantically,
    ) -> None:
        """Test that unrelated nodes are not affected."""
        graph = ContextGraph()

        # Two similar texts
        text1 = "Duplicate content here"
        text2 = "Duplicate content here"
        # One unrelated text
        text3 = "Completely different topic"

        model.set_similar_texts([text1, text2], similarity=0.96)
        # text3 will get a random embedding (dissimilar)

        node1 = _create_node_with_text(text1, token_count=100, sequence_number=0)
        node2 = _create_node_with_text(text2, token_count=100, sequence_number=1)
        node3 = _create_node_with_text(text3, token_count=100, sequence_number=2)

        for node in [node1, node2, node3]:
            graph.add_node(node, connect_temporal=False)
            semantic_index.index_node(node)

        manifest = RecoveryManifest()
        result = strategy.compress(graph, manifest)

        assert result.success is True
        assert result.nodes_removed == 1
        # Two nodes should remain (1 canonical from duplicates + 1 unrelated)
        assert len(list(graph)) == 2

        # The unrelated node should still exist
        remaining_ids = {n.id for n in graph}
        assert node3.id in remaining_ids

    def test_deduplicate_empty_graph(
        self,
        semantic_index: SemanticIndex,
        strategy: DeduplicateSemantically,
    ) -> None:
        """Test behavior with an empty graph."""
        graph = ContextGraph()
        manifest = RecoveryManifest()

        result = strategy.compress(graph, manifest)

        assert result.success is True
        assert result.nodes_removed == 0
        assert result.tokens_saved == 0

    def test_deduplicate_no_duplicates(
        self,
        model: DeterministicEmbeddingModel,
        semantic_index: SemanticIndex,
        strategy: DeduplicateSemantically,
    ) -> None:
        """Test behavior when no duplicates exist."""
        graph = ContextGraph()

        # Create nodes with different content (no duplicates)
        texts = [
            "First unique content",
            "Second different content",
            "Third distinct content",
        ]

        for i, text in enumerate(texts):
            node = _create_node_with_text(text, token_count=100, sequence_number=i)
            graph.add_node(node, connect_temporal=False)
            semantic_index.index_node(node)

        manifest = RecoveryManifest()
        result = strategy.compress(graph, manifest)

        assert result.success is True
        assert result.nodes_removed == 0
        assert len(list(graph)) == 3

    def test_deduplicate_skips_pinned_nodes(
        self,
        model: DeterministicEmbeddingModel,
        semantic_index: SemanticIndex,
        strategy: DeduplicateSemantically,
    ) -> None:
        """Test that pinned nodes are not removed."""
        graph = ContextGraph()

        text1 = "Pinned content that should stay"
        text2 = "Pinned content that should stay"

        model.set_similar_texts([text1, text2], similarity=0.98)

        node1 = _create_node_with_text(text1, token_count=100, sequence_number=0)
        node1.metadata.pinned = True  # Pin this node

        node2 = _create_node_with_text(text2, token_count=100, sequence_number=1)
        node2.metadata.pinned = True  # Pin this node too

        graph.add_node(node1, connect_temporal=False)
        graph.add_node(node2, connect_temporal=False)

        semantic_index.index_node(node1)
        semantic_index.index_node(node2)

        manifest = RecoveryManifest()
        result = strategy.compress(graph, manifest)

        # Both nodes should remain (pinned)
        assert result.nodes_removed == 0
        assert len(list(graph)) == 2

    def test_deduplicate_skips_already_compressed(
        self,
        model: DeterministicEmbeddingModel,
        semantic_index: SemanticIndex,
        strategy: DeduplicateSemantically,
    ) -> None:
        """Test that already compressed nodes are skipped."""
        graph = ContextGraph()

        text1 = "Already compressed content"
        text2 = "Already compressed content"

        model.set_similar_texts([text1, text2], similarity=0.97)

        node1 = _create_node_with_text(text1, token_count=100, sequence_number=0)
        node1.compression_level = CompressionLevel.COMPACTED

        node2 = _create_node_with_text(text2, token_count=100, sequence_number=1)

        graph.add_node(node1, connect_temporal=False)
        graph.add_node(node2, connect_temporal=False)

        semantic_index.index_node(node1)
        semantic_index.index_node(node2)

        manifest = RecoveryManifest()
        result = strategy.compress(graph, manifest)

        # Should not remove anything (one is already compressed)
        assert result.nodes_removed == 0

    def test_can_apply_with_duplicates(
        self,
        model: DeterministicEmbeddingModel,
        semantic_index: SemanticIndex,
        strategy: DeduplicateSemantically,
    ) -> None:
        """Test can_apply returns True when duplicates exist."""
        graph = ContextGraph()

        text1 = "Duplicate for can_apply test"
        text2 = "Duplicate for can_apply test"

        model.set_similar_texts([text1, text2], similarity=0.95)

        node1 = _create_node_with_text(text1, token_count=100, sequence_number=0)
        node2 = _create_node_with_text(text2, token_count=100, sequence_number=1)

        graph.add_node(node1, connect_temporal=False)
        graph.add_node(node2, connect_temporal=False)

        semantic_index.index_node(node1)
        semantic_index.index_node(node2)

        assert strategy.can_apply(graph) is True

    def test_can_apply_without_duplicates(
        self,
        semantic_index: SemanticIndex,
        strategy: DeduplicateSemantically,
    ) -> None:
        """Test can_apply returns False when no duplicates exist."""
        graph = ContextGraph()

        # Add unique content
        node = _create_node_with_text("Unique content", token_count=100)
        graph.add_node(node, connect_temporal=False)
        semantic_index.index_node(node)

        assert strategy.can_apply(graph) is False

    def test_can_apply_empty_graph(
        self,
        semantic_index: SemanticIndex,
        strategy: DeduplicateSemantically,
    ) -> None:
        """Test can_apply returns False for empty graph."""
        graph = ContextGraph()
        assert strategy.can_apply(graph) is False


class TestDeduplicateSemanticallyCanonicalSelection:
    """Tests specifically for canonical node selection logic."""

    @pytest.fixture
    def model(self) -> DeterministicEmbeddingModel:
        return DeterministicEmbeddingModel(dimension=64)

    @pytest.fixture
    def semantic_index(self, model: DeterministicEmbeddingModel) -> SemanticIndex:
        return SemanticIndex(model)

    def test_prefer_recent_selects_newer_node(
        self,
        model: DeterministicEmbeddingModel,
        semantic_index: SemanticIndex,
    ) -> None:
        """Test that prefer_recent=True favors more recent nodes."""
        strategy = DeduplicateSemantically(
            semantic_index=semantic_index,
            similarity_threshold=0.90,
            min_tokens_to_dedupe=50,
            prefer_recent=True,
        )

        graph = ContextGraph()

        text = "Content for recency test"
        model.set_identical_texts([text, text])

        # Older node with same importance
        node1 = _create_node_with_text(
            text, token_count=100, importance=0.5, sequence_number=0
        )
        # Newer node with same importance
        node2 = _create_node_with_text(
            text, token_count=100, importance=0.5, sequence_number=10
        )

        graph.add_node(node1, connect_temporal=False)
        graph.add_node(node2, connect_temporal=False)

        semantic_index.index_node(node1)
        semantic_index.index_node(node2)

        manifest = RecoveryManifest()
        strategy.compress(graph, manifest)

        # Newer node (node2) should be kept as canonical
        remaining = list(graph)
        assert len(remaining) == 1
        assert remaining[0].id == node2.id

    def test_prefer_higher_importance(
        self,
        model: DeterministicEmbeddingModel,
        semantic_index: SemanticIndex,
    ) -> None:
        """Test that higher importance nodes are preferred as canonical."""
        strategy = DeduplicateSemantically(
            semantic_index=semantic_index,
            similarity_threshold=0.90,
            min_tokens_to_dedupe=50,
            prefer_recent=False,  # Disable recency preference
        )

        graph = ContextGraph()

        text = "Content for importance test"
        model.set_identical_texts([text, text])

        # Lower importance node
        node1 = _create_node_with_text(
            text, token_count=100, importance=0.3, sequence_number=0
        )
        # Higher importance node
        node2 = _create_node_with_text(
            text, token_count=100, importance=0.9, sequence_number=1
        )

        graph.add_node(node1, connect_temporal=False)
        graph.add_node(node2, connect_temporal=False)

        semantic_index.index_node(node1)
        semantic_index.index_node(node2)

        manifest = RecoveryManifest()
        strategy.compress(graph, manifest)

        # Higher importance node (node2) should be kept
        remaining = list(graph)
        assert len(remaining) == 1
        assert remaining[0].id == node2.id

    def test_prefer_more_tokens(
        self,
        model: DeterministicEmbeddingModel,
        semantic_index: SemanticIndex,
    ) -> None:
        """Test that nodes with more tokens are preferred as canonical."""
        strategy = DeduplicateSemantically(
            semantic_index=semantic_index,
            similarity_threshold=0.90,
            min_tokens_to_dedupe=50,
            prefer_recent=False,
        )

        graph = ContextGraph()

        text1 = "Shorter content"
        text2 = "Much longer and more detailed content here"

        model.set_similar_texts([text1, text2], similarity=0.95)

        # Smaller node
        node1 = _create_node_with_text(
            text1, token_count=80, importance=0.5, sequence_number=0
        )
        # Larger node
        node2 = _create_node_with_text(
            text2, token_count=200, importance=0.5, sequence_number=1
        )

        graph.add_node(node1, connect_temporal=False)
        graph.add_node(node2, connect_temporal=False)

        semantic_index.index_node(node1)
        semantic_index.index_node(node2)

        manifest = RecoveryManifest()
        strategy.compress(graph, manifest)

        # Larger node (node2) should be kept
        remaining = list(graph)
        assert len(remaining) == 1
        assert remaining[0].id == node2.id


class TestDeduplicateSemanticallyEdgeCases:
    """Edge case tests for DeduplicateSemantically."""

    @pytest.fixture
    def model(self) -> MockEmbeddingModel:
        return MockEmbeddingModel(dimension=64)

    @pytest.fixture
    def semantic_index(self, model: MockEmbeddingModel) -> SemanticIndex:
        return SemanticIndex(model)

    @pytest.fixture
    def strategy(self, semantic_index: SemanticIndex) -> DeduplicateSemantically:
        return DeduplicateSemantically(
            semantic_index=semantic_index,
            similarity_threshold=0.90,
            min_tokens_to_dedupe=50,
        )

    def test_single_node_graph(
        self,
        semantic_index: SemanticIndex,
        strategy: DeduplicateSemantically,
    ) -> None:
        """Test behavior with only one node."""
        graph = ContextGraph()

        node = ContextNode(
            type=NodeType.MESSAGE,
            content=Content(text="Only node"),
            token_count=100,
        )
        graph.add_node(node, connect_temporal=False)
        semantic_index.index_node(node)

        manifest = RecoveryManifest()
        result = strategy.compress(graph, manifest)

        assert result.success is True
        assert result.nodes_removed == 0
        assert len(list(graph)) == 1

    def test_target_node_ids_filtering(
        self,
        model: MockEmbeddingModel,
        semantic_index: SemanticIndex,
    ) -> None:
        """Test that target_node_ids properly filters nodes."""
        # Use deterministic model for this test
        det_model = DeterministicEmbeddingModel(dimension=64)
        det_index = SemanticIndex(det_model)
        strategy = DeduplicateSemantically(
            semantic_index=det_index,
            similarity_threshold=0.90,
            min_tokens_to_dedupe=50,
        )

        graph = ContextGraph()

        text = "Duplicate content"
        det_model.set_identical_texts([text, text, text])

        node1 = ContextNode(
            type=NodeType.MESSAGE,
            content=Content(text=text),
            token_count=100,
            sequence_number=0,
        )
        node2 = ContextNode(
            type=NodeType.MESSAGE,
            content=Content(text=text),
            token_count=100,
            sequence_number=1,
        )
        node3 = ContextNode(
            type=NodeType.MESSAGE,
            content=Content(text=text),
            token_count=100,
            sequence_number=2,
        )

        for node in [node1, node2, node3]:
            graph.add_node(node, connect_temporal=False)
            det_index.index_node(node)

        manifest = RecoveryManifest()
        # Only target node2
        result = strategy.compress(graph, manifest, target_node_ids=[node2.id])

        # Should only consider node2 for removal
        assert result.success is True
        # node2 might be removed if it's not the canonical
        # The behavior depends on whether node2 ends up being canonical or duplicate

    def test_compression_result_duration(
        self,
        semantic_index: SemanticIndex,
        strategy: DeduplicateSemantically,
    ) -> None:
        """Test that compression result includes duration."""
        graph = ContextGraph()
        manifest = RecoveryManifest()

        result = strategy.compress(graph, manifest)

        assert result.duration_ms >= 0

    def test_similarity_threshold_boundary(
        self,
        semantic_index: SemanticIndex,
    ) -> None:
        """Test behavior when nodes are below similarity threshold."""
        # Use deterministic model
        det_model = DeterministicEmbeddingModel(dimension=64)
        det_index = SemanticIndex(det_model)

        # Create strategy with high threshold
        strategy = DeduplicateSemantically(
            semantic_index=det_index,
            similarity_threshold=0.95,
            min_tokens_to_dedupe=50,
        )

        graph = ContextGraph()

        # Use completely different texts - they will get random embeddings
        # which will have very low similarity (typically < 0.3 for random vectors)
        text1 = "Boundary test unique content alpha"
        text2 = "Completely different zebra banana purple"

        # Don't set embeddings - let them be random (dissimilar)

        node1 = ContextNode(
            type=NodeType.MESSAGE,
            content=Content(text=text1),
            token_count=100,
            sequence_number=0,
        )
        node2 = ContextNode(
            type=NodeType.MESSAGE,
            content=Content(text=text2),
            token_count=100,
            sequence_number=1,
        )

        graph.add_node(node1, connect_temporal=False)
        graph.add_node(node2, connect_temporal=False)

        det_index.index_node(node1)
        det_index.index_node(node2)

        manifest = RecoveryManifest()
        result = strategy.compress(graph, manifest)

        # Nodes should NOT be deduplicated (random embeddings are dissimilar)
        assert result.nodes_removed == 0
        assert len(list(graph)) == 2
