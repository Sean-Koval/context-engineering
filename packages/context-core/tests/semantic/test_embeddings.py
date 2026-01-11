"""Tests for embedding models."""

from __future__ import annotations

import numpy as np
import pytest

from context_core.semantic.embeddings import EmbeddingModel, MockEmbeddingModel


class TestMockEmbeddingModel:
    """Tests for MockEmbeddingModel."""

    def test_default_dimension(self) -> None:
        """Test default dimension is 64."""
        model = MockEmbeddingModel()
        assert model.dimension == 64

    def test_custom_dimension(self) -> None:
        """Test custom dimension."""
        model = MockEmbeddingModel(dimension=128)
        assert model.dimension == 128

    def test_embed_single_text(self) -> None:
        """Test embedding a single text."""
        model = MockEmbeddingModel(dimension=64)
        embeddings = model.embed(["Hello world"])

        assert embeddings.shape == (1, 64)
        assert embeddings.dtype == np.float32

    def test_embed_multiple_texts(self) -> None:
        """Test embedding multiple texts."""
        model = MockEmbeddingModel(dimension=64)
        texts = ["Hello", "World", "Test"]
        embeddings = model.embed(texts)

        assert embeddings.shape == (3, 64)
        assert embeddings.dtype == np.float32

    def test_embed_empty_list(self) -> None:
        """Test embedding empty list."""
        model = MockEmbeddingModel(dimension=64)
        embeddings = model.embed([])

        assert embeddings.shape == (0, 64)
        assert embeddings.dtype == np.float32

    def test_embed_deterministic(self) -> None:
        """Test that same text produces same embedding."""
        model = MockEmbeddingModel(dimension=64)
        text = "Hello world"

        emb1 = model.embed([text])
        emb2 = model.embed([text])

        np.testing.assert_array_equal(emb1, emb2)

    def test_embed_different_texts_different_embeddings(self) -> None:
        """Test that different texts produce different embeddings."""
        model = MockEmbeddingModel(dimension=64)

        emb1 = model.embed(["Hello"])
        emb2 = model.embed(["World"])

        # Should not be equal
        assert not np.allclose(emb1, emb2)

    def test_embeddings_normalized(self) -> None:
        """Test that embeddings are normalized to unit length."""
        model = MockEmbeddingModel(dimension=64)
        embeddings = model.embed(["Hello", "World", "Test"])

        norms = np.linalg.norm(embeddings, axis=1)
        np.testing.assert_array_almost_equal(norms, np.ones(3), decimal=5)

    def test_implements_protocol(self) -> None:
        """Test that MockEmbeddingModel implements EmbeddingModel protocol."""
        model = MockEmbeddingModel()
        assert isinstance(model, EmbeddingModel)


class TestEmbeddingModelProtocol:
    """Tests for EmbeddingModel protocol compliance."""

    def test_protocol_has_dimension(self) -> None:
        """Test that protocol requires dimension property."""
        model = MockEmbeddingModel()
        assert hasattr(model, "dimension")
        assert isinstance(model.dimension, int)

    def test_protocol_has_embed(self) -> None:
        """Test that protocol requires embed method."""
        model = MockEmbeddingModel()
        assert hasattr(model, "embed")
        assert callable(model.embed)


# Skip SentenceTransformerEmbedding tests if not installed
try:
    import sentence_transformers  # noqa: F401

    from context_core.semantic.embeddings import SentenceTransformerEmbedding

    HAS_SENTENCE_TRANSFORMERS = True
except ImportError:
    HAS_SENTENCE_TRANSFORMERS = False
    SentenceTransformerEmbedding = None  # type: ignore[assignment, misc]


@pytest.mark.skipif(
    not HAS_SENTENCE_TRANSFORMERS,
    reason="sentence-transformers not installed",
)
class TestSentenceTransformerEmbedding:
    """Tests for SentenceTransformerEmbedding."""

    @pytest.fixture
    def model(self) -> SentenceTransformerEmbedding:
        """Create a test model."""
        return SentenceTransformerEmbedding("all-MiniLM-L6-v2")

    def test_dimension(self, model: SentenceTransformerEmbedding) -> None:
        """Test embedding dimension."""
        # MiniLM-L6-v2 has 384 dimensions
        assert model.dimension == 384

    def test_model_name(self, model: SentenceTransformerEmbedding) -> None:
        """Test model name property."""
        assert model.model_name == "all-MiniLM-L6-v2"

    def test_embed_single(self, model: SentenceTransformerEmbedding) -> None:
        """Test embedding single text."""
        embeddings = model.embed(["Hello world"])
        assert embeddings.shape == (1, 384)
        assert embeddings.dtype == np.float32

    def test_embed_multiple(self, model: SentenceTransformerEmbedding) -> None:
        """Test embedding multiple texts."""
        embeddings = model.embed(["Hello", "World"])
        assert embeddings.shape == (2, 384)

    def test_embed_empty(self, model: SentenceTransformerEmbedding) -> None:
        """Test embedding empty list."""
        embeddings = model.embed([])
        assert embeddings.shape == (0, 384)

    def test_similar_texts_close_embeddings(
        self, model: SentenceTransformerEmbedding
    ) -> None:
        """Test that similar texts have close embeddings."""
        embeddings = model.embed(["Hello world", "Hello there world"])

        # Compute cosine similarity
        norm1 = embeddings[0] / np.linalg.norm(embeddings[0])
        norm2 = embeddings[1] / np.linalg.norm(embeddings[1])
        similarity = np.dot(norm1, norm2)

        # Similar texts should have high similarity
        assert similarity > 0.7

    def test_implements_protocol(self, model: SentenceTransformerEmbedding) -> None:
        """Test protocol compliance."""
        assert isinstance(model, EmbeddingModel)
