"""Embedding model protocol and implementations.

This module provides:
- EmbeddingModel: Protocol for embedding implementations
- SentenceTransformerEmbedding: sentence-transformers implementation
- MockEmbeddingModel: Simple mock for testing
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray


@runtime_checkable
class EmbeddingModel(Protocol):
    """Protocol for embedding models.

    All embedding implementations must provide:
    - dimension: The size of embedding vectors
    - embed: Method to convert text to vectors

    Example:
        >>> model = SentenceTransformerEmbedding()
        >>> embeddings = model.embed(["Hello world"])
        >>> embeddings.shape
        (1, 384)
    """

    @property
    def dimension(self) -> int:
        """Return embedding dimension."""
        ...

    def embed(self, texts: list[str]) -> NDArray[np.float32]:
        """Embed a batch of texts.

        Args:
            texts: List of text strings to embed.

        Returns:
            Array of shape (len(texts), dimension) containing embeddings.
        """
        ...


class SentenceTransformerEmbedding:
    """Embedding using sentence-transformers library.

    Uses the sentence-transformers library for high-quality
    text embeddings. Supports many pre-trained models.

    Attributes:
        dimension: The embedding dimension (e.g., 384 for MiniLM).

    Example:
        >>> model = SentenceTransformerEmbedding("all-MiniLM-L6-v2")
        >>> embeddings = model.embed(["Hello world", "Goodbye world"])
        >>> embeddings.shape
        (2, 384)
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        """Initialize the embedding model.

        Args:
            model_name: Name of the sentence-transformers model to use.
                Defaults to "all-MiniLM-L6-v2" which is fast and effective.

        Raises:
            ImportError: If sentence-transformers is not installed.
        """
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as e:
            msg = (
                "sentence-transformers is required for SentenceTransformerEmbedding. "
                "Install with: pip install sentence-transformers"
            )
            raise ImportError(msg) from e

        self._model_name = model_name
        self._model = SentenceTransformer(model_name)
        self._dimension: int = self._model.get_sentence_embedding_dimension()

    @property
    def dimension(self) -> int:
        """Return embedding dimension."""
        return self._dimension

    @property
    def model_name(self) -> str:
        """Return the model name."""
        return self._model_name

    def embed(self, texts: list[str]) -> NDArray[np.float32]:
        """Embed a batch of texts.

        Args:
            texts: List of text strings to embed.

        Returns:
            Array of shape (len(texts), dimension) containing embeddings.
        """
        if not texts:
            return np.array([], dtype=np.float32).reshape(0, self._dimension)

        embeddings = self._model.encode(
            texts,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return embeddings.astype(np.float32)


class MockEmbeddingModel:
    """Simple mock embedding model for testing.

    Generates deterministic embeddings based on text hash.
    Useful for unit tests that don't require real embeddings.

    Attributes:
        dimension: The embedding dimension (default 64).

    Example:
        >>> model = MockEmbeddingModel(dimension=64)
        >>> embeddings = model.embed(["test"])
        >>> embeddings.shape
        (1, 64)
    """

    def __init__(self, dimension: int = 64) -> None:
        """Initialize mock embedding model.

        Args:
            dimension: The embedding dimension to use.
        """
        self._dimension = dimension

    @property
    def dimension(self) -> int:
        """Return embedding dimension."""
        return self._dimension

    def embed(self, texts: list[str]) -> NDArray[np.float32]:
        """Generate mock embeddings based on text hash.

        The same text will always produce the same embedding,
        making this suitable for deterministic testing.

        Args:
            texts: List of text strings to embed.

        Returns:
            Array of shape (len(texts), dimension) containing embeddings.
        """
        if not texts:
            return np.array([], dtype=np.float32).reshape(0, self._dimension)

        embeddings = []
        for text in texts:
            # Generate deterministic embedding from text hash
            rng = np.random.default_rng(hash(text) % (2**32))
            embedding = rng.standard_normal(self._dimension).astype(np.float32)
            # Normalize to unit length
            embedding = embedding / (np.linalg.norm(embedding) + 1e-8)
            embeddings.append(embedding)

        return np.stack(embeddings)
