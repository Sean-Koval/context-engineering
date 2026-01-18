"""Cache key generation for tool call caching.

Provides multiple strategies for generating cache keys:
- Exact: Hash of tool name + arguments
- Normalized: Hash after normalizing paths, case, whitespace
- Semantic: Uses embeddings for similarity-based matching
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from typing import TYPE_CHECKING, Any

from context_tools.types import CacheKeyType, ToolCallSignature

if TYPE_CHECKING:
    from context_core.semantic import SemanticIndex


class CacheKeyGenerator:
    """Generate cache keys for tool call signatures.

    Supports exact, normalized, and semantic key generation strategies
    for flexible cache matching.

    Example:
        >>> generator = CacheKeyGenerator()
        >>> sig = ToolCallSignature(tool_name="read_file", arguments={"path": "x"})
        >>> key = generator.generate(sig, CacheKeyType.EXACT)
        >>> norm_key = generator.generate(sig, CacheKeyType.NORMALIZED)
    """

    # Tools with path-like arguments that should be normalized
    PATH_ARGUMENT_NAMES = {"path", "file", "filepath", "file_path", "directory", "dir"}

    # Patterns to normalize in paths
    HOME_PATTERN = re.compile(r"^(/Users/[^/]+|/home/[^/]+|~)")

    def __init__(
        self,
        semantic_index: SemanticIndex | None = None,
        normalize_paths: bool = True,
        normalize_case: bool = True,
    ) -> None:
        """Initialize CacheKeyGenerator.

        Args:
            semantic_index: Optional semantic index for embedding-based keys
            normalize_paths: Whether to normalize file paths
            normalize_case: Whether to normalize string case
        """
        self._semantic_index = semantic_index
        self._normalize_paths = normalize_paths
        self._normalize_case = normalize_case

    def generate(
        self,
        signature: ToolCallSignature,
        key_type: CacheKeyType = CacheKeyType.EXACT,
    ) -> str:
        """Generate a cache key for a tool call signature.

        Args:
            signature: Tool call signature to generate key for
            key_type: Type of key matching to use

        Returns:
            Hash string to use as cache key
        """
        if key_type == CacheKeyType.EXACT:
            return self._generate_exact_key(signature)
        elif key_type == CacheKeyType.NORMALIZED:
            return self._generate_normalized_key(signature)
        elif key_type == CacheKeyType.SEMANTIC:
            return self._generate_semantic_key(signature)
        else:
            raise ValueError(f"Unknown key type: {key_type}")

    def _generate_exact_key(self, signature: ToolCallSignature) -> str:
        """Generate exact match key from signature."""
        # Sort arguments for consistent hashing
        sorted_args = self._sort_dict(signature.arguments)
        key_data = {
            "tool": signature.tool_name,
            "args": sorted_args,
            "context": signature.context_hash,
        }
        key_str = json.dumps(key_data, sort_keys=True, default=str)
        return hashlib.sha256(key_str.encode()).hexdigest()

    def _generate_normalized_key(self, signature: ToolCallSignature) -> str:
        """Generate normalized key with path/case normalization."""
        normalized_args = self._normalize_arguments(
            signature.tool_name,
            signature.arguments,
        )
        sorted_args = self._sort_dict(normalized_args)
        key_data = {
            "tool": signature.tool_name.lower()
            if self._normalize_case
            else signature.tool_name,
            "args": sorted_args,
            "context": signature.context_hash,
        }
        key_str = json.dumps(key_data, sort_keys=True, default=str)
        return hashlib.sha256(key_str.encode()).hexdigest()

    def _generate_semantic_key(self, signature: ToolCallSignature) -> str:
        """Generate semantic key using content hash.

        Note: Actual semantic matching happens during lookup using embeddings,
        but we still need a fallback key for storage.
        """
        # Use normalized key as base for semantic entries
        return self._generate_normalized_key(signature)

    def _normalize_arguments(
        self,
        tool_name: str,
        arguments: dict[str, Any],
    ) -> dict[str, Any]:
        """Normalize arguments based on their type and name."""
        normalized: dict[str, Any] = {}

        for key, value in arguments.items():
            normalized_key = key.lower() if self._normalize_case else key
            normalized_value = self._normalize_value(key, value)
            normalized[normalized_key] = normalized_value

        return normalized

    def _normalize_value(self, key: str, value: Any) -> Any:
        """Normalize a single argument value."""
        if value is None:
            return None

        # Normalize path-like arguments
        is_path_arg = key.lower() in self.PATH_ARGUMENT_NAMES
        if self._normalize_paths and is_path_arg and isinstance(value, str):
            return self._normalize_path(value)

        # Normalize string values
        if isinstance(value, str):
            normalized = value.strip()
            # Don't lowercase file content, just paths and identifiers
            if self._normalize_case and is_path_arg:
                normalized = normalized.lower()
            return normalized

        # Normalize lists recursively
        if isinstance(value, list):
            return [self._normalize_value(key, item) for item in value]

        # Normalize dicts recursively
        if isinstance(value, dict):
            return {k: self._normalize_value(k, v) for k, v in value.items()}

        return value

    def _normalize_path(self, path: str) -> str:
        """Normalize a file path for consistent caching."""
        # Expand user home directory
        path = os.path.expanduser(path)

        # Normalize path separators and resolve . and ..
        path = os.path.normpath(path)

        # Replace home directory with placeholder for user-independence
        path = self.HOME_PATTERN.sub("~", path)

        # Lowercase on case-insensitive systems (optional)
        if self._normalize_case:
            path = path.lower()

        return path

    def _sort_dict(self, d: dict[str, Any]) -> dict[str, Any]:
        """Recursively sort dictionary for consistent serialization."""
        result: dict[str, Any] = {}
        for key in sorted(d.keys()):
            value = d[key]
            if isinstance(value, dict):
                result[key] = self._sort_dict(value)
            elif isinstance(value, list):
                result[key] = [
                    self._sort_dict(item) if isinstance(item, dict) else item
                    for item in value
                ]
            else:
                result[key] = value
        return result

    def get_embedding(self, signature: ToolCallSignature) -> list[float] | None:
        """Get embedding vector for semantic matching.

        Args:
            signature: Tool call signature to embed

        Returns:
            Embedding vector if semantic index is available, None otherwise
        """
        if self._semantic_index is None:
            return None

        # Create text representation for embedding
        text = self._signature_to_text(signature)

        # SemanticIndex exposes embedding_model which has embed() method
        # that takes a list of texts and returns a 2D array
        embedding_model = getattr(self._semantic_index, "embedding_model", None)
        if embedding_model is None:
            # Fallback: try direct embed method (for simpler interfaces)
            embed_fn = getattr(self._semantic_index, "embed", None)
            if embed_fn is not None:
                result = embed_fn(text)
                if result is not None:
                    return list(result) if hasattr(result, "__iter__") else None
            return None

        # Use embedding model's embed method
        embeddings = embedding_model.embed([text])
        if embeddings is not None and len(embeddings) > 0:
            return list(embeddings[0])
        return None

    def _signature_to_text(self, signature: ToolCallSignature) -> str:
        """Convert signature to text for embedding."""
        parts = [f"Tool: {signature.tool_name}"]
        for key, value in sorted(signature.arguments.items()):
            parts.append(f"{key}: {value}")
        return " | ".join(parts)

    def compute_similarity(
        self,
        sig1: ToolCallSignature,
        sig2: ToolCallSignature,
    ) -> float:
        """Compute semantic similarity between two signatures.

        Args:
            sig1: First signature
            sig2: Second signature

        Returns:
            Similarity score (0-1), or 0 if semantic index unavailable
        """
        if self._semantic_index is None:
            return 0.0

        # Different tools = no match
        if sig1.tool_name != sig2.tool_name:
            return 0.0

        emb1 = self.get_embedding(sig1)
        emb2 = self.get_embedding(sig2)

        if emb1 is None or emb2 is None:
            return 0.0

        # Cosine similarity
        return self._cosine_similarity(emb1, emb2)

    @staticmethod
    def _cosine_similarity(vec1: list[float], vec2: list[float]) -> float:
        """Compute cosine similarity between two vectors."""
        if len(vec1) != len(vec2):
            return 0.0

        dot_product = sum(a * b for a, b in zip(vec1, vec2, strict=True))
        norm1 = sum(a * a for a in vec1) ** 0.5
        norm2 = sum(b * b for b in vec2) ** 0.5

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return dot_product / (norm1 * norm2)
