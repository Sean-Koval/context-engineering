"""Externalize payloads strategy.

This strategy moves large content to external storage, keeping only a
preview and reference in the context. The content can be fully recovered
by fetching from the external storage.
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, runtime_checkable
from uuid import UUID

from context_compression.recovery import RecoveryManifest
from context_compression.strategies.base import BaseCompressionStrategy
from context_compression.types import CompressionResult, CompressionTier
from context_core.graph.types import CompressionLevel, NodeType

if TYPE_CHECKING:
    from context_core.graph import ContextGraph, ContextNode


@runtime_checkable
class ExternalStorage(Protocol):
    """Protocol for external storage backends.

    External storage is used to store large content that has been
    externalized from the context graph. Implementations must support
    store, retrieve, and delete operations.
    """

    def store(self, key: str, content: str) -> str:
        """Store content and return a URI.

        Args:
            key: Unique key for the content
            content: The content to store

        Returns:
            URI that can be used to retrieve the content
        """
        ...

    def retrieve(self, uri: str) -> str | None:
        """Retrieve content by URI.

        Args:
            uri: The URI from store()

        Returns:
            The stored content, or None if not found
        """
        ...

    def delete(self, uri: str) -> bool:
        """Delete content by URI.

        Args:
            uri: The URI to delete

        Returns:
            True if deleted, False if not found
        """
        ...

    def exists(self, uri: str) -> bool:
        """Check if content exists.

        Args:
            uri: The URI to check

        Returns:
            True if content exists
        """
        ...


class InMemoryExternalStorage:
    """In-memory implementation of ExternalStorage.

    Stores content in a dictionary. Useful for testing and short-lived
    sessions where persistence is not required.
    """

    def __init__(self) -> None:
        self._storage: dict[str, str] = {}
        self._uri_prefix = "memory://"

    def store(self, key: str, content: str) -> str:
        """Store content in memory."""
        uri = f"{self._uri_prefix}{key}"
        self._storage[uri] = content
        return uri

    def retrieve(self, uri: str) -> str | None:
        """Retrieve content from memory."""
        return self._storage.get(uri)

    def delete(self, uri: str) -> bool:
        """Delete content from memory."""
        return self._storage.pop(uri, None) is not None

    def exists(self, uri: str) -> bool:
        """Check if content exists in memory."""
        return uri in self._storage

    def clear(self) -> None:
        """Clear all stored content."""
        self._storage.clear()

    def __len__(self) -> int:
        """Return number of stored items."""
        return len(self._storage)


class FileSystemExternalStorage:
    """Filesystem-based implementation of ExternalStorage.

    Stores externalized content as files on disk. Each piece of content
    is stored in a file named after its content hash, enabling deduplication
    and efficient retrieval.

    The URI format is: file://{absolute_path}
    """

    def __init__(self, base_path: Path | str) -> None:
        """Initialize filesystem storage.

        Args:
            base_path: Base directory for storing files. Will be created
                       if it doesn't exist.
        """
        self._base_path = Path(base_path).resolve()
        self._base_path.mkdir(parents=True, exist_ok=True)
        self._uri_prefix = "file://"

    @property
    def base_path(self) -> Path:
        """Return the base path for storage."""
        return self._base_path

    def _uri_to_path(self, uri: str) -> Path | None:
        """Convert a URI to a filesystem path.

        Args:
            uri: The file:// URI

        Returns:
            Path object, or None if URI format is invalid
        """
        if not uri.startswith(self._uri_prefix):
            return None
        return Path(uri[len(self._uri_prefix) :])

    def store(self, key: str, content: str) -> str:
        """Store content to filesystem.

        Args:
            key: Unique key for the content (used as filename)
            content: The content to store

        Returns:
            URI that can be used to retrieve the content
        """
        file_path = self._base_path / f"{key}.txt"
        file_path.write_text(content, encoding="utf-8")
        return f"{self._uri_prefix}{file_path}"

    def retrieve(self, uri: str) -> str | None:
        """Retrieve content from filesystem.

        Args:
            uri: The URI from store()

        Returns:
            The stored content, or None if not found or error occurs
        """
        file_path = self._uri_to_path(uri)
        if file_path is None:
            return None

        try:
            return file_path.read_text(encoding="utf-8")
        except (FileNotFoundError, OSError):
            return None

    def delete(self, uri: str) -> bool:
        """Delete content from filesystem.

        Args:
            uri: The URI to delete

        Returns:
            True if deleted, False if not found or error occurs
        """
        file_path = self._uri_to_path(uri)
        if file_path is None:
            return False

        try:
            file_path.unlink()
            return True
        except (FileNotFoundError, OSError):
            return False

    def exists(self, uri: str) -> bool:
        """Check if content exists on filesystem.

        Args:
            uri: The URI to check

        Returns:
            True if content exists
        """
        file_path = self._uri_to_path(uri)
        return file_path is not None and file_path.exists()


class ExternalizePayloads(BaseCompressionStrategy):
    """Externalize large payloads to external storage.

    This strategy identifies nodes with large content (typically tool results)
    and moves the content to external storage, keeping only a preview in
    the context graph.

    Configuration:
        min_tokens: Minimum token count to externalize (default 500)
        preview_tokens: Number of tokens to keep as preview (default 100)
        target_node_types: Node types to target (default: TOOL_RESULT, ARTIFACT)

    The strategy is fully reversible - original content can be recovered
    from external storage using the URI stored in the node.
    """

    def __init__(
        self,
        storage: ExternalStorage,
        min_tokens: int = 500,
        preview_tokens: int = 100,
        target_node_types: list[NodeType] | None = None,
    ):
        """Initialize the strategy.

        Args:
            storage: External storage backend
            min_tokens: Minimum tokens to trigger externalization
            preview_tokens: Tokens to keep as preview
            target_node_types: Node types to target
        """
        self._storage = storage
        self._min_tokens = min_tokens
        self._preview_tokens = preview_tokens
        self._target_types = target_node_types or [
            NodeType.TOOL_RESULT,
            NodeType.ARTIFACT,
        ]

    @property
    def _name(self) -> str:
        return "externalize_payloads"

    @property
    def _tier(self) -> CompressionTier:
        return CompressionTier.LOSSLESS

    @property
    def _priority(self) -> int:
        return 10  # Run early in lossless tier

    def _get_content_text(self, node: ContextNode) -> str:
        """Extract text content from a node."""
        if node.content.text:
            return node.content.text
        if node.content.tool_output is not None:
            if isinstance(node.content.tool_output, str):
                return node.content.tool_output
            return json.dumps(node.content.tool_output, indent=2)
        if node.content.artifact_data is not None:
            if isinstance(node.content.artifact_data, str):
                return node.content.artifact_data
            return json.dumps(node.content.artifact_data, indent=2)
        return ""

    def _get_preview(self, content: str, max_chars: int = 500) -> str:
        """Generate a preview of content.

        Args:
            content: Full content
            max_chars: Maximum characters for preview

        Returns:
            Truncated content with ellipsis if needed
        """
        if len(content) <= max_chars:
            return content

        # Try to break at a reasonable point
        preview = content[:max_chars]
        last_newline = preview.rfind("\n")
        if last_newline > max_chars // 2:
            preview = preview[:last_newline]

        return preview + "\n... [externalized]"

    def _compute_hash(self, content: str) -> str:
        """Compute hash of content for verification."""
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    def _is_eligible(self, node: ContextNode) -> bool:
        """Check if a node is eligible for externalization."""
        # Check node type
        if node.type not in self._target_types:
            return False

        # Check compression level (don't re-externalize)
        if node.compression_level != CompressionLevel.FULL:
            return False

        # Check if already externalized
        if node.content.external_ref:
            return False

        # Check token count
        token_count = node.token_count or 0
        if token_count < self._min_tokens:
            return False

        # Check if pinned
        return not node.metadata.pinned

    def _estimate_savings_impl(
        self,
        graph: ContextGraph,
        target_node_ids: list[UUID] | None,
    ) -> int:
        """Estimate tokens that would be saved."""
        target_ids = set(target_node_ids) if target_node_ids else None
        total_savings = 0

        for node in graph:
            # Skip if not in target set
            if target_ids and node.id not in target_ids:
                continue

            if not self._is_eligible(node):
                continue

            token_count = node.token_count or 0
            # Estimate: we save (tokens - preview_tokens)
            savings = max(0, token_count - self._preview_tokens)
            total_savings += savings

        return total_savings

    def _compress_impl(
        self,
        graph: ContextGraph,
        manifest: RecoveryManifest,
        target_node_ids: list[UUID] | None,
        target_tokens: int | None,
    ) -> CompressionResult:
        """Execute externalization compression."""
        start_time = time.perf_counter()
        target_ids = set(target_node_ids) if target_node_ids else None

        original_tokens = 0
        compressed_tokens = 0
        nodes_compressed = 0
        tokens_saved = 0

        for node in graph:
            # Check target limit
            if target_tokens and tokens_saved >= target_tokens:
                break

            # Skip if not in target set
            if target_ids and node.id not in target_ids:
                continue

            if not self._is_eligible(node):
                continue

            # Get content
            content = self._get_content_text(node)
            if not content:
                continue

            token_count = node.token_count or 0
            original_tokens += token_count

            # Generate preview
            preview = self._get_preview(content)
            content_hash = self._compute_hash(content)

            # Store externally
            storage_key = f"{node.id}_{content_hash}"
            uri = self._storage.store(storage_key, content)

            # Log operation to manifest
            manifest.log_externalize(
                node_id=node.id,
                external_uri=uri,
                original_tokens=token_count,
                content_hash=content_hash,
                preview=preview[:200] if preview else None,
                storage_backend="memory",
            )

            # Update node content
            node.content.external_ref = uri
            node.content.original_tokens = token_count
            node.content.compressed_tokens = self._preview_tokens

            # Update based on node type
            if node.type == NodeType.TOOL_RESULT:
                node.content.tool_output = preview
            elif node.type == NodeType.ARTIFACT:
                node.content.artifact_data = preview
            else:
                node.content.text = preview

            # Update compression level and token count
            node.compression_level = CompressionLevel.COMPACTED
            node.token_count = self._preview_tokens

            compressed_tokens += self._preview_tokens
            tokens_saved += token_count - self._preview_tokens
            nodes_compressed += 1

        return CompressionResult(
            success=True,
            strategy_name=self.name,
            tier=self.tier,
            original_tokens=original_tokens,
            compressed_tokens=compressed_tokens,
            tokens_saved=tokens_saved,
            nodes_processed=len(graph),
            nodes_compressed=nodes_compressed,
            nodes_removed=0,
            nodes_created=0,
            duration_ms=(time.perf_counter() - start_time) * 1000,
            is_recoverable=True,
        )

    def _can_apply_impl(self, graph: ContextGraph) -> bool:
        """Check if there are eligible nodes."""
        return any(self._is_eligible(node) for node in graph)

    def recover(self, node: ContextNode) -> str | None:
        """Recover original content for an externalized node.

        Args:
            node: The node to recover

        Returns:
            Original content, or None if not recoverable
        """
        if not node.content.external_ref:
            return None
        return self._storage.retrieve(node.content.external_ref)
