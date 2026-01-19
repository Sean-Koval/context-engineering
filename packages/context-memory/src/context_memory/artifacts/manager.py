"""ArtifactManager - Versioned artifact storage with content addressing.

Provides versioned storage for artifacts (code, files, data) with:
- Content-addressed storage (SHA-256 deduplication)
- Complete version history
- Unified diff between versions
- Session-scoped artifact management
"""

from __future__ import annotations

import difflib
import hashlib
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any
from uuid import UUID, uuid4

from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from context_memory.store import MemoryStore


class ArtifactVersion(BaseModel):
    """A specific version of an artifact.

    Tracks content hash, size, and version-specific metadata.
    Content is stored separately and referenced by hash.

    Attributes:
        version: Version number (1-indexed)
        content_hash: SHA-256 hash of content for deduplication
        created_at: When this version was created
        size_bytes: Size of content in bytes
        metadata: Version-specific metadata (author, commit msg, etc.)
    """

    version: int = Field(ge=1, description="Version number")
    content_hash: str = Field(description="SHA-256 hash of content")
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        description="Creation timestamp",
    )
    size_bytes: int = Field(ge=0, description="Content size in bytes")
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Version-specific metadata",
    )


class Artifact(BaseModel):
    """A versioned artifact with complete history.

    Artifacts are identified by UUID and belong to a session.
    Each update creates a new version, preserving complete history.

    Attributes:
        id: Unique artifact identifier
        session_id: Owning session
        artifact_type: Type classification (code, file, data, etc.)
        name: Human-readable name
        current_version: Latest version number
        versions: Complete version history
        created_at: Artifact creation timestamp
        updated_at: Last modification timestamp
    """

    id: UUID = Field(default_factory=uuid4, description="Unique identifier")
    session_id: str = Field(description="Owning session")
    artifact_type: str = Field(description="Type: code, file, data, etc.")
    name: str = Field(description="Human-readable name")
    current_version: int = Field(default=1, ge=1, description="Latest version")
    versions: list[ArtifactVersion] = Field(
        default_factory=list,
        description="Complete version history",
    )
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        description="Creation timestamp",
    )
    updated_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        description="Last update timestamp",
    )


class DiffResult(BaseModel):
    """Result of comparing two artifact versions.

    Attributes:
        diff_type: Type of diff (unified_diff or binary)
        lines: Diff lines for text content
        additions: Number of added lines
        deletions: Number of deleted lines
        size_change: Size difference for binary content
    """

    diff_type: str = Field(description="Type: unified_diff or binary")
    lines: list[str] = Field(default_factory=list, description="Diff lines")
    additions: int = Field(default=0, ge=0, description="Lines added")
    deletions: int = Field(default=0, ge=0, description="Lines deleted")
    size_change: int = Field(default=0, description="Size change in bytes")


class ArtifactManager:
    """Manage versioned artifacts with content addressing.

    Provides CRUD operations for artifacts with automatic versioning
    and content deduplication via SHA-256 hashing.

    Features:
        - Content-addressed storage (identical content stored once)
        - Complete version history preservation
        - Unified diff generation between versions
        - Session-scoped artifact listing

    Example:
        >>> manager = ArtifactManager()
        >>> artifact = await manager.create_artifact(
        ...     session_id="sess-123",
        ...     name="config.json",
        ...     content='{"key": "value"}',
        ...     artifact_type="file",
        ... )
        >>> # Update creates new version
        >>> version = await manager.update_artifact(
        ...     artifact.id,
        ...     content='{"key": "new_value"}',
        ... )
        >>> # Get diff between versions
        >>> diff = await manager.diff_versions(artifact.id, 1, 2)
    """

    def __init__(self, store: MemoryStore | None = None) -> None:
        """Initialize ArtifactManager.

        Args:
            store: Optional backing store for persistence.
                If None, artifacts are stored in memory only.
        """
        self._store = store
        self._artifacts: dict[UUID, Artifact] = {}
        self._content_cache: dict[str, bytes] = {}
        self._name_index: dict[tuple[str, str], UUID] = {}  # (session_id, name) -> id

    @staticmethod
    def _content_hash(content: bytes) -> str:
        """Generate SHA-256 hash for content addressing."""
        return hashlib.sha256(content).hexdigest()

    async def create_artifact(
        self,
        session_id: str,
        name: str,
        content: bytes | str,
        artifact_type: str = "data",
        metadata: dict[str, Any] | None = None,
    ) -> Artifact:
        """Create a new artifact.

        Args:
            session_id: Session to create artifact in
            name: Human-readable artifact name
            content: Artifact content (string or bytes)
            artifact_type: Type classification (code, file, data, etc.)
            metadata: Optional version metadata

        Returns:
            Created Artifact with initial version

        Raises:
            ValueError: If artifact with same name exists in session
        """
        # Check for duplicate name
        name_key = (session_id, name)
        if name_key in self._name_index:
            raise ValueError(
                f"Artifact '{name}' already exists in session '{session_id}'"
            )

        # Normalize content to bytes
        if isinstance(content, str):
            content = content.encode("utf-8")

        content_hash = self._content_hash(content)

        version = ArtifactVersion(
            version=1,
            content_hash=content_hash,
            size_bytes=len(content),
            metadata=metadata or {},
        )

        artifact = Artifact(
            session_id=session_id,
            artifact_type=artifact_type,
            name=name,
            versions=[version],
        )

        # Store content and artifact
        self._content_cache[content_hash] = content
        self._artifacts[artifact.id] = artifact
        self._name_index[name_key] = artifact.id

        return artifact

    async def update_artifact(
        self,
        artifact_id: UUID,
        content: bytes | str,
        metadata: dict[str, Any] | None = None,
    ) -> ArtifactVersion | None:
        """Add a new version to an artifact.

        If content is identical to current version, returns existing
        version without creating a duplicate.

        Args:
            artifact_id: Artifact to update
            content: New content
            metadata: Optional version metadata

        Returns:
            New ArtifactVersion, or existing if content unchanged.
            None if artifact not found.
        """
        artifact = self._artifacts.get(artifact_id)
        if artifact is None:
            return None

        # Normalize content
        if isinstance(content, str):
            content = content.encode("utf-8")

        content_hash = self._content_hash(content)

        # Skip if content unchanged
        if artifact.versions and artifact.versions[-1].content_hash == content_hash:
            return artifact.versions[-1]

        new_version = ArtifactVersion(
            version=artifact.current_version + 1,
            content_hash=content_hash,
            size_bytes=len(content),
            metadata=metadata or {},
        )

        artifact.versions.append(new_version)
        artifact.current_version = new_version.version
        artifact.updated_at = datetime.now(UTC)

        # Store content (may dedupe with existing)
        self._content_cache[content_hash] = content

        return new_version

    async def get_artifact(
        self,
        artifact_id: UUID,
        version: int | None = None,
    ) -> tuple[Artifact, bytes] | None:
        """Get artifact and its content.

        Args:
            artifact_id: Artifact to retrieve
            version: Specific version (None for latest)

        Returns:
            Tuple of (Artifact, content bytes), or None if not found
        """
        artifact = self._artifacts.get(artifact_id)
        if artifact is None:
            return None

        if not artifact.versions:
            return None

        # Find target version
        if version is not None:
            target = next(
                (v for v in artifact.versions if v.version == version),
                None,
            )
        else:
            target = artifact.versions[-1]

        if target is None:
            return None

        content = self._content_cache.get(target.content_hash)
        if content is None:
            return None

        return artifact, content

    async def get_artifact_by_name(
        self,
        session_id: str,
        name: str,
        version: int | None = None,
    ) -> tuple[Artifact, bytes] | None:
        """Get artifact by name within a session.

        Args:
            session_id: Session to search
            name: Artifact name
            version: Specific version (None for latest)

        Returns:
            Tuple of (Artifact, content bytes), or None if not found
        """
        artifact_id = self._name_index.get((session_id, name))
        if artifact_id is None:
            return None
        return await self.get_artifact(artifact_id, version)

    async def diff_versions(
        self,
        artifact_id: UUID,
        version1: int,
        version2: int,
    ) -> DiffResult | None:
        """Generate diff between two versions.

        For text content, generates unified diff. For binary content,
        reports size change only.

        Args:
            artifact_id: Artifact to diff
            version1: First version number
            version2: Second version number

        Returns:
            DiffResult with diff details, or None if versions not found
        """
        artifact = self._artifacts.get(artifact_id)
        if artifact is None:
            return None

        v1 = next((v for v in artifact.versions if v.version == version1), None)
        v2 = next((v for v in artifact.versions if v.version == version2), None)

        if v1 is None or v2 is None:
            return None

        content1 = self._content_cache.get(v1.content_hash, b"")
        content2 = self._content_cache.get(v2.content_hash, b"")

        try:
            # Attempt text diff
            text1 = content1.decode("utf-8")
            text2 = content2.decode("utf-8")

            diff_lines = list(
                difflib.unified_diff(
                    text1.splitlines(keepends=True),
                    text2.splitlines(keepends=True),
                    fromfile=f"v{version1}",
                    tofile=f"v{version2}",
                )
            )

            additions = sum(1 for line in diff_lines if line.startswith("+"))
            deletions = sum(1 for line in diff_lines if line.startswith("-"))

            return DiffResult(
                diff_type="unified_diff",
                lines=diff_lines,
                additions=additions,
                deletions=deletions,
                size_change=len(content2) - len(content1),
            )

        except UnicodeDecodeError:
            # Binary content
            return DiffResult(
                diff_type="binary",
                size_change=len(content2) - len(content1),
            )

    async def list_artifacts(
        self,
        session_id: str,
        artifact_type: str | None = None,
    ) -> list[Artifact]:
        """List artifacts in a session.

        Args:
            session_id: Session to list
            artifact_type: Optional type filter

        Returns:
            List of matching artifacts
        """
        results: list[Artifact] = []
        for artifact in self._artifacts.values():
            if artifact.session_id != session_id:
                continue
            if artifact_type is not None and artifact.artifact_type != artifact_type:
                continue
            results.append(artifact)
        return results

    async def delete_artifact(self, artifact_id: UUID) -> bool:
        """Delete an artifact and all versions.

        Cleans up content that is no longer referenced by other artifacts.

        Args:
            artifact_id: Artifact to delete

        Returns:
            True if deleted, False if not found
        """
        artifact = self._artifacts.pop(artifact_id, None)
        if artifact is None:
            return False

        # Remove from name index
        name_key = (artifact.session_id, artifact.name)
        self._name_index.pop(name_key, None)

        # Clean up unreferenced content
        for version in artifact.versions:
            is_referenced = any(
                v.content_hash == version.content_hash
                for a in self._artifacts.values()
                for v in a.versions
            )
            if not is_referenced:
                self._content_cache.pop(version.content_hash, None)

        return True

    async def get_version_history(self, artifact_id: UUID) -> list[ArtifactVersion]:
        """Get complete version history for an artifact.

        Args:
            artifact_id: Artifact to query

        Returns:
            List of versions (oldest first), or empty if not found
        """
        artifact = self._artifacts.get(artifact_id)
        if artifact is None:
            return []
        return list(artifact.versions)

    def stats(self) -> dict[str, Any]:
        """Get manager statistics.

        Returns:
            Dict with artifact count, total versions, content size, etc.
        """
        total_versions = sum(len(a.versions) for a in self._artifacts.values())
        total_content_bytes = sum(len(c) for c in self._content_cache.values())
        unique_hashes = len(self._content_cache)

        return {
            "artifact_count": len(self._artifacts),
            "total_versions": total_versions,
            "unique_content_hashes": unique_hashes,
            "total_content_bytes": total_content_bytes,
            "deduplication_ratio": (
                total_versions / unique_hashes if unique_hashes > 0 else 1.0
            ),
        }
