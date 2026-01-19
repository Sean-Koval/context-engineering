"""Unit tests for ArtifactManager."""

from __future__ import annotations

from uuid import uuid4

import pytest

from context_memory.artifacts import (
    Artifact,
    ArtifactManager,
    ArtifactVersion,
    DiffResult,
)

# =============================================================================
# ArtifactVersion Tests
# =============================================================================


class TestArtifactVersion:
    """Tests for ArtifactVersion model."""

    def test_create_version(self) -> None:
        """Test creating a version."""
        version = ArtifactVersion(
            version=1,
            content_hash="abc123",
            size_bytes=100,
        )
        assert version.version == 1
        assert version.content_hash == "abc123"
        assert version.size_bytes == 100
        assert version.metadata == {}

    def test_version_with_metadata(self) -> None:
        """Test version with custom metadata."""
        version = ArtifactVersion(
            version=2,
            content_hash="def456",
            size_bytes=200,
            metadata={"author": "test", "message": "Updated config"},
        )
        assert version.metadata["author"] == "test"

    def test_version_validation(self) -> None:
        """Test version number validation."""
        with pytest.raises(ValueError):
            ArtifactVersion(
                version=0,  # Must be >= 1
                content_hash="abc",
                size_bytes=10,
            )


# =============================================================================
# Artifact Tests
# =============================================================================


class TestArtifact:
    """Tests for Artifact model."""

    def test_create_artifact(self) -> None:
        """Test creating an artifact."""
        artifact = Artifact(
            session_id="test-session",
            artifact_type="code",
            name="main.py",
        )
        assert artifact.session_id == "test-session"
        assert artifact.artifact_type == "code"
        assert artifact.name == "main.py"
        assert artifact.current_version == 1
        assert artifact.versions == []

    def test_artifact_has_uuid(self) -> None:
        """Test artifact gets auto-generated UUID."""
        artifact = Artifact(
            session_id="test",
            artifact_type="data",
            name="test.json",
        )
        assert artifact.id is not None


# =============================================================================
# ArtifactManager Tests
# =============================================================================


class TestArtifactManager:
    """Tests for ArtifactManager."""

    @pytest.mark.asyncio
    async def test_create_artifact(self) -> None:
        """Test creating an artifact."""
        manager = ArtifactManager()
        artifact = await manager.create_artifact(
            session_id="test-session",
            name="config.json",
            content='{"key": "value"}',
            artifact_type="file",
        )

        assert artifact.name == "config.json"
        assert artifact.session_id == "test-session"
        assert artifact.artifact_type == "file"
        assert artifact.current_version == 1
        assert len(artifact.versions) == 1

    @pytest.mark.asyncio
    async def test_create_artifact_bytes(self) -> None:
        """Test creating artifact with bytes content."""
        manager = ArtifactManager()
        artifact = await manager.create_artifact(
            session_id="test",
            name="data.bin",
            content=b"\x00\x01\x02\x03",
            artifact_type="binary",
        )

        assert artifact.versions[0].size_bytes == 4

    @pytest.mark.asyncio
    async def test_create_duplicate_name_raises(self) -> None:
        """Test creating duplicate name raises error."""
        manager = ArtifactManager()
        await manager.create_artifact(
            session_id="test",
            name="config.json",
            content="{}",
        )

        with pytest.raises(ValueError, match="already exists"):
            await manager.create_artifact(
                session_id="test",
                name="config.json",
                content="{}",
            )

    @pytest.mark.asyncio
    async def test_same_name_different_session(self) -> None:
        """Test same name allowed in different sessions."""
        manager = ArtifactManager()
        a1 = await manager.create_artifact(
            session_id="session-1",
            name="config.json",
            content="{}",
        )
        a2 = await manager.create_artifact(
            session_id="session-2",
            name="config.json",
            content="{}",
        )

        assert a1.id != a2.id

    @pytest.mark.asyncio
    async def test_update_artifact(self) -> None:
        """Test updating creates new version."""
        manager = ArtifactManager()
        artifact = await manager.create_artifact(
            session_id="test",
            name="file.txt",
            content="version 1",
        )

        version = await manager.update_artifact(
            artifact.id,
            content="version 2",
        )

        assert version is not None
        assert version.version == 2
        assert artifact.current_version == 2
        assert len(artifact.versions) == 2

    @pytest.mark.asyncio
    async def test_update_same_content_no_new_version(self) -> None:
        """Test update with same content returns existing version."""
        manager = ArtifactManager()
        artifact = await manager.create_artifact(
            session_id="test",
            name="file.txt",
            content="same content",
        )

        version = await manager.update_artifact(
            artifact.id,
            content="same content",
        )

        assert version is not None
        assert version.version == 1
        assert len(artifact.versions) == 1

    @pytest.mark.asyncio
    async def test_update_nonexistent(self) -> None:
        """Test updating nonexistent artifact returns None."""
        manager = ArtifactManager()
        result = await manager.update_artifact(uuid4(), content="test")
        assert result is None

    @pytest.mark.asyncio
    async def test_get_artifact(self) -> None:
        """Test retrieving artifact and content."""
        manager = ArtifactManager()
        artifact = await manager.create_artifact(
            session_id="test",
            name="file.txt",
            content="hello world",
        )

        result = await manager.get_artifact(artifact.id)
        assert result is not None
        retrieved_artifact, content = result
        assert retrieved_artifact.id == artifact.id
        assert content == b"hello world"

    @pytest.mark.asyncio
    async def test_get_artifact_specific_version(self) -> None:
        """Test retrieving specific version."""
        manager = ArtifactManager()
        artifact = await manager.create_artifact(
            session_id="test",
            name="file.txt",
            content="version 1",
        )
        await manager.update_artifact(artifact.id, content="version 2")
        await manager.update_artifact(artifact.id, content="version 3")

        result = await manager.get_artifact(artifact.id, version=2)
        assert result is not None
        _, content = result
        assert content == b"version 2"

    @pytest.mark.asyncio
    async def test_get_artifact_nonexistent(self) -> None:
        """Test getting nonexistent artifact returns None."""
        manager = ArtifactManager()
        result = await manager.get_artifact(uuid4())
        assert result is None

    @pytest.mark.asyncio
    async def test_get_artifact_by_name(self) -> None:
        """Test retrieving by name."""
        manager = ArtifactManager()
        await manager.create_artifact(
            session_id="test",
            name="config.json",
            content='{"key": "value"}',
        )

        result = await manager.get_artifact_by_name("test", "config.json")
        assert result is not None
        artifact, content = result
        assert artifact.name == "config.json"

    @pytest.mark.asyncio
    async def test_get_artifact_by_name_not_found(self) -> None:
        """Test get by name returns None if not found."""
        manager = ArtifactManager()
        result = await manager.get_artifact_by_name("test", "nonexistent.txt")
        assert result is None

    @pytest.mark.asyncio
    async def test_diff_versions_text(self) -> None:
        """Test diffing text content."""
        manager = ArtifactManager()
        artifact = await manager.create_artifact(
            session_id="test",
            name="file.txt",
            content="line 1\nline 2\n",
        )
        await manager.update_artifact(
            artifact.id,
            content="line 1\nline 2 modified\nline 3\n",
        )

        diff = await manager.diff_versions(artifact.id, 1, 2)
        assert diff is not None
        assert diff.diff_type == "unified_diff"
        assert diff.additions > 0
        assert diff.deletions > 0
        assert len(diff.lines) > 0

    @pytest.mark.asyncio
    async def test_diff_versions_binary(self) -> None:
        """Test diffing binary content."""
        manager = ArtifactManager()
        # Use invalid UTF-8 sequences to ensure binary handling
        artifact = await manager.create_artifact(
            session_id="test",
            name="data.bin",
            content=b"\x80\x81\x82",  # Invalid UTF-8
        )
        await manager.update_artifact(
            artifact.id,
            content=b"\x80\x81\x82\x83\x84",  # Invalid UTF-8
        )

        diff = await manager.diff_versions(artifact.id, 1, 2)
        assert diff is not None
        assert diff.diff_type == "binary"
        assert diff.size_change == 2

    @pytest.mark.asyncio
    async def test_diff_nonexistent_version(self) -> None:
        """Test diff with nonexistent version returns None."""
        manager = ArtifactManager()
        artifact = await manager.create_artifact(
            session_id="test",
            name="file.txt",
            content="test",
        )

        diff = await manager.diff_versions(artifact.id, 1, 99)
        assert diff is None

    @pytest.mark.asyncio
    async def test_list_artifacts(self) -> None:
        """Test listing artifacts."""
        manager = ArtifactManager()
        await manager.create_artifact(
            session_id="test",
            name="file1.txt",
            content="content 1",
            artifact_type="file",
        )
        await manager.create_artifact(
            session_id="test",
            name="file2.txt",
            content="content 2",
            artifact_type="code",
        )
        await manager.create_artifact(
            session_id="other",
            name="file3.txt",
            content="content 3",
        )

        # List all in session
        artifacts = await manager.list_artifacts("test")
        assert len(artifacts) == 2

        # List by type
        code_artifacts = await manager.list_artifacts("test", artifact_type="code")
        assert len(code_artifacts) == 1
        assert code_artifacts[0].artifact_type == "code"

    @pytest.mark.asyncio
    async def test_delete_artifact(self) -> None:
        """Test deleting artifact."""
        manager = ArtifactManager()
        artifact = await manager.create_artifact(
            session_id="test",
            name="file.txt",
            content="test content",
        )

        result = await manager.delete_artifact(artifact.id)
        assert result is True

        # Should not be retrievable
        get_result = await manager.get_artifact(artifact.id)
        assert get_result is None

        # Name should be available again
        new_artifact = await manager.create_artifact(
            session_id="test",
            name="file.txt",
            content="new content",
        )
        assert new_artifact is not None

    @pytest.mark.asyncio
    async def test_delete_nonexistent(self) -> None:
        """Test deleting nonexistent artifact returns False."""
        manager = ArtifactManager()
        result = await manager.delete_artifact(uuid4())
        assert result is False

    @pytest.mark.asyncio
    async def test_content_deduplication(self) -> None:
        """Test identical content is deduplicated."""
        manager = ArtifactManager()
        content = "identical content"

        await manager.create_artifact(
            session_id="test",
            name="file1.txt",
            content=content,
        )
        await manager.create_artifact(
            session_id="test",
            name="file2.txt",
            content=content,
        )

        stats = manager.stats()
        assert stats["artifact_count"] == 2
        assert stats["unique_content_hashes"] == 1

    @pytest.mark.asyncio
    async def test_get_version_history(self) -> None:
        """Test getting version history."""
        manager = ArtifactManager()
        artifact = await manager.create_artifact(
            session_id="test",
            name="file.txt",
            content="v1",
        )
        await manager.update_artifact(artifact.id, content="v2")
        await manager.update_artifact(artifact.id, content="v3")

        history = await manager.get_version_history(artifact.id)
        assert len(history) == 3
        assert history[0].version == 1
        assert history[1].version == 2
        assert history[2].version == 3

    @pytest.mark.asyncio
    async def test_stats(self) -> None:
        """Test manager statistics."""
        manager = ArtifactManager()
        await manager.create_artifact(
            session_id="test",
            name="file1.txt",
            content="content",
        )

        stats = manager.stats()
        assert stats["artifact_count"] == 1
        assert stats["total_versions"] == 1
        assert stats["total_content_bytes"] > 0


# =============================================================================
# DiffResult Tests
# =============================================================================


class TestDiffResult:
    """Tests for DiffResult model."""

    def test_default_values(self) -> None:
        """Test default diff result values."""
        diff = DiffResult(diff_type="unified_diff")
        assert diff.lines == []
        assert diff.additions == 0
        assert diff.deletions == 0
        assert diff.size_change == 0

    def test_unified_diff_values(self) -> None:
        """Test unified diff result."""
        diff = DiffResult(
            diff_type="unified_diff",
            lines=["- old line", "+ new line"],
            additions=1,
            deletions=1,
        )
        assert len(diff.lines) == 2
        assert diff.additions == 1
