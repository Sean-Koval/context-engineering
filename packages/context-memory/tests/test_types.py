"""Unit tests for context_memory.types module."""

from __future__ import annotations

from datetime import UTC, datetime
from uuid import UUID, uuid4

import pytest

from context_memory.types import (
    RetrievalResult,
    StorageKey,
    StorageMetadata,
    StorageStats,
    StorageTier,
)


class TestStorageTier:
    """Tests for StorageTier enum."""

    def test_tier_values(self) -> None:
        """Test that all tier values are correct."""
        assert StorageTier.HOT.value == "hot"
        assert StorageTier.WARM.value == "warm"
        assert StorageTier.COLD.value == "cold"

    def test_tier_is_str_enum(self) -> None:
        """Test that StorageTier is a string enum."""
        assert isinstance(StorageTier.HOT, str)
        assert StorageTier.HOT == "hot"

    def test_tier_from_string(self) -> None:
        """Test creating tier from string value."""
        assert StorageTier("hot") == StorageTier.HOT
        assert StorageTier("warm") == StorageTier.WARM
        assert StorageTier("cold") == StorageTier.COLD

    def test_invalid_tier_raises(self) -> None:
        """Test that invalid tier value raises ValueError."""
        with pytest.raises(ValueError):
            StorageTier("invalid")


class TestStorageKey:
    """Tests for StorageKey model."""

    def test_create_storage_key(self) -> None:
        """Test basic StorageKey creation."""
        node_id = uuid4()
        key = StorageKey(session_id="sess-123", node_id=node_id, version=1)

        assert key.session_id == "sess-123"
        assert key.node_id == node_id
        assert key.version == 1

    def test_default_version(self) -> None:
        """Test that version defaults to 1."""
        key = StorageKey(session_id="sess-123", node_id=uuid4())
        assert key.version == 1

    def test_str_representation(self) -> None:
        """Test string representation."""
        node_id = UUID("550e8400-e29b-41d4-a716-446655440000")
        key = StorageKey(session_id="sess-123", node_id=node_id, version=2)

        assert str(key) == "sess-123/550e8400-e29b-41d4-a716-446655440000/2"

    def test_from_string_full(self) -> None:
        """Test parsing from full string representation."""
        key = StorageKey.from_string("sess-123/550e8400-e29b-41d4-a716-446655440000/2")

        assert key.session_id == "sess-123"
        assert key.node_id == UUID("550e8400-e29b-41d4-a716-446655440000")
        assert key.version == 2

    def test_from_string_without_version(self) -> None:
        """Test parsing from string without version defaults to 1."""
        key = StorageKey.from_string("sess-123/550e8400-e29b-41d4-a716-446655440000")

        assert key.session_id == "sess-123"
        assert key.version == 1

    def test_from_string_invalid_format(self) -> None:
        """Test that invalid format raises ValueError."""
        with pytest.raises(ValueError, match="Invalid StorageKey format"):
            StorageKey.from_string("invalid")

    def test_from_string_invalid_uuid(self) -> None:
        """Test that invalid UUID raises ValueError."""
        with pytest.raises(ValueError):
            StorageKey.from_string("sess-123/not-a-uuid/1")

    def test_roundtrip(self) -> None:
        """Test string serialization roundtrip."""
        original = StorageKey(session_id="test-session", node_id=uuid4(), version=5)
        parsed = StorageKey.from_string(str(original))

        assert parsed == original

    def test_hashable(self) -> None:
        """Test that StorageKey is hashable."""
        node_id = uuid4()
        key1 = StorageKey(session_id="sess", node_id=node_id, version=1)
        key2 = StorageKey(session_id="sess", node_id=node_id, version=1)

        # Can be used in sets
        key_set = {key1, key2}
        assert len(key_set) == 1

        # Can be used as dict keys
        key_dict = {key1: "value"}
        assert key_dict[key2] == "value"

    def test_version_must_be_positive(self) -> None:
        """Test that version must be >= 1."""
        with pytest.raises(ValueError):
            StorageKey(session_id="sess", node_id=uuid4(), version=0)


class TestStorageMetadata:
    """Tests for StorageMetadata model."""

    @pytest.fixture
    def sample_key(self) -> StorageKey:
        """Create a sample StorageKey for testing."""
        return StorageKey(session_id="sess-123", node_id=uuid4(), version=1)

    def test_create_metadata(self, sample_key: StorageKey) -> None:
        """Test basic metadata creation."""
        metadata = StorageMetadata(
            key=sample_key,
            size_bytes=1024,
            token_count=100,
            node_type="MESSAGE",
        )

        assert metadata.key == sample_key
        assert metadata.size_bytes == 1024
        assert metadata.token_count == 100
        assert metadata.node_type == "MESSAGE"

    def test_default_values(self, sample_key: StorageKey) -> None:
        """Test default values are set correctly."""
        metadata = StorageMetadata(
            key=sample_key,
            size_bytes=100,
            token_count=10,
            node_type="MESSAGE",
        )

        assert metadata.tier == StorageTier.HOT
        assert metadata.importance == 0.5
        assert metadata.access_count == 0
        assert metadata.is_compressed is False
        assert metadata.original_size_bytes is None
        assert isinstance(metadata.tags, set)
        assert len(metadata.tags) == 0

    def test_touch_updates_access(self, sample_key: StorageKey) -> None:
        """Test that touch() updates access tracking."""
        metadata = StorageMetadata(
            key=sample_key,
            size_bytes=100,
            token_count=10,
            node_type="MESSAGE",
        )

        original_accessed_at = metadata.accessed_at
        original_count = metadata.access_count

        # Small delay to ensure timestamp changes
        metadata.touch()

        assert metadata.access_count == original_count + 1
        assert metadata.accessed_at >= original_accessed_at

    def test_touch_increments_count(self, sample_key: StorageKey) -> None:
        """Test that multiple touches increment count correctly."""
        metadata = StorageMetadata(
            key=sample_key,
            size_bytes=100,
            token_count=10,
            node_type="MESSAGE",
        )

        for _ in range(5):
            metadata.touch()

        assert metadata.access_count == 5

    def test_importance_bounds(self, sample_key: StorageKey) -> None:
        """Test that importance is bounded [0, 1]."""
        # Valid values
        StorageMetadata(
            key=sample_key,
            size_bytes=100,
            token_count=10,
            node_type="MESSAGE",
            importance=0.0,
        )
        StorageMetadata(
            key=sample_key,
            size_bytes=100,
            token_count=10,
            node_type="MESSAGE",
            importance=1.0,
        )

        # Invalid values
        with pytest.raises(ValueError):
            StorageMetadata(
                key=sample_key,
                size_bytes=100,
                token_count=10,
                node_type="MESSAGE",
                importance=-0.1,
            )

        with pytest.raises(ValueError):
            StorageMetadata(
                key=sample_key,
                size_bytes=100,
                token_count=10,
                node_type="MESSAGE",
                importance=1.1,
            )

    def test_tags_as_set(self, sample_key: StorageKey) -> None:
        """Test that tags is a set."""
        metadata = StorageMetadata(
            key=sample_key,
            size_bytes=100,
            token_count=10,
            node_type="MESSAGE",
            tags={"tag1", "tag2"},
        )

        assert isinstance(metadata.tags, set)
        assert "tag1" in metadata.tags
        assert "tag2" in metadata.tags

    def test_tags_from_list(self, sample_key: StorageKey) -> None:
        """Test that tags can be provided as list (for JSON compat)."""
        metadata = StorageMetadata(
            key=sample_key,
            size_bytes=100,
            token_count=10,
            node_type="MESSAGE",
            tags=["tag1", "tag2", "tag1"],  # type: ignore[arg-type]
        )

        assert isinstance(metadata.tags, set)
        assert len(metadata.tags) == 2  # Duplicates removed

    def test_json_serialization(self, sample_key: StorageKey) -> None:
        """Test JSON serialization roundtrip."""
        metadata = StorageMetadata(
            key=sample_key,
            tier=StorageTier.WARM,
            size_bytes=1024,
            token_count=100,
            node_type="TOOL_CALL",
            importance=0.8,
            tags={"important", "tool"},
        )

        json_data = metadata.model_dump(mode="json")

        # Verify JSON structure
        assert json_data["tier"] == "warm"
        assert json_data["size_bytes"] == 1024
        assert json_data["node_type"] == "TOOL_CALL"
        assert set(json_data["tags"]) == {"important", "tool"}

        # Roundtrip
        restored = StorageMetadata.model_validate(json_data)
        assert restored.tier == metadata.tier
        assert restored.size_bytes == metadata.size_bytes
        assert restored.tags == metadata.tags


class TestStorageStats:
    """Tests for StorageStats model."""

    def test_create_stats(self) -> None:
        """Test basic stats creation."""
        stats = StorageStats(
            total_items=100,
            total_size_bytes=1024000,
            total_tokens=50000,
        )

        assert stats.total_items == 100
        assert stats.total_size_bytes == 1024000
        assert stats.total_tokens == 50000

    def test_default_values(self) -> None:
        """Test default values."""
        stats = StorageStats(
            total_items=0,
            total_size_bytes=0,
            total_tokens=0,
        )

        assert stats.items_by_tier == {}
        assert stats.size_by_tier == {}
        assert stats.avg_access_count == 0.0
        assert stats.oldest_item is None
        assert stats.newest_item is None

    def test_tier_breakdown(self) -> None:
        """Test tier breakdown fields."""
        stats = StorageStats(
            total_items=100,
            total_size_bytes=1024000,
            total_tokens=50000,
            items_by_tier={"hot": 20, "warm": 50, "cold": 30},
            size_by_tier={"hot": 204800, "warm": 512000, "cold": 307200},
        )

        assert stats.items_by_tier["hot"] == 20
        assert stats.size_by_tier["warm"] == 512000

    def test_temporal_bounds(self) -> None:
        """Test temporal bound fields."""
        now = datetime.now(UTC)
        stats = StorageStats(
            total_items=10,
            total_size_bytes=1000,
            total_tokens=100,
            oldest_item=now,
            newest_item=now,
        )

        assert stats.oldest_item == now
        assert stats.newest_item == now

    def test_json_serialization(self) -> None:
        """Test JSON serialization."""
        stats = StorageStats(
            total_items=100,
            total_size_bytes=1024000,
            total_tokens=50000,
            items_by_tier={"hot": 20, "warm": 80},
            avg_access_count=5.5,
        )

        json_data = stats.model_dump(mode="json")

        assert json_data["total_items"] == 100
        assert json_data["items_by_tier"]["hot"] == 20

        # Roundtrip
        restored = StorageStats.model_validate(json_data)
        assert restored.total_items == stats.total_items


class TestRetrievalResult:
    """Tests for RetrievalResult model."""

    def test_create_result(self) -> None:
        """Test basic result creation."""
        # Using a dict as mock node since we don't want to import ContextNode
        mock_node = {"id": str(uuid4()), "type": "MESSAGE"}

        result = RetrievalResult(
            node=mock_node,
            score=0.85,
            source_tier=StorageTier.WARM,
            retrieval_method="semantic",
            latency_ms=15.5,
        )

        assert result.node == mock_node
        assert result.score == 0.85
        assert result.source_tier == StorageTier.WARM
        assert result.retrieval_method == "semantic"
        assert result.latency_ms == 15.5

    def test_score_bounds(self) -> None:
        """Test that score is bounded [0, 1]."""
        mock_node = {"id": "test"}

        # Valid scores
        RetrievalResult(
            node=mock_node,
            score=0.0,
            source_tier=StorageTier.HOT,
            retrieval_method="test",
            latency_ms=1.0,
        )
        RetrievalResult(
            node=mock_node,
            score=1.0,
            source_tier=StorageTier.HOT,
            retrieval_method="test",
            latency_ms=1.0,
        )

        # Invalid scores
        with pytest.raises(ValueError):
            RetrievalResult(
                node=mock_node,
                score=-0.1,
                source_tier=StorageTier.HOT,
                retrieval_method="test",
                latency_ms=1.0,
            )

        with pytest.raises(ValueError):
            RetrievalResult(
                node=mock_node,
                score=1.1,
                source_tier=StorageTier.HOT,
                retrieval_method="test",
                latency_ms=1.0,
            )

    def test_latency_non_negative(self) -> None:
        """Test that latency must be non-negative."""
        mock_node = {"id": "test"}

        with pytest.raises(ValueError):
            RetrievalResult(
                node=mock_node,
                score=0.5,
                source_tier=StorageTier.HOT,
                retrieval_method="test",
                latency_ms=-1.0,
            )

    def test_retrieval_methods(self) -> None:
        """Test various retrieval method values."""
        mock_node = {"id": "test"}

        for method in ["semantic", "entity", "temporal", "ensemble", "exact"]:
            result = RetrievalResult(
                node=mock_node,
                score=0.5,
                source_tier=StorageTier.WARM,
                retrieval_method=method,
                latency_ms=10.0,
            )
            assert result.retrieval_method == method
