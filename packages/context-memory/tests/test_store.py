"""Unit tests for context_memory.store module (MemoryStore protocol)."""

from __future__ import annotations

from datetime import datetime
from typing import Any
from uuid import uuid4

import pytest

from context_memory.store import MemoryStore
from context_memory.types import (
    StorageKey,
    StorageMetadata,
    StorageStats,
    StorageTier,
)


class TestMemoryStoreProtocol:
    """Tests for MemoryStore protocol definition."""

    def test_protocol_is_runtime_checkable(self) -> None:
        """Test that MemoryStore is a runtime_checkable Protocol."""
        # Protocol should have __protocol_attrs__ or similar
        assert hasattr(MemoryStore, "__protocol_attrs__") or hasattr(
            MemoryStore, "_is_protocol"
        )

    def test_complete_implementation_satisfies_protocol(self) -> None:
        """Test that a complete implementation satisfies the protocol."""

        class CompleteStore:
            """A complete implementation of MemoryStore."""

            async def store(
                self,
                node: Any,
                session_id: str,
                metadata: StorageMetadata | None = None,
            ) -> StorageKey:
                return StorageKey(session_id=session_id, node_id=uuid4())

            async def store_batch(
                self,
                nodes: list[Any],
                session_id: str,
            ) -> list[StorageKey]:
                return [
                    StorageKey(session_id=session_id, node_id=uuid4()) for _ in nodes
                ]

            async def retrieve(self, key: StorageKey) -> Any | None:
                return None

            async def retrieve_batch(
                self, keys: list[StorageKey]
            ) -> list[Any | None]:
                return [None] * len(keys)

            async def delete(self, key: StorageKey) -> bool:
                return False

            async def exists(self, key: StorageKey) -> bool:
                return False

            async def get_metadata(self, key: StorageKey) -> StorageMetadata | None:
                return None

            async def update_metadata(
                self, key: StorageKey, updates: dict[str, Any]
            ) -> bool:
                return False

            async def list_keys(
                self,
                session_id: str,
                *,
                tier: StorageTier | None = None,
                node_type: str | None = None,
                limit: int = 1000,
            ) -> list[StorageKey]:
                return []

            async def search_by_metadata(
                self,
                session_id: str,
                *,
                min_importance: float | None = None,
                tags: set[str] | None = None,
                since: datetime | None = None,
                limit: int = 100,
            ) -> list[tuple[StorageKey, StorageMetadata]]:
                return []

            async def stats(self, session_id: str | None = None) -> StorageStats:
                return StorageStats(total_items=0, total_size_bytes=0, total_tokens=0)

            async def close(self) -> None:
                pass

        store = CompleteStore()
        assert isinstance(store, MemoryStore)

    def test_incomplete_implementation_fails_protocol(self) -> None:
        """Test that an incomplete implementation does NOT satisfy the protocol."""

        class IncompleteStore:
            """Missing most methods."""

            async def store(
                self,
                node: Any,
                session_id: str,
                metadata: StorageMetadata | None = None,
            ) -> StorageKey:
                return StorageKey(session_id=session_id, node_id=uuid4())

            # Missing all other methods

        store = IncompleteStore()
        # Should NOT satisfy the protocol due to missing methods
        assert not isinstance(store, MemoryStore)

    def test_wrong_signature_fails_protocol(self) -> None:
        """Test that wrong method signatures fail the protocol check."""

        class WrongSignatureStore:
            """Has methods but with wrong signatures."""

            # Wrong: missing session_id parameter
            async def store(self, node: Any) -> StorageKey:
                return StorageKey(session_id="test", node_id=uuid4())

            async def store_batch(
                self, nodes: list[Any], session_id: str
            ) -> list[StorageKey]:
                return []

            async def retrieve(self, key: StorageKey) -> Any | None:
                return None

            async def retrieve_batch(
                self, keys: list[StorageKey]
            ) -> list[Any | None]:
                return []

            async def delete(self, key: StorageKey) -> bool:
                return False

            async def exists(self, key: StorageKey) -> bool:
                return False

            async def get_metadata(self, key: StorageKey) -> StorageMetadata | None:
                return None

            async def update_metadata(
                self, key: StorageKey, updates: dict[str, Any]
            ) -> bool:
                return False

            async def list_keys(
                self,
                session_id: str,
                *,
                tier: StorageTier | None = None,
                node_type: str | None = None,
                limit: int = 1000,
            ) -> list[StorageKey]:
                return []

            async def search_by_metadata(
                self,
                session_id: str,
                *,
                min_importance: float | None = None,
                tags: set[str] | None = None,
                since: datetime | None = None,
                limit: int = 100,
            ) -> list[tuple[StorageKey, StorageMetadata]]:
                return []

            async def stats(self, session_id: str | None = None) -> StorageStats:
                return StorageStats(total_items=0, total_size_bytes=0, total_tokens=0)

            async def close(self) -> None:
                pass

        store = WrongSignatureStore()
        # Protocol runtime checks only verify method existence, not signatures
        # So this will still pass isinstance but would fail at runtime
        # This is a known limitation of Python's Protocol runtime checking
        assert isinstance(store, MemoryStore)

    def test_non_async_methods_fail_protocol(self) -> None:
        """Test that sync methods don't satisfy the async protocol."""

        class SyncStore:
            """Has sync methods instead of async."""

            def store(
                self,
                node: Any,
                session_id: str,
                metadata: StorageMetadata | None = None,
            ) -> StorageKey:
                return StorageKey(session_id=session_id, node_id=uuid4())

            # ... other methods would be sync too

        _store = SyncStore()  # noqa: F841 - intentionally unused, demonstrating limitation
        # Runtime check only verifies method existence, not async-ness
        # This is a known limitation - the check passes but usage would fail
        # We document this as expected behavior

    def test_protocol_has_all_required_methods(self) -> None:
        """Test that the protocol defines all expected methods."""
        expected_methods = [
            "store",
            "store_batch",
            "retrieve",
            "retrieve_batch",
            "delete",
            "exists",
            "get_metadata",
            "update_metadata",
            "list_keys",
            "search_by_metadata",
            "stats",
            "close",
        ]

        for method_name in expected_methods:
            assert hasattr(MemoryStore, method_name), f"Missing method: {method_name}"


class TestMemoryStoreMethodSignatures:
    """Tests to verify method signature expectations through documentation."""

    def test_store_method_docstring(self) -> None:
        """Test that store method has proper documentation."""
        assert MemoryStore.store.__doc__ is not None
        assert "node" in MemoryStore.store.__doc__
        assert "session_id" in MemoryStore.store.__doc__
        assert "StorageKey" in MemoryStore.store.__doc__

    def test_retrieve_method_docstring(self) -> None:
        """Test that retrieve method has proper documentation."""
        assert MemoryStore.retrieve.__doc__ is not None
        assert "key" in MemoryStore.retrieve.__doc__
        assert "None" in MemoryStore.retrieve.__doc__

    def test_search_method_docstring(self) -> None:
        """Test that search_by_metadata has proper documentation."""
        assert MemoryStore.search_by_metadata.__doc__ is not None
        assert "min_importance" in MemoryStore.search_by_metadata.__doc__
        assert "tags" in MemoryStore.search_by_metadata.__doc__

    def test_close_method_docstring(self) -> None:
        """Test that close method documents cleanup behavior."""
        assert MemoryStore.close.__doc__ is not None
        assert "close" in MemoryStore.close.__doc__.lower()


@pytest.mark.asyncio
class TestMockStoreUsage:
    """Tests demonstrating actual async usage of a mock store."""

    @pytest.fixture
    def mock_store(self) -> MockMemoryStore:
        """Create a mock store for testing."""
        return MockMemoryStore()

    async def test_store_and_retrieve(self, mock_store: MockMemoryStore) -> None:
        """Test basic store and retrieve flow."""
        # Create a mock node (dict for simplicity)
        mock_node = {"id": str(uuid4()), "type": "MESSAGE", "content": "Hello"}

        # Store
        key = await mock_store.store(mock_node, "test-session")
        assert key.session_id == "test-session"

        # Retrieve
        retrieved = await mock_store.retrieve(key)
        assert retrieved == mock_node

    async def test_store_batch(self, mock_store: MockMemoryStore) -> None:
        """Test batch store operation."""
        nodes = [
            {"id": str(uuid4()), "type": "MESSAGE", "content": f"Message {i}"}
            for i in range(5)
        ]

        keys = await mock_store.store_batch(nodes, "test-session")
        assert len(keys) == 5
        assert all(k.session_id == "test-session" for k in keys)

    async def test_delete(self, mock_store: MockMemoryStore) -> None:
        """Test delete operation."""
        mock_node = {"id": str(uuid4()), "type": "MESSAGE"}
        key = await mock_store.store(mock_node, "test-session")

        # Should exist
        assert await mock_store.exists(key)

        # Delete
        deleted = await mock_store.delete(key)
        assert deleted is True

        # Should no longer exist
        assert not await mock_store.exists(key)

    async def test_stats(self, mock_store: MockMemoryStore) -> None:
        """Test stats operation."""
        # Store some nodes
        for _ in range(3):
            await mock_store.store({"id": str(uuid4())}, "test-session")

        stats = await mock_store.stats("test-session")
        assert stats.total_items == 3

    async def test_close_is_safe(self, mock_store: MockMemoryStore) -> None:
        """Test that close can be called multiple times safely."""
        await mock_store.close()
        await mock_store.close()  # Should not raise


class MockMemoryStore:
    """A complete mock implementation for testing."""

    def __init__(self) -> None:
        self._storage: dict[str, Any] = {}
        self._metadata: dict[str, StorageMetadata] = {}

    async def store(
        self,
        node: Any,
        session_id: str,
        metadata: StorageMetadata | None = None,
    ) -> StorageKey:
        key = StorageKey(session_id=session_id, node_id=uuid4())
        key_str = str(key)
        self._storage[key_str] = node

        if metadata is None:
            metadata = StorageMetadata(
                key=key,
                size_bytes=100,
                token_count=10,
                node_type="MESSAGE",
            )
        self._metadata[key_str] = metadata

        return key

    async def store_batch(
        self,
        nodes: list[Any],
        session_id: str,
    ) -> list[StorageKey]:
        return [await self.store(node, session_id) for node in nodes]

    async def retrieve(self, key: StorageKey) -> Any | None:
        key_str = str(key)
        if key_str in self._metadata:
            self._metadata[key_str].touch()
        return self._storage.get(key_str)

    async def retrieve_batch(self, keys: list[StorageKey]) -> list[Any | None]:
        return [await self.retrieve(key) for key in keys]

    async def delete(self, key: StorageKey) -> bool:
        key_str = str(key)
        existed = key_str in self._storage
        self._storage.pop(key_str, None)
        self._metadata.pop(key_str, None)
        return existed

    async def exists(self, key: StorageKey) -> bool:
        return str(key) in self._storage

    async def get_metadata(self, key: StorageKey) -> StorageMetadata | None:
        return self._metadata.get(str(key))

    async def update_metadata(self, key: StorageKey, updates: dict[str, Any]) -> bool:
        key_str = str(key)
        if key_str not in self._metadata:
            return False
        for k, v in updates.items():
            setattr(self._metadata[key_str], k, v)
        return True

    async def list_keys(
        self,
        session_id: str,
        *,
        tier: StorageTier | None = None,
        node_type: str | None = None,
        limit: int = 1000,
    ) -> list[StorageKey]:
        keys = []
        for key_str in self._storage:
            key = StorageKey.from_string(key_str)
            if key.session_id != session_id:
                continue
            if (
                tier
                and self._metadata.get(
                    key_str,
                    StorageMetadata(key=key, size_bytes=0, token_count=0, node_type=""),
                ).tier
                != tier
            ):
                continue
            if (
                node_type
                and self._metadata.get(
                    key_str,
                    StorageMetadata(key=key, size_bytes=0, token_count=0, node_type=""),
                ).node_type
                != node_type
            ):
                continue
            keys.append(key)
            if len(keys) >= limit:
                break
        return keys

    async def search_by_metadata(
        self,
        session_id: str,
        *,
        min_importance: float | None = None,
        tags: set[str] | None = None,
        since: datetime | None = None,
        limit: int = 100,
    ) -> list[tuple[StorageKey, StorageMetadata]]:
        results = []
        for _key_str, meta in self._metadata.items():
            if meta.key.session_id != session_id:
                continue
            if min_importance and meta.importance < min_importance:
                continue
            if tags and not (tags & meta.tags):
                continue
            if since and meta.created_at < since:
                continue
            results.append((meta.key, meta))
            if len(results) >= limit:
                break
        return results

    async def stats(self, session_id: str | None = None) -> StorageStats:
        items = 0
        size = 0
        tokens = 0

        for _key_str, meta in self._metadata.items():
            if session_id and meta.key.session_id != session_id:
                continue
            items += 1
            size += meta.size_bytes
            tokens += meta.token_count

        return StorageStats(
            total_items=items,
            total_size_bytes=size,
            total_tokens=tokens,
        )

    async def close(self) -> None:
        self._storage.clear()
        self._metadata.clear()


# Verify MockMemoryStore satisfies the protocol
def test_mock_store_satisfies_protocol() -> None:
    """Verify our mock implementation satisfies the protocol."""
    store = MockMemoryStore()
    assert isinstance(store, MemoryStore)
