"""Unit tests for retrieval strategies."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from uuid import UUID, uuid4

import pytest
import pytest_asyncio

from context_memory.backends.filesystem import FileSystemStore
from context_memory.retrieval import (
    EnsembleRetriever,
    EntityRetrieval,
    MemoryRetriever,
    RetrievalQuery,
    SemanticRetrieval,
    TemporalRetrieval,
)
from context_memory.types import RetrievalResult, StorageTier

# =============================================================================
# Mock Classes
# =============================================================================


class MockNodeMetadata:
    """Mock node metadata."""

    def __init__(self, importance: float = 0.5, tags: set[str] | None = None) -> None:
        self.importance = importance
        self.tags = tags or set()


class MockNodeType:
    """Mock node type enum."""

    def __init__(self, value: str = "message") -> None:
        self.value = value


class MockNode:
    """Mock ContextNode for testing."""

    def __init__(
        self,
        content: str = "test",
        token_count: int = 10,
        importance: float = 0.5,
        node_type: str = "message",
    ) -> None:
        self.id = uuid4()
        self.type = MockNodeType(node_type)
        self.content = content
        self.token_count = token_count
        self.metadata = MockNodeMetadata(importance)

    def model_dump(self, mode: str = "python") -> dict:
        return {
            "id": str(self.id),
            "type": self.type.value,
            "content": {"text": self.content, "role": "user"},
            "token_count": self.token_count,
            "metadata": {
                "importance": self.metadata.importance,
                "tags": [],
            },
        }


class MockSearchResult:
    """Mock semantic search result."""

    def __init__(self, node_id: UUID, score: float) -> None:
        self.id = node_id
        self.score = score


class MockSemanticIndex:
    """Mock SemanticIndex for testing."""

    def __init__(self) -> None:
        self._results: list[MockSearchResult] = []

    def add_result(self, node_id: UUID, score: float) -> None:
        """Add a search result."""
        self._results.append(MockSearchResult(node_id, score))

    def search(
        self,
        query: str,
        k: int = 10,
        min_score: float = 0.0,
        filter: dict[str, Any] | None = None,
    ) -> list[MockSearchResult]:
        """Return mock search results."""
        return [r for r in self._results if r.score >= min_score][:k]


class MockEntity:
    """Mock Entity for testing."""

    def __init__(
        self,
        entity_id: UUID,
        node_ids: list[UUID],
        importance: float = 0.5,
        mention_count: int = 1,
    ) -> None:
        self.id = entity_id
        self.node_ids = node_ids
        self.importance = importance
        self.mention_count = mention_count


class MockEntityTracker:
    """Mock EntityTracker for testing."""

    def __init__(self) -> None:
        self._entities: dict[UUID, MockEntity] = {}
        self._extracted: list[MockEntity] = []

    def add_entity(self, entity: MockEntity) -> None:
        """Add an entity."""
        self._entities[entity.id] = entity

    def set_extracted_entities(self, entities: list[MockEntity]) -> None:
        """Set entities to return from extract_from_text."""
        self._extracted = entities

    def get_entity(self, entity_id: UUID) -> MockEntity | None:
        """Get entity by ID."""
        return self._entities.get(entity_id)

    def extract_from_text(
        self, text: str, node_id: UUID, timestamp: datetime | None = None
    ) -> list[MockEntity]:
        """Return pre-set extracted entities."""
        return self._extracted


# =============================================================================
# Fixtures
# =============================================================================


@pytest_asyncio.fixture
async def store(tmp_path: Path) -> FileSystemStore:
    """Create a FileSystemStore for testing."""
    s = FileSystemStore(tmp_path / "store")
    yield s
    await s.close()


# =============================================================================
# RetrievalQuery Tests
# =============================================================================


class TestRetrievalQuery:
    """Tests for RetrievalQuery model."""

    def test_default_values(self) -> None:
        """Test default query values."""
        query = RetrievalQuery(session_id="test")
        assert query.session_id == "test"
        assert query.query_text is None
        assert query.entity_ids is None
        assert query.max_results == 10

    def test_custom_values(self) -> None:
        """Test custom query values."""
        query = RetrievalQuery(
            session_id="test",
            query_text="search text",
            min_importance=0.7,
            max_results=5,
        )
        assert query.query_text == "search text"
        assert query.min_importance == 0.7
        assert query.max_results == 5


# =============================================================================
# SemanticRetrieval Tests
# =============================================================================


class TestSemanticRetrieval:
    """Tests for SemanticRetrieval strategy."""

    def test_name(self) -> None:
        """Test strategy name."""
        index = MockSemanticIndex()
        strategy = SemanticRetrieval(semantic_index=index)
        assert strategy.name == "semantic"

    @pytest.mark.asyncio
    async def test_empty_query_text(self, store: FileSystemStore) -> None:
        """Test returns empty for no query text."""
        index = MockSemanticIndex()
        strategy = SemanticRetrieval(semantic_index=index)

        query = RetrievalQuery(session_id="test")
        results = await strategy.retrieve(query, store)
        assert results == []

    @pytest.mark.asyncio
    async def test_retrieves_matching_nodes(self, store: FileSystemStore) -> None:
        """Test retrieves nodes matching semantic search."""
        # Store a node
        node = MockNode(content="test content")
        key = await store.store(node, "test-session")

        # Setup mock index to return this node
        index = MockSemanticIndex()
        index.add_result(key.node_id, score=0.9)

        strategy = SemanticRetrieval(semantic_index=index, min_similarity=0.5)
        query = RetrievalQuery(session_id="test-session", query_text="test")

        results = await strategy.retrieve(query, store)
        assert len(results) == 1
        assert results[0].score == 0.9
        assert results[0].retrieval_method == "semantic"

    @pytest.mark.asyncio
    async def test_respects_min_similarity(self, store: FileSystemStore) -> None:
        """Test filters by minimum similarity."""
        node = MockNode(content="test")
        key = await store.store(node, "test-session")

        index = MockSemanticIndex()
        index.add_result(key.node_id, score=0.4)  # Below threshold

        strategy = SemanticRetrieval(semantic_index=index, min_similarity=0.6)
        query = RetrievalQuery(session_id="test-session", query_text="test")

        results = await strategy.retrieve(query, store)
        assert len(results) == 0


# =============================================================================
# EntityRetrieval Tests
# =============================================================================


class TestEntityRetrieval:
    """Tests for EntityRetrieval strategy."""

    def test_name(self) -> None:
        """Test strategy name."""
        tracker = MockEntityTracker()
        strategy = EntityRetrieval(entity_tracker=tracker)
        assert strategy.name == "entity"

    @pytest.mark.asyncio
    async def test_empty_entities(self, store: FileSystemStore) -> None:
        """Test returns empty when no entities."""
        tracker = MockEntityTracker()
        strategy = EntityRetrieval(entity_tracker=tracker)

        query = RetrievalQuery(session_id="test")
        results = await strategy.retrieve(query, store)
        assert results == []

    @pytest.mark.asyncio
    async def test_retrieves_by_entity_id(self, store: FileSystemStore) -> None:
        """Test retrieves nodes by entity ID."""
        # Store a node
        node = MockNode(content="test")
        key = await store.store(node, "test-session")

        # Create entity referencing this node
        entity_id = uuid4()
        entity = MockEntity(
            entity_id=entity_id,
            node_ids=[key.node_id],
            importance=0.8,
            mention_count=5,
        )

        tracker = MockEntityTracker()
        tracker.add_entity(entity)

        strategy = EntityRetrieval(entity_tracker=tracker)
        query = RetrievalQuery(
            session_id="test-session",
            entity_ids=[str(entity_id)],
        )

        results = await strategy.retrieve(query, store)
        assert len(results) == 1
        assert results[0].retrieval_method == "entity"


# =============================================================================
# TemporalRetrieval Tests
# =============================================================================


class TestTemporalRetrieval:
    """Tests for TemporalRetrieval strategy."""

    def test_name(self) -> None:
        """Test strategy name."""
        strategy = TemporalRetrieval()
        assert strategy.name == "temporal"

    @pytest.mark.asyncio
    async def test_retrieves_recent_items(self, store: FileSystemStore) -> None:
        """Test retrieves recently stored items."""
        # Store nodes
        node1 = MockNode(content="recent")
        await store.store(node1, "test-session")

        strategy = TemporalRetrieval(default_window_hours=24)
        query = RetrievalQuery(session_id="test-session")

        results = await strategy.retrieve(query, store)
        assert len(results) == 1
        assert results[0].retrieval_method == "temporal"

    @pytest.mark.asyncio
    async def test_respects_time_window(self, store: FileSystemStore) -> None:
        """Test filters by time window."""
        # Store a node
        node = MockNode(content="test")
        key = await store.store(node, "test-session")

        # Backdate the node
        old_time = datetime.now(UTC) - timedelta(hours=48)
        await store.update_metadata(key, {"created_at": old_time.isoformat()})

        strategy = TemporalRetrieval(default_window_hours=24)
        query = RetrievalQuery(session_id="test-session")

        results = await strategy.retrieve(query, store)
        assert len(results) == 0

    @pytest.mark.asyncio
    async def test_custom_time_range(self, store: FileSystemStore) -> None:
        """Test with explicit time range in query."""
        node = MockNode(content="test")
        key = await store.store(node, "test-session")

        # Backdate slightly
        past = datetime.now(UTC) - timedelta(hours=2)
        await store.update_metadata(key, {"created_at": past.isoformat()})

        strategy = TemporalRetrieval()
        query = RetrievalQuery(
            session_id="test-session",
            since=datetime.now(UTC) - timedelta(hours=5),
            until=datetime.now(UTC),
        )

        results = await strategy.retrieve(query, store)
        assert len(results) == 1


# =============================================================================
# EnsembleRetriever Tests
# =============================================================================


class MockStrategy:
    """Mock strategy for ensemble testing."""

    def __init__(self, name: str, results: list[RetrievalResult]) -> None:
        self._name = name
        self._results = results

    @property
    def name(self) -> str:
        return self._name

    async def retrieve(
        self,
        query: RetrievalQuery,
        store: Any,
        current_context: Any = None,
    ) -> list[RetrievalResult]:
        return self._results


class TestEnsembleRetriever:
    """Tests for EnsembleRetriever."""

    def test_name(self) -> None:
        """Test ensemble name."""
        ensemble = EnsembleRetriever(strategies=[])
        assert ensemble.name == "ensemble"

    @pytest.mark.asyncio
    async def test_combines_strategies(self, store: FileSystemStore) -> None:
        """Test combines results from multiple strategies."""
        # Store nodes
        node1 = MockNode(content="node1")
        node2 = MockNode(content="node2")
        await store.store(node1, "test-session")
        await store.store(node2, "test-session")

        # Create mock results
        result1 = RetrievalResult(
            node=node1,
            score=0.9,
            source_tier=StorageTier.WARM,
            retrieval_method="strategy1",
            latency_ms=1.0,
        )
        result2 = RetrievalResult(
            node=node2,
            score=0.8,
            source_tier=StorageTier.WARM,
            retrieval_method="strategy2",
            latency_ms=1.0,
        )

        strategy1 = MockStrategy("s1", [result1])
        strategy2 = MockStrategy("s2", [result2])

        ensemble = EnsembleRetriever(
            strategies=[(strategy1, 1.0), (strategy2, 1.0)],
            k=60,
        )

        query = RetrievalQuery(session_id="test-session")
        results = await ensemble.retrieve(query, store)

        assert len(results) == 2
        assert all(r.retrieval_method == "ensemble" for r in results)

    @pytest.mark.asyncio
    async def test_empty_strategies(self, store: FileSystemStore) -> None:
        """Test handles no strategies."""
        ensemble = EnsembleRetriever(strategies=[])
        query = RetrievalQuery(session_id="test")
        results = await ensemble.retrieve(query, store)
        assert results == []

    @pytest.mark.asyncio
    async def test_weights_affect_ranking(self, store: FileSystemStore) -> None:
        """Test that weights affect final ranking."""
        node1 = MockNode(content="high-weight")
        node2 = MockNode(content="low-weight")

        result1 = RetrievalResult(
            node=node1,
            score=0.7,
            source_tier=StorageTier.WARM,
            retrieval_method="high",
            latency_ms=1.0,
        )
        result2 = RetrievalResult(
            node=node2,
            score=0.9,
            source_tier=StorageTier.WARM,
            retrieval_method="low",
            latency_ms=1.0,
        )

        high_weight_strategy = MockStrategy("high", [result1])
        low_weight_strategy = MockStrategy("low", [result2])

        # High weight strategy should boost node1
        ensemble = EnsembleRetriever(
            strategies=[(high_weight_strategy, 10.0), (low_weight_strategy, 0.1)],
            k=60,
        )

        query = RetrievalQuery(session_id="test")
        results = await ensemble.retrieve(query, store)

        # Node1 from high-weight strategy should rank first
        assert results[0].node.content == "high-weight"


# =============================================================================
# MemoryRetriever Tests
# =============================================================================


class TestMemoryRetriever:
    """Tests for MemoryRetriever convenience class."""

    @pytest.mark.asyncio
    async def test_creation(self, store: FileSystemStore) -> None:
        """Test retriever creation."""
        retriever = MemoryRetriever(store=store, default_session_id="test")
        assert retriever._default_session_id == "test"

    @pytest.mark.asyncio
    async def test_add_temporal(self, store: FileSystemStore) -> None:
        """Test adding temporal strategy."""
        retriever = MemoryRetriever(store=store, default_session_id="test")
        retriever.add_temporal(weight=0.5)

        # Store a node
        node = MockNode(content="test")
        await store.store(node, "test")

        results = await retriever.search("query")
        assert len(results) == 1

    @pytest.mark.asyncio
    async def test_requires_session_id(self, store: FileSystemStore) -> None:
        """Test raises error without session_id."""
        retriever = MemoryRetriever(store=store)

        with pytest.raises(ValueError, match="session_id required"):
            await retriever.search("query")
