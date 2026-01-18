"""Tests for cache invalidation policies."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from context_tools.cache.policies import (
    DEFAULT_POLICIES,
    DependencyTracker,
    InvalidationPolicy,
    InvalidationTrigger,
    PolicyRegistry,
    StalenessCalculator,
)
from context_tools.types import ToolCallSignature

# =============================================================================
# InvalidationPolicy Tests
# =============================================================================


class TestInvalidationPolicy:
    """Tests for InvalidationPolicy class."""

    def test_default_values(self) -> None:
        """Test default policy values."""
        policy = InvalidationPolicy(tool_name="test_tool")

        assert policy.tool_name == "test_tool"
        assert policy.ttl_seconds == 3600
        assert policy.cache_errors is False
        assert policy.trigger == InvalidationTrigger.TIME_BASED
        assert policy.invalidation_pattern is None
        assert policy.version_path is None
        assert policy.staleness_threshold == 1.0
        assert policy.dependencies == []

    def test_custom_values(self) -> None:
        """Test custom policy configuration."""
        policy = InvalidationPolicy(
            tool_name="read_file",
            ttl_seconds=300,
            cache_errors=True,
            trigger=InvalidationTrigger.WRITE_THROUGH,
            invalidation_pattern=r"path.*\.py$",
            staleness_threshold=0.8,
            dependencies=["write_file", "edit_file"],
        )

        assert policy.ttl_seconds == 300
        assert policy.cache_errors is True
        assert policy.trigger == InvalidationTrigger.WRITE_THROUGH
        assert policy.invalidation_pattern == r"path.*\.py$"
        assert policy.staleness_threshold == 0.8
        assert policy.dependencies == ["write_file", "edit_file"]

    def test_matches_invalidation_pattern_no_pattern(self) -> None:
        """Test pattern matching when no pattern is set."""
        policy = InvalidationPolicy(tool_name="test")
        arguments = {"path": "/some/file.py"}

        # Should return True when no pattern set
        assert policy.matches_invalidation_pattern(arguments) is True

    def test_matches_invalidation_pattern_match(self) -> None:
        """Test pattern matching with matching arguments."""
        policy = InvalidationPolicy(
            tool_name="test",
            invalidation_pattern=r"\.py",  # Matches .py anywhere in stringified args
        )
        arguments = {"path": "/some/file.py"}

        assert policy.matches_invalidation_pattern(arguments) is True

    def test_matches_invalidation_pattern_no_match(self) -> None:
        """Test pattern matching with non-matching arguments."""
        policy = InvalidationPolicy(
            tool_name="test",
            invalidation_pattern=r"\.py$",
        )
        arguments = {"path": "/some/file.txt"}

        assert policy.matches_invalidation_pattern(arguments) is False

    def test_matches_invalidation_pattern_case_insensitive(self) -> None:
        """Test pattern matching is case insensitive."""
        policy = InvalidationPolicy(
            tool_name="test",
            invalidation_pattern=r"test",
        )
        arguments = {"query": "TEST string"}

        assert policy.matches_invalidation_pattern(arguments) is True

    def test_extract_version_simple_path(self) -> None:
        """Test version extraction with simple path."""
        policy = InvalidationPolicy(
            tool_name="api_call",
            version_path="version",
        )
        result = {"version": "1.2.3", "data": "content"}

        assert policy.extract_version(result) == "1.2.3"

    def test_extract_version_nested_path(self) -> None:
        """Test version extraction with nested path."""
        policy = InvalidationPolicy(
            tool_name="api_call",
            version_path="data.version",
        )
        result = {"data": {"version": "2.0.0", "content": "stuff"}}

        assert policy.extract_version(result) == "2.0.0"

    def test_extract_version_no_path(self) -> None:
        """Test version extraction when no path configured."""
        policy = InvalidationPolicy(tool_name="test")
        result = {"version": "1.0.0"}

        assert policy.extract_version(result) is None

    def test_extract_version_missing_key(self) -> None:
        """Test version extraction when key doesn't exist."""
        policy = InvalidationPolicy(
            tool_name="test",
            version_path="missing.key",
        )
        result = {"data": "no version"}

        assert policy.extract_version(result) is None


class TestInvalidationTrigger:
    """Tests for InvalidationTrigger enum."""

    def test_trigger_values(self) -> None:
        """Test all trigger enum values."""
        assert InvalidationTrigger.TIME_BASED == "time_based"
        assert InvalidationTrigger.WRITE_THROUGH == "write_through"
        assert InvalidationTrigger.VERSION_BASED == "version_based"
        assert InvalidationTrigger.MANUAL == "manual"


# =============================================================================
# DependencyTracker Tests
# =============================================================================


class TestDependencyTracker:
    """Tests for DependencyTracker class."""

    @pytest.fixture
    def tracker(self) -> DependencyTracker:
        """Create a fresh dependency tracker."""
        return DependencyTracker()

    def test_record_write_file(self, tracker: DependencyTracker) -> None:
        """Test recording a write operation."""
        tracker.record_write("write_file", {"path": "/tmp/test.py"})

        assert "/tmp/test.py" in tracker._modified_paths
        assert len(tracker._writes["write_file"]) == 1

    def test_record_non_write_tool(self, tracker: DependencyTracker) -> None:
        """Test that non-write tools are not recorded."""
        tracker.record_write("read_file", {"path": "/tmp/test.py"})

        assert "/tmp/test.py" not in tracker._modified_paths
        assert len(tracker._writes["read_file"]) == 0

    def test_get_affected_tools_write_file(self, tracker: DependencyTracker) -> None:
        """Test getting affected tools for write_file."""
        affected = tracker.get_affected_tools("write_file")

        assert "read_file" in affected
        assert "search" in affected
        assert "grep" in affected

    def test_get_affected_tools_delete_file(self, tracker: DependencyTracker) -> None:
        """Test getting affected tools for delete_file."""
        affected = tracker.get_affected_tools("delete_file")

        assert "read_file" in affected
        assert "list_directory" in affected
        assert "search" in affected

    def test_get_affected_tools_unknown(self, tracker: DependencyTracker) -> None:
        """Test getting affected tools for unknown tool."""
        affected = tracker.get_affected_tools("unknown_tool")

        assert affected == set()

    def test_should_invalidate_path_modified_after_cache(
        self,
        tracker: DependencyTracker,
    ) -> None:
        """Test invalidation when path modified after caching."""
        cached_at = datetime.now(UTC) - timedelta(minutes=5)

        # Record write after cache time
        tracker.record_write("write_file", {"path": "/tmp/test.py"})

        sig = ToolCallSignature(
            tool_name="read_file",
            arguments={"path": "/tmp/test.py"},
        )

        assert tracker.should_invalidate(sig, cached_at) is True

    def test_should_invalidate_path_modified_before_cache(
        self,
        tracker: DependencyTracker,
    ) -> None:
        """Test no invalidation when path modified before caching."""
        # Record write first
        tracker.record_write("write_file", {"path": "/tmp/test.py"})

        # Cache after the write
        cached_at = datetime.now(UTC) + timedelta(seconds=1)

        sig = ToolCallSignature(
            tool_name="read_file",
            arguments={"path": "/tmp/test.py"},
        )

        assert tracker.should_invalidate(sig, cached_at) is False

    def test_should_invalidate_different_path(
        self,
        tracker: DependencyTracker,
    ) -> None:
        """Test no invalidation for different path."""
        cached_at = datetime.now(UTC) - timedelta(minutes=5)
        tracker.record_write("write_file", {"path": "/tmp/other.py"})

        sig = ToolCallSignature(
            tool_name="read_file",
            arguments={"path": "/tmp/test.py"},
        )

        assert tracker.should_invalidate(sig, cached_at) is False

    def test_should_invalidate_list_directory_child_modified(
        self,
        tracker: DependencyTracker,
    ) -> None:
        """Test list_directory invalidation when child file modified."""
        cached_at = datetime.now(UTC) - timedelta(minutes=5)
        tracker.record_write("create_file", {"path": "/tmp/dir/newfile.py"})

        sig = ToolCallSignature(
            tool_name="list_directory",
            arguments={"path": "/tmp/dir"},
        )

        assert tracker.should_invalidate(sig, cached_at) is True

    def test_get_modified_paths_since(self, tracker: DependencyTracker) -> None:
        """Test getting paths modified since a timestamp."""
        old_time = datetime.now(UTC) - timedelta(hours=1)

        tracker.record_write("write_file", {"path": "/tmp/new.py"})

        paths = tracker.get_modified_paths_since(old_time)
        assert "/tmp/new.py" in paths

    def test_clear(self, tracker: DependencyTracker) -> None:
        """Test clearing all tracked writes."""
        tracker.record_write("write_file", {"path": "/tmp/test.py"})
        tracker.clear()

        assert len(tracker._modified_paths) == 0
        assert len(tracker._writes) == 0

    def test_trim_writes_under_limit(self, tracker: DependencyTracker) -> None:
        """Test that writes under limit are kept."""
        for i in range(10):
            tracker.record_write("write_file", {"path": f"/tmp/file{i}.py"})

        assert len(tracker._modified_paths) == 10

    def test_trim_writes_over_limit(self) -> None:
        """Test that old writes are trimmed when over limit."""
        tracker = DependencyTracker(max_tracked_writes=5)

        for i in range(10):
            tracker.record_write("write_file", {"path": f"/tmp/file{i}.py"})

        # Should keep only the 5 most recent
        assert len(tracker._modified_paths) <= 10  # paths not trimmed
        total_writes = sum(len(w) for w in tracker._writes.values())
        assert total_writes <= 5

    def test_extract_path_various_keys(self, tracker: DependencyTracker) -> None:
        """Test path extraction from various argument keys."""
        # Test different path key names
        for key in ["path", "file_path", "filepath", "file", "directory", "dir"]:
            args = {key: "/tmp/test.py"}
            assert tracker._extract_path(args) == "/tmp/test.py"

    def test_extract_path_no_path_key(self, tracker: DependencyTracker) -> None:
        """Test path extraction when no path key exists."""
        args = {"query": "search term"}
        assert tracker._extract_path(args) is None


# =============================================================================
# StalenessCalculator Tests
# =============================================================================


class TestStalenessCalculator:
    """Tests for StalenessCalculator class."""

    @pytest.fixture
    def calculator(self) -> StalenessCalculator:
        """Create a staleness calculator."""
        return StalenessCalculator()

    def test_fresh_entry(self, calculator: StalenessCalculator) -> None:
        """Test staleness of a fresh entry."""
        score = calculator.calculate(
            age_seconds=0,
            ttl_seconds=3600,
            last_access_seconds=0,
            access_count=1,
            writes_since_cache=0,
        )

        assert score < 0.1  # Should be very fresh

    def test_near_expiry_entry(self, calculator: StalenessCalculator) -> None:
        """Test staleness of entry near TTL."""
        score = calculator.calculate(
            age_seconds=3500,  # Near 3600 TTL
            ttl_seconds=3600,
            last_access_seconds=3500,
            access_count=1,
            writes_since_cache=0,
        )

        # Age is ~97% of TTL, so age component is high
        # Score combines age (0.4 weight), access (0.3), writes (0.3)
        assert score > 0.4  # Should be moderately stale

    def test_expired_entry(self, calculator: StalenessCalculator) -> None:
        """Test staleness of expired entry."""
        score = calculator.calculate(
            age_seconds=7200,  # Past 3600 TTL
            ttl_seconds=3600,
            last_access_seconds=7200,
            access_count=0,
            writes_since_cache=0,
        )

        # Age capped at 1.0, access component is high due to no recent access
        # Combined with weights: 0.4*1.0 + 0.3*access + 0.3*0 >= 0.5
        assert score >= 0.5  # Should be stale

    def test_recently_accessed_entry(self, calculator: StalenessCalculator) -> None:
        """Test staleness with recent access."""
        # Same age, but one recently accessed
        score_stale = calculator.calculate(
            age_seconds=1800,
            ttl_seconds=3600,
            last_access_seconds=1800,  # Not accessed since creation
            access_count=1,
            writes_since_cache=0,
        )

        score_fresh = calculator.calculate(
            age_seconds=1800,
            ttl_seconds=3600,
            last_access_seconds=60,  # Accessed 1 minute ago
            access_count=10,  # Frequently accessed
            writes_since_cache=0,
        )

        assert score_fresh < score_stale

    def test_writes_increase_staleness(self, calculator: StalenessCalculator) -> None:
        """Test that writes increase staleness."""
        score_no_writes = calculator.calculate(
            age_seconds=1000,
            ttl_seconds=3600,
            last_access_seconds=100,
            access_count=5,
            writes_since_cache=0,
        )

        score_many_writes = calculator.calculate(
            age_seconds=1000,
            ttl_seconds=3600,
            last_access_seconds=100,
            access_count=5,
            writes_since_cache=10,
        )

        assert score_many_writes > score_no_writes

    def test_no_ttl_uses_default_decay(self, calculator: StalenessCalculator) -> None:
        """Test staleness calculation without TTL."""
        score = calculator.calculate(
            age_seconds=43200,  # 12 hours
            ttl_seconds=None,
            last_access_seconds=43200,
            access_count=1,
            writes_since_cache=0,
        )

        # Should use 24-hour default, so 12 hours = ~0.5 age score
        assert 0.3 < score < 0.7

    def test_staleness_clamped_to_range(self, calculator: StalenessCalculator) -> None:
        """Test that staleness is clamped to 0-1 range."""
        # Extreme values
        score_high = calculator.calculate(
            age_seconds=1000000,
            ttl_seconds=100,
            last_access_seconds=1000000,
            access_count=0,
            writes_since_cache=100,
        )

        score_low = calculator.calculate(
            age_seconds=0,
            ttl_seconds=10000,
            last_access_seconds=0,
            access_count=1000,
            writes_since_cache=0,
        )

        assert 0.0 <= score_high <= 1.0
        assert 0.0 <= score_low <= 1.0


# =============================================================================
# PolicyRegistry Tests
# =============================================================================


class TestPolicyRegistry:
    """Tests for PolicyRegistry class."""

    def test_register_and_get_policy(self) -> None:
        """Test registering and retrieving a policy."""
        registry = PolicyRegistry(use_defaults=False)
        policy = InvalidationPolicy(
            tool_name="custom_tool",
            ttl_seconds=600,
            trigger=InvalidationTrigger.WRITE_THROUGH,
        )

        registry.register(policy)
        retrieved = registry.get("custom_tool")

        assert retrieved.tool_name == "custom_tool"
        assert retrieved.ttl_seconds == 600
        assert retrieved.trigger == InvalidationTrigger.WRITE_THROUGH

    def test_get_unregistered_returns_default(self) -> None:
        """Test getting policy for unregistered tool."""
        registry = PolicyRegistry(use_defaults=False)
        policy = registry.get("unknown_tool")

        assert policy.tool_name == "unknown_tool"
        assert policy.ttl_seconds == 3600  # Default TTL

    def test_set_default_policy(self) -> None:
        """Test setting custom default policy."""
        registry = PolicyRegistry(use_defaults=False)
        default = InvalidationPolicy(
            tool_name="__default__",
            ttl_seconds=7200,
            cache_errors=True,
        )

        registry.set_default(default)
        policy = registry.get("new_tool")

        assert policy.ttl_seconds == 7200
        assert policy.cache_errors is True

    def test_list_policies(self) -> None:
        """Test listing all registered policies."""
        registry = PolicyRegistry(use_defaults=False)
        registry.register(InvalidationPolicy(tool_name="tool1"))
        registry.register(InvalidationPolicy(tool_name="tool2"))

        policies = registry.list_policies()

        assert len(policies) == 2
        tool_names = {p.tool_name for p in policies}
        assert "tool1" in tool_names
        assert "tool2" in tool_names

    def test_defaults_loaded(self) -> None:
        """Test that default policies are loaded."""
        registry = PolicyRegistry(use_defaults=True)

        # Should have default policies for common tools
        read_policy = registry.get("read_file")
        assert read_policy.ttl_seconds == 300
        assert read_policy.trigger == InvalidationTrigger.WRITE_THROUGH

        search_policy = registry.get("search")
        assert search_policy.ttl_seconds == 3600

    def test_defaults_not_loaded_when_disabled(self) -> None:
        """Test that defaults are not loaded when disabled."""
        registry = PolicyRegistry(use_defaults=False)
        policies = registry.list_policies()

        assert len(policies) == 0

    def test_override_default_policy(self) -> None:
        """Test overriding a default policy."""
        registry = PolicyRegistry(use_defaults=True)

        # Override read_file policy
        custom = InvalidationPolicy(
            tool_name="read_file",
            ttl_seconds=60,
            trigger=InvalidationTrigger.MANUAL,
        )
        registry.register(custom)

        policy = registry.get("read_file")
        assert policy.ttl_seconds == 60
        assert policy.trigger == InvalidationTrigger.MANUAL


# =============================================================================
# Default Policies Tests
# =============================================================================


class TestDefaultPolicies:
    """Tests for default policy configurations."""

    def test_read_file_policy(self) -> None:
        """Test read_file default policy."""
        policy = DEFAULT_POLICIES["read_file"]

        assert policy.ttl_seconds == 300
        assert policy.trigger == InvalidationTrigger.WRITE_THROUGH
        assert "write_file" in policy.dependencies

    def test_search_policy(self) -> None:
        """Test search default policy."""
        policy = DEFAULT_POLICIES["search"]

        assert policy.ttl_seconds == 3600
        assert policy.trigger == InvalidationTrigger.TIME_BASED

    def test_web_fetch_policy(self) -> None:
        """Test web_fetch default policy."""
        policy = DEFAULT_POLICIES["web_fetch"]

        assert policy.ttl_seconds == 1800
        assert policy.cache_errors is False

    def test_list_directory_policy(self) -> None:
        """Test list_directory default policy."""
        policy = DEFAULT_POLICIES["list_directory"]

        assert policy.ttl_seconds == 60
        assert policy.trigger == InvalidationTrigger.WRITE_THROUGH

    def test_run_command_policy(self) -> None:
        """Test run_command default policy (no caching)."""
        policy = DEFAULT_POLICIES["run_command"]

        assert policy.ttl_seconds is None  # No caching
        assert policy.trigger == InvalidationTrigger.MANUAL

    def test_api_call_policy(self) -> None:
        """Test api_call default policy with version support."""
        policy = DEFAULT_POLICIES["api_call"]

        assert policy.version_path == "data.version"
        assert policy.ttl_seconds == 600

    def test_all_default_policies_valid(self) -> None:
        """Test that all default policies are valid."""
        for name, policy in DEFAULT_POLICIES.items():
            assert policy.tool_name == name
            assert policy.staleness_threshold >= 0.0
            assert policy.staleness_threshold <= 1.0
            assert isinstance(policy.trigger, InvalidationTrigger)
