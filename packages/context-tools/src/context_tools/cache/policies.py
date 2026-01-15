"""Cache invalidation policies for tool call caching.

Provides configurable invalidation strategies:
- Time-based (TTL): Entries expire after a set duration
- Write-through: Entries invalidate when underlying data changes
- Version-based: Entries invalidate when version changes
- Manual: Only explicit invalidation removes entries
- Dependency-based: Track dependencies between tool calls
"""

from __future__ import annotations

import re
from collections import defaultdict
from datetime import UTC, datetime
from enum import Enum
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from context_tools.types import ToolCallSignature


class InvalidationTrigger(str, Enum):
    """When to invalidate cache entries."""

    TIME_BASED = "time_based"  # After TTL expires
    WRITE_THROUGH = "write_through"  # When underlying data changes
    VERSION_BASED = "version_based"  # When version field changes
    MANUAL = "manual"  # Only on explicit invalidation


class InvalidationPolicy(BaseModel):
    """Policy for cache invalidation per tool.

    Configures how and when cache entries for a specific tool
    should be invalidated.

    Attributes:
        tool_name: Name of the tool this policy applies to
        ttl_seconds: Time-to-live in seconds (None = no expiry)
        cache_errors: Whether to cache error results
        trigger: Type of invalidation trigger
        invalidation_pattern: Regex pattern for write-through invalidation
        version_path: JSONPath-like path to version field for version-based
        staleness_threshold: Max staleness score before invalidation (0-1)
        dependencies: List of tool names whose results affect this cache
    """

    tool_name: str = Field(description="Name of the tool")
    ttl_seconds: int | None = Field(
        default=3600,
        description="Time-to-live in seconds (None = no expiry)",
    )
    cache_errors: bool = Field(
        default=False,
        description="Whether to cache error results",
    )
    trigger: InvalidationTrigger = Field(
        default=InvalidationTrigger.TIME_BASED,
        description="Type of invalidation trigger",
    )
    invalidation_pattern: str | None = Field(
        default=None,
        description="Regex pattern for write-through invalidation",
    )
    version_path: str | None = Field(
        default=None,
        description="JSONPath-like path to version field",
    )
    staleness_threshold: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Max staleness score before invalidation",
    )
    dependencies: list[str] = Field(
        default_factory=list,
        description="Tool names whose results affect this cache",
    )

    def matches_invalidation_pattern(self, arguments: dict[str, Any]) -> bool:
        """Check if arguments match the invalidation pattern.

        Args:
            arguments: Tool call arguments to check

        Returns:
            True if pattern matches (or no pattern set)
        """
        if self.invalidation_pattern is None:
            return True

        # Convert arguments to string for pattern matching
        args_str = str(arguments)
        return bool(re.search(self.invalidation_pattern, args_str, re.IGNORECASE))

    def extract_version(self, result: Any) -> str | None:
        """Extract version from result using version_path.

        Args:
            result: Tool result to extract version from

        Returns:
            Version string if found, None otherwise
        """
        if self.version_path is None:
            return None

        # Simple JSONPath-like extraction (supports "key" and "key.subkey")
        try:
            value = result
            for part in self.version_path.split("."):
                if isinstance(value, dict):
                    value = value.get(part)
                else:
                    return None
            return str(value) if value is not None else None
        except (AttributeError, TypeError):
            return None


class DependencyTracker:
    """Track dependencies between tool calls for write-through invalidation.

    Maintains a graph of dependencies where:
    - write_file -> invalidates read_file entries for same path
    - delete_file -> invalidates read_file and list_directory entries
    - Any file modification -> invalidates search results

    Example:
        >>> tracker = DependencyTracker()
        >>> tracker.record_write("write_file", {"path": "/tmp/test.py"})
        >>> sig = ToolCallSignature(tool_name="read_file", arguments={...})
        >>> tracker.should_invalidate(sig, cached_at)  # Returns True if stale
    """

    # Tools that modify state (writers)
    WRITE_TOOLS = frozenset(
        {
            "write_file",
            "edit_file",
            "delete_file",
            "create_file",
            "rename_file",
            "move_file",
            "mkdir",
            "rmdir",
        }
    )

    # Mapping of write tools to the read tools they invalidate
    INVALIDATION_MAP: dict[str, set[str]] = {
        "write_file": {"read_file", "search", "grep"},
        "edit_file": {"read_file", "search", "grep"},
        "create_file": {"read_file", "list_directory", "search"},
        "delete_file": {"read_file", "list_directory", "search"},
        "rename_file": {"read_file", "list_directory", "search"},
        "move_file": {"read_file", "list_directory", "search"},
        "mkdir": {"list_directory"},
        "rmdir": {"list_directory"},
    }

    def __init__(self, max_tracked_writes: int = 1000) -> None:
        """Initialize DependencyTracker.

        Args:
            max_tracked_writes: Maximum number of writes to track
        """
        self._max_writes = max_tracked_writes

        # Track recent writes: tool_name -> list of (timestamp, arguments)
        self._writes: dict[str, list[tuple[datetime, dict[str, Any]]]] = defaultdict(
            list
        )

        # Track file paths that have been modified
        self._modified_paths: dict[str, datetime] = {}

    def record_write(
        self,
        tool_name: str,
        arguments: dict[str, Any],
    ) -> None:
        """Record a write operation.

        Args:
            tool_name: Name of the write tool
            arguments: Arguments passed to the tool
        """
        if tool_name not in self.WRITE_TOOLS:
            return

        now = datetime.now(UTC)
        self._writes[tool_name].append((now, arguments.copy()))

        # Track modified path
        path = self._extract_path(arguments)
        if path:
            self._modified_paths[path] = now

        # Trim old entries
        self._trim_writes()

    def get_affected_tools(self, tool_name: str) -> set[str]:
        """Get tools that this write tool affects.

        Args:
            tool_name: Name of write tool

        Returns:
            Set of tool names that should be invalidated
        """
        return self.INVALIDATION_MAP.get(tool_name, set())

    def should_invalidate(
        self,
        signature: ToolCallSignature,
        cached_at: datetime,
    ) -> bool:
        """Check if a cached entry should be invalidated.

        Args:
            signature: The cached tool call signature
            cached_at: When the entry was cached

        Returns:
            True if entry should be invalidated
        """
        path = self._extract_path(signature.arguments)
        if path is None:
            return False

        # Check if path was modified after caching
        modified_at = self._modified_paths.get(path)
        if modified_at and modified_at > cached_at:
            return True

        # Check parent directory modifications for list_directory
        if signature.tool_name == "list_directory":
            for modified_path, modified_time in self._modified_paths.items():
                if modified_time > cached_at and modified_path.startswith(path):
                    return True

        return False

    def get_modified_paths_since(self, since: datetime) -> list[str]:
        """Get paths modified since a timestamp.

        Args:
            since: Timestamp to check from

        Returns:
            List of modified paths
        """
        return [
            path
            for path, modified_at in self._modified_paths.items()
            if modified_at > since
        ]

    def clear(self) -> None:
        """Clear all tracked writes."""
        self._writes.clear()
        self._modified_paths.clear()

    def _extract_path(self, arguments: dict[str, Any]) -> str | None:
        """Extract file path from arguments."""
        for key in ("path", "file_path", "filepath", "file", "directory", "dir"):
            if key in arguments:
                value = arguments[key]
                if isinstance(value, str):
                    return value
        return None

    def _trim_writes(self) -> None:
        """Remove old write entries to prevent unbounded growth."""
        total = sum(len(writes) for writes in self._writes.values())
        if total <= self._max_writes:
            return

        # Remove oldest entries
        all_writes = []
        for tool_name, writes in self._writes.items():
            for timestamp, args in writes:
                all_writes.append((timestamp, tool_name, args))

        all_writes.sort(key=lambda x: x[0])

        # Keep only the most recent
        to_keep = all_writes[-self._max_writes :]

        # Rebuild
        self._writes.clear()
        for timestamp, tool_name, args in to_keep:
            self._writes[tool_name].append((timestamp, args))


class StalenessCalculator:
    """Calculate staleness score for cache entries.

    Staleness is a value from 0 (fresh) to 1 (stale) based on:
    - Age as fraction of TTL
    - Time since last access
    - Number of writes to related resources
    - Access frequency (frequently accessed = more valuable)

    Example:
        >>> calc = StalenessCalculator()
        >>> score = calc.calculate(
        ...     age_seconds=1800,
        ...     ttl_seconds=3600,
        ...     last_access_seconds=600,
        ...     access_count=5,
        ...     writes_since_cache=0,
        ... )
        >>> print(f"Staleness: {score:.2f}")
    """

    def __init__(
        self,
        age_weight: float = 0.4,
        access_weight: float = 0.3,
        write_weight: float = 0.3,
    ) -> None:
        """Initialize StalenessCalculator.

        Args:
            age_weight: Weight for age-based staleness
            access_weight: Weight for access pattern staleness
            write_weight: Weight for write activity staleness
        """
        self._age_weight = age_weight
        self._access_weight = access_weight
        self._write_weight = write_weight

    def calculate(
        self,
        age_seconds: float,
        ttl_seconds: int | None,
        last_access_seconds: float,
        access_count: int,
        writes_since_cache: int,
    ) -> float:
        """Calculate staleness score.

        Args:
            age_seconds: How old the entry is
            ttl_seconds: TTL for the entry (None = no TTL)
            last_access_seconds: Seconds since last access
            access_count: Number of times entry was accessed
            writes_since_cache: Number of relevant writes since caching

        Returns:
            Staleness score from 0 (fresh) to 1 (stale)
        """
        # Age component: approaches 1 as age approaches TTL
        if ttl_seconds is not None and ttl_seconds > 0:
            age_score = min(age_seconds / ttl_seconds, 1.0)
        else:
            # Without TTL, use a default decay over 24 hours
            age_score = min(age_seconds / 86400, 1.0)

        # Access component: recently accessed = less stale
        # Uses exponential decay with half-life of 1 hour
        access_score = 1.0 - (1.0 / (1.0 + last_access_seconds / 3600))
        # Frequently accessed entries are more valuable
        if access_count > 0:
            access_score *= 1.0 / (1.0 + access_count * 0.1)

        # Write component: more writes = more likely stale
        write_score = min(writes_since_cache / 5, 1.0)

        # Weighted combination
        staleness = (
            self._age_weight * age_score
            + self._access_weight * access_score
            + self._write_weight * write_score
        )

        return min(max(staleness, 0.0), 1.0)


class PolicyRegistry:
    """Registry for managing invalidation policies.

    Provides a central place to register and retrieve policies
    for different tools, with support for default policies.

    Example:
        >>> registry = PolicyRegistry()
        >>> registry.register(InvalidationPolicy(
        ...     tool_name="my_tool",
        ...     ttl_seconds=600,
        ...     trigger=InvalidationTrigger.WRITE_THROUGH,
        ... ))
        >>> policy = registry.get("my_tool")
    """

    def __init__(self, use_defaults: bool = True) -> None:
        """Initialize PolicyRegistry.

        Args:
            use_defaults: Whether to load default policies
        """
        self._policies: dict[str, InvalidationPolicy] = {}
        self._default_policy = InvalidationPolicy(
            tool_name="__default__",
            ttl_seconds=3600,
            trigger=InvalidationTrigger.TIME_BASED,
        )

        if use_defaults:
            self._load_defaults()

    def register(self, policy: InvalidationPolicy) -> None:
        """Register a policy for a tool.

        Args:
            policy: The invalidation policy to register
        """
        self._policies[policy.tool_name] = policy

    def get(self, tool_name: str) -> InvalidationPolicy:
        """Get policy for a tool.

        Args:
            tool_name: Name of the tool

        Returns:
            Policy for the tool (or default if not registered)
        """
        if tool_name in self._policies:
            return self._policies[tool_name]

        # Return a copy of default with the tool name set
        return InvalidationPolicy(
            tool_name=tool_name,
            ttl_seconds=self._default_policy.ttl_seconds,
            cache_errors=self._default_policy.cache_errors,
            trigger=self._default_policy.trigger,
        )

    def set_default(self, policy: InvalidationPolicy) -> None:
        """Set the default policy for unregistered tools.

        Args:
            policy: Default policy to use
        """
        self._default_policy = policy

    def list_policies(self) -> list[InvalidationPolicy]:
        """List all registered policies.

        Returns:
            List of registered policies
        """
        return list(self._policies.values())

    def _load_defaults(self) -> None:
        """Load default policies for common tools."""
        for policy in DEFAULT_POLICIES.values():
            self._policies[policy.tool_name] = policy


# Default policies for common tools
DEFAULT_POLICIES: dict[str, InvalidationPolicy] = {
    "read_file": InvalidationPolicy(
        tool_name="read_file",
        ttl_seconds=300,  # 5 minutes - files can change
        trigger=InvalidationTrigger.WRITE_THROUGH,
        invalidation_pattern=r"path",
        dependencies=["write_file", "edit_file", "delete_file"],
    ),
    "search": InvalidationPolicy(
        tool_name="search",
        ttl_seconds=3600,  # 1 hour
        trigger=InvalidationTrigger.TIME_BASED,
        dependencies=["write_file", "create_file", "delete_file"],
    ),
    "grep": InvalidationPolicy(
        tool_name="grep",
        ttl_seconds=3600,  # 1 hour
        trigger=InvalidationTrigger.TIME_BASED,
        dependencies=["write_file", "edit_file"],
    ),
    "web_fetch": InvalidationPolicy(
        tool_name="web_fetch",
        ttl_seconds=1800,  # 30 minutes
        trigger=InvalidationTrigger.TIME_BASED,
        cache_errors=False,
    ),
    "list_directory": InvalidationPolicy(
        tool_name="list_directory",
        ttl_seconds=60,  # 1 minute - directories change often
        trigger=InvalidationTrigger.WRITE_THROUGH,
        dependencies=["create_file", "delete_file", "mkdir", "rmdir"],
    ),
    "api_call": InvalidationPolicy(
        tool_name="api_call",
        ttl_seconds=600,  # 10 minutes
        trigger=InvalidationTrigger.TIME_BASED,
        version_path="data.version",
    ),
    "get_symbol_info": InvalidationPolicy(
        tool_name="get_symbol_info",
        ttl_seconds=300,  # 5 minutes
        trigger=InvalidationTrigger.WRITE_THROUGH,
        dependencies=["write_file", "edit_file"],
    ),
    "run_command": InvalidationPolicy(
        tool_name="run_command",
        ttl_seconds=None,  # No caching by default - commands may have side effects
        cache_errors=False,
        trigger=InvalidationTrigger.MANUAL,
    ),
}
