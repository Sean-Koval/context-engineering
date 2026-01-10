"""Token budget management with section-based allocation and pre-rot detection.

This module provides:
- BudgetSection: Named section of token budget with allocation tracking
- BudgetAlert: Alert triggered by budget threshold crossings
- OverflowStrategy: Strategies for handling budget overflow
- TokenBudget: Main budget manager with sections and alerts
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime
from enum import Enum
from typing import TYPE_CHECKING

from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from collections.abc import Mapping


class BudgetSection(BaseModel):
    """A named section of the token budget.

    Sections allow allocating portions of the total budget to different
    purposes (e.g., system prompts, conversation history, tools).

    Attributes:
        name: Unique identifier for the section.
        allocated: Maximum tokens for this section.
        used: Currently used tokens (default 0).
        priority: Compression priority (higher = compress last).
    """

    name: str
    allocated: int = Field(ge=0, description="Maximum tokens for this section")
    used: int = Field(default=0, ge=0, description="Currently used tokens")
    priority: int = Field(default=0, description="Higher = compress last")

    @property
    def available(self) -> int:
        """Tokens available in this section."""
        return max(0, self.allocated - self.used)

    @property
    def usage_ratio(self) -> float:
        """Usage ratio for this section (0.0 to 1.0+)."""
        if self.allocated <= 0:
            return 0.0
        return self.used / self.allocated


class BudgetAlert(BaseModel):
    """An alert triggered by budget threshold crossings.

    Alerts are fired when global usage crosses warning or trigger thresholds.

    Attributes:
        level: Alert severity ("warning" or "critical").
        section: Section name, or None for global alerts.
        message: Human-readable alert description.
        usage_ratio: Usage ratio when alert was triggered.
        triggered_at: Timestamp of alert creation.
    """

    level: str = Field(description="Alert level: 'warning' or 'critical'")
    section: str | None = Field(
        default=None, description="Section name or None for global"
    )
    message: str = Field(description="Human-readable alert message")
    usage_ratio: float = Field(ge=0, description="Usage ratio when triggered")
    triggered_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class OverflowStrategy(str, Enum):
    """Strategy for handling budget overflow.

    Determines what happens when an allocation would exceed the budget.
    """

    ERROR = "error"
    """Raise an exception on overflow."""

    TRUNCATE_OLD = "truncate_old"
    """Remove oldest content to make room."""

    COMPRESS = "compress"
    """Trigger compression to free space."""

    OVERFLOW = "overflow"
    """Allow temporary overflow."""


class TokenBudget:
    """Manages token budget allocation across sections.

    TokenBudget provides:
    - Section-based allocation (system, conversation, tools, memory)
    - Pre-rot threshold monitoring with configurable thresholds
    - Overflow handling strategies
    - Alert callbacks for threshold crossings

    The key insight is that context quality degrades before hitting hard
    token limits. Pre-rot thresholds (default: warning at 50%, trigger at 65%)
    allow proactive compression before degradation occurs.

    Example:
        >>> budget = TokenBudget(
        ...     total_tokens=128_000,
        ...     pre_rot_warning=0.50,
        ...     pre_rot_trigger=0.65,
        ...     on_alert=lambda a: print(f"Alert: {a.message}"),
        ... )
        >>> budget.allocate("conversation", 50_000)
        True
        >>> budget.usage_ratio
        0.390625  # 50k / 128k
    """

    DEFAULT_SECTIONS: dict[str, dict[str, float | int]] = {
        "system": {"allocation_ratio": 0.10, "priority": 100},
        "conversation": {"allocation_ratio": 0.60, "priority": 50},
        "tools": {"allocation_ratio": 0.20, "priority": 30},
        "memory": {"allocation_ratio": 0.10, "priority": 20},
    }
    """Default section configuration.

    - system: High priority, 10% allocation for system prompts
    - conversation: Medium priority, 60% allocation for chat history
    - tools: Lower priority, 20% allocation for tool calls/results
    - memory: Lowest priority, 10% allocation for retrieved memories
    """

    def __init__(
        self,
        total_tokens: int,
        pre_rot_warning: float = 0.50,
        pre_rot_trigger: float = 0.65,
        overflow_strategy: OverflowStrategy = OverflowStrategy.COMPRESS,
        on_alert: Callable[[BudgetAlert], None] | None = None,
    ) -> None:
        """Initialize token budget.

        Args:
            total_tokens: Total available tokens for the budget.
            pre_rot_warning: Usage ratio to trigger warning (default 0.50).
            pre_rot_trigger: Usage ratio to trigger compression (default 0.65).
            overflow_strategy: How to handle overflow (default: COMPRESS).
            on_alert: Optional callback for alert notifications.

        Raises:
            ValueError: If thresholds are invalid (trigger must exceed warning).
        """
        if total_tokens <= 0:
            raise ValueError("total_tokens must be positive")
        if not 0 <= pre_rot_warning <= 1:
            raise ValueError("pre_rot_warning must be between 0 and 1")
        if not 0 <= pre_rot_trigger <= 1:
            raise ValueError("pre_rot_trigger must be between 0 and 1")
        if pre_rot_trigger <= pre_rot_warning:
            raise ValueError("pre_rot_trigger must exceed pre_rot_warning")

        self.total_tokens = total_tokens
        self.pre_rot_warning = pre_rot_warning
        self.pre_rot_trigger = pre_rot_trigger
        self.overflow_strategy = overflow_strategy
        self._on_alert = on_alert

        self._sections: dict[str, BudgetSection] = {}
        self._alerts: list[BudgetAlert] = []
        self._warning_fired = False
        self._trigger_fired = False

        # Initialize default sections
        self._init_default_sections()

    def _init_default_sections(self) -> None:
        """Initialize default budget sections based on DEFAULT_SECTIONS."""
        for name, config in self.DEFAULT_SECTIONS.items():
            ratio = config["allocation_ratio"]
            priority = config["priority"]
            self.add_section(
                name=name,
                allocated=int(self.total_tokens * ratio),  # type: ignore[arg-type]
                priority=int(priority),  # type: ignore[arg-type]
            )

    def add_section(
        self,
        name: str,
        allocated: int,
        priority: int = 0,
    ) -> BudgetSection:
        """Add or update a budget section.

        Args:
            name: Unique name for the section.
            allocated: Maximum tokens for the section.
            priority: Compression priority (higher = compress last).

        Returns:
            The created BudgetSection.
        """
        section = BudgetSection(
            name=name,
            allocated=allocated,
            priority=priority,
        )
        self._sections[name] = section
        return section

    def remove_section(self, name: str) -> bool:
        """Remove a budget section.

        Args:
            name: Section name to remove.

        Returns:
            True if section was removed, False if not found.
        """
        if name in self._sections:
            del self._sections[name]
            return True
        return False

    def allocate(
        self,
        section: str,
        tokens: int,
        force: bool = False,
    ) -> bool:
        """Allocate tokens to a section.

        Args:
            section: Section name to allocate to.
            tokens: Number of tokens to allocate (can be negative to deallocate).
            force: If True, allow overflow regardless of strategy.

        Returns:
            True if allocation succeeded, False if would overflow.

        Raises:
            ValueError: If section doesn't exist, or ERROR strategy and overflow.
        """
        if section not in self._sections:
            raise ValueError(f"Unknown section: {section}")

        sec = self._sections[section]
        new_used = sec.used + tokens

        # Prevent negative usage
        if new_used < 0:
            new_used = 0

        # Check for overflow
        if new_used > sec.allocated and not force:
            if self.overflow_strategy == OverflowStrategy.ERROR:
                raise ValueError(
                    f"Section '{section}' would overflow: "
                    f"{new_used} > {sec.allocated} allocated"
                )
            if self.overflow_strategy not in (
                OverflowStrategy.OVERFLOW,
                OverflowStrategy.TRUNCATE_OLD,
            ):
                # COMPRESS strategy - return False to signal need for compression
                return False

        sec.used = new_used
        self._check_thresholds()
        return True

    def deallocate(self, section: str, tokens: int) -> None:
        """Release tokens from a section.

        Args:
            section: Section name to deallocate from.
            tokens: Number of tokens to release.

        Raises:
            ValueError: If section doesn't exist.
        """
        if section not in self._sections:
            raise ValueError(f"Unknown section: {section}")

        sec = self._sections[section]
        sec.used = max(0, sec.used - tokens)

        # Reset trigger states if we've dropped below thresholds
        if self.usage_ratio < self.pre_rot_warning:
            self._warning_fired = False
        if self.usage_ratio < self.pre_rot_trigger:
            self._trigger_fired = False

    def set_section_usage(self, section: str, tokens: int) -> None:
        """Set the absolute usage of a section.

        Args:
            section: Section name.
            tokens: Absolute token usage to set.

        Raises:
            ValueError: If section doesn't exist or tokens is negative.
        """
        if section not in self._sections:
            raise ValueError(f"Unknown section: {section}")
        if tokens < 0:
            raise ValueError("tokens must be non-negative")

        self._sections[section].used = tokens
        self._check_thresholds()

    def _check_thresholds(self) -> None:
        """Check and fire alerts for threshold crossings."""
        ratio = self.usage_ratio

        if ratio >= self.pre_rot_trigger and not self._trigger_fired:
            self._trigger_fired = True
            alert = BudgetAlert(
                level="critical",
                section=None,
                message=f"Pre-rot trigger reached: {ratio:.1%} usage",
                usage_ratio=ratio,
            )
            self._alerts.append(alert)
            if self._on_alert:
                self._on_alert(alert)

        elif ratio >= self.pre_rot_warning and not self._warning_fired:
            self._warning_fired = True
            alert = BudgetAlert(
                level="warning",
                section=None,
                message=f"Pre-rot warning: {ratio:.1%} usage",
                usage_ratio=ratio,
            )
            self._alerts.append(alert)
            if self._on_alert:
                self._on_alert(alert)

    @property
    def sections(self) -> Mapping[str, BudgetSection]:
        """Read-only view of all sections."""
        return self._sections

    @property
    def alerts(self) -> list[BudgetAlert]:
        """List of all alerts triggered."""
        return list(self._alerts)

    @property
    def used_tokens(self) -> int:
        """Total tokens used across all sections."""
        return sum(s.used for s in self._sections.values())

    @property
    def available_tokens(self) -> int:
        """Total tokens available (may be negative if over budget)."""
        return self.total_tokens - self.used_tokens

    @property
    def usage_ratio(self) -> float:
        """Overall usage ratio (0.0 to 1.0+, can exceed 1.0 if over budget)."""
        if self.total_tokens <= 0:
            return 0.0
        return self.used_tokens / self.total_tokens

    @property
    def is_pre_rot_warning(self) -> bool:
        """True if usage has crossed warning threshold."""
        return self.usage_ratio >= self.pre_rot_warning

    @property
    def is_pre_rot_trigger(self) -> bool:
        """True if usage has crossed trigger threshold."""
        return self.usage_ratio >= self.pre_rot_trigger

    def get_section(self, name: str) -> BudgetSection | None:
        """Get a section by name.

        Args:
            name: Section name.

        Returns:
            The BudgetSection if found, None otherwise.
        """
        return self._sections.get(name)

    def get_compression_priority(self) -> list[str]:
        """Get sections in compression priority order.

        Lower priority sections should be compressed first.

        Returns:
            List of section names, lowest priority first.
        """
        return [
            s.name for s in sorted(self._sections.values(), key=lambda s: s.priority)
        ]

    def reset_alerts(self) -> None:
        """Reset alert state, allowing alerts to fire again."""
        self._warning_fired = False
        self._trigger_fired = False
        self._alerts.clear()

    def to_dict(self) -> dict:
        """Serialize budget state to a dictionary.

        Returns:
            Dictionary representation of the budget state.
        """
        return {
            "total_tokens": self.total_tokens,
            "pre_rot_warning": self.pre_rot_warning,
            "pre_rot_trigger": self.pre_rot_trigger,
            "overflow_strategy": self.overflow_strategy.value,
            "sections": {
                name: sec.model_dump() for name, sec in self._sections.items()
            },
            "alerts": [a.model_dump(mode="json") for a in self._alerts],
            "used_tokens": self.used_tokens,
            "usage_ratio": self.usage_ratio,
        }

    @classmethod
    def from_dict(
        cls,
        data: dict,
        on_alert: Callable[[BudgetAlert], None] | None = None,
    ) -> TokenBudget:
        """Deserialize budget from a dictionary.

        Args:
            data: Dictionary from to_dict().
            on_alert: Optional callback for new alerts.

        Returns:
            Reconstructed TokenBudget instance.
        """
        budget = cls(
            total_tokens=data["total_tokens"],
            pre_rot_warning=data["pre_rot_warning"],
            pre_rot_trigger=data["pre_rot_trigger"],
            overflow_strategy=OverflowStrategy(data["overflow_strategy"]),
            on_alert=on_alert,
        )

        # Clear default sections and restore saved ones
        budget._sections.clear()
        for name, sec_data in data.get("sections", {}).items():
            budget._sections[name] = BudgetSection.model_validate(sec_data)

        # Restore alerts
        budget._alerts = [BudgetAlert.model_validate(a) for a in data.get("alerts", [])]

        # Restore trigger states based on current usage
        if budget.usage_ratio >= budget.pre_rot_warning:
            budget._warning_fired = True
        if budget.usage_ratio >= budget.pre_rot_trigger:
            budget._trigger_fired = True

        return budget

    def summary(self) -> str:
        """Generate human-readable budget summary.

        Returns:
            Multi-line string describing budget state.
        """
        status = "OK"
        if self.is_pre_rot_trigger:
            status = "CRITICAL"
        elif self.is_pre_rot_warning:
            status = "WARNING"

        lines = [
            f"Token Budget: {self.used_tokens:,} / {self.total_tokens:,} "
            f"({self.usage_ratio:.1%})",
            f"Status: {status}",
            "",
            "Sections:",
        ]

        for sec in sorted(self._sections.values(), key=lambda s: -s.priority):
            sec_status = "OVER" if sec.used > sec.allocated else "OK"
            lines.append(
                f"  {sec.name}: {sec.used:,} / {sec.allocated:,} "
                f"({sec.usage_ratio:.1%}) [{sec_status}]"
            )

        return "\n".join(lines)
