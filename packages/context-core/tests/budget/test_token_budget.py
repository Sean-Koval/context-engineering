"""Tests for token budget management."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from context_core.budget import (
    BudgetAlert,
    BudgetSection,
    OverflowStrategy,
    TokenBudget,
)


class TestBudgetSection:
    """Tests for BudgetSection model."""

    def test_basic_creation(self) -> None:
        """Can create a basic budget section."""
        section = BudgetSection(name="test", allocated=1000)
        assert section.name == "test"
        assert section.allocated == 1000
        assert section.used == 0
        assert section.priority == 0

    def test_with_all_fields(self) -> None:
        """Can create section with all fields."""
        section = BudgetSection(
            name="conversation",
            allocated=5000,
            used=2500,
            priority=50,
        )
        assert section.name == "conversation"
        assert section.allocated == 5000
        assert section.used == 2500
        assert section.priority == 50

    def test_available_property(self) -> None:
        """available returns allocated - used."""
        section = BudgetSection(name="test", allocated=1000, used=400)
        assert section.available == 600

    def test_available_not_negative(self) -> None:
        """available never goes below 0."""
        section = BudgetSection(name="test", allocated=1000, used=1500)
        assert section.available == 0

    def test_usage_ratio(self) -> None:
        """usage_ratio calculates correctly."""
        section = BudgetSection(name="test", allocated=1000, used=250)
        assert section.usage_ratio == 0.25

    def test_usage_ratio_zero_allocation(self) -> None:
        """usage_ratio is 0.0 when allocated is 0."""
        section = BudgetSection(name="test", allocated=0, used=0)
        assert section.usage_ratio == 0.0

    def test_usage_ratio_over_one(self) -> None:
        """usage_ratio can exceed 1.0 if over budget."""
        section = BudgetSection(name="test", allocated=1000, used=1500)
        assert section.usage_ratio == 1.5

    def test_validates_allocated_non_negative(self) -> None:
        """allocated must be non-negative."""
        with pytest.raises(ValidationError):
            BudgetSection(name="test", allocated=-100)

    def test_validates_used_non_negative(self) -> None:
        """used must be non-negative."""
        with pytest.raises(ValidationError):
            BudgetSection(name="test", allocated=1000, used=-100)


class TestBudgetAlert:
    """Tests for BudgetAlert model."""

    def test_basic_creation(self) -> None:
        """Can create a basic alert."""
        alert = BudgetAlert(
            level="warning",
            message="Test warning",
            usage_ratio=0.55,
        )
        assert alert.level == "warning"
        assert alert.section is None
        assert alert.message == "Test warning"
        assert alert.usage_ratio == 0.55

    def test_with_section(self) -> None:
        """Can create alert for specific section."""
        alert = BudgetAlert(
            level="critical",
            section="conversation",
            message="Section critical",
            usage_ratio=0.9,
        )
        assert alert.section == "conversation"

    def test_triggered_at_auto_set(self) -> None:
        """triggered_at is automatically set."""
        before = datetime.now(UTC)
        alert = BudgetAlert(
            level="warning",
            message="Test",
            usage_ratio=0.5,
        )
        after = datetime.now(UTC)
        assert before <= alert.triggered_at <= after

    def test_validates_usage_ratio_non_negative(self) -> None:
        """usage_ratio must be non-negative."""
        with pytest.raises(ValidationError):
            BudgetAlert(
                level="warning",
                message="Test",
                usage_ratio=-0.1,
            )


class TestOverflowStrategy:
    """Tests for OverflowStrategy enum."""

    def test_values(self) -> None:
        """All expected strategies exist."""
        assert OverflowStrategy.ERROR.value == "error"
        assert OverflowStrategy.TRUNCATE_OLD.value == "truncate_old"
        assert OverflowStrategy.COMPRESS.value == "compress"
        assert OverflowStrategy.OVERFLOW.value == "overflow"

    def test_from_string(self) -> None:
        """Can create from string value."""
        assert OverflowStrategy("error") == OverflowStrategy.ERROR
        assert OverflowStrategy("compress") == OverflowStrategy.COMPRESS


class TestTokenBudget:
    """Tests for TokenBudget class."""

    def test_basic_creation(self) -> None:
        """Can create budget with defaults."""
        budget = TokenBudget(total_tokens=100_000)
        assert budget.total_tokens == 100_000
        assert budget.pre_rot_warning == 0.50
        assert budget.pre_rot_trigger == 0.65
        assert budget.overflow_strategy == OverflowStrategy.COMPRESS

    def test_custom_thresholds(self) -> None:
        """Can set custom thresholds."""
        budget = TokenBudget(
            total_tokens=100_000,
            pre_rot_warning=0.40,
            pre_rot_trigger=0.60,
        )
        assert budget.pre_rot_warning == 0.40
        assert budget.pre_rot_trigger == 0.60

    def test_invalid_total_tokens(self) -> None:
        """total_tokens must be positive."""
        with pytest.raises(ValueError, match="must be positive"):
            TokenBudget(total_tokens=0)
        with pytest.raises(ValueError, match="must be positive"):
            TokenBudget(total_tokens=-1000)

    def test_invalid_warning_threshold(self) -> None:
        """pre_rot_warning must be between 0 and 1."""
        with pytest.raises(ValueError, match="between 0 and 1"):
            TokenBudget(total_tokens=1000, pre_rot_warning=-0.1)
        with pytest.raises(ValueError, match="between 0 and 1"):
            TokenBudget(total_tokens=1000, pre_rot_warning=1.5)

    def test_invalid_trigger_threshold(self) -> None:
        """pre_rot_trigger must be between 0 and 1."""
        with pytest.raises(ValueError, match="between 0 and 1"):
            TokenBudget(total_tokens=1000, pre_rot_trigger=-0.1)
        with pytest.raises(ValueError, match="between 0 and 1"):
            TokenBudget(total_tokens=1000, pre_rot_trigger=1.5)

    def test_trigger_must_exceed_warning(self) -> None:
        """pre_rot_trigger must exceed pre_rot_warning."""
        with pytest.raises(ValueError, match="must exceed"):
            TokenBudget(
                total_tokens=1000,
                pre_rot_warning=0.60,
                pre_rot_trigger=0.50,
            )
        with pytest.raises(ValueError, match="must exceed"):
            TokenBudget(
                total_tokens=1000,
                pre_rot_warning=0.50,
                pre_rot_trigger=0.50,
            )


class TestTokenBudgetDefaultSections:
    """Tests for default section initialization."""

    def test_default_sections_created(self) -> None:
        """Default sections are created on init."""
        budget = TokenBudget(total_tokens=100_000)
        assert "system" in budget.sections
        assert "conversation" in budget.sections
        assert "tools" in budget.sections
        assert "memory" in budget.sections

    def test_default_section_allocations(self) -> None:
        """Default sections have correct allocations."""
        budget = TokenBudget(total_tokens=100_000)

        # system: 10%
        assert budget.sections["system"].allocated == 10_000
        # conversation: 60%
        assert budget.sections["conversation"].allocated == 60_000
        # tools: 20%
        assert budget.sections["tools"].allocated == 20_000
        # memory: 10%
        assert budget.sections["memory"].allocated == 10_000

    def test_default_section_priorities(self) -> None:
        """Default sections have correct priorities."""
        budget = TokenBudget(total_tokens=100_000)

        assert budget.sections["system"].priority == 100
        assert budget.sections["conversation"].priority == 50
        assert budget.sections["tools"].priority == 30
        assert budget.sections["memory"].priority == 20


class TestTokenBudgetSectionManagement:
    """Tests for section add/remove/get operations."""

    def test_add_section(self) -> None:
        """Can add a new section."""
        budget = TokenBudget(total_tokens=100_000)
        section = budget.add_section("custom", allocated=5000, priority=10)

        assert section.name == "custom"
        assert section.allocated == 5000
        assert "custom" in budget.sections

    def test_add_section_replaces_existing(self) -> None:
        """Adding section with same name replaces it."""
        budget = TokenBudget(total_tokens=100_000)
        budget.add_section("custom", allocated=5000)
        budget.add_section("custom", allocated=10000)

        assert budget.sections["custom"].allocated == 10000

    def test_remove_section(self) -> None:
        """Can remove a section."""
        budget = TokenBudget(total_tokens=100_000)
        budget.add_section("custom", allocated=5000)

        result = budget.remove_section("custom")
        assert result is True
        assert "custom" not in budget.sections

    def test_remove_nonexistent_section(self) -> None:
        """Removing nonexistent section returns False."""
        budget = TokenBudget(total_tokens=100_000)
        result = budget.remove_section("nonexistent")
        assert result is False

    def test_get_section(self) -> None:
        """Can get a section by name."""
        budget = TokenBudget(total_tokens=100_000)
        section = budget.get_section("conversation")
        assert section is not None
        assert section.name == "conversation"

    def test_get_nonexistent_section(self) -> None:
        """Getting nonexistent section returns None."""
        budget = TokenBudget(total_tokens=100_000)
        section = budget.get_section("nonexistent")
        assert section is None


class TestTokenBudgetAllocation:
    """Tests for token allocation operations."""

    def test_allocate_basic(self) -> None:
        """Can allocate tokens to a section."""
        budget = TokenBudget(total_tokens=100_000)
        result = budget.allocate("conversation", 30_000)

        assert result is True
        assert budget.sections["conversation"].used == 30_000

    def test_allocate_cumulative(self) -> None:
        """Allocations are cumulative."""
        budget = TokenBudget(total_tokens=100_000)
        budget.allocate("conversation", 10_000)
        budget.allocate("conversation", 15_000)

        assert budget.sections["conversation"].used == 25_000

    def test_allocate_unknown_section_raises(self) -> None:
        """Allocating to unknown section raises ValueError."""
        budget = TokenBudget(total_tokens=100_000)
        with pytest.raises(ValueError, match="Unknown section"):
            budget.allocate("nonexistent", 1000)

    def test_allocate_overflow_compress_returns_false(self) -> None:
        """COMPRESS strategy returns False on overflow."""
        budget = TokenBudget(
            total_tokens=100_000,
            overflow_strategy=OverflowStrategy.COMPRESS,
        )
        # Try to allocate more than section allows
        result = budget.allocate("system", 20_000)  # > 10_000 allocated
        assert result is False
        assert budget.sections["system"].used == 0  # Not allocated

    def test_allocate_overflow_error_raises(self) -> None:
        """ERROR strategy raises ValueError on overflow."""
        budget = TokenBudget(
            total_tokens=100_000,
            overflow_strategy=OverflowStrategy.ERROR,
        )
        with pytest.raises(ValueError, match="would overflow"):
            budget.allocate("system", 20_000)

    def test_allocate_overflow_allowed(self) -> None:
        """OVERFLOW strategy allows exceeding allocation."""
        budget = TokenBudget(
            total_tokens=100_000,
            overflow_strategy=OverflowStrategy.OVERFLOW,
        )
        result = budget.allocate("system", 20_000)
        assert result is True
        assert budget.sections["system"].used == 20_000

    def test_allocate_force_ignores_strategy(self) -> None:
        """force=True ignores overflow strategy."""
        budget = TokenBudget(
            total_tokens=100_000,
            overflow_strategy=OverflowStrategy.COMPRESS,
        )
        result = budget.allocate("system", 20_000, force=True)
        assert result is True
        assert budget.sections["system"].used == 20_000

    def test_allocate_negative_clamps_to_zero(self) -> None:
        """Negative allocation clamps usage to zero."""
        budget = TokenBudget(total_tokens=100_000)
        budget.allocate("conversation", 10_000)
        budget.allocate("conversation", -20_000)

        assert budget.sections["conversation"].used == 0


class TestTokenBudgetDeallocation:
    """Tests for token deallocation operations."""

    def test_deallocate_basic(self) -> None:
        """Can deallocate tokens from a section."""
        budget = TokenBudget(total_tokens=100_000)
        budget.allocate("conversation", 30_000, force=True)
        budget.deallocate("conversation", 10_000)

        assert budget.sections["conversation"].used == 20_000

    def test_deallocate_unknown_section_raises(self) -> None:
        """Deallocating from unknown section raises ValueError."""
        budget = TokenBudget(total_tokens=100_000)
        with pytest.raises(ValueError, match="Unknown section"):
            budget.deallocate("nonexistent", 1000)

    def test_deallocate_clamps_to_zero(self) -> None:
        """Deallocation doesn't go below zero."""
        budget = TokenBudget(total_tokens=100_000)
        budget.allocate("conversation", 1000, force=True)
        budget.deallocate("conversation", 5000)

        assert budget.sections["conversation"].used == 0

    def test_set_section_usage(self) -> None:
        """Can set absolute usage for a section."""
        budget = TokenBudget(total_tokens=100_000)
        budget.set_section_usage("conversation", 25_000)

        assert budget.sections["conversation"].used == 25_000

    def test_set_section_usage_unknown_raises(self) -> None:
        """Setting usage on unknown section raises ValueError."""
        budget = TokenBudget(total_tokens=100_000)
        with pytest.raises(ValueError, match="Unknown section"):
            budget.set_section_usage("nonexistent", 1000)

    def test_set_section_usage_negative_raises(self) -> None:
        """Setting negative usage raises ValueError."""
        budget = TokenBudget(total_tokens=100_000)
        with pytest.raises(ValueError, match="must be non-negative"):
            budget.set_section_usage("conversation", -1000)


class TestTokenBudgetProperties:
    """Tests for budget property calculations."""

    def test_used_tokens(self) -> None:
        """used_tokens sums all sections."""
        budget = TokenBudget(total_tokens=100_000)
        budget.allocate("system", 5_000, force=True)
        budget.allocate("conversation", 30_000, force=True)
        budget.allocate("tools", 10_000, force=True)

        assert budget.used_tokens == 45_000

    def test_available_tokens(self) -> None:
        """available_tokens is total - used."""
        budget = TokenBudget(total_tokens=100_000)
        budget.allocate("conversation", 40_000, force=True)

        assert budget.available_tokens == 60_000

    def test_available_tokens_can_be_negative(self) -> None:
        """available_tokens can go negative if over budget."""
        budget = TokenBudget(
            total_tokens=100_000,
            overflow_strategy=OverflowStrategy.OVERFLOW,
        )
        budget.allocate("conversation", 120_000, force=True)

        assert budget.available_tokens == -20_000

    def test_usage_ratio(self) -> None:
        """usage_ratio is used / total."""
        budget = TokenBudget(total_tokens=100_000)
        budget.allocate("conversation", 50_000, force=True)

        assert budget.usage_ratio == 0.5


class TestTokenBudgetPreRotDetection:
    """Tests for pre-rot threshold detection."""

    def test_is_pre_rot_warning_false_initially(self) -> None:
        """is_pre_rot_warning is False at low usage."""
        budget = TokenBudget(total_tokens=100_000)
        assert budget.is_pre_rot_warning is False

    def test_is_pre_rot_warning_true_at_threshold(self) -> None:
        """is_pre_rot_warning is True at/above warning threshold."""
        budget = TokenBudget(total_tokens=100_000, pre_rot_warning=0.50)
        budget.allocate("conversation", 50_000, force=True)

        assert budget.is_pre_rot_warning is True

    def test_is_pre_rot_trigger_false_initially(self) -> None:
        """is_pre_rot_trigger is False at low usage."""
        budget = TokenBudget(total_tokens=100_000)
        assert budget.is_pre_rot_trigger is False

    def test_is_pre_rot_trigger_true_at_threshold(self) -> None:
        """is_pre_rot_trigger is True at/above trigger threshold."""
        budget = TokenBudget(total_tokens=100_000, pre_rot_trigger=0.65)
        budget.allocate("conversation", 65_000, force=True)

        assert budget.is_pre_rot_trigger is True


class TestTokenBudgetAlerts:
    """Tests for alert generation."""

    def test_warning_alert_fired(self) -> None:
        """Warning alert is fired when crossing warning threshold."""
        alerts_received: list[BudgetAlert] = []
        budget = TokenBudget(
            total_tokens=100_000,
            pre_rot_warning=0.50,
            on_alert=alerts_received.append,
        )

        budget.allocate("conversation", 50_000, force=True)

        assert len(alerts_received) == 1
        assert alerts_received[0].level == "warning"
        assert "50.0%" in alerts_received[0].message

    def test_critical_alert_fired(self) -> None:
        """Critical alert is fired when crossing trigger threshold."""
        alerts_received: list[BudgetAlert] = []
        budget = TokenBudget(
            total_tokens=100_000,
            pre_rot_warning=0.50,
            pre_rot_trigger=0.65,
            on_alert=alerts_received.append,
        )

        # First cross warning threshold
        budget.allocate("conversation", 52_000, force=True)  # 52% > 50%
        assert len(alerts_received) == 1
        assert alerts_received[0].level == "warning"

        # Then cross trigger threshold
        budget.allocate("conversation", 15_000, force=True)  # 67% > 65%
        assert len(alerts_received) == 2
        assert alerts_received[1].level == "critical"

    def test_warning_fired_only_once(self) -> None:
        """Warning alert is only fired once per threshold crossing."""
        alerts_received: list[BudgetAlert] = []
        budget = TokenBudget(
            total_tokens=100_000,
            pre_rot_warning=0.50,
            pre_rot_trigger=0.90,  # High so we don't trigger it
            on_alert=alerts_received.append,
        )

        budget.allocate("conversation", 25_000, force=True)  # 25%
        budget.allocate("conversation", 30_000, force=True)  # 55%

        warning_alerts = [a for a in alerts_received if a.level == "warning"]
        assert len(warning_alerts) == 1

    def test_alerts_reset_after_dropping_below_threshold(self) -> None:
        """Alerts can fire again after usage drops below threshold."""
        alerts_received: list[BudgetAlert] = []
        budget = TokenBudget(
            total_tokens=100_000,
            pre_rot_warning=0.50,
            pre_rot_trigger=0.90,
            on_alert=alerts_received.append,
        )

        # Cross warning threshold
        budget.allocate("conversation", 55_000, force=True)
        assert len(alerts_received) == 1

        # Drop below threshold
        budget.deallocate("conversation", 20_000)

        # Cross again
        budget.allocate("conversation", 20_000, force=True)
        assert len(alerts_received) == 2

    def test_reset_alerts(self) -> None:
        """reset_alerts clears alert history."""
        budget = TokenBudget(total_tokens=100_000)
        budget.allocate("conversation", 65_000, force=True)

        assert len(budget.alerts) > 0

        budget.reset_alerts()

        assert len(budget.alerts) == 0


class TestTokenBudgetCompressionPriority:
    """Tests for compression priority ordering."""

    def test_get_compression_priority_order(self) -> None:
        """Sections are returned in priority order (lowest first)."""
        budget = TokenBudget(total_tokens=100_000)
        priority_order = budget.get_compression_priority()

        # memory (20) < tools (30) < conversation (50) < system (100)
        assert priority_order == ["memory", "tools", "conversation", "system"]

    def test_get_compression_priority_with_custom_sections(self) -> None:
        """Custom sections are included in priority order."""
        budget = TokenBudget(total_tokens=100_000)
        budget.add_section("cache", allocated=5000, priority=5)

        priority_order = budget.get_compression_priority()
        assert priority_order[0] == "cache"  # Lowest priority = first


class TestTokenBudgetSerialization:
    """Tests for serialization/deserialization."""

    def test_to_dict(self) -> None:
        """to_dict includes all state."""
        budget = TokenBudget(
            total_tokens=100_000,
            pre_rot_warning=0.45,
            pre_rot_trigger=0.70,
            overflow_strategy=OverflowStrategy.ERROR,
        )
        budget.allocate("conversation", 30_000, force=True)

        data = budget.to_dict()

        assert data["total_tokens"] == 100_000
        assert data["pre_rot_warning"] == 0.45
        assert data["pre_rot_trigger"] == 0.70
        assert data["overflow_strategy"] == "error"
        assert "conversation" in data["sections"]
        assert data["sections"]["conversation"]["used"] == 30_000
        assert data["used_tokens"] == 30_000
        assert data["usage_ratio"] == 0.3

    def test_from_dict_roundtrip(self) -> None:
        """Can roundtrip through serialization."""
        original = TokenBudget(
            total_tokens=100_000,
            pre_rot_warning=0.45,
            pre_rot_trigger=0.70,
        )
        original.allocate("conversation", 30_000, force=True)
        original.add_section("custom", allocated=5000, priority=15)
        original.allocate("custom", 2000, force=True)

        data = original.to_dict()
        restored = TokenBudget.from_dict(data)

        assert restored.total_tokens == 100_000
        assert restored.pre_rot_warning == 0.45
        assert restored.pre_rot_trigger == 0.70
        assert restored.sections["conversation"].used == 30_000
        assert "custom" in restored.sections
        assert restored.sections["custom"].used == 2000

    def test_from_dict_restores_threshold_states(self) -> None:
        """Deserialization restores threshold fired states."""
        original = TokenBudget(total_tokens=100_000)
        original.allocate("conversation", 55_000, force=True)  # Cross warning

        data = original.to_dict()
        restored = TokenBudget.from_dict(data)

        assert restored._warning_fired is True


class TestTokenBudgetSummary:
    """Tests for human-readable summary."""

    def test_summary_ok_status(self) -> None:
        """Summary shows OK for low usage."""
        budget = TokenBudget(total_tokens=100_000)
        budget.allocate("conversation", 10_000, force=True)

        summary = budget.summary()
        assert "Status: OK" in summary
        assert "10,000" in summary  # used tokens

    def test_summary_warning_status(self) -> None:
        """Summary shows WARNING at warning threshold."""
        budget = TokenBudget(total_tokens=100_000)
        budget.allocate("conversation", 55_000, force=True)

        summary = budget.summary()
        assert "Status: WARNING" in summary

    def test_summary_critical_status(self) -> None:
        """Summary shows CRITICAL at trigger threshold."""
        budget = TokenBudget(total_tokens=100_000)
        budget.allocate("conversation", 70_000, force=True)

        summary = budget.summary()
        assert "Status: CRITICAL" in summary

    def test_summary_includes_sections(self) -> None:
        """Summary lists all sections with usage."""
        budget = TokenBudget(total_tokens=100_000)
        summary = budget.summary()

        assert "system:" in summary
        assert "conversation:" in summary
        assert "tools:" in summary
        assert "memory:" in summary

    def test_summary_shows_over_budget_sections(self) -> None:
        """Summary marks over-budget sections."""
        budget = TokenBudget(total_tokens=100_000)
        budget.allocate("system", 15_000, force=True)  # > 10_000 allocated

        summary = budget.summary()
        assert "[OVER]" in summary
