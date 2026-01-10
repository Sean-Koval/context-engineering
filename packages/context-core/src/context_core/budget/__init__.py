"""Budget module - Token allocation and monitoring.

Components:
- TokenBudget: Manage token allocation across sections
- BudgetSection: Individual budget allocations
- BudgetAlert: Alert for threshold crossings
- OverflowStrategy: Strategies for handling overflow
- Pre-rot detection: Proactive degradation monitoring
"""

from __future__ import annotations

from context_core.budget.token_budget import (
    BudgetAlert,
    BudgetSection,
    OverflowStrategy,
    TokenBudget,
)

__all__ = [
    "BudgetAlert",
    "BudgetSection",
    "OverflowStrategy",
    "TokenBudget",
]
