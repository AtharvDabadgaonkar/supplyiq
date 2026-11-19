"""Provider contract and the recommendation payload."""
from __future__ import annotations

import abc
from dataclasses import dataclass, field, asdict
from typing import Any, Literal

ActionType = Literal[
    "expedite_order",
    "rebalance_stock",
    "switch_supplier",
    "dual_source",
    "adjust_order_qty",
    "no_action",
]

ACTION_TYPES: tuple[ActionType, ...] = (
    "expedite_order",
    "rebalance_stock",
    "switch_supplier",
    "dual_source",
    "adjust_order_qty",
    "no_action",
)

# Every numeric field on a Recommendation is computed by the simulator. Tests
# assert that no provider can write to any of these.
NUMERIC_FIELDS: tuple[str, ...] = (
    "baseline_fill_rate",
    "scenario_fill_rate",
    "action_fill_rate",
    "baseline_stockout_weeks",
    "action_stockout_weeks",
    "baseline_inventory_value",
    "action_inventory_value",
    "baseline_landed_cost",
    "action_landed_cost",
    "action_cost_delta",
    "service_gain_pp",
    "units",
    "lead_time_days_delta",
)


class ProviderUnavailable(RuntimeError):
    """Raised when a provider cannot be used (no key, missing SDK, bad config)."""


@dataclass(frozen=True)
class Recommendation:
    """One proposed action for one at-risk planning item.

    Every field below is produced by the simulator. A provider receives this
    object, reads it, and returns prose. It cannot construct or alter one.
    """

    sku: str
    store: str
    supplier_id: str
    action_type: ActionType
    # --- simulator-computed numbers ----------------------------------------
    baseline_fill_rate: float = 0.0
    scenario_fill_rate: float = 0.0
    action_fill_rate: float = 0.0
    baseline_stockout_weeks: int = 0
    action_stockout_weeks: int = 0
    baseline_inventory_value: float = 0.0
    action_inventory_value: float = 0.0
    baseline_landed_cost: float = 0.0
    action_landed_cost: float = 0.0
    action_cost_delta: float = 0.0
    service_gain_pp: float = 0.0
    units: int = 0
    lead_time_days_delta: float = 0.0
    # --- context -----------------------------------------------------------
    abc_class: str = ""
    xyz_class: str = ""
    risk_reason: str = ""
    scenario: dict[str, Any] = field(default_factory=dict)

    def as_payload(self) -> dict[str, Any]:
        """Read-only dict handed to a provider. No mutation path back."""
        return asdict(self)

    @property
    def cost_per_service_point(self) -> float | None:
        """USD of extra cost per percentage point of fill rate gained."""
        if self.service_gain_pp <= 0:
            return None
        return self.action_cost_delta / self.service_gain_pp


@dataclass(frozen=True)
class Explanation:
    """What a provider returns: prose only, plus how it was produced."""

    text: str
    provider: str
    model: str | None = None
    # True when this came from the rule-based fallback rather than an LLM.
    is_mock: bool = True
    # Set when a live provider failed and the mock answered instead.
    fallback_reason: str | None = None


class AgentProvider(abc.ABC):
    """Turns a computed Recommendation into a short planner-facing explanation."""

    name: str = "base"
    is_mock: bool = True

    @abc.abstractmethod
    def available(self) -> bool:
        """True when this provider can actually be called right now."""

    @abc.abstractmethod
    def explain(self, rec: Recommendation) -> Explanation:
        """Return prose for one recommendation. Must not invent numbers."""

    def explain_many(self, recs: list[Recommendation]) -> list[Explanation]:
        return [self.explain(r) for r in recs]
