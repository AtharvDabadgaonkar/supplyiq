"""ACT — recommendations for at-risk items, priced by the simulator.

The rule this module exists to enforce: **no number here is estimated.** For
each at-risk item we build a candidate action, express it as a set of
`simulate.Overrides`, re-run the inventory twin with that override applied, and
read the service and cost outcome off the result. A recommendation is then the
difference between two simulator runs that shared the same scenario and the
same lead-time draws.

That is why `simulate.item_seed` keys randomness to the item rather than to its
row: pricing one recommendation simulates a one- or two-item slice, and that
slice has to see exactly the supply luck it saw in the full run, or the
comparison is noise.

The language model's only job, downstream in `supplyiq.agent`, is to put the
resulting numbers into a sentence.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np
import pandas as pd

from supplyiq.agent.base import ActionType, Recommendation
from supplyiq.config import load_assumptions
from supplyiq.inventory import apply_order_constraints
from supplyiq.simulate import (
    Overrides,
    Scenario,
    SimulationResult,
    baseline_scenario,
    draw_lead_times,
    simulate,
)

ITEM_KEYS = ["sku", "store"]
ItemKey = tuple[str, str]


# --------------------------------------------------------------------------- #
# Finding the items worth a planner's attention
# --------------------------------------------------------------------------- #
def at_risk(
    result: SimulationResult,
    policy: pd.DataFrame,
    a: dict[str, Any] | None = None,
    top_n: int | None = None,
) -> pd.DataFrame:
    """Rank items by how much service and money the scenario puts at risk.

    Risk value is lost revenue, not lost units: ten units of a $40 hobby item
    matters more than ten units of a $2 food item, and a planner's attention is
    the scarce resource being allocated here.
    """
    a = a or load_assumptions()
    top_n = top_n or int(a["agent"]["at_risk_top_n"])

    merged = result.per_item.merge(
        policy[ITEM_KEYS + ["avg_sell_price", "service_level", "supplier_id", "country",
                            "alternate_supplier_id", "order_quantity", "mean_weekly_demand",
                            "lead_time_mean_weeks", "tariff_rate"]],
        on=ITEM_KEYS,
        how="left",
        suffixes=("", "_policy"),
    )
    merged["revenue_at_risk"] = merged["short_units"] * merged["avg_sell_price"].astype(float)
    merged["service_gap_pp"] = (merged["service_level"] - merged["fill_rate"]) * 100

    flagged = merged[(merged["short_units"] > 0) | (merged["fill_rate"] < merged["service_level"])]
    return (
        flagged.sort_values(["revenue_at_risk", "short_units"], ascending=False)
        .head(top_n)
        .reset_index(drop=True)
    )


def risk_reason(row: pd.Series, scenario: Scenario) -> str:
    """A plain sentence a planner would recognise, built only from computed values."""
    bits = []
    if row["short_units"] > 0:
        bits.append(f"short {row['short_units']:.0f} units")
    if row["stockout_weeks"] > 0:
        bits.append(f"out of stock in {row['stockout_weeks']:.0f} of {row.get('weeks', 12):.0f} weeks")
    gap = row["service_gap_pp"]
    if gap > 0:
        bits.append(f"{gap:.1f}pp below its service target")
    cause = "" if scenario.is_baseline else f" under {scenario.describe().lower()}"
    return (", ".join(bits) if bits else "below its service target") + cause


# --------------------------------------------------------------------------- #
# Candidate actions -> simulator overrides
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Candidate:
    """One proposed action, expressed as something the simulator can run."""

    action_type: ActionType
    overrides: Overrides
    units: int
    lead_time_days_delta: float
    extra_items: tuple[ItemKey, ...] = ()      # other items the action touches
    note: str = ""


def _supplier(a: dict[str, Any], supplier_id: str) -> dict[str, Any]:
    for s in a["suppliers"]["network"]:
        if s["supplier_id"] == supplier_id:
            return s
    raise KeyError(f"unknown supplier {supplier_id!r}")


def build_candidates(
    row: pd.Series,
    policy: pd.DataFrame,
    a: dict[str, Any] | None = None,
) -> list[Candidate]:
    """Every action worth simulating for one at-risk item."""
    a = a or load_assumptions()
    key: ItemKey = (row["sku"], row["store"])
    costs = a["costs"]
    item = policy.set_index(ITEM_KEYS).loc[key]
    base_lead_days = float(item["lead_time_mean_weeks"]) * 7.0
    order_qty = float(item["order_quantity"])
    shortfall = float(row["short_units"])

    out: list[Candidate] = []

    # --- expedite: air-freight the inbound order ---------------------------
    exp = costs["expedite"]
    expedited_lead = base_lead_days * float(exp["lead_time_multiplier"])
    extra_freight = float(item["freight_per_unit"]) * (float(exp["freight_multiplier"]) - 1.0)
    out.append(
        Candidate(
            action_type="expedite_order",
            overrides=Overrides(
                lead_time_days={key: expedited_lead},
                freight_per_unit={key: float(item["freight_per_unit"]) * float(exp["freight_multiplier"])},
            ),
            units=int(round(order_qty)),
            lead_time_days_delta=expedited_lead - base_lead_days,
            note=f"air freight at {exp['freight_multiplier']}x cost, {exp['lead_time_multiplier']}x lead time",
        )
    )

    # --- rebalance: pull stock from the best-covered sibling store ----------
    donor = _best_donor(row, policy)
    if donor is not None:
        donor_key, movable = donor
        move = float(min(movable, max(shortfall, order_qty * 0.25)))
        if move >= 1:
            out.append(
                Candidate(
                    action_type="rebalance_stock",
                    overrides=Overrides(
                        initial_on_hand={
                            key: float(item["reorder_point"]) + order_qty + move,
                            donor_key: _opening_stock(policy, donor_key, a) - move,
                        },
                        extra_cost={key: move * float(costs["rebalance"]["cost_per_unit"])},
                    ),
                    units=int(round(move)),
                    lead_time_days_delta=float(costs["rebalance"]["transit_days"]) - base_lead_days,
                    extra_items=(donor_key,),
                    note=f"move {move:.0f} units from {donor_key[1]}",
                )
            )

    # --- switch supplier: move the SKU to its alternate source --------------
    alternate_id = item.get("alternate_supplier_id")
    if isinstance(alternate_id, str) and alternate_id:
        alt = _supplier(a, alternate_id)
        alt_lead = float(alt["mean_lead_time_days"])
        alt_freight = float(costs["freight_per_unit_by_country"][alt["country"]])
        out.append(
            Candidate(
                action_type="switch_supplier",
                overrides=Overrides(
                    lead_time_days={key: alt_lead},
                    freight_per_unit={key: alt_freight},
                ),
                units=int(round(order_qty)),
                lead_time_days_delta=alt_lead - base_lead_days,
                note=f"source from {alternate_id} ({alt['country']})",
            )
        )

        # --- dual source: split the order across both lanes -----------------
        # Modelled as a 50/50 blend of the two lanes' lead time and freight.
        # A true two-lane simulation would need a second pipeline per item;
        # the blend is an approximation and is labelled as one.
        blended_lead = 0.5 * base_lead_days + 0.5 * alt_lead
        blended_freight = 0.5 * float(item["freight_per_unit"]) + 0.5 * alt_freight
        out.append(
            Candidate(
                action_type="dual_source",
                overrides=Overrides(
                    lead_time_days={key: blended_lead},
                    freight_per_unit={key: blended_freight},
                ),
                units=int(round(order_qty)),
                lead_time_days_delta=blended_lead - base_lead_days,
                note=f"50/50 split with {alternate_id}; blended lane (approximation)",
            )
        )

    # --- adjust order quantity ---------------------------------------------
    bigger = apply_order_constraints(
        order_qty * 1.5, float(item["moq_units"]), float(item["order_multiple_units"])
    )
    if bigger > order_qty:
        out.append(
            Candidate(
                action_type="adjust_order_qty",
                overrides=Overrides(order_quantity={key: bigger}),
                units=int(round(bigger - order_qty)),
                lead_time_days_delta=0.0,
                note=f"raise order quantity {order_qty:.0f} -> {bigger:.0f}",
            )
        )

    out.append(Candidate(action_type="no_action", overrides=Overrides(), units=0,
                         lead_time_days_delta=0.0, note="hold the current plan"))
    return out


def _opening_stock(policy: pd.DataFrame, key: ItemKey, a: dict[str, Any]) -> float:
    item = policy.set_index(ITEM_KEYS).loc[key]
    if a["policy"]["initial_stock_policy"] == "order_up_to":
        return float(item["reorder_point"]) + float(item["order_quantity"])
    return float(item["mean_weekly_demand"]) * float(a["policy"]["initial_stock_weeks_of_cover"])


def _best_donor(row: pd.Series, policy: pd.DataFrame) -> tuple[ItemKey, float] | None:
    """The same SKU at another store with the most cover to spare."""
    siblings = policy[(policy["sku"] == row["sku"]) & (policy["store"] != row["store"])]
    if siblings.empty:
        return None
    best = siblings.sort_values("safety_stock_weeks", ascending=False).iloc[0]
    movable = float(best["safety_stock"]) * 0.5     # never strip a donor below half its buffer
    if movable < 1:
        return None
    return (best["sku"], best["store"]), movable


# --------------------------------------------------------------------------- #
# Pricing a candidate: two simulator runs, same scenario, same draws
# --------------------------------------------------------------------------- #
def _slice(policy: pd.DataFrame, demand: pd.DataFrame, keys: Iterable[ItemKey]):
    keyset = set(keys)
    mask_p = [(s, st) in keyset for s, st in zip(policy["sku"], policy["store"])]
    mask_d = [(s, st) in keyset for s, st in zip(demand["sku"], demand["store"])]
    return (
        policy[mask_p].sort_values(ITEM_KEYS).reset_index(drop=True),
        demand[mask_d].sort_values(ITEM_KEYS + ["week_index"]).reset_index(drop=True),
    )


def price_candidate(
    candidate: Candidate,
    key: ItemKey,
    policy: pd.DataFrame,
    demand: pd.DataFrame,
    scenario: Scenario,
    a: dict[str, Any],
    baseline_kpis: dict[str, float] | None = None,
) -> dict[str, float]:
    """Simulate the affected items with and without the action.

    Only the items the action touches are simulated, which is exact because an
    override provably affects no other item (`test_an_override_touches_only_the
    _item_it_names`) and because lead-time draws are keyed per item.
    """
    keys = [key, *candidate.extra_items]
    sub_policy, sub_demand = _slice(policy, demand, keys)
    weeks = sub_demand["week_index"].nunique()
    draws = draw_lead_times(sub_policy, weeks, int(a["simulation"]["seed"]))

    before = simulate(sub_policy, sub_demand, scenario, a, lead_time_draws=draws)
    after = simulate(
        sub_policy, sub_demand, scenario, a, overrides=candidate.overrides, lead_time_draws=draws
    )

    def pooled(result: SimulationResult, column: str) -> float:
        return float(result.per_item[column].sum())

    demand_units = pooled(before, "demand_units")
    fill_before = pooled(before, "shipped_units") / demand_units if demand_units else 1.0
    fill_after = pooled(after, "shipped_units") / demand_units if demand_units else 1.0

    # Net cost over the window. Goods ordered but not yet consumed are still an
    # asset, so the closing position is credited back; otherwise an action that
    # merely shifts an order across the window boundary reads as a saving.
    # Opening position is identical in both runs, so it cancels.
    def net_cost(result: SimulationResult) -> float:
        return (
            pooled(result, "cost_to_serve")
            + pooled(result, "cost_of_goods_ordered")
            - pooled(result, "ending_position_value")
        )

    cost_before = net_cost(before)
    cost_after = net_cost(after)

    return {
        "scenario_fill_rate": fill_before,
        "action_fill_rate": fill_after,
        "service_gain_pp": (fill_after - fill_before) * 100.0,
        "baseline_stockout_weeks": int(pooled(before, "stockout_weeks")),
        "action_stockout_weeks": int(pooled(after, "stockout_weeks")),
        "baseline_inventory_value": pooled(before, "working_capital"),
        "action_inventory_value": pooled(after, "working_capital"),
        "baseline_landed_cost": cost_before,
        "action_landed_cost": cost_after,
        "action_cost_delta": cost_after - cost_before,
        "short_units_before": pooled(before, "short_units"),
        "short_units_after": pooled(after, "short_units"),
    }


# --------------------------------------------------------------------------- #
# The recommendation set
# --------------------------------------------------------------------------- #
def recommend(
    policy: pd.DataFrame,
    demand: pd.DataFrame,
    scenario: Scenario | None = None,
    a: dict[str, Any] | None = None,
    result: SimulationResult | None = None,
    top_n: int | None = None,
    best_only: bool = True,
) -> list[Recommendation]:
    """Rank at-risk items, price every action for each, return the best per item.

    `best_only=False` returns every priced option, which is what the UI shows
    when a planner wants to see what else was considered.
    """
    a = a or load_assumptions()
    scenario = scenario or baseline_scenario()
    if result is None:
        draws = draw_lead_times(
            policy.sort_values(ITEM_KEYS).reset_index(drop=True),
            demand["week_index"].nunique(),
            int(a["simulation"]["seed"]),
        )
        result = simulate(policy, demand, scenario, a, lead_time_draws=draws)

    risky = at_risk(result, policy, a, top_n)
    baseline_fill = float(result.kpis["fill_rate"])

    out: list[Recommendation] = []
    for _, row in risky.iterrows():
        key: ItemKey = (row["sku"], row["store"])
        priced: list[Recommendation] = []
        for candidate in build_candidates(row, policy, a):
            numbers = price_candidate(candidate, key, policy, demand, scenario, a)
            priced.append(
                Recommendation(
                    sku=row["sku"],
                    store=row["store"],
                    supplier_id=row["supplier_id"],
                    action_type=candidate.action_type,
                    baseline_fill_rate=baseline_fill,
                    units=candidate.units,
                    lead_time_days_delta=candidate.lead_time_days_delta,
                    abc_class=row.get("abc_class", ""),
                    xyz_class=row.get("xyz_class", ""),
                    risk_reason=risk_reason(row, scenario),
                    scenario=scenario.as_dict(),
                    **{
                        k: v
                        for k, v in numbers.items()
                        if k not in ("short_units_before", "short_units_after")
                    },
                )
            )
        out.extend([choose_best(priced)] if best_only else priced)
    return out


def choose_best(options: list[Recommendation]) -> Recommendation:
    """Cheapest service point wins; if nothing buys service, the cheapest option does.

    Deliberately not "largest service gain": a planner has a budget, and an
    action that buys 2pp for $5k is usually a worse call than one that buys
    1.5pp for $400. `no_action` wins when nothing clears the bar.
    """
    helpful = [o for o in options if o.service_gain_pp > 1e-9]
    if not helpful:
        return next((o for o in options if o.action_type == "no_action"), options[0])
    return min(helpful, key=lambda o: (o.cost_per_service_point or float("inf")))


def to_frame(recommendations: list[Recommendation]) -> pd.DataFrame:
    """Recommendations as a table, for the UI and for `make metrics`."""
    if not recommendations:
        return pd.DataFrame()
    rows = []
    for rec in recommendations:
        payload = rec.as_payload()
        payload["scenario"] = rec.scenario.get("name", "baseline")
        payload["cost_per_service_point"] = rec.cost_per_service_point
        rows.append(payload)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Human in the loop: the decision log
# --------------------------------------------------------------------------- #
DECISION_COLUMNS = [
    "decided_at",
    "sku",
    "store",
    "supplier_id",
    "action_type",
    "decision",
    "scenario",
    "units",
    "service_gain_pp",
    "action_cost_delta",
    "cost_per_service_point",
    "scenario_fill_rate",
    "action_fill_rate",
    "explanation",
    "explanation_source",
    "decided_by",
    "note",
]


class DecisionLog:
    """Approve/reject decisions, with the numbers that were on screen at the time.

    Deliberately append-only: the record of what a planner was shown when they
    decided is the point of it, so a later re-run of the simulator must not
    rewrite history.

    On Streamlit Community Cloud the filesystem resets on redeploy, so the app
    keeps this in `st.session_state` and offers a CSV download. That limitation
    is stated in the README rather than papered over.
    """

    def __init__(self, rows: list[dict[str, Any]] | None = None) -> None:
        self._rows: list[dict[str, Any]] = list(rows or [])

    def __len__(self) -> int:
        return len(self._rows)

    def record(
        self,
        rec: Recommendation,
        decision: str,
        explanation: str = "",
        explanation_source: str = "",
        decided_by: str = "planner",
        note: str = "",
        decided_at: str | None = None,
    ) -> dict[str, Any]:
        if decision not in ("approved", "rejected"):
            raise ValueError(f"decision must be 'approved' or 'rejected', got {decision!r}")
        from supplyiq.data import utc_now_iso

        row = {
            "decided_at": decided_at or utc_now_iso(),
            "sku": rec.sku,
            "store": rec.store,
            "supplier_id": rec.supplier_id,
            "action_type": rec.action_type,
            "decision": decision,
            "scenario": rec.scenario.get("name", "baseline"),
            "units": rec.units,
            "service_gain_pp": rec.service_gain_pp,
            "action_cost_delta": rec.action_cost_delta,
            "cost_per_service_point": rec.cost_per_service_point,
            "scenario_fill_rate": rec.scenario_fill_rate,
            "action_fill_rate": rec.action_fill_rate,
            "explanation": explanation,
            "explanation_source": explanation_source,
            "decided_by": decided_by,
            "note": note,
        }
        self._rows.append(row)
        return row

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(self._rows, columns=DECISION_COLUMNS)

    def to_csv(self) -> str:
        return self.to_frame().to_csv(index=False)

    def summary(self) -> dict[str, Any]:
        """What the planner actually committed to."""
        frame = self.to_frame()
        if frame.empty:
            return {
                "decisions": 0,
                "approved": 0,
                "rejected": 0,
                "approval_rate": float("nan"),
                "approved_cost": 0.0,
                "approved_service_gain_pp": 0.0,
            }
        approved = frame[frame["decision"] == "approved"]
        return {
            "decisions": int(len(frame)),
            "approved": int(len(approved)),
            "rejected": int((frame["decision"] == "rejected").sum()),
            "approval_rate": float((frame["decision"] == "approved").mean()),
            "approved_cost": float(approved["action_cost_delta"].sum()),
            "approved_service_gain_pp": float(approved["service_gain_pp"].sum()),
            "by_action": {
                k: int(v) for k, v in approved["action_type"].value_counts().items()
            },
        }


def explain_all(
    recommendations: list[Recommendation], provider=None
) -> list[tuple[Recommendation, Any]]:
    """Attach a written explanation to each recommendation.

    The provider only sees the computed payload and returns prose. Anything it
    writes that contains a number the simulator did not produce is rejected by
    `supplyiq.agent.guard` and replaced with the rule-based text.
    """
    from supplyiq.agent import get_provider

    provider = provider or get_provider()
    return [(rec, provider.explain(rec)) for rec in recommendations]
