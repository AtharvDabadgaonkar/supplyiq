"""MEASURE — does the SupplyIQ policy actually beat what a planner does today?

The comparison
--------------
Both policies are run through the SAME simulator, over the SAME holdout weeks,
with the SAME lead-time draws. The only thing that differs is how the reorder
point and order quantity were set.

  current   naive forecast (next week = last week's sales) and a FLAT number of
            weeks of cover for every SKU, regardless of demand variability or
            how long and erratic the supply lane is. This is not a strawman:
            four weeks of cover is generous against this network's median 2.3
            week lead time. It is thin only on the long overseas lanes -- which
            is exactly the blind spot a flat rule has.

  supplyiq  ETS demand statistics, ABC-differentiated service levels, and
            safety stock that accounts for demand AND lead-time variability.

Why the comparison is between FRONTIERS, not points
---------------------------------------------------
"We cut inventory 20%" means nothing if service fell, and "we lifted service"
means nothing if inventory rose. Those are the two ways this kind of result is
usually oversold.

Comparing the two policies at their default settings is also not enough: it
compares one arbitrary point on each curve. So both policies are swept --
SupplyIQ across service-level targets, the current rule across weeks of cover --
and the question asked at every point is: to reach the fill rate this flat rule
achieves, how much working capital does each one need?

Note what the sweep turned up. Even with its service level driven down to 0.50,
where z = 0 and safety stock vanishes entirely, the SupplyIQ policy still beats
the flat rule's fill rate. The gap therefore is NOT safety stock. It is the
reorder point: SupplyIQ reorders against each item's own lead time plus review
period, while a flat four weeks of cover under-protects exactly the long
overseas lanes. That is a more interesting finding than an inventory
percentage, and it is the one the backtest reports.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from supplyiq.config import load_assumptions
from supplyiq.inventory import apply_order_constraints, build_policy
from supplyiq.simulate import (
    SimulationResult,
    baseline_scenario,
    draw_lead_times,
    simulate,
)

ITEM_KEYS = ["sku", "store"]
WEEKS_PER_YEAR = 52.0


# --------------------------------------------------------------------------- #
# The policy a planner runs today
# --------------------------------------------------------------------------- #
def naive_demand_rate(train: pd.DataFrame, a: dict[str, Any] | None = None) -> pd.DataFrame:
    """Next week = last week's sales, per the configured baseline forecast.

    A single week is noisy, which is the point: it is what the baseline rule
    actually uses, and the noise is part of what the comparison is measuring.
    """
    a = a or load_assumptions()
    method = str(a["baseline_policy"]["forecast"])
    if method != "naive":
        raise ValueError(f"baseline_policy.forecast: only 'naive' is implemented, got {method!r}")
    last_week = int(train["week_index"].max())
    rate = (
        train[train["week_index"] == last_week]
        .groupby(ITEM_KEYS, observed=True)["units_sold"]
        .sum()
        .rename("naive_rate")
        .reset_index()
    )
    # An item that happened to sell nothing in that one week still needs stock;
    # fall back to its own mean so the baseline is not sabotaged by one zero.
    fallback = (
        train.groupby(ITEM_KEYS, observed=True)["units_sold"].mean().rename("mean_rate").reset_index()
    )
    merged = rate.merge(fallback, on=ITEM_KEYS, how="outer")
    merged["naive_rate"] = merged["naive_rate"].fillna(0.0)
    merged["demand_rate"] = np.where(
        merged["naive_rate"] > 0, merged["naive_rate"], merged["mean_rate"]
    )
    return merged[ITEM_KEYS + ["demand_rate"]]


def current_policy(
    supplyiq_policy: pd.DataFrame, train: pd.DataFrame, a: dict[str, Any] | None = None
) -> pd.DataFrame:
    """Build the flat weeks-of-cover policy, in the same shape the simulator takes."""
    a = a or load_assumptions()
    cfg = a["baseline_policy"]
    cover = float(cfg["weeks_of_cover"])
    flat_ss_weeks = float(cfg["flat_safety_stock_weeks"])

    out = supplyiq_policy.copy()
    rates = naive_demand_rate(train, a)
    out = out.merge(rates, on=ITEM_KEYS, how="left", validate="one_to_one")
    out["demand_rate"] = out["demand_rate"].fillna(0.0)

    # Flat rule: the same cover for every item, whatever its lead time or
    # variability. No z, no sigma, no lead-time term.
    out["safety_stock"] = out["demand_rate"] * flat_ss_weeks
    out["reorder_point"] = out["demand_rate"] * cover
    out["order_quantity"] = [
        apply_order_constraints(rate * cover, moq, mult)
        for rate, moq, mult in zip(out["demand_rate"], out["moq_units"], out["order_multiple_units"])
    ]
    out["order_up_to"] = out["reorder_point"] + out["order_quantity"]
    out["policy_name"] = "current"
    out["service_level"] = np.nan          # the flat rule does not target one
    out["average_inventory_units"] = out["safety_stock"] + out["order_quantity"] / 2.0
    out["average_inventory_value"] = out["average_inventory_units"] * out["landed_cost"].astype(float)
    return out


# --------------------------------------------------------------------------- #
# Running the comparison
# --------------------------------------------------------------------------- #
@dataclass
class BacktestResult:
    current: SimulationResult
    supplyiq: SimulationResult
    sweep: pd.DataFrame            # SupplyIQ across service levels
    cover_sweep: pd.DataFrame      # the flat rule across weeks of cover
    frontier: pd.DataFrame         # capital needed by each, at equal fill rate
    iso_service: dict[str, float]
    a: dict[str, Any]

    def frontier_summary(self) -> dict[str, Any]:
        """The defensible headline: capital needed at EQUAL fill rate.

        Only the fill rates both policies can actually reach are counted. The
        rest are reported as not comparable rather than quietly dropped.
        """
        if self.frontier.empty:
            return {"comparable_points": 0}
        usable = self.frontier[self.frontier["comparable"]]
        if usable.empty:
            return {"comparable_points": 0, "note": "no overlapping fill rates"}
        return {
            "comparable_points": int(len(usable)),
            "not_comparable_points": int((~self.frontier["comparable"]).sum()),
            "fill_rate_range": [
                float(usable["fill_rate"].min()),
                float(usable["fill_rate"].max()),
            ],
            "capital_saving_pct_min": float(usable["capital_saving_pct"].min()),
            "capital_saving_pct_median": float(usable["capital_saving_pct"].median()),
            "capital_saving_pct_max": float(usable["capital_saving_pct"].max()),
            "capital_saving_at_best_match": float(usable["capital_saving"].iloc[0]),
            "note": (
                "Working capital each policy needs to reach the same simulated fill rate. "
                "Fill rates below the SupplyIQ policy's floor (reached with zero safety "
                "stock) are excluded because it cannot be tuned that low."
            ),
        }

    def headline(self) -> dict[str, float]:
        """The numbers that go in the README, as straight deltas."""
        cur, new = self.current.kpis, self.supplyiq.kpis
        stockouts_avoided = cur["stockout_weeks"] - new["stockout_weeks"]
        return {
            "current_fill_rate": cur["fill_rate"],
            "supplyiq_fill_rate": new["fill_rate"],
            "fill_rate_change_pp": (new["fill_rate"] - cur["fill_rate"]) * 100,
            "current_stockout_weeks": cur["stockout_weeks"],
            "supplyiq_stockout_weeks": new["stockout_weeks"],
            "stockout_weeks_avoided": stockouts_avoided,
            "stockout_weeks_avoided_pct": (
                stockouts_avoided / cur["stockout_weeks"] * 100 if cur["stockout_weeks"] else 0.0
            ),
            "current_short_units": cur["short_units"],
            "supplyiq_short_units": new["short_units"],
            "current_working_capital": cur["working_capital"],
            "supplyiq_working_capital": new["working_capital"],
            "working_capital_change_pct": (
                (new["working_capital"] - cur["working_capital"]) / cur["working_capital"] * 100
                if cur["working_capital"]
                else 0.0
            ),
            "current_cost_to_serve": cur["cost_to_serve"],
            "supplyiq_cost_to_serve": new["cost_to_serve"],
            "cost_to_serve_change_pct": (
                (new["cost_to_serve"] - cur["cost_to_serve"]) / cur["cost_to_serve"] * 100
                if cur["cost_to_serve"]
                else 0.0
            ),
        }


def _simulate_policy(
    policy: pd.DataFrame, demand: pd.DataFrame, a: dict[str, Any]
) -> SimulationResult:
    ordered = policy.sort_values(ITEM_KEYS).reset_index(drop=True)
    draws = draw_lead_times(ordered, demand["week_index"].nunique(), int(a["simulation"]["seed"]))
    return simulate(ordered, demand, baseline_scenario(), a, lead_time_draws=draws)


def cover_sweep(
    supplyiq_policy: pd.DataFrame,
    train: pd.DataFrame,
    demand: pd.DataFrame,
    a: dict[str, Any],
    covers: list[float] | None = None,
) -> pd.DataFrame:
    """Simulate the flat rule across weeks of cover, so it is judged at its best too.

    Giving the incumbent only its default setting would be a strawman: the
    honest question is whether a flat rule can reach the same service as a
    variability-aware one by simply holding more.
    """
    covers = covers or [2, 3, 4, 5, 6, 8, 10, 12, 16]
    rows = []
    for cover in covers:
        tuned = dict(a)
        tuned["baseline_policy"] = {**a["baseline_policy"], "weeks_of_cover": float(cover)}
        policy = current_policy(supplyiq_policy, train, tuned)
        result = _simulate_policy(policy, demand, a)
        rows.append(
            {
                "weeks_of_cover": float(cover),
                "fill_rate": result.kpis["fill_rate"],
                "stockout_weeks": result.kpis["stockout_weeks"],
                "working_capital": result.kpis["working_capital"],
                "avg_inventory_value": result.kpis["avg_inventory_value"],
                "cost_to_serve": result.kpis["cost_to_serve"],
            }
        )
    return pd.DataFrame(rows)


def _capital_at_fill(sweep: pd.DataFrame, target: float) -> float | None:
    """Linear interpolation of working capital at a target fill rate."""
    ordered = sweep.sort_values("fill_rate").reset_index(drop=True)
    below = ordered[ordered["fill_rate"] <= target]
    above = ordered[ordered["fill_rate"] >= target]
    if below.empty or above.empty:
        return None
    low, high = below.iloc[-1], above.iloc[0]
    span = high["fill_rate"] - low["fill_rate"]
    if span <= 0:
        return float(low["working_capital"])
    weight = (target - low["fill_rate"]) / span
    return float(low["working_capital"] + weight * (high["working_capital"] - low["working_capital"]))


def build_frontier(sweep: pd.DataFrame, covers: pd.DataFrame) -> pd.DataFrame:
    """At each fill rate the flat rule reaches, what does each policy cost to hold?"""
    rows = []
    for row in covers.itertuples(index=False):
        target = float(row.fill_rate)
        supplyiq_capital = _capital_at_fill(sweep, target)
        rows.append(
            {
                "fill_rate": target,
                "weeks_of_cover": float(row.weeks_of_cover),
                "current_working_capital": float(row.working_capital),
                "supplyiq_working_capital": supplyiq_capital,
                "capital_saving": (
                    float(row.working_capital) - supplyiq_capital
                    if supplyiq_capital is not None
                    else None
                ),
                "capital_saving_pct": (
                    (float(row.working_capital) - supplyiq_capital) / float(row.working_capital) * 100
                    if supplyiq_capital
                    else None
                ),
                "comparable": supplyiq_capital is not None,
            }
        )
    return pd.DataFrame(rows)


def service_sweep(
    segmented: pd.DataFrame,
    sku_master: pd.DataFrame,
    item_lead_times: pd.DataFrame,
    demand: pd.DataFrame,
    a: dict[str, Any],
    levels: list[float] | None = None,
) -> pd.DataFrame:
    """Simulate the SupplyIQ policy across service-level targets.

    This is what makes an at-equal-service claim possible instead of a
    convenient one.
    """
    # The grid runs down to 0.50, where z = 0 and safety stock vanishes. A
    # narrower grid failed to bracket the current policy's achieved fill rate,
    # which would have made the equal-service comparison impossible to state.
    levels = levels or [
        0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.93, 0.95, 0.97, 0.98, 0.99, 0.995
    ]
    rows = []
    for level in levels:
        policy = build_policy(
            segmented, sku_master, item_lead_times, a, service_level=level,
            differentiate_by_abc=False,
        )
        result = _simulate_policy(policy, demand, a)
        rows.append(
            {
                "service_level": level,
                "fill_rate": result.kpis["fill_rate"],
                "stockout_weeks": result.kpis["stockout_weeks"],
                "working_capital": result.kpis["working_capital"],
                "avg_inventory_value": result.kpis["avg_inventory_value"],
                "cost_to_serve": result.kpis["cost_to_serve"],
                "safety_stock_units": float(policy["safety_stock"].sum()),
            }
        )
    return pd.DataFrame(rows)


def iso_service_comparison(
    sweep: pd.DataFrame, current: SimulationResult
) -> dict[str, float]:
    """Interpolate the sweep to the fill rate the current policy achieved.

    Returns the working capital SupplyIQ needs to match that service, and the
    saving against what the current policy ties up to achieve it.
    """
    target = float(current.kpis["fill_rate"])
    ordered = sweep.sort_values("fill_rate").reset_index(drop=True)

    below = ordered[ordered["fill_rate"] <= target]
    above = ordered[ordered["fill_rate"] >= target]
    if below.empty or above.empty:
        # The sweep does not bracket the target; report the closest point and
        # say so rather than extrapolating.
        closest = ordered.iloc[(ordered["fill_rate"] - target).abs().argmin()]
        return {
            "matched": False,
            "why_not": (
                "The SupplyIQ policy cannot be tuned down to this fill rate: at a service "
                "level of 0.50 its safety stock is already zero, and its reorder point "
                "still covers each item's own lead time. The equal-service comparison is "
                "made over the range where both policies can operate -- see `frontier`."
            ),
            "target_fill_rate": target,
            "matched_fill_rate": float(closest["fill_rate"]),
            "matched_service_level": float(closest["service_level"]),
            "supplyiq_working_capital": float(closest["working_capital"]),
            "current_working_capital": float(current.kpis["working_capital"]),
            "working_capital_saving": float(
                current.kpis["working_capital"] - closest["working_capital"]
            ),
            "working_capital_saving_pct": float(
                (current.kpis["working_capital"] - closest["working_capital"])
                / current.kpis["working_capital"]
                * 100
            ),
        }

    low, high = below.iloc[-1], above.iloc[0]
    span = high["fill_rate"] - low["fill_rate"]
    weight = 0.0 if span <= 0 else (target - low["fill_rate"]) / span
    capital = float(low["working_capital"] + weight * (high["working_capital"] - low["working_capital"]))
    _ = "matched"
    level = float(low["service_level"] + weight * (high["service_level"] - low["service_level"]))
    current_capital = float(current.kpis["working_capital"])
    return {
        "matched": True,
        "target_fill_rate": target,
        "matched_fill_rate": target,
        "matched_service_level": level,
        "supplyiq_working_capital": capital,
        "current_working_capital": current_capital,
        "working_capital_saving": current_capital - capital,
        "working_capital_saving_pct": (current_capital - capital) / current_capital * 100,
    }


def run(
    planning,
    tables: dict[str, pd.DataFrame],
    a: dict[str, Any] | None = None,
    with_sweep: bool = True,
) -> BacktestResult:
    """Backtest the SupplyIQ policy against the current policy on the holdout."""
    a = a or load_assumptions()
    from supplyiq.suppliers import item_lead_times as build_lead_times

    supplyiq_policy = planning.policy.copy()
    supplyiq_policy["policy_name"] = "supplyiq"
    current = current_policy(supplyiq_policy, planning.train, a)

    current_result = _simulate_policy(current, planning.test, a)
    supplyiq_result = _simulate_policy(supplyiq_policy, planning.test, a)

    if with_sweep:
        lead_times = build_lead_times(tables["sku_master"], planning.performance, a)
        sweep = service_sweep(
            planning.segmented, tables["sku_master"], lead_times, planning.test, a
        )
        covers = cover_sweep(supplyiq_policy, planning.train, planning.test, a)
        frontier = build_frontier(sweep, covers)
        iso = iso_service_comparison(sweep, current_result)
    else:
        sweep = covers = frontier = pd.DataFrame()
        iso = {}

    return BacktestResult(
        current=current_result,
        supplyiq=supplyiq_result,
        sweep=sweep,
        cover_sweep=covers,
        frontier=frontier,
        iso_service=iso,
        a=a,
    )


def comparison_table(result: BacktestResult) -> pd.DataFrame:
    """Side-by-side KPIs, the shape the Backtest Results page renders."""
    rows = []
    for name, sim in (("current", result.current), ("supplyiq", result.supplyiq)):
        k = sim.kpis
        rows.append(
            {
                "policy": name,
                "fill_rate": k["fill_rate"],
                "stockout_weeks": k["stockout_weeks"],
                "items_with_a_stockout": k["items_with_a_stockout"],
                "short_units": k["short_units"],
                "in_stock_rate": k["in_stock_rate"],
                "avg_inventory_value": k["avg_inventory_value"],
                "working_capital": k["working_capital"],
                "cost_to_serve": k["cost_to_serve"],
                "orders_placed": k["orders_placed"],
            }
        )
    return pd.DataFrame(rows)
