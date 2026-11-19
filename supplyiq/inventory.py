"""PLAN — segmentation and inventory policy.

ABC/XYZ
-------
ABC splits items by revenue contribution (Pareto): A items earn the first 80%
of revenue, B the next 15%, C the rest. XYZ splits by how predictable demand
is, using the coefficient of variation of weekly demand. The pair is what lets
a planner treat an AX item (big, steady, worth tight control) differently from
a CZ item (small, erratic, not worth the effort).

Safety stock
------------
The formula accounts for BOTH sources of uncertainty, which is the point:

    SS = z * sqrt( L * sd_d^2  +  d^2 * sd_L^2 )

    z     service factor, the standard normal quantile of the target cycle
          service level
    L     mean lead time, in weeks
    sd_d  standard deviation of WEEKLY demand
    d     mean WEEKLY demand
    sd_L  standard deviation of lead time, in weeks

The first term is demand risk over the lead time; the second is lead-time risk
at the average demand rate. Dropping the second term -- which a flat
weeks-of-cover rule does implicitly -- is what leaves long, erratic overseas
lanes under-protected while over-stocking short reliable ones.

Reorder point, order quantity
-----------------------------
    ROP = d * L + SS                    (plus the review period, see below)
    EOQ = sqrt( 2 * D * S / H )         D annual demand, S cost per PO,
                                        H annual holding cost per unit

Under periodic review the exposure is the lead time PLUS the review period,
because a stockout can open at any point until the next review. `protection
_weeks` carries that, and it is why `review_period_weeks` is in the config.

Fill rate
---------
Cycle service level (the chance of no stockout in a cycle) is not the same as
fill rate (the share of demand met from stock), and planners are measured on
the latter. We convert with the standard normal loss function:

    G(z)       = phi(z) - z * (1 - Phi(z))
    E[short]   = sd_dl * G(z)             expected units short per cycle
    fill rate  = 1 - E[short] / Q
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

from supplyiq.config import load_assumptions

ITEM_KEYS = ["sku", "store"]
WEEKS_PER_YEAR = 52.0


# --------------------------------------------------------------------------- #
# Statistical helpers
# --------------------------------------------------------------------------- #
def service_factor(service_level: float) -> float:
    """z for a cycle service level. 0.95 -> 1.6449."""
    if not 0.0 < service_level < 1.0:
        raise ValueError(f"service level {service_level} must be strictly between 0 and 1")
    from statistics import NormalDist

    return float(NormalDist().inv_cdf(service_level))


def normal_loss(z: float) -> float:
    """Standard normal loss function G(z) = phi(z) - z * (1 - Phi(z)).

    Expected shortfall, in standard deviations, once demand exceeds z.
    """
    from statistics import NormalDist

    nd = NormalDist()
    return float(nd.pdf(z) - z * (1.0 - nd.cdf(z)))


def demand_sd_over_leadtime(
    mean_demand: float, sd_demand: float, lead_time: float, sd_lead_time: float
) -> float:
    """sqrt( L * sd_d^2 + d^2 * sd_L^2 ) -- demand risk plus lead-time risk."""
    if lead_time < 0 or sd_lead_time < 0:
        raise ValueError("lead time and its standard deviation must be non-negative")
    return float(
        math.sqrt(lead_time * sd_demand**2 + (mean_demand**2) * (sd_lead_time**2))
    )


def safety_stock(
    mean_demand: float,
    sd_demand: float,
    lead_time: float,
    sd_lead_time: float,
    service_level: float,
    minimum: float = 0.0,
) -> float:
    """Safety stock in units, covering demand AND lead-time variability."""
    z = service_factor(service_level)
    return float(
        max(minimum, z * demand_sd_over_leadtime(mean_demand, sd_demand, lead_time, sd_lead_time))
    )


def reorder_point(mean_demand: float, protection_weeks: float, safety: float) -> float:
    """Expected demand over the protection window, plus safety stock."""
    return float(mean_demand * protection_weeks + safety)


def economic_order_quantity(
    annual_demand: float, ordering_cost: float, holding_cost_per_unit_year: float
) -> float:
    """Classic EOQ. Returns 0 for an item with no demand."""
    if annual_demand <= 0 or holding_cost_per_unit_year <= 0:
        return 0.0
    return float(math.sqrt(2.0 * annual_demand * ordering_cost / holding_cost_per_unit_year))


def apply_order_constraints(
    quantity: float,
    moq: float,
    multiple: float,
    max_units: float | None = None,
) -> float:
    """Round an order up to the supplier's MOQ and order multiple.

    `max_units` is the coverage cap (max_weeks_of_supply x weekly demand). It is
    applied last and may pull the quantity below the raw EOQ, because holding
    many months of a cheap slow mover costs more in practice -- shelf space,
    obsolescence, cash -- than the EOQ trade-off captures. The MOQ still wins
    over the cap: you cannot order less than the supplier will ship.
    """
    q = max(float(quantity), float(moq))
    if multiple and multiple > 0:
        q = math.ceil(q / multiple) * multiple
    if max_units is not None and max_units > 0 and q > max_units:
        capped = math.floor(max_units / multiple) * multiple if multiple and multiple > 0 else max_units
        q = max(float(moq), float(capped))
        if multiple and multiple > 0:
            q = math.ceil(q / multiple) * multiple
    return float(q)


def projected_fill_rate(
    order_quantity: float, sd_over_leadtime: float, service_level: float
) -> float:
    """Share of demand met from stock, given the cycle service level and Q."""
    if order_quantity <= 0:
        return 0.0
    if sd_over_leadtime <= 0:
        return 1.0
    z = service_factor(service_level)
    expected_short = sd_over_leadtime * normal_loss(z)
    return float(min(1.0, max(0.0, 1.0 - expected_short / order_quantity)))


# --------------------------------------------------------------------------- #
# Segmentation
# --------------------------------------------------------------------------- #
def abc_segment(revenue: pd.Series, cutoffs: dict[str, float]) -> pd.Series:
    """Pareto classes on cumulative revenue share, descending."""
    order = revenue.sort_values(ascending=False)
    total = float(order.sum())
    if total <= 0:
        return pd.Series("C", index=revenue.index, dtype="object")
    cumulative = order.cumsum() / total
    # An item is in A if the cumulative share UP TO AND INCLUDING it is within
    # the A cutoff, so the boundary item that crosses 80% is still an A.
    previous = cumulative.shift(1).fillna(0.0)
    classes = np.where(
        previous < cutoffs["A"], "A", np.where(previous < cutoffs["B"], "B", "C")
    )
    return pd.Series(classes, index=order.index).reindex(revenue.index)


def xyz_segment(cv: pd.Series, cutoffs: dict[str, float]) -> pd.Series:
    """Predictability classes on the coefficient of variation of weekly demand."""
    return pd.Series(
        np.where(
            cv.isna(), "Z", np.where(cv < cutoffs["X"], "X", np.where(cv < cutoffs["Y"], "Y", "Z"))
        ),
        index=cv.index,
        dtype="object",
    )


def segment(stats: pd.DataFrame, revenue: pd.Series, a: dict[str, Any] | None = None) -> pd.DataFrame:
    """Attach ABC, XYZ and the combined class to the per-item demand stats."""
    a = a or load_assumptions()
    out = stats.copy()
    out["revenue"] = revenue.to_numpy()
    out["abc_class"] = abc_segment(out["revenue"], a["policy"]["abc_revenue_cutoffs"]).to_numpy()
    out["xyz_class"] = xyz_segment(out["demand_cv"], a["policy"]["xyz_cv_cutoffs"]).to_numpy()
    out["abc_xyz"] = out["abc_class"] + out["xyz_class"]
    return out


# --------------------------------------------------------------------------- #
# The policy table
# --------------------------------------------------------------------------- #
def build_policy(
    stats: pd.DataFrame,
    sku_master: pd.DataFrame,
    item_lead_times: pd.DataFrame,
    a: dict[str, Any] | None = None,
    service_level: float | None = None,
    differentiate_by_abc: bool = True,
) -> pd.DataFrame:
    """Compute the full inventory policy, one row per planning item.

    `service_level` overrides the configured target for the whole table, which
    is how the UI's service-level slider works. With `differentiate_by_abc`
    set, A/B/C items get the per-class targets from the config instead -- the
    thing a flat policy cannot do.
    """
    a = a or load_assumptions()
    policy_cfg = a["policy"]
    costs = a["costs"]

    df = stats.merge(
        item_lead_times[
            ITEM_KEYS + ["supplier_id", "lead_time_mean_weeks", "lead_time_sd_weeks",
                         "quoted_lead_time_days", "unit_cost", "freight_per_unit",
                         "tariff_rate", "landed_cost", "avg_sell_price", "category"]
        ],
        on=ITEM_KEYS,
        how="left",
        validate="one_to_one",
    )

    # Supplier constraints and country. Country is carried so a scenario can be
    # scoped to a sourcing origin ("what if China goes to 35% duty").
    sup = (
        sku_master[["supplier_id"]].drop_duplicates().merge(
            _supplier_constraints(a), on="supplier_id", how="left"
        )
    )
    df = df.merge(sup, on="supplier_id", how="left", validate="many_to_one")

    # Service level: explicit override, else per-ABC, else the flat target.
    if service_level is not None:
        df["service_level"] = float(service_level)
    elif differentiate_by_abc and "abc_class" in df.columns:
        df["service_level"] = df["abc_class"].map(policy_cfg["service_level_by_abc"]).astype(float)
    else:
        df["service_level"] = float(policy_cfg["service_level_target"])

    review = float(policy_cfg["review_period_weeks"])
    df["review_period_weeks"] = review
    df["protection_weeks"] = df["lead_time_mean_weeks"] + review

    df["z"] = df["service_level"].map(service_factor)
    df["sd_over_leadtime"] = [
        demand_sd_over_leadtime(d, sd, lt, sdlt)
        for d, sd, lt, sdlt in zip(
            df["mean_weekly_demand"],
            df["sd_weekly_demand"],
            df["protection_weeks"],
            df["lead_time_sd_weeks"],
        )
    ]
    df["safety_stock"] = np.maximum(
        float(policy_cfg["min_safety_stock_units"]), df["z"] * df["sd_over_leadtime"]
    )
    df["cycle_stock"] = df["mean_weekly_demand"] * df["protection_weeks"]
    df["reorder_point"] = df["cycle_stock"] + df["safety_stock"]

    # EOQ on landed cost: holding cost is a share of what the unit is worth
    # landed in the warehouse, not of its ex-works price.
    df["annual_demand"] = df["mean_weekly_demand"] * WEEKS_PER_YEAR
    df["holding_cost_per_unit_year"] = (
        df["landed_cost"].astype(float) * float(costs["holding_cost_rate_annual"])
    )
    df["eoq_raw"] = [
        economic_order_quantity(d, float(costs["ordering_cost_per_po"]), h)
        for d, h in zip(df["annual_demand"], df["holding_cost_per_unit_year"])
    ]
    max_wos = float(policy_cfg["max_weeks_of_supply"])
    df["max_order_units"] = df["mean_weekly_demand"] * max_wos
    df["order_quantity"] = [
        apply_order_constraints(q, moq, mult, cap)
        for q, moq, mult, cap in zip(
            df["eoq_raw"], df["moq_units"], df["order_multiple_units"], df["max_order_units"]
        )
    ]
    df["eoq_capped_by_coverage"] = df["order_quantity"] < df["eoq_raw"]
    df["order_at_moq"] = df["order_quantity"] <= df["moq_units"]
    df["order_up_to"] = df["reorder_point"] + df["order_quantity"]

    df["projected_fill_rate"] = [
        projected_fill_rate(q, sd, sl)
        for q, sd, sl in zip(df["order_quantity"], df["sd_over_leadtime"], df["service_level"])
    ]

    # Working capital and flow metrics.
    df["average_inventory_units"] = df["safety_stock"] + df["order_quantity"] / 2.0
    df["average_inventory_value"] = df["average_inventory_units"] * df["landed_cost"].astype(float)
    df["safety_stock_value"] = df["safety_stock"] * df["landed_cost"].astype(float)
    df["inventory_turns"] = np.where(
        df["average_inventory_units"] > 0,
        df["annual_demand"] / df["average_inventory_units"],
        np.nan,
    )
    df["weeks_of_supply"] = np.where(
        df["mean_weekly_demand"] > 0,
        df["average_inventory_units"] / df["mean_weekly_demand"],
        np.nan,
    )
    df["safety_stock_weeks"] = np.where(
        df["mean_weekly_demand"] > 0, df["safety_stock"] / df["mean_weekly_demand"], np.nan
    )
    return df


def _supplier_constraints(a: dict[str, Any]) -> pd.DataFrame:
    rows = [
        {
            "supplier_id": s["supplier_id"],
            "country": s["country"],
            "moq_units": float(s["moq_units"]),
            "order_multiple_units": float(s["order_multiple_units"]),
            "alternate_supplier_id": a["suppliers"]["alternate_source"].get(s["supplier_id"]),
        }
        for s in a["suppliers"]["network"]
    ]
    return pd.DataFrame(rows)


def policy_summary(policy: pd.DataFrame) -> dict[str, float]:
    """Portfolio roll-up, volume-weighted where a mean of ratios would mislead."""
    total_annual = float(policy["annual_demand"].sum())
    total_inventory_units = float(policy["average_inventory_units"].sum())
    return {
        "items": int(len(policy)),
        "total_safety_stock_units": float(policy["safety_stock"].sum()),
        "total_safety_stock_value": float(policy["safety_stock_value"].sum()),
        "total_average_inventory_units": total_inventory_units,
        "total_average_inventory_value": float(policy["average_inventory_value"].sum()),
        "weighted_fill_rate": float(
            np.average(policy["projected_fill_rate"], weights=policy["annual_demand"].clip(lower=1e-9))
        ),
        "portfolio_inventory_turns": (
            total_annual / total_inventory_units if total_inventory_units > 0 else float("nan")
        ),
        "median_weeks_of_supply": float(policy["weeks_of_supply"].median()),
        "median_safety_stock_weeks": float(policy["safety_stock_weeks"].median()),
    }


def segment_summary(policy: pd.DataFrame) -> pd.DataFrame:
    """ABC/XYZ grid: where the revenue is, and where the stock is tied up."""
    grouped = (
        policy.groupby(["abc_class", "xyz_class"], observed=True)
        .agg(
            items=("sku", "size"),
            revenue=("revenue", "sum"),
            annual_demand=("annual_demand", "sum"),
            safety_stock_value=("safety_stock_value", "sum"),
            average_inventory_value=("average_inventory_value", "sum"),
            mean_demand_cv=("demand_cv", "mean"),
            mean_service_level=("service_level", "mean"),
            mean_fill_rate=("projected_fill_rate", "mean"),
        )
        .reset_index()
    )
    grouped["revenue_share"] = grouped["revenue"] / grouped["revenue"].sum()
    grouped["inventory_value_share"] = (
        grouped["average_inventory_value"] / grouped["average_inventory_value"].sum()
    )
    return grouped.sort_values(["abc_class", "xyz_class"]).reset_index(drop=True)
