"""Inventory Policy — PLAN. Segmentation, safety stock, and what it costs."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import plotly.express as px  # noqa: E402
import plotly.graph_objects as go  # noqa: E402
import streamlit as st  # noqa: E402

from app.components import (  # noqa: E402
    COLORS,
    get_artifact,
    get_assumptions,
    get_metrics,
    how_calculated,
    kpi_row,
    money,
    page_setup,
    pct,
    require_artifacts,
    show,
    sidebar_context,
    simulated_note,
)
from supplyiq.inventory import demand_sd_over_leadtime, safety_stock, service_factor  # noqa: E402

page_setup("Inventory Policy")
prov = sidebar_context()
a = get_assumptions()
m = get_metrics()
require_artifacts(["policy"])

st.title("Inventory Policy")
st.markdown(
    "Safety stock, reorder point and order quantity per item — accounting for demand "
    "**and** lead-time variability."
)

policy = get_artifact("policy")
inv = m.get("inventory_policy", {})

kpi_row(
    [
        {"label": "Projected fill rate", "value": pct(inv.get("weighted_fill_rate", 0), 2),
         "help": "Demand-weighted, from the normal loss function."},
        {"label": "Inventory turns", "value": f"{inv.get('portfolio_inventory_turns', 0):.1f}"},
        {"label": "Median weeks of supply", "value": f"{inv.get('median_weeks_of_supply', 0):.1f}"},
        {"label": "Safety stock value",
         "value": money(inv.get("total_safety_stock_value", 0))},
    ]
)

st.success(
    f"**Lead-time variability adds {inv.get('leadtime_risk_uplift_pct', 0):.1f}% to total "
    f"safety stock** ({inv.get('safety_stock_units_demand_risk_only', 0):,.0f} → "
    f"{inv.get('safety_stock_units_with_leadtime_risk', 0):,.0f} units). A flat "
    "weeks-of-cover rule ignores that term entirely, which is exactly how long overseas "
    "lanes end up under-protected.",
    icon="🎯",
)
simulated_note(["lead_time_mean_weeks", "lead_time_sd_weeks", "unit_cost", "landed_cost"])

st.divider()

# --------------------------------------------------------------------------- #
# Segmentation
# --------------------------------------------------------------------------- #
st.subheader("ABC / XYZ segmentation")
left, right = st.columns(2)

with left:
    grid = (
        policy.groupby(["abc_class", "xyz_class"], observed=True)
        .agg(items=("sku", "size"), revenue=("revenue", "sum"),
             inventory=("average_inventory_value", "sum"))
        .reset_index()
    )
    pivot = grid.pivot(index="abc_class", columns="xyz_class", values="items").fillna(0)
    fig = px.imshow(
        pivot, text_auto=True, aspect="auto",
        labels=dict(x="XYZ (demand variability)", y="ABC (revenue)", color="items"),
        color_continuous_scale="Blues",
    )
    show(fig, height=280, legend=False)
    how_calculated(
        "ABC by cumulative revenue share (Pareto: A to "
        f"{a['policy']['abc_revenue_cutoffs']['A']:.0%}, B to "
        f"{a['policy']['abc_revenue_cutoffs']['B']:.0%}). XYZ by coefficient of variation "
        f"of weekly demand (X below {a['policy']['xyz_cv_cutoffs']['X']}, Y below "
        f"{a['policy']['xyz_cv_cutoffs']['Y']}). Both on training weeks only."
    )

with right:
    share = (
        policy.groupby("abc_class", observed=True)
        .agg(revenue=("revenue", "sum"), inventory=("average_inventory_value", "sum"))
        .reset_index()
    )
    share["revenue_share"] = share["revenue"] / share["revenue"].sum()
    share["inventory_share"] = share["inventory"] / share["inventory"].sum()
    fig = go.Figure()
    fig.add_trace(go.Bar(x=share["abc_class"], y=share["revenue_share"],
                         name="share of revenue", marker_color=COLORS["primary"]))
    fig.add_trace(go.Bar(x=share["abc_class"], y=share["inventory_share"],
                         name="share of inventory value", marker_color=COLORS["accent"]))
    fig.update_layout(barmode="group", yaxis_tickformat=".0%", xaxis_title="ABC class")
    show(fig, height=280)
    how_calculated(
        "Where the money comes from versus where it is tied up. C items earning a few "
        "percent of revenue while holding more than that share of inventory is the "
        "classic place to release working capital."
    )

st.divider()

# --------------------------------------------------------------------------- #
# Live service-level what-if
# --------------------------------------------------------------------------- #
st.subheader("What a service-level target costs")
st.caption(
    "Recomputed live from the safety-stock formula — not a lookup table."
)
target = st.select_slider(
    "Cycle service level (applied flat to every item)",
    options=[0.80, 0.85, 0.90, 0.93, 0.95, 0.97, 0.98, 0.99, 0.995],
    value=float(a["policy"]["service_level_target"]),
    format_func=lambda v: f"{v:.1%}",
)

z = service_factor(target)
recomputed = policy.copy()
recomputed["new_safety_stock"] = z * recomputed["sd_over_leadtime"].astype(float)
recomputed["new_safety_stock_value"] = (
    recomputed["new_safety_stock"] * recomputed["landed_cost"].astype(float)
)
current_value = float(policy["safety_stock_value"].sum())
new_value = float(recomputed["new_safety_stock_value"].sum())

s1, s2, s3 = st.columns(3)
s1.metric("Service factor z", f"{z:.4f}")
s2.metric("Safety stock value", money(new_value),
          delta=money(new_value - current_value), delta_color="inverse")
s3.metric("Units", f"{recomputed['new_safety_stock'].sum():,.0f}",
          delta=f"{recomputed['new_safety_stock'].sum() - policy['safety_stock'].sum():+,.0f}")
how_calculated(
    "safety stock = z × √(L × σ_demand² + demand² × σ_leadtime²), where z is the standard "
    "normal quantile of the target. Changing the target rescales every item by z, so the "
    "whole portfolio recomputes instantly."
)

st.divider()

# --------------------------------------------------------------------------- #
# What drives safety stock
# --------------------------------------------------------------------------- #
st.subheader("What drives each item's buffer")
sample = policy.sample(min(400, len(policy)), random_state=0)
fig = px.scatter(
    sample, x="lead_time_mean_weeks", y="safety_stock_weeks", color="abc_class",
    size="revenue", size_max=24, opacity=0.75,
    hover_data=["sku", "store", "supplier_id", "demand_cv"],
    labels={
        "lead_time_mean_weeks": "supplier lead time (weeks)",
        "safety_stock_weeks": "safety stock (weeks of cover)",
        "abc_class": "ABC",
    },
    category_orders={"abc_class": ["A", "B", "C"]},
)
show(fig, height=380)
how_calculated(
    "Each point is a planning item; size is annual revenue. Cover rises with lead time "
    "because the lead-time variance term grows with it — the relationship a flat rule "
    "cannot express."
)

st.divider()
st.subheader("The policy table")
columns = [
    "sku", "store", "abc_xyz", "supplier_id", "mean_weekly_demand", "demand_cv",
    "lead_time_mean_weeks", "lead_time_sd_weeks", "service_level", "safety_stock",
    "reorder_point", "order_quantity", "projected_fill_rate", "weeks_of_supply",
    "inventory_turns", "average_inventory_value",
]
st.dataframe(policy[columns].round(2), hide_index=True, width="stretch", height=360)
how_calculated(
    "Precomputed by `make metrics` and committed, so the deployed app loads a Parquet "
    "file instead of refitting models. Reorder point = demand × (lead time + review "
    f"period of {a['policy']['review_period_weeks']} week), plus safety stock."
)

with st.expander("Why some C items still hold a lot of cover", expanded=False):
    at_moq = int(inv.get("items_ordering_at_moq", 0))
    st.markdown(
        f"""
**{at_moq} of {len(policy)} items order at their supplier's minimum order quantity.**
For a slow mover, the MOQ — not the policy — sets the inventory. You cannot order less
than the supplier will ship, so a 24-unit MOQ on an item selling 2 units a week is
12 weeks of cover whatever the service target says.

That is why the coverage cap (`max_weeks_of_supply = {a['policy']['max_weeks_of_supply']}`)
binds on most items but C items still sit above it. The lever here is renegotiating MOQs
or consolidating slow movers, not tuning the service level.
"""
    )
