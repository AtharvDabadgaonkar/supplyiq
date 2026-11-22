"""Backtest Results — MEASURE. Does the policy beat what a planner runs today?"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd  # noqa: E402
import plotly.graph_objects as go  # noqa: E402
import streamlit as st  # noqa: E402

from app.components import (  # noqa: E402
    COLORS,
    get_artifact,
    get_assumptions,
    get_metrics,
    how_calculated,
    money,
    page_setup,
    pct,
    require_artifacts,
    show,
    sidebar_context,
)

page_setup("Backtest Results")
prov = sidebar_context()
a = get_assumptions()
m = get_metrics()
require_artifacts(["backtest_comparison", "backtest_sweep", "backtest_cover_sweep",
                   "backtest_frontier"])

st.title("Backtest Results")
st.markdown(
    "The SupplyIQ policy against the policy a planner runs today — same simulator, "
    "same weeks, same lead-time draws. Only the reorder point and order quantity differ."
)

bt = m.get("backtest", {})
default = bt.get("at_default_settings", {})
equal = bt.get("at_equal_service", {})
comparison = get_artifact("backtest_comparison")
sweep = get_artifact("backtest_sweep")
covers = get_artifact("backtest_cover_sweep")
frontier = get_artifact("backtest_frontier")

c1, c2 = st.columns(2)
with c1:
    st.markdown("##### Current policy")
    st.caption(bt.get("current_policy", ""))
with c2:
    st.markdown("##### SupplyIQ policy")
    st.caption(bt.get("supplyiq_policy", ""))

st.divider()

# --------------------------------------------------------------------------- #
# At default settings
# --------------------------------------------------------------------------- #
st.subheader("At their default settings")
k1, k2, k3, k4 = st.columns(4)
k1.metric("Fill rate", pct(default.get("supplyiq_fill_rate", 0), 2),
          delta=f"{default.get('fill_rate_change_pp', 0):+.2f} pp")
k2.metric("Stockout weeks", f"{default.get('supplyiq_stockout_weeks', 0):.0f}",
          delta=f"−{default.get('stockout_weeks_avoided', 0):.0f} "
                f"({default.get('stockout_weeks_avoided_pct', 0):.0f}%)",
          delta_color="inverse")
k3.metric("Cost to serve", money(default.get("supplyiq_cost_to_serve", 0)),
          delta=f"{default.get('cost_to_serve_change_pct', 0):+.1f}%", delta_color="inverse")
k4.metric("Working capital", money(default.get("supplyiq_working_capital", 0)),
          delta=f"{default.get('working_capital_change_pct', 0):+.1f}%", delta_color="inverse")

st.dataframe(
    comparison.round(4), hide_index=True, width="stretch"
)
how_calculated(
    "Both policies simulated over the same holdout weeks with the same pre-drawn lead "
    "times. Fill rate is pooled on units; working capital counts on-hand plus in-transit "
    "inventory at landed cost."
)

st.warning(
    "**Read this honestly.** At their default settings SupplyIQ buys a lot more service "
    f"({default.get('fill_rate_change_pp', 0):+.2f} pp, "
    f"{default.get('stockout_weeks_avoided', 0):.0f} stockout weeks avoided) for "
    f"{default.get('working_capital_change_pct', 0):+.1f}% more working capital. "
    "More service for more stock is not by itself an achievement. The comparison that "
    "means something is below.",
    icon="⚠️",
)

st.divider()

# --------------------------------------------------------------------------- #
# The frontier
# --------------------------------------------------------------------------- #
st.subheader("The like-for-like comparison")
st.markdown(
    "Both policies are swept — SupplyIQ across service-level targets, the flat rule "
    "across weeks of cover — and the question asked at every point is: **to reach this "
    "fill rate, how much working capital does each one need?**"
)

fig = go.Figure()
fig.add_trace(
    go.Scatter(
        x=covers["fill_rate"], y=covers["working_capital"], mode="lines+markers",
        name="current policy (flat weeks of cover)", line=dict(color=COLORS["current"], width=3),
        text=[f"{w:.0f} wk cover" for w in covers["weeks_of_cover"]],
        hovertemplate="%{text}<br>fill %{x:.2%}<br>capital %{y:$,.0f}<extra></extra>",
    )
)
fig.add_trace(
    go.Scatter(
        x=sweep["fill_rate"], y=sweep["working_capital"], mode="lines+markers",
        name="SupplyIQ policy (service-level target)",
        line=dict(color=COLORS["supplyiq"], width=3),
        text=[f"SL {s:.1%}" for s in sweep["service_level"]],
        hovertemplate="%{text}<br>fill %{x:.2%}<br>capital %{y:$,.0f}<extra></extra>",
    )
)
if equal.get("comparable_points"):
    low, high = equal["fill_rate_range"]
    fig.add_vrect(
        x0=low, x1=high, fillcolor=COLORS["good"], opacity=0.10, line_width=0,
        annotation_text="range where both policies can operate",
        annotation_position="top left", annotation_font_size=11,
    )
fig.update_layout(
    xaxis_title="simulated fill rate", yaxis_title="working capital ($)",
    xaxis_tickformat=".1%",
)
show(fig, height=420)
how_calculated(
    "Lower is better at any given fill rate. The SupplyIQ curve sits below and to the "
    "right: it reaches the same service for less capital, and it cannot be pushed as far "
    "left because even with zero safety stock its reorder point still covers each item's "
    "own lead time."
)

if equal.get("comparable_points"):
    st.success(
        f"**At equal service, the SupplyIQ policy needs "
        f"{equal['capital_saving_pct_median']:.0f}% less working capital** — a range of "
        f"{equal['capital_saving_pct_min']:.1f}% to {equal['capital_saving_pct_max']:.1f}% "
        f"across the {equal['comparable_points']} fill rates between "
        f"{equal['fill_rate_range'][0]:.1%} and {equal['fill_rate_range'][1]:.1%} that "
        f"both policies can actually reach.",
        icon="📉",
    )
    st.caption(
        f"{equal['not_comparable_points']} swept points fall outside that range and are "
        "excluded rather than averaged in — see the table below, where they are marked "
        "not comparable."
    )

st.dataframe(
    frontier.round(4), hide_index=True, width="stretch",
    column_config={
        "comparable": st.column_config.CheckboxColumn("comparable", disabled=True),
    },
)
how_calculated(
    "For each weeks-of-cover setting of the flat rule, the SupplyIQ curve is linearly "
    "interpolated to the same fill rate. Rows where that fill rate lies outside the swept "
    "range are marked not comparable — the interpolation is never extrapolated."
)

st.divider()

# --------------------------------------------------------------------------- #
# The finding
# --------------------------------------------------------------------------- #
st.subheader("The finding worth the write-up")
st.info(bt.get("key_finding", ""), icon="🔑")

floor = sweep.sort_values("service_level").iloc[0]
f1, f2, f3 = st.columns(3)
f1.metric("SupplyIQ at service level 0.50", pct(float(floor["fill_rate"]), 2))
f2.metric("Its safety stock", f"{float(floor['safety_stock_units']):,.0f} units")
f3.metric("Current policy fill rate", pct(default.get("current_fill_rate", 0), 2),
          delta="beaten with zero buffer", delta_color="off")
how_calculated(
    "Driving the service level to 0.50 sets z = 0, so safety stock is exactly zero. The "
    "policy still beats a flat four weeks of cover, which locates the advantage in the "
    "reorder point rather than the buffer."
)

with st.expander("Why the originally specified comparison could not be made", expanded=False):
    iso = bt.get("at_equal_service", {})
    st.markdown(
        f"""
The brief asked for *inventory reduction at the same service level*. That comparison
turned out to be impossible as stated: the SupplyIQ policy **cannot be tuned down** to the
fill rate the flat rule achieves at its default setting. At a service level of 0.50 its
safety stock is already zero, and its reorder point still covers each item's own lead
time plus the review period.

Rather than quietly pick a flattering pair of points, the backtest sweeps **both**
policies and compares the curves over the range where they overlap. The flat rule is
swept too, so it is judged at its best rather than as a strawman — it *can* reach 99.9%
fill, it simply has to hold {money(float(covers['working_capital'].max()))} to do it.

{iso.get('why_not', '')}
"""
    )
