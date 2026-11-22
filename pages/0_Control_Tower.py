"""Control Tower — the one screen an operations leader opens.

KPI tiles, the top at-risk items, and the whole product in one line of
narrative: risk found -> scenario -> action -> impact.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd  # noqa: E402
import plotly.graph_objects as go  # noqa: E402
import streamlit as st  # noqa: E402

from app.components import (  # noqa: E402
    COLORS,
    agent_badge,
    data_badge,
    get_artifact,
    get_assumptions,
    get_metrics,
    get_table,
    how_calculated,
    kpi_row,
    money,
    page_setup,
    pct,
    require_artifacts,
    show,
    sidebar_context,
)

page_setup("Control Tower")
prov = sidebar_context()
a = get_assumptions()
m = get_metrics()
require_artifacts(["policy", "scenario_summary", "recommendations", "backtest_comparison"])

st.title("Control Tower")
st.markdown(
    "Service level, inventory cost and risk, on one screen. "
    "**Sense → Plan → Simulate → Act → Measure.**"
)
c1, c2 = st.columns(2)
with c1:
    data_badge(prov)
with c2:
    agent_badge()

# --------------------------------------------------------------------------- #
# KPI tiles
# --------------------------------------------------------------------------- #
policy = get_artifact("policy")
scenarios = get_artifact("scenario_summary")
baseline = scenarios[scenarios["scenario"] == "baseline"].iloc[0]
supplier_perf = get_artifact("supplier_performance")
inv = m.get("inventory_policy", {})
fc = m.get("forecast_accuracy", {})
candidate = fc.get("models", {}).get(fc.get("candidate_model", ""), {})

st.subheader("Where the network stands")
kpi_row(
    [
        {
            "label": "OTIF",
            "value": pct(float(baseline["otif"])),
            "help": "On time AND in full, inbound POs, simulated baseline window.",
        },
        {
            "label": "Fill rate",
            "value": pct(float(baseline["fill_rate"]), 2),
            "help": "Units shipped ÷ units demanded, pooled across all items.",
        },
        {
            "label": "In-stock rate",
            "value": pct(float(baseline["in_stock_rate"]), 2),
            "help": "Share of item-weeks with no unmet demand.",
        },
        {
            "label": "Forecast WAPE",
            "value": f"{candidate.get('wape', float('nan')):.3f}",
            "help": "Weighted absolute percentage error on the holdout. Lower is better.",
        },
    ]
)
kpi_row(
    [
        {
            "label": "Inventory turns",
            "value": f"{inv.get('portfolio_inventory_turns', float('nan')):.1f}",
            "help": "Annual demand ÷ average inventory units, portfolio level.",
        },
        {
            "label": "Weeks of supply",
            "value": f"{inv.get('median_weeks_of_supply', float('nan')):.1f}",
            "help": "Median across planning items.",
        },
        {
            "label": "Working capital",
            "value": money(float(baseline["working_capital"])),
            "help": "Average on-hand plus in-transit inventory, at landed cost.",
        },
        {
            "label": "Cost to serve",
            "value": money(float(baseline["cost_to_serve"])),
            "help": "Holding + stockout + ordering costs. Excludes the price of goods.",
        },
    ]
)
how_calculated(
    "All tiles come from `reports/metrics.json` and the precomputed baseline simulation "
    "over the holdout weeks. Nothing on this page is typed by hand."
)

st.divider()

# --------------------------------------------------------------------------- #
# The one-screen story
# --------------------------------------------------------------------------- #
st.subheader("Risk → Scenario → Action → Impact")

recs = get_artifact("recommendations")
agent_meta = m.get("agent_recommendations", {})
backtest = m.get("backtest", {})
worst = m.get("scenarios", {}).get("worst_case", {})

s1, s2, s3, s4 = st.columns(4)
with s1:
    st.markdown("##### 1 · Risk found")
    st.metric("Items at risk", f"{len(recs)}")
    st.caption(
        f"Ranked by revenue at risk under the "
        f"*{str(recs['scenario'].iloc[0]).replace('_', ' ')}* scenario."
    )
with s2:
    st.markdown("##### 2 · Scenario")
    st.metric(
        "Worst preset",
        f"{worst.get('fill_rate_vs_baseline_pp', 0):+.2f} pp",
        help=f"Scenario: {worst.get('scenario', 'n/a')}",
    )
    st.caption(f"{worst.get('stockout_weeks', 0)} stockout weeks vs baseline.")
with s3:
    st.markdown("##### 3 · Action")
    st.metric("Service recoverable", f"{agent_meta.get('total_service_gain_pp', 0):.1f} pp")
    st.caption(
        f"For {money(agent_meta.get('total_action_cost', 0))} — "
        f"median {money(agent_meta.get('median_cost_per_service_point', 0))} per point."
    )
with s4:
    st.markdown("##### 4 · Impact")
    at_equal = backtest.get("at_equal_service", {})
    st.metric(
        "Capital at equal service",
        f"−{at_equal.get('capital_saving_pct_median', 0):.0f}%",
        help="SupplyIQ policy vs. a flat weeks-of-cover rule, at the same fill rate.",
    )
    st.caption(
        f"{backtest.get('at_default_settings', {}).get('stockout_weeks_avoided', 0):.0f} "
        "stockout weeks avoided in the backtest."
    )

how_calculated(
    "Each step reads the module that produced it: at-risk items from the simulator, "
    "the scenario from the precomputed sweep, action value from simulator re-runs with "
    "the action applied, and impact from the backtest against the current policy."
)

st.divider()

# --------------------------------------------------------------------------- #
# Scenario impact
# --------------------------------------------------------------------------- #
left, right = st.columns([3, 2])
with left:
    st.subheader("What each disruption does to service")
    plot = scenarios[scenarios["scenario"] != "baseline"].copy()
    plot["fill_drop_pp"] = -plot["delta_fill_rate"] * 100
    fig = go.Figure(
        go.Bar(
            x=plot["fill_drop_pp"],
            y=plot["description"],
            orientation="h",
            marker_color=COLORS["bad"],
            hovertemplate="%{y}<br>%{x:.2f} pp fill-rate drop<extra></extra>",
        )
    )
    fig.update_layout(xaxis_title="Fill-rate drop vs baseline (pp)", yaxis_title=None)
    show(fig, height=300, legend=False)
    how_calculated(
        "Each scenario re-runs the weekly inventory simulation over the holdout, sharing "
        "the baseline's lead-time draws so the difference is the scenario, not noise."
    )

with right:
    st.subheader("Top items at risk")
    table = recs[["sku", "store", "action_type", "service_gain_pp", "action_cost_delta"]].copy()
    table.columns = ["SKU", "Store", "Recommended", "Service +pp", "Cost $"]
    st.dataframe(
        table.round(2),
        hide_index=True,
        width="stretch",
        height=300,
    )
    how_calculated(
        "Items whose simulated fill rate falls below their service target, ranked by "
        "revenue at risk. The recommendation is the cheapest option per point of fill "
        "rate, priced by re-running the simulator."
    )

st.divider()

# --------------------------------------------------------------------------- #
# Backtest headline
# --------------------------------------------------------------------------- #
st.subheader("Does the policy actually work?")
default = backtest.get("at_default_settings", {})
b1, b2, b3, b4 = st.columns(4)
b1.metric(
    "Fill rate",
    pct(default.get("supplyiq_fill_rate", 0), 2),
    delta=f"{default.get('fill_rate_change_pp', 0):+.2f} pp vs current policy",
)
b2.metric(
    "Stockout weeks",
    f"{default.get('supplyiq_stockout_weeks', 0):.0f}",
    delta=f"−{default.get('stockout_weeks_avoided', 0):.0f}",
    delta_color="inverse",
)
b3.metric(
    "Cost to serve",
    money(default.get("supplyiq_cost_to_serve", 0)),
    delta=f"{default.get('cost_to_serve_change_pct', 0):+.1f}%",
    delta_color="inverse",
)
b4.metric(
    "Working capital",
    money(default.get("supplyiq_working_capital", 0)),
    delta=f"{default.get('working_capital_change_pct', 0):+.1f}%",
    delta_color="inverse",
)
st.info(
    f"**At their default settings** the SupplyIQ policy buys more service for more "
    f"capital. The like-for-like comparison is on the Backtest page: at the **same fill "
    f"rate**, it needs **{at_equal.get('capital_saving_pct_median', 0):.0f}% less working "
    f"capital** (range {at_equal.get('capital_saving_pct_min', 0):.1f}–"
    f"{at_equal.get('capital_saving_pct_max', 0):.1f}%).",
    icon="📊",
)
how_calculated(
    "Both policies run through the same simulator, over the same weeks, with the same "
    "lead-time draws. Only the reorder point and order quantity differ."
)

st.caption(
    f"Built from `reports/metrics.json` · data origin `{prov.origin}` · "
    f"phases complete {m.get('phases_complete', [])}"
)
