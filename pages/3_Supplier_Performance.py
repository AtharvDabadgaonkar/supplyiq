"""Supplier Performance — SENSE. The procurement conversation, with evidence."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd  # noqa: E402
import plotly.express as px  # noqa: E402
import plotly.graph_objects as go  # noqa: E402
import streamlit as st  # noqa: E402

from app.components import (  # noqa: E402
    COLORS,
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
    simulated_note,
)

page_setup("Supplier Performance")
prov = sidebar_context()
a = get_assumptions()
m = get_metrics()
require_artifacts(["supplier_performance", "policy"])

st.title("Supplier Performance")
st.markdown(
    "Lead time, its variability, and OTIF — the three numbers that decide how much "
    "stock you have to hold against each lane."
)

perf = get_artifact("supplier_performance")
policy = get_artifact("policy")
service = m.get("supplier_service", {})

kpi_row(
    [
        {"label": "Network OTIF", "value": pct(service.get("otif", 0)),
         "help": "On time AND in full, across every inbound PO in the history."},
        {"label": "On-time rate", "value": pct(service.get("on_time_rate", 0))},
        {"label": "In-full rate", "value": pct(service.get("in_full_rate", 0))},
        {"label": "Mean lead time",
         "value": f"{service.get('lead_time_mean_days', 0):.1f} d",
         "help": "Order to receipt, measured from the PO history."},
    ]
)
simulated_note(["mean_lead_time_days", "lead_time_sd_days", "otif_target", "tariff_rate"])

st.divider()

# --------------------------------------------------------------------------- #
# The scatter that makes the argument
# --------------------------------------------------------------------------- #
st.subheader("Lead time vs. reliability")
volume = (
    policy.groupby("supplier_id", observed=True)
    .agg(items=("sku", "size"), safety_stock_value=("safety_stock_value", "sum"))
    .reset_index()
)
plot = perf.merge(volume, on="supplier_id", how="left")
fig = px.scatter(
    plot,
    x="lead_time_mean_days",
    y="lead_time_sd_days",
    size="safety_stock_value",
    color="country",
    text="supplier_id",
    size_max=48,
    hover_data=["otif", "late_delivery_rate", "items"],
    labels={
        "lead_time_mean_days": "mean lead time (days)",
        "lead_time_sd_days": "lead-time standard deviation (days)",
    },
)
fig.update_traces(textposition="top center", textfont_size=10)
show(fig, height=420)
how_calculated(
    "Both axes are measured from realised PO receipts, not read back from the supplier "
    "master. Bubble size is the safety-stock value held against that supplier — the "
    "working capital its lead-time spread costs you. Top-right is expensive to source from."
)

st.divider()

left, right = st.columns(2)
with left:
    st.subheader("OTIF against target")
    ordered = perf.sort_values("otif")
    fig = go.Figure()
    fig.add_trace(go.Bar(y=ordered["supplier_id"], x=ordered["otif"], orientation="h",
                         name="realised OTIF", marker_color=COLORS["primary"]))
    fig.add_trace(go.Scatter(y=ordered["supplier_id"], x=ordered["otif_target"],
                             mode="markers", name="target",
                             marker=dict(color=COLORS["accent"], size=11, symbol="diamond")))
    fig.update_layout(xaxis_tickformat=".0%", xaxis_title="OTIF", yaxis_title=None)
    show(fig, height=320)
    how_calculated(
        "Realised OTIF is the share of POs that arrived on or before the promised date "
        "AND complete. On-time is judged against the supplier's quoted date, not its own "
        "average — judging against the average would make half of all POs late by "
        "construction."
    )

with right:
    st.subheader("Cost of the lane")
    lane = (
        policy.groupby(["supplier_id", "country"], observed=True)
        .agg(unit_cost=("unit_cost", "mean"), freight=("freight_per_unit", "mean"),
             landed=("landed_cost", "mean"), tariff_rate=("tariff_rate", "mean"))
        .reset_index()
    )
    lane["duty"] = lane["unit_cost"] * lane["tariff_rate"]
    fig = go.Figure()
    for name, column, color in (
        ("unit cost", "unit_cost", COLORS["primary"]),
        ("freight", "freight", COLORS["accent"]),
        ("duty", "duty", COLORS["bad"]),
    ):
        fig.add_trace(go.Bar(x=lane["supplier_id"], y=lane[column], name=name,
                             marker_color=color))
    fig.update_layout(barmode="stack", yaxis_title="$ per unit", xaxis_title=None)
    show(fig, height=320)
    how_calculated(
        "landed cost = unit cost + freight + unit cost × tariff rate. Unit cost is derived "
        f"from real sell prices at a {a['costs']['gross_margin_target']:.0%} gross margin; "
        "freight and duty are simulated per country from `config/assumptions.yaml`."
    )

st.divider()

st.subheader("Supplier scorecard")
columns = [
    "supplier_id", "name", "country", "pos", "otif", "otif_target", "otif_vs_target_pp",
    "on_time_rate", "in_full_rate", "late_delivery_rate", "lead_time_mean_days",
    "lead_time_sd_days", "lead_time_p95_days", "lead_time_cv", "quoted_lead_time_days",
    "alternate_supplier_id",
]
st.dataframe(
    perf[[c for c in columns if c in perf.columns]].round(3),
    hide_index=True,
    width="stretch",
)
how_calculated(
    "Aggregated from the inbound PO log. `lead_time_cv` is the standard deviation over "
    "the mean — the single best predictor of how much safety stock a lane forces you to "
    "carry. `alternate_supplier_id` is the second source the Planning Agent switches to."
)

with st.expander("What is real and what is simulated here", expanded=False):
    st.markdown(
        f"""
Neither source dataset contains a vendor dimension or inbound purchase lead times, so the
**supplier master is simulated** from `config/assumptions.yaml` — lead-time mean and
standard deviation, OTIF target, MOQ, order multiple, and the cost stack.

When the real Kaggle data is present, DataCo contributes the **real** shipping-mode mix,
order regions and late-delivery rate, which anchor the PO log to observed behaviour.

Current data origin: **`{prov.origin}`**. Everything on this page is labelled accordingly.
"""
    )
