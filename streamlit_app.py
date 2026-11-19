"""SupplyIQ — entry point for Streamlit Community Cloud.

Must stay at the repo root: Streamlit Cloud takes this file as the main module
and treats `pages/` beside it as the multipage app.

Phase 1 scope: this page proves the deployment path works end to end — data
loads from the committed sample, provenance and agent mode are visible, and the
assumptions registry is inspectable. The planning pages arrive in Phase 5.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from app.components import (  # noqa: E402
    agent_badge,
    data_badge,
    get_assumptions,
    get_table,
    how_calculated,
    page_setup,
    sidebar_context,
    table_summary,
)
from supplyiq import __version__  # noqa: E402
from supplyiq.config import simulated_sections  # noqa: E402
from supplyiq.data import split_weeks  # noqa: E402

page_setup("Control Tower")
prov = sidebar_context()
a = get_assumptions()

st.title("Control Tower")
st.markdown(
    "Balancing **service level** against **inventory cost** and **risk**, "
    "across five steps: Sense → Plan → Simulate → Act → Measure."
)

c1, c2 = st.columns(2)
with c1:
    data_badge(prov)
with c2:
    agent_badge()

st.divider()

# --------------------------------------------------------------------------- #
# Phase 1: what is wired up
# --------------------------------------------------------------------------- #
st.subheader("Build status")
st.caption(
    "KPI tiles, the at-risk SKU table and the risk→scenario→action→impact story "
    "land in Phase 5. This page currently proves the data and deployment path."
)

demand = get_table("demand", ("sku", "store", "week_index", "week_start", "units_sold", "revenue"))
sku_master = get_table("sku_master")
suppliers = get_table("suppliers")
pos = get_table("purchase_orders", ("po_id", "supplier_id", "late_delivery", "in_full"))

last_train, first_test = split_weeks(demand, int(a["sample"]["holdout_weeks"]))
otif = float(((pos["late_delivery"] == 0) & (pos["in_full"] == 1)).mean())

m1, m2, m3, m4 = st.columns(4)
m1.metric("Planning items", f"{len(sku_master):,}")
m2.metric("Weeks of history", f"{int(demand['week_index'].max()) + 1:,}")
m3.metric("Suppliers", f"{len(suppliers):,}")
m4.metric("Historical OTIF", f"{otif:.1%}")
how_calculated(
    "OTIF = share of inbound POs that arrived on or before the promised date AND complete. "
    "Inbound lead times are simulated; see Assumptions below."
)

st.divider()

# --------------------------------------------------------------------------- #
# Data health
# --------------------------------------------------------------------------- #
st.subheader("Data health")
left, right = st.columns([3, 2])
with left:
    st.dataframe(table_summary(), hide_index=True, width="stretch")
    how_calculated(
        "Read directly from the committed Parquet files in `data/sample/`. "
        "The app never reads `data/raw/`."
    )
with right:
    st.markdown("**Provenance**")
    st.json(
        {
            "origin": prov.origin,
            "built_at": prov.built_at,
            "demand_source": prov.demand_source,
            "orders_source": prov.orders_source,
            "row_counts": prov.row_counts,
        },
        expanded=True,
    )
    st.caption(prov.notes)

st.markdown(
    f"**Time-based split:** train on weeks 0–{last_train}, "
    f"hold out weeks {first_test}–{int(demand['week_index'].max())} "
    f"({a['sample']['holdout_weeks']} weeks). "
    "Forecast evaluation never sees the holdout."
)

st.divider()

# --------------------------------------------------------------------------- #
# Demand shape — the one chart Phase 1 earns
# --------------------------------------------------------------------------- #
st.subheader("Weekly demand, total")
weekly = (
    demand.groupby("week_start", observed=True)
    .agg(units=("units_sold", "sum"), revenue=("revenue", "sum"))
    .reset_index()
)
weekly["split"] = [
    "holdout" if w >= first_test else "history"
    for w in sorted(demand["week_index"].unique())
]
st.line_chart(weekly, x="week_start", y="units", color="split", height=300)
how_calculated(
    "Units summed across all planning items per retail week. "
    "`holdout` marks the final "
    f"{a['sample']['holdout_weeks']} weeks, which are reserved for evaluation."
)

st.divider()

# --------------------------------------------------------------------------- #
# Assumptions registry
# --------------------------------------------------------------------------- #
st.subheader("Assumptions")
st.markdown(
    "Every planning parameter lives in `config/assumptions.yaml`; nothing is "
    f"hard-coded. Sections marked simulated: **{', '.join(simulated_sections(a))}**."
)
with st.expander("Supplier master (simulated)", expanded=False):
    st.dataframe(
        suppliers[
            [
                "supplier_id", "name", "country", "mean_lead_time_days",
                "lead_time_sd_days", "otif_target", "freight_per_unit",
                "tariff_rate", "alternate_supplier_id",
            ]
        ],
        hide_index=True,
        width="stretch",
    )
    how_calculated(
        "Declared in `config/assumptions.yaml`. Inbound lead times are drawn from a "
        "lognormal matched to each supplier's mean and standard deviation; the quoted "
        "date is that distribution's on-time quantile."
    )
with st.expander("Cost model (simulated)", expanded=False):
    st.markdown(
        "```\n"
        "unit_cost   = avg_sell_price * (1 - gross_margin_target) * supplier.unit_cost_index\n"
        "landed_cost = unit_cost + freight_per_unit + unit_cost * tariff_rate\n"
        "```"
    )
    st.dataframe(
        pd.DataFrame(
            {
                "country": list(a["costs"]["tariff_rate_by_country"]),
                "tariff_rate": list(a["costs"]["tariff_rate_by_country"].values()),
                "freight_per_unit": [
                    a["costs"]["freight_per_unit_by_country"][c]
                    for c in a["costs"]["tariff_rate_by_country"]
                ],
            }
        ),
        hide_index=True,
        width="stretch",
    )
with st.expander("Full assumptions file", expanded=False):
    st.json(a, expanded=False)

st.caption(f"SupplyIQ v{__version__} · Phase 1 of 5")
