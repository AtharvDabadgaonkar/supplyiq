"""Scenario Planner — SIMULATE. The sliders re-run the twin live."""
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
    get_table,
    delta_or_none,
    how_calculated,
    money,
    page_setup,
    pct,
    require_artifacts,
    show,
    sidebar_context,
)
from supplyiq.data import train_test  # noqa: E402
from supplyiq.simulate import (  # noqa: E402
    Scenario,
    baseline_scenario,
    draw_lead_times,
    simulate,
)

page_setup("Scenario Planner")
prov = sidebar_context()
a = get_assumptions()
require_artifacts(["policy"])

st.title("Scenario Planner")
st.markdown(
    "Move a slider and the weekly inventory simulation re-runs over the holdout. "
    "The baseline and your scenario share the same lead-time draws, so every "
    "difference you see is the disruption, not simulation noise."
)


@st.cache_data(show_spinner=False)
def _inputs():
    policy = get_artifact("policy").sort_values(["sku", "store"]).reset_index(drop=True)
    demand = get_table("demand", ("sku", "store", "week_index", "units_sold"))
    _, test = train_test(demand, int(a["sample"]["holdout_weeks"]))
    draws = draw_lead_times(
        policy, test["week_index"].nunique(), int(a["simulation"]["seed"])
    )
    return policy, test, draws


policy, test, draws = _inputs()
cfg = a["scenarios"]

# --------------------------------------------------------------------------- #
# Controls
# --------------------------------------------------------------------------- #
with st.form("scenario"):
    c1, c2, c3 = st.columns(3)
    with c1:
        delay = st.slider(
            "Supplier delay (days)",
            min_value=int(cfg["supplier_delay_days"]["min"]),
            max_value=int(cfg["supplier_delay_days"]["max"]),
            value=int(cfg["supplier_delay_days"]["default"]),
            step=int(cfg["supplier_delay_days"]["step"]),
            help="Added to every drawn lead time for the suppliers in scope.",
        )
    with c2:
        spike = st.slider(
            "Demand change (%)",
            min_value=float(cfg["demand_spike_pct"]["min"]),
            max_value=float(cfg["demand_spike_pct"]["max"]),
            value=float(cfg["demand_spike_pct"]["default"]),
            step=float(cfg["demand_spike_pct"]["step"]),
            help="Demand is multiplied by (1 + this / 100).",
        )
    with c3:
        tariff = st.slider(
            "Tariff change (percentage points)",
            min_value=float(cfg["tariff_change_pct"]["min"]),
            max_value=float(cfg["tariff_change_pct"]["max"]),
            value=float(cfg["tariff_change_pct"]["default"]),
            step=float(cfg["tariff_change_pct"]["step"]),
            help="PERCENTAGE POINTS on the duty rate: +10 takes 25% to 35%, not 27.5%.",
        )

    s1, s2 = st.columns([1, 2])
    with s1:
        scope_kind = st.radio(
            "Apply delay and tariff to", ["All suppliers", "A country", "One supplier"],
            horizontal=False,
        )
    with s2:
        countries = sorted(policy["country"].dropna().unique())
        suppliers = sorted(policy["supplier_id"].dropna().unique())
        country_scope = supplier_scope = None
        if scope_kind == "A country":
            picked = st.multiselect("Countries", countries, default=countries[:1])
            country_scope = tuple(picked) if picked else None
        elif scope_kind == "One supplier":
            picked = st.multiselect("Suppliers", suppliers, default=suppliers[:1])
            supplier_scope = tuple(picked) if picked else None

    submitted = st.form_submit_button("Run scenario", type="primary", width="stretch")

scenario = Scenario(
    name="scenario",
    supplier_delay_days=float(delay),
    demand_spike_pct=float(spike),
    tariff_change_pct=float(tariff),
    country_scope=country_scope,
    supplier_scope=supplier_scope,
)

base = simulate(policy, test, baseline_scenario(), a, lead_time_draws=draws)
scen = simulate(policy, test, scenario, a, lead_time_draws=draws)
delta = scen.compare_to(base)

if scenario.is_baseline:
    st.info(
        "No disruption selected — the scenario reproduces the baseline exactly. "
        "That identity is asserted by the test suite.",
        icon="✅",
    )
else:
    st.warning(f"**Scenario:** {scenario.describe()}", icon="🌪️")

# --------------------------------------------------------------------------- #
# Outcome
# --------------------------------------------------------------------------- #
st.subheader("Baseline vs scenario")
k1, k2, k3, k4 = st.columns(4)
k1.metric("Fill rate", pct(scen.kpis["fill_rate"], 2),
          delta=delta_or_none(delta["fill_rate"], f"{delta['fill_rate'] * 100:+.2f} pp"))
k2.metric("Stockout weeks", f"{scen.kpis['stockout_weeks']:.0f}",
          delta=delta_or_none(delta["stockout_weeks"], f"{delta['stockout_weeks']:+.0f}"),
          delta_color="inverse")
k3.metric("OTIF", pct(scen.kpis["otif"]),
          delta=delta_or_none(delta["otif"], f"{delta['otif'] * 100:+.1f} pp"))
k4.metric("Items hit", f"{scen.kpis['items_with_a_stockout']:.0f}",
          delta=delta_or_none(delta["items_with_a_stockout"],
                              f"{delta['items_with_a_stockout']:+.0f}"),
          delta_color="inverse")

k5, k6, k7, k8 = st.columns(4)
k5.metric("Working capital", money(scen.kpis["working_capital"]),
          delta=delta_or_none(delta["working_capital"], money(delta["working_capital"]), 0.5),
          delta_color="inverse")
k6.metric("In transit", money(scen.kpis["avg_in_transit_value"]),
          delta=delta_or_none(delta["avg_in_transit_value"],
                              money(delta["avg_in_transit_value"]), 0.5),
          delta_color="off",
          help="A delay strands cash on a ship rather than freeing it up.")
k7.metric("Landed cost / unit", f"${scen.kpis['landed_cost_per_unit']:.3f}",
          delta=delta_or_none(delta["landed_cost_per_unit"],
                              f"{delta['landed_cost_per_unit']:+.3f}", 1e-4),
          delta_color="inverse")
k8.metric("Cost to serve", money(scen.kpis["cost_to_serve"]),
          delta=delta_or_none(delta["cost_to_serve"], money(delta["cost_to_serve"]), 0.5),
          delta_color="inverse")
how_calculated(
    "Two runs of the weekly simulation over the holdout weeks, sharing one set of "
    "pre-drawn lead times. Cost to serve is holding + stockout + ordering costs and "
    "excludes the price of the goods: over a finite window a delay pushes reorders past "
    "the end of it, so including goods would make a disruption look cheaper."
)

st.divider()

# --------------------------------------------------------------------------- #
# Where it bites
# --------------------------------------------------------------------------- #
left, right = st.columns([3, 2])

with left:
    st.subheader("Service by ABC class")
    merged = scen.per_item.merge(
        base.per_item[["sku", "store", "fill_rate", "short_units"]],
        on=["sku", "store"], suffixes=("", "_base"),
    )
    by_class = (
        merged.groupby("abc_class", observed=True)
        .apply(
            lambda g: pd.Series(
                {
                    "baseline": g["shipped_units_base"].sum() / g["demand_units"].sum()
                    if "shipped_units_base" in g else g["fill_rate_base"].mean(),
                    "scenario": g["shipped_units"].sum() / g["demand_units"].sum(),
                }
            ),
            include_groups=False,
        )
        .reset_index()
    )
    fig = go.Figure()
    fig.add_trace(go.Bar(x=by_class["abc_class"], y=by_class["baseline"],
                         name="baseline", marker_color=COLORS["muted"]))
    fig.add_trace(go.Bar(x=by_class["abc_class"], y=by_class["scenario"],
                         name="scenario", marker_color=COLORS["primary"]))
    fig.update_layout(barmode="group", yaxis_tickformat=".1%",
                      yaxis_title="fill rate", xaxis_title="ABC class")
    show(fig, height=320)
    how_calculated(
        "Fill rate pooled on units within each class. A items are protected hardest by "
        "the policy, so a disruption that still hits them is the one to act on."
    )

with right:
    st.subheader("Worst-hit items")
    hit = merged.assign(extra_short=lambda d: d["short_units"] - d["short_units_base"])
    hit = hit[hit["extra_short"] > 0].sort_values("extra_short", ascending=False).head(12)
    if hit.empty:
        st.success("No item is worse off under this scenario.", icon="✅")
    else:
        table = hit[["sku", "store", "supplier_id", "extra_short", "fill_rate"]].copy()
        table.columns = ["SKU", "Store", "Supplier", "Extra units short", "Fill rate"]
        st.dataframe(table.round(3), hide_index=True, width="stretch", height=320)
    how_calculated(
        "Units short under the scenario minus units short at baseline, per planning item."
    )

st.divider()
st.subheader("Weekly stockouts through the window")
ledger = simulate(policy, test, scenario, a, lead_time_draws=draws, keep_ledger=True).ledger
weekly = (
    ledger.assign(hit=lambda d: d["short"] > 1e-9)
    .groupby("week_index", observed=True)
    .agg(items_short=("hit", "sum"), units_short=("short", "sum"))
    .reset_index()
)
fig = go.Figure()
fig.add_trace(go.Bar(x=weekly["week_index"], y=weekly["items_short"],
                     name="items short", marker_color=COLORS["bad"]))
fig.update_layout(xaxis_title="holdout week", yaxis_title="items with unmet demand")
show(fig, height=260, legend=False)
how_calculated(
    "Items with any unmet demand in each simulated week. Delays typically show up a few "
    "weeks in, once the pipeline that was already in transit has been consumed."
)

with st.expander("What the three levers mean exactly", expanded=False):
    st.markdown(
        """
**Supplier delay (days)** — added to each drawn lead time for suppliers in scope. An
order placed in week *w* arrives in week *w + ceil(lead days / 7)*.

**Demand change (%)** — demand in every holdout week is multiplied by (1 + pct/100).

**Tariff change (percentage points)** — added to the ad-valorem duty rate and floored at
zero. **+10 takes a 25% duty to 35%, not to 27.5%.** On a 25% base those two readings
differ by a factor of four, which is why it is stated here, in the config, and in a test.

A tariff change moves cost and working capital but not service — it is a price event, not
a supply event, and the simulation keeps them separate.
"""
    )
