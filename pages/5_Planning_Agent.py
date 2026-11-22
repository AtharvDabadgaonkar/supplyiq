"""Planning Agent — ACT. Recommendations a planner approves or rejects.

Every number on this page is simulator output. The language model writes only
the explanation, and if the text it returns contains a figure the simulator did
not produce, it is discarded in favour of the rule-based one.
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
    get_artifact,
    get_assumptions,
    get_metrics,
    get_provider_status,
    get_table,
    how_calculated,
    money,
    page_setup,
    pct,
    require_artifacts,
    show,
    sidebar_context,
)
from supplyiq.act import DecisionLog, recommend  # noqa: E402
from supplyiq.agent import MockProvider, Recommendation, get_provider  # noqa: E402
from supplyiq.data import train_test  # noqa: E402
from supplyiq.pipeline import AGENT_SCENARIO, preset_scenarios  # noqa: E402
from supplyiq.simulate import draw_lead_times, simulate  # noqa: E402

page_setup("Planning Agent")
prov = sidebar_context()
a = get_assumptions()
m = get_metrics()
require_artifacts(["policy", "recommendations"])

st.title("Planning Agent")
st.markdown(
    "For each at-risk item: what to do, what it costs, and what service it buys — "
    "every figure computed by re-running the inventory simulation with the action applied."
)
agent_badge()

LOG_KEY = "decision_log"
if LOG_KEY not in st.session_state:
    st.session_state[LOG_KEY] = DecisionLog()
log: DecisionLog = st.session_state[LOG_KEY]


@st.cache_data(show_spinner=False)
def _inputs():
    policy = get_artifact("policy").sort_values(["sku", "store"]).reset_index(drop=True)
    demand = get_table("demand", ("sku", "store", "week_index", "units_sold"))
    _, test = train_test(demand, int(a["sample"]["holdout_weeks"]))
    return policy, test


@st.cache_resource(show_spinner="Pricing actions against the simulator…")
def _recommendations(scenario_name: str, top_n: int) -> list[Recommendation]:
    policy, test = _inputs()
    scenario = next(
        (s for s in preset_scenarios() if s.name == scenario_name), preset_scenarios()[0]
    )
    draws = draw_lead_times(policy, test["week_index"].nunique(), int(a["simulation"]["seed"]))
    result = simulate(policy, test, scenario, a, lead_time_draws=draws)
    return recommend(policy, test, scenario, a, result=result, top_n=top_n)


names = [s.name for s in preset_scenarios()]
c1, c2 = st.columns([3, 1])
with c1:
    scenario_name = st.selectbox(
        "Scenario to plan against",
        names,
        index=names.index(AGENT_SCENARIO) if AGENT_SCENARIO in names else 0,
        format_func=lambda n: n.replace("_", " "),
        help="With no disruption there is usually nothing for the agent to recommend.",
    )
with c2:
    top_n = st.number_input("Items to review", 3, 25, int(a["agent"]["at_risk_top_n"]))

recommendations = _recommendations(scenario_name, int(top_n))
provider = get_provider()

if not recommendations:
    st.success("No item falls below its service target under this scenario.", icon="✅")
    st.stop()

# --------------------------------------------------------------------------- #
# Summary
# --------------------------------------------------------------------------- #
helpful = [r for r in recommendations if r.service_gain_pp > 0]
k1, k2, k3, k4 = st.columns(4)
k1.metric("Items at risk", len(recommendations))
k2.metric("Service recoverable", f"{sum(r.service_gain_pp for r in helpful):.1f} pp")
k3.metric("Total cost", money(sum(r.action_cost_delta for r in helpful)))
k4.metric(
    "Median $ / point",
    money(
        pd.Series([r.cost_per_service_point for r in helpful]).median()
        if helpful
        else 0
    ),
)
how_calculated(
    "Each candidate action is expressed as a simulator override and the inventory twin is "
    "re-run with and without it, sharing the scenario and the lead-time draws. The "
    "recommendation shown is the cheapest option per point of fill rate — not the largest "
    "gain, because a planner has a budget."
)

st.divider()

# --------------------------------------------------------------------------- #
# The cost/service frontier of the recommendations
# --------------------------------------------------------------------------- #
if helpful:
    st.subheader("What each action buys")
    frame = pd.DataFrame(
        {
            "item": [f"{r.sku} · {r.store}" for r in helpful],
            "action": [r.action_type.replace("_", " ") for r in helpful],
            "service_gain_pp": [r.service_gain_pp for r in helpful],
            "cost": [r.action_cost_delta for r in helpful],
            "cost_per_point": [r.cost_per_service_point for r in helpful],
            "abc": [r.abc_class for r in helpful],
        }
    )
    fig = go.Figure()
    for action in frame["action"].unique():
        sub = frame[frame["action"] == action]
        fig.add_trace(
            go.Scatter(
                x=sub["service_gain_pp"], y=sub["cost"], mode="markers+text",
                text=sub["item"], textposition="top center", textfont_size=9,
                name=action, marker=dict(size=14, opacity=0.85),
            )
        )
    fig.update_layout(
        xaxis_title="fill-rate gain (pp)", yaxis_title="cost of the action ($)"
    )
    show(fig, height=380)
    how_calculated(
        "Bottom-right is what you want: a lot of service for little money. Each point is "
        "the difference between two simulator runs that shared the same supply luck."
    )
    st.divider()

# --------------------------------------------------------------------------- #
# Review queue
# --------------------------------------------------------------------------- #
st.subheader("Review queue")
decided = {(r["sku"], r["store"], r["action_type"]) for r in log.to_frame().to_dict("records")}

for i, rec in enumerate(recommendations):
    key = (rec.sku, rec.store, rec.action_type)
    explanation = provider.explain(rec)
    with st.container(border=True):
        head, metrics_col, buttons = st.columns([3, 4, 2])
        with head:
            st.markdown(f"**{rec.sku}** · {rec.store}")
            st.caption(f"{rec.abc_class}{rec.xyz_class} · {rec.supplier_id}")
            st.markdown(f"`{rec.action_type.replace('_', ' ')}`")
        with metrics_col:
            mc1, mc2, mc3 = st.columns(3)
            mc1.metric("Fill rate", pct(rec.action_fill_rate, 1),
                       delta=f"{rec.service_gain_pp:+.1f} pp")
            mc2.metric("Cost", money(rec.action_cost_delta))
            mc3.metric(
                "$ / point",
                money(rec.cost_per_service_point) if rec.cost_per_service_point else "—",
            )
        with buttons:
            if key in decided:
                st.success("Decided", icon="✓")
            else:
                if st.button("Approve", key=f"a{i}", type="primary", width="stretch"):
                    log.record(rec, "approved", explanation=explanation.text,
                               explanation_source=explanation.provider)
                    st.rerun()
                if st.button("Reject", key=f"r{i}", width="stretch"):
                    log.record(rec, "rejected", explanation=explanation.text,
                               explanation_source=explanation.provider)
                    st.rerun()
        st.markdown(f"> {explanation.text}")
        source = "rule-based" if explanation.is_mock else f"LLM · {explanation.provider}"
        if explanation.fallback_reason:
            st.caption(f"_Explanation: {source} — {explanation.fallback_reason}_")
        else:
            st.caption(f"_Explanation: {source}. Numbers are simulator output._")

st.divider()

# --------------------------------------------------------------------------- #
# Decision log
# --------------------------------------------------------------------------- #
st.subheader("Decision log")
summary = log.summary()
d1, d2, d3, d4 = st.columns(4)
d1.metric("Decisions", summary["decisions"])
d2.metric("Approved", summary["approved"])
d3.metric("Committed cost", money(summary["approved_cost"]))
d4.metric("Service committed", f"{summary['approved_service_gain_pp']:.1f} pp")

if summary["decisions"]:
    st.dataframe(
        log.to_frame()[
            ["decided_at", "sku", "store", "action_type", "decision",
             "service_gain_pp", "action_cost_delta", "explanation_source"]
        ].round(2),
        hide_index=True,
        width="stretch",
    )
    st.download_button(
        "Download decisions (CSV)",
        log.to_csv(),
        file_name="supplyiq_decisions.csv",
        mime="text/csv",
    )
else:
    st.caption("No decisions yet. Approve or reject a recommendation above.")

st.info(
    "**This log is not permanent.** Streamlit Community Cloud resets the filesystem on "
    "redeploy, so decisions live in your browser session only. Download the CSV to keep "
    "them. A real deployment would write to a database — that is on the Phase 2 roadmap.",
    icon="ℹ️",
)

with st.expander("How the agent is kept honest", expanded=False):
    status = get_provider_status()
    st.markdown(
        f"""
**The language model never produces a number.** It receives a payload of simulator
output and returns prose. Generated text is then scanned by `supplyiq/agent/guard.py`:
every figure in it must reconcile with a value on the recommendation, and if any does
not, the text is thrown away and the rule-based explanation is used instead.

That makes "the LLM does not invent numbers" an enforced property rather than a line in
a prompt. There is a test that feeds the guard fabricated text and asserts it is rejected.

**Current provider:** `{status['active']}` — {status['badge']}.
With no API key configured the app runs entirely on the rule-based writer, which is why
the live demo works without one.
"""
    )
