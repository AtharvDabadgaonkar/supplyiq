"""Demand Forecast — SENSE. Two models, one honest scoreboard."""
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
    get_table,
    how_calculated,
    kpi_row,
    page_setup,
    require_artifacts,
    show,
    sidebar_context,
)

page_setup("Demand Forecast")
prov = sidebar_context()
a = get_assumptions()
m = get_metrics()
require_artifacts(["forecast_accuracy", "forecasts", "future_forecast"])

st.title("Demand Forecast")
st.markdown(
    "A seasonal-naive baseline against one better model, scored on weeks neither "
    "of them ever saw."
)

fc = m.get("forecast_accuracy", {})
models = fc.get("models", {})
baseline_name, candidate_name = fc.get("baseline_model"), fc.get("candidate_model")
baseline, candidate = models.get(baseline_name, {}), models.get(candidate_name, {})

kpi_row(
    [
        {"label": f"WAPE · {candidate_name}", "value": f"{candidate.get('wape', 0):.3f}",
         "delta": f"{-fc.get('wape_improvement_pp', 0):.2f} pp vs baseline",
         "delta_color": "inverse",
         "help": "Weighted absolute percentage error. Lower is better."},
        {"label": f"WAPE · {baseline_name}", "value": f"{baseline.get('wape', 0):.3f}",
         "help": "The baseline every forecast must beat to justify itself."},
        {"label": "Bias", "value": f"{candidate.get('bias_pct', 0):+.3f}",
         "help": "Positive means over-forecasting. Near zero is what you want."},
        {"label": "Relative improvement",
         "value": f"{fc.get('wape_improvement_pct_of_baseline', 0):.0f}%",
         "help": "Share of the baseline's error removed."},
    ]
)

st.warning(
    f"**Why WAPE and not MAPE.** {fc.get('mape_note', '')} "
    "A model scored only on MAPE here would look better than it is.",
    icon="⚠️",
)

st.divider()

# --------------------------------------------------------------------------- #
# Scoreboard
# --------------------------------------------------------------------------- #
left, right = st.columns([2, 3])
with left:
    st.subheader("Model scoreboard")
    rows = []
    for name, acc in models.items():
        rows.append(
            {
                "Model": name,
                "WAPE": round(acc.get("wape", 0), 4),
                "MAPE": round(acc.get("mape", 0), 4),
                "Bias": round(acc.get("bias_pct", 0), 4),
                "RMSE": round(acc.get("rmse", 0), 2),
                "MAE": round(acc.get("mae", 0), 2),
            }
        )
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    how_calculated(
        "Pooled over every item and holdout week — total absolute error divided by total "
        "actual demand — not an average of per-item percentages, which would let a tiny "
        "slow mover swamp the number."
    )

with right:
    st.subheader("Where each model wins and loses")
    accuracy = get_artifact("forecast_accuracy")
    fig = go.Figure()
    for i, name in enumerate(accuracy["model"].unique()):
        subset = accuracy[accuracy["model"] == name]
        fig.add_trace(
            go.Box(
                x=subset["wape"].clip(upper=3),
                name=str(name),
                orientation="h",
                boxmean=True,
                marker_color=COLORS["primary"] if name == candidate_name else COLORS["muted"],
            )
        )
    fig.update_layout(xaxis_title="Per-item WAPE (clipped at 3.0)", yaxis_title=None)
    show(fig, height=260, legend=False)
    how_calculated(
        "Distribution of per-item WAPE across all planning items. The long right tail is "
        "intermittent slow movers, which neither model forecasts well — an honest limit "
        "of weekly statistical forecasting on sparse demand."
    )

st.divider()

# --------------------------------------------------------------------------- #
# One SKU at a time
# --------------------------------------------------------------------------- #
st.subheader("Actual vs forecast, item by item")

demand = get_table("demand", ("sku", "store", "week_index", "week_start", "units_sold"))
forecasts = get_artifact("forecasts")
future = get_artifact("future_forecast")
accuracy = get_artifact("forecast_accuracy")

c1, c2, c3 = st.columns([2, 1, 2])
items = sorted(demand["sku"].unique())
with c1:
    sku = st.selectbox("SKU", items, index=0)
with c2:
    stores = sorted(demand.loc[demand["sku"] == sku, "store"].unique())
    store = st.selectbox("Store", stores, index=0)
with c3:
    scored = accuracy[(accuracy["sku"] == sku) & (accuracy["store"] == store)]
    if len(scored):
        best = scored.sort_values("wape").iloc[0]
        st.metric(
            "Best model for this item",
            str(best["model"]),
            delta=f"WAPE {best['wape']:.3f}",
            delta_color="off",
        )

history = demand[(demand["sku"] == sku) & (demand["store"] == store)].sort_values("week_index")
holdout_start = int(a["sample"]["n_weeks"]) - int(a["sample"]["holdout_weeks"])
week_lookup = dict(zip(demand["week_index"], demand["week_start"]))

fig = go.Figure()
fig.add_trace(
    go.Scatter(
        x=history["week_start"], y=history["units_sold"], name="actual",
        mode="lines", line=dict(color="#24292f", width=1.6),
    )
)
for name, color in ((baseline_name, COLORS["muted"]), (candidate_name, COLORS["primary"])):
    sub = forecasts[
        (forecasts["sku"] == sku) & (forecasts["store"] == store) & (forecasts["model"] == name)
    ].sort_values("week_index")
    if len(sub):
        fig.add_trace(
            go.Scatter(
                x=[week_lookup.get(w) for w in sub["week_index"]],
                y=sub["forecast"], name=f"{name} (holdout)",
                mode="lines", line=dict(color=color, width=2.2, dash="dot"),
            )
        )
fut = future[(future["sku"] == sku) & (future["store"] == store)].sort_values("week_index")
if len(fut):
    last_date = pd.Timestamp(history["week_start"].max())
    fig.add_trace(
        go.Scatter(
            x=[last_date + pd.Timedelta(weeks=int(s)) for s in fut["step"]],
            y=fut["forecast"], name="forward plan", mode="lines",
            line=dict(color=COLORS["accent"], width=2.2),
        )
    )
if len(history):
    fig.add_vline(
        x=pd.Timestamp(week_lookup.get(holdout_start, history["week_start"].max())).timestamp()
        * 1000,
        line_dash="dash", line_color=COLORS["muted"],
        annotation_text="holdout starts", annotation_position="top left",
    )
fig.update_layout(xaxis_title=None, yaxis_title="units per week")
show(fig, height=380)
how_calculated(
    "Both models are fitted only on weeks before the dashed line and asked for the "
    f"{a['sample']['holdout_weeks']}-week holdout in one shot. The forward plan continues "
    f"{a['sample']['forecast_horizon_weeks']} weeks past all history and is what the "
    "inventory policy consumes."
)

st.divider()
with st.expander("How the models work", expanded=False):
    st.markdown(
        f"""
**`{baseline_name}`** — last year's same week, falling back to the last observed value
when there is under a year of history. Cheap, surprisingly hard to beat, and the bar any
real model has to clear.

**`{candidate_name}`** — Holt-Winters exponential smoothing with a damped trend and
additive annual seasonality. It degrades in steps rather than failing: seasonal →
trend-only → seasonal naive, so a degenerate series still returns a usable number.

**No third model.** Two is the brief, and the value of this project is the decision
downstream of the forecast, not a leaderboard position.
"""
    )
