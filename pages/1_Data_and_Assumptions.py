"""Data & Assumptions — the provenance and reproducibility page.

Exists from Phase 1 so that "every number is reproducible" is something a
reviewer can check in the UI, not just a claim in the README.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from app.components import (  # noqa: E402
    data_badge,
    get_assumptions,
    get_table,
    how_calculated,
    page_setup,
    sidebar_context,
)
from supplyiq.schemas import ALL_TABLES, describe  # noqa: E402

page_setup("Data & Assumptions")
prov = sidebar_context()
a = get_assumptions()

st.title("Data & Assumptions")
data_badge(prov)

st.subheader("What is real and what is simulated")
st.markdown(
    "Real data and simulated data are never mixed without a label. "
    "The table below is the authoritative list."
)
rows = []
for spec in ALL_TABLES:
    for col in spec.columns:
        rows.append(
            {
                "table": spec.name,
                "column": col,
                "origin": "simulated" if col in spec.simulated_columns else prov.origin,
            }
        )
origin_df = pd.DataFrame(rows)
c1, c2 = st.columns([2, 3])
with c1:
    st.dataframe(
        origin_df.groupby(["table", "origin"]).size().rename("columns").reset_index(),
        hide_index=True,
        width="stretch",
    )
with c2:
    st.dataframe(
        origin_df[origin_df["origin"] == "simulated"], hide_index=True, width="stretch", height=280
    )
how_calculated(
    "Taken from `simulated_columns` on each TableSpec in `supplyiq/schemas.py`, "
    "which the ingest tests assert against."
)

st.divider()
st.subheader("Schema contract")
st.dataframe(pd.DataFrame(describe()), hide_index=True, width="stretch")
how_calculated(
    "Both ingest paths — real Kaggle files and the synthetic fallback — are validated "
    "against these specs before anything is written to `data/sample/`."
)

st.divider()
st.subheader("Reproducing these numbers")
st.markdown(
    f"""
1. `make setup` — create the virtual environment and install pinned dependencies.
2. `make ingest` — rebuild `data/sample/` (seed `{a['sample']['random_seed']}`, so it is byte-identical).
3. `make metrics` — recompute every reported figure into `reports/metrics.json`.
4. `make test` — run the test suite, including the no-leakage and safety-stock checks.

The README quotes its numbers from `reports/metrics.json`; none are typed by hand.
"""
)

st.divider()
st.subheader("Sample preview")
tab1, tab2, tab3 = st.tabs(["Demand", "Planning items", "Purchase orders"])
with tab1:
    st.dataframe(get_table("demand").head(200), hide_index=True, width="stretch")
with tab2:
    st.dataframe(get_table("sku_master").head(200), hide_index=True, width="stretch")
with tab3:
    st.dataframe(get_table("purchase_orders").head(200), hide_index=True, width="stretch")
