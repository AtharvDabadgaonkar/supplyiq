"""Data & Assumptions — provenance and reproducibility, checkable in the UI."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from app.components import (  # noqa: E402
    data_badge,
    get_assumptions,
    get_metrics,
    get_table,
    how_calculated,
    page_setup,
    sidebar_context,
    table_summary,
)
from supplyiq.config import simulated_sections  # noqa: E402
from supplyiq.schemas import ALL_TABLES, describe  # noqa: E402

page_setup("Data & Assumptions")
prov = sidebar_context()
a = get_assumptions()
m = get_metrics()

st.title("Data & Assumptions")
st.markdown(
    "Every number in this app should be checkable. This page is where you check it."
)
data_badge(prov)

st.subheader("What is real and what is simulated")
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
        hide_index=True, width="stretch",
    )
with c2:
    st.dataframe(
        origin_df[origin_df["origin"] == "simulated"], hide_index=True,
        width="stretch", height=280,
    )
how_calculated(
    "Taken from `simulated_columns` on each TableSpec in `supplyiq/schemas.py`, which the "
    "ingest tests assert against. Simulated sections of the config: "
    f"**{', '.join(simulated_sections(a))}**."
)

st.caption(prov.notes)

st.divider()
st.subheader("Data health")
left, right = st.columns([3, 2])
with left:
    st.dataframe(table_summary(), hide_index=True, width="stretch")
    how_calculated(
        "Read from the committed Parquet in `data/sample/`. The app never touches "
        "`data/raw/`, which is gitignored."
    )
with right:
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

st.divider()
st.subheader("Schema contract")
st.dataframe(pd.DataFrame(describe()), hide_index=True, width="stretch")
how_calculated(
    "Both ingest paths — real Kaggle files and the synthetic fallback — are validated "
    "against these specs before anything is written to `data/sample/`."
)

st.divider()
st.subheader("Reproducing every number")
st.markdown(
    f"""
```bash
make setup     # venv + pinned dependencies, builds data/sample/
make ingest    # rebuild the sample (seed {a['sample']['random_seed']}, byte-identical)
make metrics   # recompute reports/metrics.json and the precomputed artifacts
make test      # the full suite, including the no-leakage and safety-stock checks
```

The README's figures are rendered from `reports/metrics.json` by
`scripts/render_readme.py`, and a test fails the build if that block goes stale. No
number in this project is typed by hand.
"""
)

st.divider()
st.subheader("Assumptions registry")
st.caption(
    "Every planning parameter lives in `config/assumptions.yaml`. A test fails the build "
    "if one is hard-coded in the package instead."
)
tab1, tab2, tab3 = st.tabs(["Suppliers (simulated)", "Costs (simulated)", "Full file"])
with tab1:
    st.dataframe(get_table("suppliers"), hide_index=True, width="stretch")
with tab2:
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
        hide_index=True, width="stretch",
    )
with tab3:
    st.json(a, expanded=False)

st.divider()
st.subheader("Sample preview")
t1, t2, t3 = st.tabs(["Demand", "Planning items", "Purchase orders"])
with t1:
    st.dataframe(get_table("demand").head(200), hide_index=True, width="stretch")
with t2:
    st.dataframe(get_table("sku_master").head(200), hide_index=True, width="stretch")
with t3:
    st.dataframe(get_table("purchase_orders").head(200), hide_index=True, width="stretch")
