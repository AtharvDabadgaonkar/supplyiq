"""UI pieces every page shares: caching, badges, captions, error handling.

All data loading goes through the cached helpers here so Streamlit Cloud reads
each Parquet file once per session.
"""
from __future__ import annotations

from typing import Any

import pandas as pd
import streamlit as st

from supplyiq import paths
from supplyiq.agent import provider_status
from supplyiq.config import load_assumptions
from supplyiq.data import (
    DataNotBuiltError,
    Provenance,
    is_built,
    load_table,
    missing_tables,
    read_provenance,
)
from supplyiq.schemas import TABLES_BY_NAME

PAGE_ICON = "🛳️"


def page_setup(title: str, icon: str = PAGE_ICON, wide: bool = True) -> None:
    st.set_page_config(
        page_title=f"SupplyIQ · {title}",
        page_icon=icon,
        layout="wide" if wide else "centered",
        initial_sidebar_state="expanded",
    )


# --------------------------------------------------------------------------- #
# Cached loaders
# --------------------------------------------------------------------------- #
@st.cache_data(show_spinner=False)
def get_assumptions() -> dict[str, Any]:
    return load_assumptions()


@st.cache_data(show_spinner=False)
def get_provenance() -> Provenance | None:
    try:
        return read_provenance()
    except DataNotBuiltError:
        return None


@st.cache_data(show_spinner="Loading data…")
def get_table(name: str, columns: tuple[str, ...] | None = None) -> pd.DataFrame:
    """Cached table read. `columns` is a tuple so it is hashable for the cache."""
    return load_table(name, columns=list(columns) if columns else None)


@st.cache_data(show_spinner=False)
def get_provider_status() -> dict[str, Any]:
    return provider_status()


# --------------------------------------------------------------------------- #
# Guards and badges
# --------------------------------------------------------------------------- #
def require_data() -> Provenance:
    """Stop the page with a friendly message if the sample has not been built.

    Streamlit Cloud resets the filesystem on redeploy, so a missing sample means
    the Parquet files were not committed — never a crash traceback for the user.
    """
    if not is_built():
        st.error("#### Data not available")
        st.markdown(
            "The processed sample is missing, so there is nothing to plan with yet.\n\n"
            f"**Missing files:** `{'`, `'.join(missing_tables()) or 'provenance.json'}`\n\n"
            "If you are running this locally, build it with:"
        )
        st.code("make ingest", language="bash")
        st.caption(
            f"Expected location: `{paths.SAMPLE_DIR.relative_to(paths.REPO_ROOT)}/`. "
            "On a deployed app this means the sample was not committed to the repo."
        )
        st.stop()
    prov = get_provenance()
    assert prov is not None
    return prov


def data_badge(prov: Provenance) -> None:
    """The real-vs-simulated label. Shown on every page, by design."""
    if prov.is_real:
        st.success(f"**Data:** {prov.badge}", icon="✅")
    else:
        st.warning(f"**Data:** {prov.badge}", icon="⚠️")


def agent_badge() -> None:
    status = get_provider_status()
    if status["is_mock"]:
        st.info(f"**Agent:** {status['badge']}", icon="🤖")
    else:
        st.success(f"**Agent:** {status['badge']} ({status['model']})", icon="🤖")


def how_calculated(text: str) -> None:
    """The one-line 'how it's calculated' caption that every chart must carry."""
    st.caption(f"**How it's calculated:** {text}")


def simulated_note(fields: list[str] | tuple[str, ...]) -> None:
    if fields:
        st.caption(f"⚠️ Simulated fields on this view: `{'`, `'.join(fields)}`")


def sidebar_context() -> Provenance:
    """Shared sidebar: provenance, agent mode, sample scope."""
    prov = require_data()
    a = get_assumptions()
    with st.sidebar:
        st.markdown("### SupplyIQ")
        st.caption("AI Supply Chain Control Tower & Disruption Planner")
        st.divider()
        st.markdown(f"**Data origin:** `{prov.origin}`")
        st.caption(prov.badge)
        status = get_provider_status()
        st.markdown(f"**Agent:** `{status['active']}`")
        st.caption(status["badge"])
        st.divider()
        s = a["sample"]
        st.markdown("**Sample scope**")
        st.caption(
            f"{s['state']} · {len(s['categories'])} categories · "
            f"{s['n_planning_items']} planning items · {s['n_weeks']} weeks · "
            f"{s['holdout_weeks']}-week holdout"
        )
    return prov


def table_summary() -> pd.DataFrame:
    """Row counts and file sizes for the data health view."""
    rows = []
    for name, spec in TABLES_BY_NAME.items():
        path = paths.SAMPLE_DIR / spec.filename
        rows.append(
            {
                "table": name,
                "file": spec.filename,
                "exists": path.exists(),
                "size_mb": round(path.stat().st_size / 1e6, 3) if path.exists() else 0.0,
                "columns": len(spec.columns),
                "simulated_columns": len(spec.simulated_columns),
                "primary_key": ", ".join(spec.primary_key),
            }
        )
    return pd.DataFrame(rows)
