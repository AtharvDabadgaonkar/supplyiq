"""Shared Streamlit UI: cached loading, badges, captions, charts.

Every page goes through the cached helpers here, so each Parquet file is read
once per session. On the free Streamlit Community Cloud tier that is the
difference between a snappy app and one that runs out of memory.
"""
from __future__ import annotations

from typing import Any, Iterable

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
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
from supplyiq.pipeline import ARTIFACTS, load_artifact, missing_artifacts
from supplyiq.schemas import TABLES_BY_NAME

PAGE_ICON = "🛳️"

# Colour-blind safe, consistent across every page.
COLORS = {
    "primary": "#1f6feb",
    "accent": "#e8710a",
    "good": "#1a7f37",
    "bad": "#cf222e",
    "muted": "#8c959f",
    "current": "#8c959f",
    "supplyiq": "#1f6feb",
}
SEQUENCE = ["#1f6feb", "#e8710a", "#1a7f37", "#8250df", "#cf222e", "#9a6700"]


def page_setup(title: str, icon: str = PAGE_ICON, wide: bool = True) -> None:
    """Set page config. Tolerant of being called under st.navigation."""
    try:
        st.set_page_config(
            page_title=f"SupplyIQ · {title}",
            page_icon=icon,
            layout="wide" if wide else "centered",
            initial_sidebar_state="expanded",
        )
    except Exception:
        # Already configured by the entry point; harmless.
        pass


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
    return load_table(name, columns=list(columns) if columns else None)


@st.cache_data(show_spinner="Loading precomputed results…")
def get_artifact(name: str, columns: tuple[str, ...] | None = None) -> pd.DataFrame:
    return load_artifact(name, columns=list(columns) if columns else None)


@st.cache_data(show_spinner=False)
def get_metrics() -> dict[str, Any]:
    import json

    if not paths.METRICS_FILE.exists():
        return {}
    return json.loads(paths.METRICS_FILE.read_text())


@st.cache_data(show_spinner=False)
def get_provider_status() -> dict[str, Any]:
    return provider_status()


# --------------------------------------------------------------------------- #
# Guards and badges
# --------------------------------------------------------------------------- #
def require_data() -> Provenance:
    """Stop with a friendly message rather than a traceback if data is missing."""
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


def require_artifacts(names: Iterable[str] = ()) -> None:
    """Stop with a friendly message if `make metrics` has not been run."""
    missing = missing_artifacts()
    needed = [n for n in (names or ARTIFACTS) if n in missing]
    if needed:
        st.warning(
            f"#### Precomputed results missing\n\nThis page needs "
            f"`{'`, `'.join(needed)}`, which are built by `make metrics`."
        )
        st.code("make metrics", language="bash")
        st.stop()


def data_badge(prov: Provenance, compact: bool = False) -> None:
    text = prov.badge if not compact else ("Real data" if prov.is_real else "Synthetic sample")
    if prov.is_real:
        st.success(f"**Data:** {text}", icon="✅")
    else:
        st.warning(f"**Data:** {text}", icon="⚠️")


def agent_badge() -> None:
    status = get_provider_status()
    if status["is_mock"]:
        st.info(f"**Agent:** {status['badge']}", icon="🤖")
    else:
        st.success(f"**Agent:** {status['badge']} ({status['model']})", icon="🤖")


def how_calculated(text: str) -> None:
    """Every chart carries one of these. No exceptions."""
    st.caption(f"**How it's calculated:** {text}")


def simulated_note(fields: Iterable[str]) -> None:
    fields = list(fields)
    if fields:
        st.caption(f"⚠️ Simulated inputs on this view: `{'`, `'.join(fields)}`")


def sidebar_context() -> Provenance:
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


# --------------------------------------------------------------------------- #
# Charts
# --------------------------------------------------------------------------- #
def style(fig: go.Figure, height: int = 340, legend: bool = True) -> go.Figure:
    fig.update_layout(
        height=height,
        margin=dict(l=10, r=10, t=30, b=10),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        showlegend=legend,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0, title=None),
        colorway=SEQUENCE,
        hovermode="x unified",
    )
    fig.update_xaxes(showgrid=False, zeroline=False)
    fig.update_yaxes(gridcolor="rgba(140,149,159,0.2)", zeroline=False)
    return fig


def show(fig: go.Figure, height: int = 340, legend: bool = True) -> None:
    st.plotly_chart(style(fig, height, legend), use_container_width=True)


def kpi_row(tiles: list[dict[str, Any]], columns: int | None = None) -> None:
    """KPI tiles. `delta_good` flags whether an increase is good or bad."""
    cols = st.columns(columns or len(tiles))
    for col, tile in zip(cols, tiles):
        delta = tile.get("delta")
        col.metric(
            tile["label"],
            tile["value"],
            delta=delta,
            delta_color=tile.get("delta_color", "normal"),
            help=tile.get("help"),
        )


def bar(df: pd.DataFrame, x: str, y: str, color: str | None = None, **kw) -> go.Figure:
    return px.bar(df, x=x, y=y, color=color, **kw)


def delta_or_none(value: float, text: str, tolerance: float = 1e-9) -> str | None:
    """Suppress a delta that is exactly zero.

    Streamlit renders `delta=0` as a green upward arrow, which reads as an
    improvement when nothing changed -- actively misleading on the zero-
    disruption scenario, where nothing changing is the whole point.
    """
    return None if abs(float(value)) < tolerance else text


def pct(value: float, digits: int = 1) -> str:
    return f"{value:.{digits}%}"


def money(value: float, digits: int = 0) -> str:
    sign = "-" if value < 0 else ""
    v = abs(float(value))
    if v >= 1_000_000:
        return f"{sign}${v / 1e6:.2f}M"
    if v >= 10_000:
        return f"{sign}${v / 1e3:,.0f}k"
    return f"{sign}${v:,.{digits}f}"
