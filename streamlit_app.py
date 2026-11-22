"""SupplyIQ — Control Tower. Entry point for Streamlit Community Cloud.

Must stay at the repo root: Streamlit Cloud takes this file as the main module.
Navigation is declared explicitly with st.navigation so the pages get real
titles and ordering rather than being named after their filenames.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import streamlit as st  # noqa: E402

from app.components import PAGE_ICON  # noqa: E402

st.set_page_config(
    page_title="SupplyIQ · Control Tower",
    page_icon=PAGE_ICON,
    layout="wide",
    initial_sidebar_state="expanded",
)

PAGES = [
    st.Page("pages/0_Control_Tower.py", title="Control Tower", icon="🗼", default=True),
    st.Page("pages/1_Demand_Forecast.py", title="Demand Forecast", icon="📈"),
    st.Page("pages/2_Inventory_Policy.py", title="Inventory Policy", icon="📦"),
    st.Page("pages/3_Supplier_Performance.py", title="Supplier Performance", icon="🚚"),
    st.Page("pages/4_Scenario_Planner.py", title="Scenario Planner", icon="🌪️"),
    st.Page("pages/5_Planning_Agent.py", title="Planning Agent", icon="🤖"),
    st.Page("pages/6_Backtest_Results.py", title="Backtest Results", icon="🧪"),
    st.Page("pages/7_Data_and_Assumptions.py", title="Data & Assumptions", icon="📋"),
]

st.navigation(PAGES).run()
