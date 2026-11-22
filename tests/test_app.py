"""Smoke tests: every Streamlit page loads without raising.

These are what catch a deploy breaking — a bad import, a missing file, an API
change in Streamlit — before Streamlit Community Cloud shows a traceback.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from supplyiq.paths import REPO_ROOT

AppTest = pytest.importorskip("streamlit.testing.v1").AppTest

ENTRY_POINT = REPO_ROOT / "streamlit_app.py"
PAGES = sorted((REPO_ROOT / "pages").glob("*.py"))
ALL_SCRIPTS = [ENTRY_POINT, *PAGES]


def ids(paths: list[Path]) -> list[str]:
    return [p.name for p in paths]


@pytest.mark.slow
@pytest.mark.parametrize("script", ALL_SCRIPTS, ids=ids(ALL_SCRIPTS))
def test_page_runs_without_exception(script: Path, built: bool):
    if not built:
        pytest.skip("data/sample not built; run `make ingest`")
    at = AppTest.from_file(str(script), default_timeout=60).run()
    assert not at.exception, f"{script.name} raised: {at.exception}"


@pytest.mark.slow
def test_entry_point_shows_the_data_provenance_badge(built: bool):
    if not built:
        pytest.skip("data/sample not built")
    at = AppTest.from_file(str(ENTRY_POINT), default_timeout=60).run()
    text = " ".join(
        [*(w.value for w in at.warning), *(s.value for s in at.success), *(m.value for m in at.markdown)]
    )
    assert "Data:" in text
    assert "simulated" in text.lower()


@pytest.mark.slow
def test_entry_point_shows_the_agent_mode_badge(built: bool):
    if not built:
        pytest.skip("data/sample not built")
    at = AppTest.from_file(str(ENTRY_POINT), default_timeout=60).run()
    text = " ".join([*(i.value for i in at.info), *(s.value for s in at.success)])
    assert "Agent:" in text


@pytest.mark.slow
def test_every_page_carries_a_how_its_calculated_caption(built: bool):
    """A chart without a stated method is not allowed on any page."""
    if not built:
        pytest.skip("data/sample not built")
    for script in ALL_SCRIPTS:
        at = AppTest.from_file(str(script), default_timeout=60).run()
        captions = " ".join(c.value for c in at.caption)
        assert "How it's calculated" in captions, script.name


def test_entry_point_is_at_the_repo_root():
    """Streamlit Community Cloud requires this exact path."""
    assert ENTRY_POINT.exists()
    assert ENTRY_POINT.parent == REPO_ROOT


def test_pages_directory_sits_beside_the_entry_point():
    assert (REPO_ROOT / "pages").is_dir()
    assert PAGES, "no pages found"


def test_secrets_example_is_committed_and_the_real_file_is_not():
    assert (REPO_ROOT / ".streamlit" / "secrets.toml.example").exists()
    gitignore = (REPO_ROOT / ".gitignore").read_text()
    assert ".streamlit/secrets.toml" in gitignore


def test_requirements_are_fully_pinned():
    """An unpinned dependency means a deploy can break without a code change."""
    lines = [
        line.strip()
        for line in (REPO_ROOT / "requirements.txt").read_text().splitlines()
        if line.strip() and not line.startswith("#")
    ]
    assert lines
    for line in lines:
        assert "==" in line, f"not pinned: {line}"


def test_runtime_requirements_exclude_modelling_libraries():
    """Forecasts are precomputed, so the cloud app must not carry heavy deps."""
    runtime = (REPO_ROOT / "requirements.txt").read_text().lower()
    for heavy in ("prophet", "torch", "tensorflow", "statsmodels", "lightgbm", "scikit-learn"):
        assert heavy not in runtime, f"{heavy} does not belong in requirements.txt"


def test_dev_requirements_include_the_modelling_libraries():
    dev = (REPO_ROOT / "requirements-dev.txt").read_text().lower()
    assert "statsmodels" in dev
    assert "pytest" in dev
    assert "-r requirements.txt" in dev


def test_raw_data_is_gitignored():
    gitignore = (REPO_ROOT / ".gitignore").read_text()
    assert "data/raw/*" in gitignore


# --------------------------------------------------------------------------- #
# Phase 5: the pages must actually contain their content, not just not crash
# --------------------------------------------------------------------------- #
EXPECTED_PAGES = {
    "0_Control_Tower.py": "Control Tower",
    "1_Demand_Forecast.py": "Demand Forecast",
    "2_Inventory_Policy.py": "Inventory Policy",
    "3_Supplier_Performance.py": "Supplier Performance",
    "4_Scenario_Planner.py": "Scenario Planner",
    "5_Planning_Agent.py": "Planning Agent",
    "6_Backtest_Results.py": "Backtest Results",
    "7_Data_and_Assumptions.py": "Data & Assumptions",
}


def test_every_expected_page_exists():
    present = {p.name for p in PAGES}
    assert present == set(EXPECTED_PAGES), present.symmetric_difference(EXPECTED_PAGES)


def test_entry_point_declares_navigation():
    """Without st.navigation the sidebar names pages after their filenames."""
    source = ENTRY_POINT.read_text()
    assert "st.navigation" in source
    for filename in EXPECTED_PAGES:
        assert filename in source, filename


@pytest.mark.slow
@pytest.mark.parametrize("filename,title", sorted(EXPECTED_PAGES.items()))
def test_page_renders_its_title(filename: str, title: str, built: bool):
    if not built:
        pytest.skip("data/sample not built")
    at = AppTest.from_file(str(REPO_ROOT / "pages" / filename), default_timeout=90).run()
    assert not at.exception, f"{filename} raised: {at.exception}"
    headings = " ".join(h.value for h in at.title) + " ".join(m.value for m in at.markdown)
    assert title.replace(" & ", " & ") in headings or title in headings, filename


@pytest.mark.slow
def test_scenario_planner_exposes_all_three_levers(built: bool):
    if not built:
        pytest.skip("data/sample not built")
    at = AppTest.from_file(str(REPO_ROOT / "pages" / "4_Scenario_Planner.py"),
                           default_timeout=90).run()
    assert not at.exception
    labels = " ".join(s.label for s in at.slider)
    assert "Supplier delay" in labels
    assert "Demand change" in labels
    assert "Tariff change" in labels


@pytest.mark.slow
def test_scenario_planner_opens_on_the_no_disruption_case(built: bool):
    """Defaults must be the null scenario, matching the simulator guarantee."""
    if not built:
        pytest.skip("data/sample not built")
    at = AppTest.from_file(str(REPO_ROOT / "pages" / "4_Scenario_Planner.py"),
                           default_timeout=90).run()
    for slider in at.slider:
        assert float(slider.value) == 0.0, slider.label
    text = " ".join(i.value for i in at.info)
    assert "reproduces the baseline exactly" in text


@pytest.mark.slow
def test_planning_agent_shows_the_demo_mode_badge(built: bool, monkeypatch):
    if not built:
        pytest.skip("data/sample not built")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    at = AppTest.from_file(str(REPO_ROOT / "pages" / "5_Planning_Agent.py"),
                           default_timeout=120).run()
    assert not at.exception
    text = " ".join(i.value for i in at.info)
    assert "rule-based explanations" in text


@pytest.mark.slow
def test_planning_agent_warns_that_decisions_are_not_persisted(built: bool):
    if not built:
        pytest.skip("data/sample not built")
    at = AppTest.from_file(str(REPO_ROOT / "pages" / "5_Planning_Agent.py"),
                           default_timeout=120).run()
    text = " ".join(i.value for i in at.info)
    assert "not permanent" in text


@pytest.mark.slow
def test_backtest_page_states_the_capital_increase_at_default_settings(built: bool):
    """The unflattering number must be on the page, not only in the repo."""
    if not built:
        pytest.skip("data/sample not built")
    at = AppTest.from_file(str(REPO_ROOT / "pages" / "6_Backtest_Results.py"),
                           default_timeout=90).run()
    assert not at.exception
    warnings = " ".join(w.value for w in at.warning)
    assert "more working capital" in warnings
    assert "not by itself an achievement" in warnings


@pytest.mark.slow
def test_every_page_labels_the_data_origin(built: bool):
    """A reviewer must never see a number without knowing if it is real."""
    if not built:
        pytest.skip("data/sample not built")
    for filename in EXPECTED_PAGES:
        at = AppTest.from_file(str(REPO_ROOT / "pages" / filename), default_timeout=90).run()
        sidebar = " ".join(m.value for m in at.sidebar.markdown)
        assert "Data origin" in sidebar, filename
