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
