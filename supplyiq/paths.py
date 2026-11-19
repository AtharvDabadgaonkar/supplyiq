"""Single place that knows where things live.

Every path is derived from the repo root so the app behaves the same locally
and on Streamlit Community Cloud (where the working directory is the repo root
but should never be assumed).
"""
from __future__ import annotations

from pathlib import Path

# supplyiq/paths.py -> supplyiq/ -> repo root
REPO_ROOT = Path(__file__).resolve().parent.parent

CONFIG_DIR = REPO_ROOT / "config"
ASSUMPTIONS_FILE = CONFIG_DIR / "assumptions.yaml"

DATA_DIR = REPO_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
SAMPLE_DIR = DATA_DIR / "sample"
PROVENANCE_FILE = SAMPLE_DIR / "provenance.json"

REPORTS_DIR = REPO_ROOT / "reports"
METRICS_FILE = REPORTS_DIR / "metrics.json"

DOCS_DIR = REPO_ROOT / "docs"


def ensure_dirs() -> None:
    """Create the output directories if they are missing."""
    for d in (SAMPLE_DIR, RAW_DIR, REPORTS_DIR):
        d.mkdir(parents=True, exist_ok=True)
