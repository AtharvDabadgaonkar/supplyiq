from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from supplyiq.config import load_assumptions  # noqa: E402
from supplyiq.data import is_built, load_all, read_provenance  # noqa: E402


@pytest.fixture(scope="session")
def assumptions() -> dict:
    return load_assumptions()


@pytest.fixture(scope="session")
def built() -> bool:
    return is_built()


@pytest.fixture(scope="session")
def tables(built):
    if not built:
        pytest.skip("data/sample not built; run `make ingest`")
    return load_all()


@pytest.fixture(scope="session")
def provenance(built):
    if not built:
        pytest.skip("data/sample not built; run `make ingest`")
    return read_provenance()
