from __future__ import annotations

import sys
import warnings
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


@pytest.fixture(scope="session")
def planning(tables, assumptions):
    """The PLAN/SENSE artifacts, built once per test session.

    `with_evaluation=False` keeps it fast: the ETS fit over the whole panel
    takes ~25s and only the forecast tests need it.
    """
    from supplyiq.pipeline import build

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return build(tables, assumptions, with_evaluation=False)


@pytest.fixture(scope="session")
def policy(planning):
    return planning.policy


@pytest.fixture(scope="session")
def segmented(planning):
    return planning.segmented


@pytest.fixture(scope="session")
def performance(planning):
    return planning.performance
