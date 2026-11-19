"""Reading the processed sample, plus data provenance.

The app never touches data/raw/. It reads only the committed Parquet in
data/sample/, which keeps Streamlit Cloud startup fast and memory low.
Callers may pass `columns=` to load a projection instead of the whole table.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Literal

import pandas as pd

from supplyiq import paths
from supplyiq.schemas import TABLES_BY_NAME, TableSpec, validate

Origin = Literal["kaggle", "synthetic"]


class DataNotBuiltError(FileNotFoundError):
    """Raised when the processed sample is missing. Carries a planner-friendly hint."""

    def __init__(self, missing: Iterable[str]) -> None:
        self.missing = list(missing)
        super().__init__(
            "The processed data sample has not been built yet.\n"
            f"Missing: {', '.join(self.missing)}\n"
            "Run `make ingest` (or `make setup`) to build it."
        )


@dataclass(frozen=True)
class Provenance:
    """What the committed sample actually is. Written by scripts/ingest.py."""

    origin: Origin
    built_at: str
    demand_source: str
    orders_source: str
    simulated_sections: list[str]
    sample_scope: dict
    row_counts: dict[str, int]
    notes: str

    @property
    def is_real(self) -> bool:
        return self.origin == "kaggle"

    @property
    def badge(self) -> str:
        """Short label shown in the UI next to any chart built on this data."""
        if self.is_real:
            return "Real data: M5 + DataCo (Kaggle) · supplier, cost and tariff fields simulated"
        return "Synthetic sample standing in for M5 + DataCo · all fields simulated"

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2) + "\n"


def write_provenance(p: Provenance, path: Path | None = None) -> Path:
    target = path or paths.PROVENANCE_FILE
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(p.to_json(), encoding="utf-8")
    return target


def read_provenance(path: Path | None = None) -> Provenance:
    target = path or paths.PROVENANCE_FILE
    if not target.exists():
        raise DataNotBuiltError([target.name])
    raw = json.loads(target.read_text(encoding="utf-8"))
    return Provenance(**raw)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# --------------------------------------------------------------------------- #
# Table access
# --------------------------------------------------------------------------- #
def table_path(name: str, directory: Path | None = None) -> Path:
    spec = TABLES_BY_NAME[name]
    return (directory or paths.SAMPLE_DIR) / spec.filename


def missing_tables(directory: Path | None = None) -> list[str]:
    return [n for n in TABLES_BY_NAME if not table_path(n, directory).exists()]


def is_built(directory: Path | None = None) -> bool:
    return not missing_tables(directory) and (
        (directory or paths.SAMPLE_DIR) / paths.PROVENANCE_FILE.name
    ).exists()


def load_table(
    name: str,
    columns: list[str] | None = None,
    directory: Path | None = None,
    check: bool = False,
) -> pd.DataFrame:
    """Load one processed table.

    `columns` reads a projection straight out of Parquet, which matters on the
    512 MB Streamlit Cloud tier. `check` runs the full schema validation and is
    used by tests and the ingest script, not by the UI.
    """
    spec: TableSpec = TABLES_BY_NAME[name]
    path = table_path(name, directory)
    if not path.exists():
        raise DataNotBuiltError([spec.filename])
    df = pd.read_parquet(path, columns=columns)
    if check and columns is None:
        validate(df, spec)
    return df


def write_table(df: pd.DataFrame, name: str, directory: Path | None = None) -> Path:
    """Validate against the spec, then write Parquet. Ingest's only writer."""
    spec = TABLES_BY_NAME[name]
    validate(df, spec)
    out_dir = directory or paths.SAMPLE_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / spec.filename
    df.to_parquet(path, index=False, compression="snappy")
    return path


def load_all(directory: Path | None = None, check: bool = False) -> dict[str, pd.DataFrame]:
    missing = missing_tables(directory)
    if missing:
        raise DataNotBuiltError(missing)
    return {n: load_table(n, directory=directory, check=check) for n in TABLES_BY_NAME}


# --------------------------------------------------------------------------- #
# Train / test split — the one definition used everywhere
# --------------------------------------------------------------------------- #
def split_weeks(demand: pd.DataFrame, holdout_weeks: int) -> tuple[int, int]:
    """Return (last_train_week_index, first_test_week_index).

    Time-based: training is every week strictly before the cut, the holdout is
    the final `holdout_weeks`. No SKU-level randomisation, so no leakage.
    """
    if holdout_weeks < 1:
        raise ValueError("holdout_weeks must be >= 1")
    max_week = int(demand["week_index"].max())
    first_test = max_week - holdout_weeks + 1
    if first_test <= 0:
        raise ValueError(
            f"holdout_weeks={holdout_weeks} leaves no training data "
            f"(history is {max_week + 1} weeks)"
        )
    return first_test - 1, first_test


def train_test(
    demand: pd.DataFrame, holdout_weeks: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split demand into (train, test) on week_index."""
    _, first_test = split_weeks(demand, holdout_weeks)
    train = demand.loc[demand["week_index"] < first_test].copy()
    test = demand.loc[demand["week_index"] >= first_test].copy()
    return train, test
