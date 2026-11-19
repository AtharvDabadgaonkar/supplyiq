"""One definition of how the planning artifacts are built.

`make metrics` precomputes these and commits the results; the Streamlit app
loads the committed Parquet rather than recomputing (the free Cloud tier has
neither the RAM nor the patience). The tests build them through this same
function, so what is tested is what ships.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from supplyiq import paths
from supplyiq.config import load_assumptions
from supplyiq.data import load_all, train_test
from supplyiq.forecast import Evaluation, demand_stats, evaluate, future_forecast
from supplyiq.inventory import build_policy, segment
from supplyiq.suppliers import item_lead_times, overall_performance, supplier_performance

ITEM_KEYS = ["sku", "store"]

# Precomputed artifacts, written next to the data sample.
ARTIFACT_DIR = paths.DATA_DIR / "derived"
ARTIFACTS = {
    "policy": "policy.parquet",
    "supplier_performance": "supplier_performance.parquet",
    "forecast_accuracy": "forecast_accuracy.parquet",
    "forecasts": "forecasts.parquet",
    "future_forecast": "future_forecast.parquet",
}


@dataclass
class Planning:
    """Everything the PLAN and SENSE modules produce, for one data sample."""

    train: pd.DataFrame
    test: pd.DataFrame
    stats: pd.DataFrame
    segmented: pd.DataFrame
    performance: pd.DataFrame
    overall: dict[str, float]
    policy: pd.DataFrame
    evaluation: Evaluation | None = None
    future: pd.DataFrame | None = None


def item_revenue(train: pd.DataFrame, stats: pd.DataFrame) -> pd.Series:
    """Revenue per planning item, aligned to the row order of `stats`."""
    revenue = (
        train.groupby(ITEM_KEYS, observed=True)["revenue"].sum().rename("revenue").reset_index()
    )
    aligned = stats[ITEM_KEYS].merge(revenue, on=ITEM_KEYS, how="left", validate="one_to_one")
    return aligned["revenue"].fillna(0.0)


def build(
    tables: dict[str, pd.DataFrame] | None = None,
    a: dict[str, Any] | None = None,
    with_evaluation: bool = True,
    service_level: float | None = None,
) -> Planning:
    """Build segmentation, supplier performance and the inventory policy.

    Everything that feeds the policy -- demand mean, demand sd, revenue -- is
    computed on the TRAINING weeks only. Using all weeks would quietly leak the
    holdout into the policy the backtest is meant to judge.
    """
    a = a or load_assumptions()
    tables = tables or load_all()
    holdout = int(a["sample"]["holdout_weeks"])

    train, test = train_test(tables["demand"], holdout)
    stats = demand_stats(train)
    segmented = segment(stats, item_revenue(train, stats), a)

    performance = supplier_performance(tables["purchase_orders"], tables["suppliers"])
    lead_times = item_lead_times(tables["sku_master"], performance, a)
    policy = build_policy(
        segmented, tables["sku_master"], lead_times, a, service_level=service_level
    )

    evaluation = evaluate(tables["demand"], a) if with_evaluation else None
    future = (
        future_forecast(tables["demand"], str(a["forecast"]["candidate_model"]), a)
        if with_evaluation
        else None
    )

    return Planning(
        train=train,
        test=test,
        stats=stats,
        segmented=segmented,
        performance=performance,
        overall=overall_performance(tables["purchase_orders"]),
        policy=policy,
        evaluation=evaluation,
        future=future,
    )


# --------------------------------------------------------------------------- #
# Persisting the precomputed artifacts
# --------------------------------------------------------------------------- #
def artifact_path(name: str, directory: Path | None = None) -> Path:
    return (directory or ARTIFACT_DIR) / ARTIFACTS[name]


def missing_artifacts(directory: Path | None = None) -> list[str]:
    return [n for n in ARTIFACTS if not artifact_path(n, directory).exists()]


def write_artifacts(planning: Planning, directory: Path | None = None) -> dict[str, Path]:
    """Write the precomputed tables the app loads at runtime."""
    out_dir = directory or ARTIFACT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    written: dict[str, Path] = {}

    frames: dict[str, pd.DataFrame] = {
        "policy": planning.policy,
        "supplier_performance": planning.performance,
    }
    if planning.evaluation is not None:
        frames["forecast_accuracy"] = planning.evaluation.per_item
        frames["forecasts"] = planning.evaluation.forecasts
    if planning.future is not None:
        frames["future_forecast"] = planning.future

    for name, df in frames.items():
        path = artifact_path(name, out_dir)
        # Booleans and object columns round-trip badly through Parquet unless cast.
        safe = df.copy()
        for column in safe.columns:
            if safe[column].dtype == object:
                safe[column] = safe[column].astype("string")
        safe.to_parquet(path, index=False, compression="snappy")
        written[name] = path
    return written


def load_artifact(name: str, directory: Path | None = None, columns: list[str] | None = None):
    path = artifact_path(name, directory)
    if not path.exists():
        raise FileNotFoundError(
            f"{path.name} has not been built. Run `make metrics` to precompute it."
        )
    return pd.read_parquet(path, columns=columns)
