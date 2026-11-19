"""The precomputed artifacts the deployed app loads instead of recomputing."""
from __future__ import annotations

import pandas as pd
import pytest

from supplyiq.pipeline import (
    ARTIFACTS,
    Planning,
    artifact_path,
    build,
    item_revenue,
    load_artifact,
    missing_artifacts,
)


def test_pipeline_builds_every_artifact(planning: Planning, tables):
    assert len(planning.policy) == len(tables["sku_master"])
    assert len(planning.performance) == len(tables["suppliers"])
    assert len(planning.stats) == len(tables["sku_master"])
    assert planning.overall["pos"] == len(tables["purchase_orders"])


def test_pipeline_uses_training_weeks_only(planning, assumptions):
    """The policy must not be fitted on the weeks the backtest will judge it on."""
    holdout = int(assumptions["sample"]["holdout_weeks"])
    assert planning.train["week_index"].max() < planning.test["week_index"].min()
    assert planning.test["week_index"].nunique() == holdout
    assert (planning.stats["weeks_observed"] == planning.train["week_index"].nunique()).all()


def test_item_revenue_aligns_to_the_stats_row_order(planning):
    revenue = item_revenue(planning.train, planning.stats)
    assert len(revenue) == len(planning.stats)
    assert revenue.notna().all()
    # Spot-check one item against a direct sum.
    row = planning.stats.iloc[0]
    direct = planning.train[
        (planning.train["sku"] == row["sku"]) & (planning.train["store"] == row["store"])
    ]["revenue"].sum()
    assert revenue.iloc[0] == pytest.approx(float(direct))


def test_service_level_override_flows_through_the_pipeline(tables, assumptions):
    built = build(tables, assumptions, with_evaluation=False, service_level=0.99)
    assert (built.policy["service_level"] == 0.99).all()


# --------------------------------------------------------------------------- #
# Committed artifacts — these are what make the Cloud app fast
# --------------------------------------------------------------------------- #
def test_artifacts_are_precomputed_and_committed():
    missing = missing_artifacts()
    if missing:
        pytest.skip(f"artifacts not built: {missing}; run `make metrics`")
    for name in ARTIFACTS:
        assert artifact_path(name).exists()


def test_artifacts_load_and_are_not_empty():
    if missing_artifacts():
        pytest.skip("artifacts not built; run `make metrics`")
    for name in ARTIFACTS:
        df = load_artifact(name)
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0, name


def test_committed_policy_matches_a_fresh_build(policy):
    """A stale artifact would make the UI disagree with the code."""
    if missing_artifacts():
        pytest.skip("artifacts not built; run `make metrics`")
    stored = load_artifact("policy")
    assert len(stored) == len(policy)
    for column in ("safety_stock", "reorder_point", "order_quantity", "projected_fill_rate"):
        pd.testing.assert_series_equal(
            stored[column].reset_index(drop=True).astype(float),
            policy[column].reset_index(drop=True).astype(float),
            check_names=False,
            rtol=1e-6,
        )


def test_artifacts_stay_within_the_deployment_budget():
    if missing_artifacts():
        pytest.skip("artifacts not built")
    total = sum(artifact_path(n).stat().st_size for n in ARTIFACTS)
    assert total < 10e6, f"derived artifacts are {total / 1e6:.1f} MB"


def test_loading_a_missing_artifact_gives_a_useful_message(tmp_path):
    with pytest.raises(FileNotFoundError, match="make metrics"):
        load_artifact("policy", directory=tmp_path)


def test_forecast_artifacts_cover_the_holdout(assumptions):
    if missing_artifacts():
        pytest.skip("artifacts not built")
    forecasts = load_artifact("forecasts")
    holdout = int(assumptions["sample"]["holdout_weeks"])
    assert forecasts["week_index"].nunique() == holdout
    assert set(forecasts["model"].unique()) == {
        assumptions["forecast"]["baseline_model"],
        assumptions["forecast"]["candidate_model"],
    }


def test_future_forecast_artifact_looks_forward(assumptions, tables):
    if missing_artifacts():
        pytest.skip("artifacts not built")
    future = load_artifact("future_forecast")
    assert future["week_index"].min() > int(tables["demand"]["week_index"].max())
    assert future["step"].max() == assumptions["sample"]["forecast_horizon_weeks"]
