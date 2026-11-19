"""SENSE — forecast accuracy, and the guarantee that no future data is used."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from supplyiq.data import split_weeks, train_test
from supplyiq.forecast import (
    Accuracy,
    accuracy,
    demand_stats,
    ets,
    evaluate,
    forecast_panel,
    future_forecast,
    naive,
    score_panel,
    seasonal_naive,
)


# --------------------------------------------------------------------------- #
# Metrics, against hand calculations
# --------------------------------------------------------------------------- #
def test_accuracy_matches_a_hand_calculation():
    actual = [10, 0, 5, 8]
    forecast = [9, 2, 5, 10]
    #   abs errors: 1, 2, 0, 2  -> sum 5 ; sum(actual) = 23
    #   WAPE  = 5 / 23
    #   MAPE  over non-zero weeks only: mean(1/10, 0/5, 2/8)
    #   bias  = (9+2+5+10 - 23) / 23 = 3/23
    result = accuracy(actual, forecast)
    assert result.wape == pytest.approx(5 / 23)
    assert result.mape == pytest.approx(np.mean([1 / 10, 0 / 5, 2 / 8]))
    assert result.bias_pct == pytest.approx(3 / 23)
    assert result.mae == pytest.approx(5 / 4)
    assert result.rmse == pytest.approx(np.sqrt((1 + 4 + 0 + 4) / 4))
    assert result.n_zero_weeks == 1


def test_mape_can_include_zero_weeks_when_configured():
    with np.errstate(divide="ignore", invalid="ignore"):
        result = accuracy([10, 0], [9, 2], exclude_zero_actuals=False)
    assert not np.isfinite(result.mape)
    assert np.isfinite(result.wape), "WAPE stays defined where MAPE blows up"


def test_perfect_forecast_scores_zero_error():
    result = accuracy([3, 7, 0, 12], [3, 7, 0, 12])
    assert result.wape == 0.0
    assert result.mape == 0.0
    assert result.bias_pct == 0.0


def test_bias_sign_is_positive_when_over_forecasting():
    assert accuracy([10, 10], [12, 12]).bias_pct > 0
    assert accuracy([10, 10], [8, 8]).bias_pct < 0


def test_accuracy_rejects_misaligned_series():
    with pytest.raises(ValueError, match="not aligned"):
        accuracy([1, 2, 3], [1, 2])


def test_accuracy_rejects_an_empty_series():
    with pytest.raises(ValueError, match="empty"):
        accuracy([], [])


# --------------------------------------------------------------------------- #
# Model behaviour
# --------------------------------------------------------------------------- #
def test_seasonal_naive_repeats_last_season():
    history = np.arange(1, 105, dtype=float)  # 104 weeks
    out = seasonal_naive(history, horizon=3, season=52)
    np.testing.assert_allclose(out, history[-52:][:3])


def test_seasonal_naive_falls_back_to_naive_below_one_season():
    history = np.array([4.0, 7.0])
    np.testing.assert_allclose(seasonal_naive(history, 3, 52), [7.0, 7.0, 7.0])


def test_seasonal_naive_wraps_past_the_season_length():
    history = np.arange(1, 5, dtype=float)
    out = seasonal_naive(history, horizon=6, season=4)
    np.testing.assert_allclose(out, [1, 2, 3, 4, 1, 2])


def test_naive_repeats_the_last_value():
    np.testing.assert_allclose(naive(np.array([1.0, 9.0]), 3), [9.0, 9.0, 9.0])


def test_ets_falls_back_rather_than_raising_on_a_degenerate_series():
    """A constant or near-empty series must still return a usable forecast."""
    for history in (np.zeros(60), np.ones(60), np.array([5.0, 5.0, 5.0])):
        out = ets(history, horizon=4, season=52)
        assert out.shape == (4,)
        assert np.all(np.isfinite(out))


def test_ets_tracks_a_clean_seasonal_signal():
    season = 12
    t = np.arange(season * 6)
    history = 50 + 20 * np.sin(2 * np.pi * t / season)
    out = ets(history, horizon=season, season=season, min_weeks_for_seasonal=2 * season)
    expected = 50 + 20 * np.sin(2 * np.pi * np.arange(len(t), len(t) + season) / season)
    assert np.mean(np.abs(out - expected)) < 5.0


# --------------------------------------------------------------------------- #
# NO LEAKAGE — the guarantee the whole SENSE module rests on
# --------------------------------------------------------------------------- #
def test_forecasts_do_not_change_when_the_holdout_is_corrupted(tables, assumptions):
    """If a model saw the future, destroying the future would move its output."""
    demand = tables["demand"]
    holdout = int(assumptions["sample"]["holdout_weeks"])
    train, _ = train_test(demand, holdout)

    corrupted = demand.copy()
    _, first_test = split_weeks(demand, holdout)
    mask = corrupted["week_index"] >= first_test
    corrupted.loc[mask, "units_sold"] = 999_999
    corrupted_train, _ = train_test(corrupted, holdout)

    sample = sorted(train["sku"].unique())[:5]
    a = forecast_panel(train[train["sku"].isin(sample)], "seasonal_naive", 4, 52)
    b = forecast_panel(corrupted_train[corrupted_train["sku"].isin(sample)], "seasonal_naive", 4, 52)
    pd.testing.assert_frame_equal(a, b)


def test_training_frame_contains_no_holdout_week(tables, assumptions):
    demand = tables["demand"]
    holdout = int(assumptions["sample"]["holdout_weeks"])
    _, first_test = split_weeks(demand, holdout)
    train, test = train_test(demand, holdout)
    assert train["week_index"].max() < first_test
    assert test["week_index"].min() == first_test


def test_forecast_weeks_land_exactly_on_the_holdout(tables, assumptions):
    demand = tables["demand"]
    holdout = int(assumptions["sample"]["holdout_weeks"])
    train, test = train_test(demand, holdout)
    fc = forecast_panel(train, "seasonal_naive", holdout, 52)
    assert sorted(fc["week_index"].unique()) == sorted(test["week_index"].unique())


def test_demand_stats_are_computed_on_training_weeks_only(tables, assumptions):
    """Mean and sd feed safety stock; computing them on all weeks would leak."""
    demand = tables["demand"]
    holdout = int(assumptions["sample"]["holdout_weeks"])
    train, _ = train_test(demand, holdout)
    stats = demand_stats(train)
    assert (stats["weeks_observed"] == train["week_index"].nunique()).all()


# --------------------------------------------------------------------------- #
# Panel mechanics
# --------------------------------------------------------------------------- #
def test_forecast_panel_covers_every_item_for_every_step(tables, assumptions):
    train, _ = train_test(tables["demand"], assumptions["sample"]["holdout_weeks"])
    horizon = 4
    fc = forecast_panel(train, "seasonal_naive", horizon, 52)
    assert len(fc) == len(tables["sku_master"]) * horizon
    per_item = fc.groupby(["sku", "store"], observed=True).size()
    assert per_item.min() == per_item.max() == horizon


def test_forecasts_are_never_negative(tables, assumptions):
    train, _ = train_test(tables["demand"], assumptions["sample"]["holdout_weeks"])
    for model in ("seasonal_naive", "ets"):
        fc = forecast_panel(train.head(2000), model, 4, 52)
        assert (fc["forecast"] >= 0).all(), model


def test_unknown_model_is_rejected(tables, assumptions):
    train, _ = train_test(tables["demand"], assumptions["sample"]["holdout_weeks"])
    with pytest.raises(KeyError, match="unknown model"):
        forecast_panel(train, "arima-deluxe", 4, 52)


def test_score_panel_rejects_a_forecast_with_no_matching_actual(tables, assumptions):
    train, test = train_test(tables["demand"], assumptions["sample"]["holdout_weeks"])
    fc = forecast_panel(train, "seasonal_naive", 4, 52)
    fc.loc[0, "week_index"] = 99_999
    with pytest.raises(ValueError, match="no matching actual"):
        score_panel(fc, test)


def test_pooled_accuracy_is_not_a_mean_of_ratios(tables, assumptions):
    """A pooled WAPE must be volume weighted, not the average of per-item WAPEs."""
    train, test = train_test(tables["demand"], assumptions["sample"]["holdout_weeks"])
    fc = forecast_panel(train, "seasonal_naive", assumptions["sample"]["holdout_weeks"], 52)
    per_item, pooled = score_panel(fc, test)
    naive_mean = float(per_item["wape"].mean())
    assert pooled.wape != pytest.approx(naive_mean, rel=1e-6)


def test_future_forecast_starts_after_all_history(tables, assumptions):
    demand = tables["demand"]
    fc = future_forecast(demand, "seasonal_naive", assumptions)
    assert fc["week_index"].min() == int(demand["week_index"].max()) + 1
    assert fc["step"].max() == assumptions["sample"]["forecast_horizon_weeks"]


# --------------------------------------------------------------------------- #
# The evaluation as a whole
# --------------------------------------------------------------------------- #
@pytest.mark.slow
def test_candidate_model_beats_the_baseline(tables, assumptions):
    """If ETS cannot beat seasonal-naive, the SENSE module is not earning its place."""
    ev = evaluate(tables["demand"], assumptions)
    baseline = ev.overall[ev.baseline_model]
    candidate = ev.overall[ev.candidate_model]
    assert candidate.wape < baseline.wape, (baseline.wape, candidate.wape)
    assert ev.winner == ev.candidate_model
    assert ev.improvement_pp() > 0


@pytest.mark.slow
def test_evaluation_reports_are_internally_consistent(tables, assumptions):
    ev = evaluate(tables["demand"], assumptions)
    holdout = int(assumptions["sample"]["holdout_weeks"])
    assert ev.holdout_weeks[1] - ev.holdout_weeks[0] + 1 == holdout
    assert ev.train_weeks[1] + 1 == ev.holdout_weeks[0]
    for model, acc in ev.overall.items():
        assert isinstance(acc, Accuracy)
        assert acc.n_weeks == len(tables["sku_master"]) * holdout
        assert 0 <= acc.wape < 5
        assert acc.total_actual > 0


@pytest.mark.slow
def test_bias_is_small_for_both_models(tables, assumptions):
    """A model that is systematically high or low would wreck the inventory policy."""
    ev = evaluate(tables["demand"], assumptions)
    for model, acc in ev.overall.items():
        assert abs(acc.bias_pct) < 0.25, (model, acc.bias_pct)
