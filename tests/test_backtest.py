"""MEASURE — does the SupplyIQ policy beat what a planner runs today?"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from supplyiq import backtest
from supplyiq.backtest import (
    _capital_at_fill,
    build_frontier,
    comparison_table,
    cover_sweep,
    current_policy,
    naive_demand_rate,
)


@pytest.fixture(scope="module")
def result(planning, tables, assumptions):
    return backtest.run(planning, tables, assumptions)


# --------------------------------------------------------------------------- #
# The incumbent policy
# --------------------------------------------------------------------------- #
def test_naive_rate_uses_the_last_training_week(planning, assumptions):
    rates = naive_demand_rate(planning.train, assumptions)
    last_week = int(planning.train["week_index"].max())
    actual = (
        planning.train[planning.train["week_index"] == last_week]
        .groupby(["sku", "store"], observed=True)["units_sold"]
        .sum()
    )
    merged = rates.set_index(["sku", "store"]).join(actual.rename("actual"))
    nonzero = merged[merged["actual"] > 0]
    np.testing.assert_allclose(nonzero["demand_rate"], nonzero["actual"], rtol=1e-9)


def test_naive_rate_falls_back_to_the_mean_on_a_zero_week(planning, assumptions):
    """One unlucky zero week must not sabotage the incumbent; that would be a strawman."""
    rates = naive_demand_rate(planning.train, assumptions)
    assert (rates["demand_rate"] >= 0).all()
    assert rates["demand_rate"].notna().all()


def test_current_policy_is_flat_across_lead_times(planning, assumptions):
    """The defining weakness of the incumbent: cover ignores the supply lane."""
    policy = current_policy(planning.policy, planning.train, assumptions)
    cover = policy["reorder_point"] / policy["demand_rate"].replace(0, np.nan)
    assert cover.dropna().std() < 1e-6, "weeks of cover must be identical for every item"
    assert cover.dropna().iloc[0] == pytest.approx(
        float(assumptions["baseline_policy"]["weeks_of_cover"])
    )


def test_current_policy_ignores_demand_variability(planning, assumptions):
    policy = current_policy(planning.policy, planning.train, assumptions)
    weeks = policy["safety_stock"] / policy["demand_rate"].replace(0, np.nan)
    assert weeks.dropna().std() < 1e-6
    assert policy["service_level"].isna().all(), "a flat rule targets no service level"


def test_current_policy_respects_supplier_constraints(planning, assumptions):
    policy = current_policy(planning.policy, planning.train, assumptions)
    assert (policy["order_quantity"] >= policy["moq_units"]).all()
    np.testing.assert_allclose(
        policy["order_quantity"] % policy["order_multiple_units"], 0.0, atol=1e-9
    )


def test_an_unsupported_baseline_forecast_is_rejected(planning, assumptions):
    broken = {**assumptions, "baseline_policy": {**assumptions["baseline_policy"],
                                                 "forecast": "prophet"}}
    with pytest.raises(ValueError, match="only 'naive' is implemented"):
        naive_demand_rate(planning.train, broken)


# --------------------------------------------------------------------------- #
# The comparison
# --------------------------------------------------------------------------- #
def test_both_policies_are_judged_on_the_same_weeks(result, planning):
    assert result.current.kpis["weeks"] == result.supplyiq.kpis["weeks"]
    assert result.current.kpis["items"] == result.supplyiq.kpis["items"]
    assert result.current.kpis["weeks"] == planning.test["week_index"].nunique()


def test_supplyiq_improves_service(result):
    head = result.headline()
    assert head["supplyiq_fill_rate"] > head["current_fill_rate"]
    assert head["fill_rate_change_pp"] > 0
    assert head["stockout_weeks_avoided"] > 0
    assert head["supplyiq_short_units"] < head["current_short_units"]


def test_headline_deltas_are_internally_consistent(result):
    head = result.headline()
    assert head["stockout_weeks_avoided"] == pytest.approx(
        head["current_stockout_weeks"] - head["supplyiq_stockout_weeks"]
    )
    assert head["fill_rate_change_pp"] == pytest.approx(
        (head["supplyiq_fill_rate"] - head["current_fill_rate"]) * 100
    )


def test_comparison_table_has_one_row_per_policy(result):
    table = comparison_table(result)
    assert list(table["policy"]) == ["current", "supplyiq"]
    assert table["fill_rate"].between(0, 1).all()


# --------------------------------------------------------------------------- #
# Sweeps: both policies judged across their settings, not at one point
# --------------------------------------------------------------------------- #
def test_supplyiq_sweep_is_monotonic(result):
    sweep = result.sweep.sort_values("service_level")
    assert list(sweep["fill_rate"]) == sorted(sweep["fill_rate"])
    assert list(sweep["working_capital"]) == sorted(sweep["working_capital"])
    assert list(sweep["safety_stock_units"]) == sorted(sweep["safety_stock_units"])


def test_sweep_reaches_zero_safety_stock_at_the_bottom(result):
    """The floor matters: it is what shows the advantage is not the safety stock."""
    lowest = result.sweep.sort_values("service_level").iloc[0]
    assert lowest["service_level"] == pytest.approx(0.50)
    assert lowest["safety_stock_units"] == pytest.approx(0.0, abs=1e-6)


def test_the_incumbent_is_swept_too(result, assumptions):
    """Judging the incumbent only at its default setting would be a strawman."""
    covers = result.cover_sweep.sort_values("weeks_of_cover")
    assert len(covers) >= 5
    assert list(covers["fill_rate"]) == sorted(covers["fill_rate"])
    assert list(covers["working_capital"]) == sorted(covers["working_capital"])
    assert float(assumptions["baseline_policy"]["weeks_of_cover"]) in set(covers["weeks_of_cover"])


def test_the_flat_rule_can_reach_high_service_by_holding_more(result):
    """It is not that the incumbent cannot hit the service -- it is what that costs."""
    best = result.cover_sweep.sort_values("fill_rate").iloc[-1]
    assert best["fill_rate"] > 0.99


def test_supplyiq_needs_less_capital_at_every_comparable_fill_rate(result):
    frontier = result.frontier
    usable = frontier[frontier["comparable"]]
    assert len(usable) > 0
    assert (usable["capital_saving"] > 0).all(), usable
    assert (usable["supplyiq_working_capital"] < usable["current_working_capital"]).all()


def test_non_comparable_points_are_flagged_not_dropped(result):
    """Honesty check: fill rates SupplyIQ cannot be tuned down to stay visible."""
    frontier = result.frontier
    assert (~frontier["comparable"]).any(), "expected some out-of-range points"
    not_comparable = frontier[~frontier["comparable"]]
    assert not_comparable["supplyiq_working_capital"].isna().all()
    assert not_comparable["capital_saving"].isna().all()


def test_frontier_summary_excludes_the_non_comparable_points(result):
    summary = result.frontier_summary()
    assert summary["comparable_points"] == int(result.frontier["comparable"].sum())
    assert summary["not_comparable_points"] == int((~result.frontier["comparable"]).sum())
    assert summary["capital_saving_pct_min"] > 0
    assert "cannot be tuned that low" in summary["note"]


def test_iso_service_reports_why_it_could_not_match(result):
    """The sweep genuinely cannot bracket the incumbent's fill rate. Say so."""
    iso = result.iso_service
    if not iso.get("matched", True):
        assert "why_not" in iso
        assert "safety stock is already zero" in iso["why_not"]


def test_capital_interpolation_is_linear_between_sweep_points():
    sweep = pd.DataFrame(
        {"fill_rate": [0.90, 0.98], "working_capital": [100.0, 200.0], "service_level": [0.5, 0.9]}
    )
    assert _capital_at_fill(sweep, 0.94) == pytest.approx(150.0)
    assert _capital_at_fill(sweep, 0.90) == pytest.approx(100.0)
    assert _capital_at_fill(sweep, 0.99) is None, "must not extrapolate past the sweep"


def test_frontier_marks_uncomparable_rows():
    sweep = pd.DataFrame(
        {"fill_rate": [0.98, 0.99], "working_capital": [100.0, 150.0], "service_level": [0.8, 0.95]}
    )
    covers = pd.DataFrame(
        {"weeks_of_cover": [2.0, 6.0], "fill_rate": [0.80, 0.985], "working_capital": [50.0, 200.0]}
    )
    frontier = build_frontier(sweep, covers)
    assert list(frontier["comparable"]) == [False, True]
    assert frontier.loc[1, "capital_saving"] == pytest.approx(200.0 - 125.0)


# --------------------------------------------------------------------------- #
# The finding the backtest actually reports
# --------------------------------------------------------------------------- #
def test_supplyiq_beats_the_flat_rule_even_with_no_safety_stock(result):
    """The headline insight: the gap is the reorder point, not the buffer."""
    floor = result.sweep.sort_values("service_level").iloc[0]
    assert floor["safety_stock_units"] == pytest.approx(0.0, abs=1e-6)
    assert floor["fill_rate"] > result.current.kpis["fill_rate"], (
        "with zero safety stock SupplyIQ should still beat a flat four weeks of cover"
    )
