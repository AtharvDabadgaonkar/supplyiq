"""PLAN — segmentation and inventory policy, checked against hand calculations."""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from supplyiq.inventory import (
    WEEKS_PER_YEAR,
    abc_segment,
    apply_order_constraints,
    build_policy,
    demand_sd_over_leadtime,
    economic_order_quantity,
    normal_loss,
    policy_summary,
    projected_fill_rate,
    reorder_point,
    safety_stock,
    segment,
    segment_summary,
    service_factor,
    xyz_segment,
)


# --------------------------------------------------------------------------- #
# THE hand-calculated safety-stock case
# --------------------------------------------------------------------------- #
#   d    = 100 units / week      mean weekly demand
#   sd_d = 20 units              weekly demand standard deviation
#   L    = 4 weeks               mean lead time
#   sd_L = 1 week                lead-time standard deviation
#   SL   = 95%   ->  z = 1.6448536...
#
#   sd over lead time = sqrt(L * sd_d^2 + d^2 * sd_L^2)
#                     = sqrt(4 * 400   + 10000 * 1)
#                     = sqrt(1600 + 10000) = sqrt(11600) = 107.7032961427
#   SS                = 1.6448536270 * 107.7032961427 = 177.1561572949
HAND = dict(mean_demand=100.0, sd_demand=20.0, lead_time=4.0, sd_lead_time=1.0, service_level=0.95)


def test_service_factor_matches_the_standard_normal_table():
    assert service_factor(0.95) == pytest.approx(1.6448536, abs=1e-6)
    assert service_factor(0.99) == pytest.approx(2.3263479, abs=1e-6)
    assert service_factor(0.50) == pytest.approx(0.0, abs=1e-12)
    assert service_factor(0.90) == pytest.approx(1.2815516, abs=1e-6)


def test_service_factor_rejects_impossible_service_levels():
    for bad in (0.0, 1.0, -0.1, 1.5):
        with pytest.raises(ValueError, match="between 0 and 1"):
            service_factor(bad)


def test_sd_over_leadtime_matches_the_hand_calculation():
    got = demand_sd_over_leadtime(
        HAND["mean_demand"], HAND["sd_demand"], HAND["lead_time"], HAND["sd_lead_time"]
    )
    assert got == pytest.approx(math.sqrt(11600), abs=1e-9)
    assert got == pytest.approx(107.7032961, abs=1e-6)


def test_safety_stock_matches_the_hand_calculation():
    """The spec's required check: one case worked out by hand, to six figures."""
    got = safety_stock(**HAND)
    expected = 1.6448536269514722 * math.sqrt(4 * 20**2 + 100**2 * 1**2)
    assert got == pytest.approx(expected, abs=1e-6)
    assert got == pytest.approx(177.1561573, abs=1e-6)


def test_safety_stock_accounts_for_lead_time_variability():
    """Dropping the lead-time term is the error this whole formula exists to avoid.

    With these inputs, ignoring lead-time variability understates safety stock
    by a factor of ~2.7 -- the gap between a flat weeks-of-cover rule and a
    policy that knows the supplier is erratic.
    """
    with_lt_risk = safety_stock(**HAND)
    without = safety_stock(**{**HAND, "sd_lead_time": 0.0})
    assert without == pytest.approx(1.6448536269514722 * math.sqrt(4 * 400), abs=1e-6)
    assert with_lt_risk > without
    assert with_lt_risk / without == pytest.approx(math.sqrt(11600 / 1600), abs=1e-6)


def test_safety_stock_rises_with_the_service_level():
    levels = [0.90, 0.95, 0.975, 0.99]
    values = [safety_stock(**{**HAND, "service_level": s}) for s in levels]
    assert values == sorted(values)
    assert all(v > 0 for v in values)


def test_safety_stock_is_zero_for_perfectly_predictable_demand():
    assert safety_stock(100.0, 0.0, 4.0, 0.0, 0.95) == 0.0


def test_safety_stock_respects_a_configured_minimum():
    assert safety_stock(100.0, 0.0, 4.0, 0.0, 0.95, minimum=25.0) == 25.0


def test_safety_stock_rejects_negative_lead_time():
    with pytest.raises(ValueError, match="non-negative"):
        safety_stock(100.0, 20.0, -1.0, 1.0, 0.95)


def test_reorder_point_is_lead_time_demand_plus_safety_stock():
    ss = safety_stock(**HAND)
    # protection window = lead time (4) + review period (1) = 5 weeks
    assert reorder_point(100.0, 5.0, ss) == pytest.approx(500.0 + ss)


# --------------------------------------------------------------------------- #
# EOQ and order constraints
# --------------------------------------------------------------------------- #
def test_eoq_matches_the_textbook_formula():
    #   D = 5200/yr, S = $120/order, H = $1.10/unit/yr
    #   EOQ = sqrt(2 * 5200 * 120 / 1.10) = 1065.15
    assert economic_order_quantity(5200, 120, 1.10) == pytest.approx(
        math.sqrt(2 * 5200 * 120 / 1.10), abs=1e-9
    )
    assert economic_order_quantity(5200, 120, 1.10) == pytest.approx(1065.1500, abs=1e-3)


def test_eoq_is_zero_without_demand_or_holding_cost():
    assert economic_order_quantity(0, 120, 1.1) == 0.0
    assert economic_order_quantity(5200, 120, 0) == 0.0


def test_eoq_scales_with_the_square_root_of_demand():
    assert economic_order_quantity(4 * 1000, 50, 2) == pytest.approx(
        2 * economic_order_quantity(1000, 50, 2)
    )


def test_order_constraints_round_up_to_moq_and_multiple():
    assert apply_order_constraints(137, moq=100, multiple=25) == 150
    assert apply_order_constraints(10, moq=100, multiple=25) == 100
    assert apply_order_constraints(100, moq=0, multiple=0) == 100


def test_coverage_cap_pulls_the_quantity_down():
    assert apply_order_constraints(1000, moq=24, multiple=6, max_units=48) == 48


def test_moq_wins_over_the_coverage_cap():
    """You cannot order less than the supplier will ship, whatever the cap says."""
    assert apply_order_constraints(1000, moq=72, multiple=12, max_units=30) == 72


# --------------------------------------------------------------------------- #
# Fill rate vs cycle service level
# --------------------------------------------------------------------------- #
def test_normal_loss_matches_known_values():
    assert normal_loss(0.0) == pytest.approx(0.3989423, abs=1e-6)
    assert normal_loss(1.6448536270) == pytest.approx(0.0208929590, abs=1e-9)
    assert normal_loss(2.3263478740) == pytest.approx(0.0033886635, abs=1e-9)


def test_fill_rate_matches_the_hand_calculation():
    #   sd over lead time 107.7032961, z(0.95) = 1.6448536, G(z) = 0.0208929590
    #   expected shortage = 107.7032961 * 0.0208929590 = 2.2502405535
    #   Q = 500  ->  fill rate = 1 - 2.2502405535 / 500 = 0.9954995189
    sd_dl = demand_sd_over_leadtime(100.0, 20.0, 4.0, 1.0)
    got = projected_fill_rate(order_quantity=500, sd_over_leadtime=sd_dl, service_level=0.95)
    assert got == pytest.approx(1 - sd_dl * normal_loss(service_factor(0.95)) / 500, abs=1e-12)
    assert got == pytest.approx(0.9954995189, abs=1e-9)


def test_fill_rate_exceeds_the_cycle_service_level():
    """A planner is measured on fill rate; quoting the cycle service level understates it."""
    sd_dl = demand_sd_over_leadtime(100.0, 20.0, 4.0, 1.0)
    assert projected_fill_rate(500, sd_dl, 0.95) > 0.95


def test_fill_rate_improves_with_a_larger_order_quantity():
    sd_dl = demand_sd_over_leadtime(100.0, 20.0, 4.0, 1.0)
    assert projected_fill_rate(1000, sd_dl, 0.95) > projected_fill_rate(250, sd_dl, 0.95)


def test_fill_rate_edge_cases():
    assert projected_fill_rate(0, 100.0, 0.95) == 0.0
    assert projected_fill_rate(500, 0.0, 0.95) == 1.0
    assert 0.0 <= projected_fill_rate(1, 10_000.0, 0.95) <= 1.0


# --------------------------------------------------------------------------- #
# Segmentation
# --------------------------------------------------------------------------- #
def test_abc_segment_follows_cumulative_revenue():
    revenue = pd.Series([800, 100, 50, 30, 20], index=list("abcde"))  # total 1000
    classes = abc_segment(revenue, {"A": 0.80, "B": 0.95})
    assert classes["a"] == "A"          # 0 -> 80% cumulative, inside the A band
    assert classes["b"] == "B"          # starts at 80%
    assert classes["c"] == "B"          # starts at 90%
    assert classes["e"] == "C"


def test_abc_segment_handles_zero_revenue():
    classes = abc_segment(pd.Series([0.0, 0.0]), {"A": 0.8, "B": 0.95})
    assert set(classes) == {"C"}


def test_abc_segment_preserves_the_input_index_order():
    revenue = pd.Series([10, 900, 90], index=["x", "y", "z"])
    classes = abc_segment(revenue, {"A": 0.80, "B": 0.95})
    assert list(classes.index) == ["x", "y", "z"]
    assert classes["y"] == "A"


def test_xyz_segment_uses_the_coefficient_of_variation():
    cv = pd.Series([0.2, 0.75, 1.5, np.nan])
    classes = xyz_segment(cv, {"X": 0.5, "Y": 1.0})
    assert list(classes) == ["X", "Y", "Z", "Z"]


def test_abc_splits_revenue_roughly_on_the_configured_cutoffs(policy, assumptions):
    cutoffs = assumptions["policy"]["abc_revenue_cutoffs"]
    revenue_share = policy.groupby("abc_class")["revenue"].sum() / policy["revenue"].sum()
    assert revenue_share["A"] == pytest.approx(cutoffs["A"], abs=0.05)
    assert revenue_share["A"] + revenue_share["B"] == pytest.approx(cutoffs["B"], abs=0.05)


# --------------------------------------------------------------------------- #
# The policy table over the real sample
# --------------------------------------------------------------------------- #
def test_policy_covers_every_planning_item(policy, tables):
    assert len(policy) == len(tables["sku_master"])
    assert not policy.duplicated(subset=["sku", "store"]).any()


def test_policy_columns_are_internally_consistent(policy):
    np.testing.assert_allclose(
        policy["reorder_point"],
        policy["mean_weekly_demand"] * policy["protection_weeks"] + policy["safety_stock"],
        rtol=1e-9,
    )
    np.testing.assert_allclose(
        policy["order_up_to"], policy["reorder_point"] + policy["order_quantity"], rtol=1e-9
    )
    np.testing.assert_allclose(
        policy["average_inventory_units"],
        policy["safety_stock"] + policy["order_quantity"] / 2.0,
        rtol=1e-9,
    )
    np.testing.assert_allclose(
        policy["annual_demand"], policy["mean_weekly_demand"] * WEEKS_PER_YEAR, rtol=1e-9
    )


def test_protection_window_includes_the_review_period(policy, assumptions):
    review = float(assumptions["policy"]["review_period_weeks"])
    np.testing.assert_allclose(
        policy["protection_weeks"], policy["lead_time_mean_weeks"] + review, rtol=1e-9
    )


def test_every_safety_stock_matches_the_formula(policy):
    """Recompute every row independently and compare."""
    expected = [
        safety_stock(d, sd, lt, sdlt, sl)
        for d, sd, lt, sdlt, sl in zip(
            policy["mean_weekly_demand"],
            policy["sd_weekly_demand"],
            policy["protection_weeks"],
            policy["lead_time_sd_weeks"],
            policy["service_level"],
        )
    ]
    np.testing.assert_allclose(policy["safety_stock"], expected, rtol=1e-9)


def test_policy_values_are_non_negative_and_finite(policy):
    for column in (
        "safety_stock", "reorder_point", "order_quantity", "average_inventory_value",
        "projected_fill_rate", "sd_over_leadtime",
    ):
        values = policy[column].to_numpy(dtype=float)
        assert np.all(np.isfinite(values)), column
        assert np.all(values >= 0), column


def test_order_quantities_respect_supplier_constraints(policy):
    assert (policy["order_quantity"] >= policy["moq_units"]).all()
    remainder = policy["order_quantity"] % policy["order_multiple_units"]
    np.testing.assert_allclose(remainder, 0.0, atol=1e-9)


def test_service_levels_are_differentiated_by_abc_class(policy, assumptions):
    configured = assumptions["policy"]["service_level_by_abc"]
    by_class = policy.groupby("abc_class")["service_level"].unique()
    for abc_class, levels in by_class.items():
        assert len(levels) == 1
        assert levels[0] == pytest.approx(configured[abc_class])
    assert configured["A"] > configured["C"], "A items must be protected harder than C items"


def test_a_items_carry_more_safety_stock_cover_than_c_items(policy):
    median_cover = policy.groupby("abc_class")["safety_stock_weeks"].median()
    assert median_cover["A"] > median_cover["C"]


def test_a_flat_service_level_override_applies_to_every_row(segmented, tables, assumptions):
    from supplyiq.suppliers import item_lead_times, supplier_performance

    performance = supplier_performance(tables["purchase_orders"], tables["suppliers"])
    lead_times = item_lead_times(tables["sku_master"], performance, assumptions)
    flat = build_policy(segmented, tables["sku_master"], lead_times, assumptions, service_level=0.99)
    assert (flat["service_level"] == 0.99).all()
    assert flat["safety_stock"].sum() > 0


def test_raising_the_service_level_raises_safety_stock(segmented, tables, assumptions):
    from supplyiq.suppliers import item_lead_times, supplier_performance

    performance = supplier_performance(tables["purchase_orders"], tables["suppliers"])
    lead_times = item_lead_times(tables["sku_master"], performance, assumptions)
    low = build_policy(segmented, tables["sku_master"], lead_times, assumptions, service_level=0.90)
    high = build_policy(segmented, tables["sku_master"], lead_times, assumptions, service_level=0.99)
    assert high["safety_stock"].sum() > low["safety_stock"].sum()
    assert high["average_inventory_value"].sum() > low["average_inventory_value"].sum()


def test_longer_lead_time_suppliers_need_more_safety_stock(policy):
    """The procurement story: slow, erratic lanes cost working capital."""
    cover = policy.groupby("supplier_id").agg(
        lead_time=("lead_time_mean_weeks", "first"),
        ss_weeks=("safety_stock_weeks", "median"),
    )
    assert cover["lead_time"].corr(cover["ss_weeks"]) > 0.5


def test_policy_summary_is_consistent_with_the_table(policy):
    summary = policy_summary(policy)
    assert summary["items"] == len(policy)
    assert summary["total_safety_stock_units"] == pytest.approx(policy["safety_stock"].sum())
    assert summary["total_average_inventory_value"] == pytest.approx(
        policy["average_inventory_value"].sum()
    )
    assert 0 < summary["weighted_fill_rate"] <= 1


def test_portfolio_metrics_are_in_a_plausible_retail_range(policy):
    """Guards the assumptions: turns of 1 or 60 means the cost inputs are wrong."""
    summary = policy_summary(policy)
    assert 3 < summary["portfolio_inventory_turns"] < 30, summary["portfolio_inventory_turns"]
    assert 2 < summary["median_weeks_of_supply"] < 20, summary["median_weeks_of_supply"]


def test_segment_summary_shares_sum_to_one(policy):
    grid = segment_summary(policy)
    assert grid["revenue_share"].sum() == pytest.approx(1.0)
    assert grid["inventory_value_share"].sum() == pytest.approx(1.0)
    assert grid["items"].sum() == len(policy)
