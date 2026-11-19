"""SIMULATE — the inventory twin and the disruption scenarios."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from supplyiq.simulate import (
    Overrides,
    Scenario,
    baseline_scenario,
    draw_lead_times,
    in_scope,
    lognormal_params,
    roll_up,
    run_scenarios,
    scenario_from_config,
    simulate,
)


@pytest.fixture(scope="module")
def draws(planning):
    return draw_lead_times(planning.policy, planning.test["week_index"].nunique(), seed=7312026)


def run(planning, scenario=None, **kw):
    return simulate(planning.policy, planning.test, scenario, kw.pop("a", None), **kw)


# --------------------------------------------------------------------------- #
# THE required guarantee: zero disruption reproduces the baseline exactly
# --------------------------------------------------------------------------- #
def test_zero_disruption_reproduces_the_baseline_exactly(planning, assumptions, draws):
    base = simulate(planning.policy, planning.test, baseline_scenario(), assumptions,
                    lead_time_draws=draws)
    zero = simulate(
        planning.policy,
        planning.test,
        Scenario(name="explicit_zero", supplier_delay_days=0, demand_spike_pct=0.0,
                 tariff_change_pct=0.0),
        assumptions,
        lead_time_draws=draws,
    )
    pd.testing.assert_frame_equal(base.per_item, zero.per_item)
    assert base.kpis == zero.kpis


def test_configured_scenario_defaults_are_the_null_case(assumptions):
    scenario = scenario_from_config(assumptions)
    assert scenario.is_baseline
    assert scenario.describe() == "No disruption"


def test_the_simulation_is_deterministic_for_a_given_seed(planning, assumptions):
    first = run(planning, a=assumptions)
    second = run(planning, a=assumptions)
    pd.testing.assert_frame_equal(first.per_item, second.per_item)


def test_baseline_and_scenario_share_the_same_lead_time_draws(planning, assumptions):
    """Common random numbers: the difference must be the scenario, not luck."""
    results = run_scenarios(
        planning.policy,
        planning.test,
        [baseline_scenario(), Scenario("delay", supplier_delay_days=7)],
        assumptions,
    )
    assert results["baseline"].kpis["fill_rate"] >= results["delay"].kpis["fill_rate"]


# --------------------------------------------------------------------------- #
# Conservation and lost-sales accounting
# --------------------------------------------------------------------------- #
def test_every_unit_of_demand_is_shipped_or_lost(planning, assumptions):
    result = run(planning, a=assumptions)
    np.testing.assert_allclose(
        result.per_item["demand_units"],
        result.per_item["shipped_units"] + result.per_item["short_units"],
        rtol=1e-9,
    )


def test_unmet_demand_is_lost_not_backordered(planning, assumptions):
    """A backorder would reappear as later demand; lost sales must not."""
    result = run(planning, a=assumptions, keep_ledger=True)
    ledger = result.ledger
    totals = ledger.groupby(["sku", "store"], observed=True)["demand"].sum()
    expected = result.per_item.set_index(["sku", "store"])["demand_units"]
    np.testing.assert_allclose(
        totals.sort_index().to_numpy(), expected.sort_index().to_numpy(), rtol=1e-9
    )


def test_stock_on_hand_is_never_negative(planning, assumptions):
    result = run(planning, a=assumptions, keep_ledger=True)
    assert (result.ledger["on_hand"] >= -1e-9).all()


def test_shipments_never_exceed_demand_or_available_stock(planning, assumptions):
    result = run(planning, a=assumptions, keep_ledger=True)
    ledger = result.ledger
    assert (ledger["shipped"] <= ledger["demand"] + 1e-9).all()
    assert (ledger["shipped"] >= -1e-9).all()


def test_fill_rate_is_pooled_on_units(planning, assumptions):
    result = run(planning, a=assumptions)
    expected = (
        result.per_item["shipped_units"].sum() / result.per_item["demand_units"].sum()
    )
    assert result.kpis["fill_rate"] == pytest.approx(float(expected))


def test_stockout_weeks_count_weeks_with_unmet_demand(planning, assumptions):
    result = run(planning, a=assumptions, keep_ledger=True)
    expected = (
        result.ledger.assign(hit=lambda d: d["short"] > 1e-9)
        .groupby(["sku", "store"], observed=True)["hit"]
        .sum()
    )
    got = result.per_item.set_index(["sku", "store"])["stockout_weeks"]
    np.testing.assert_allclose(
        expected.sort_index().to_numpy(dtype=float), got.sort_index().to_numpy(dtype=float)
    )


# --------------------------------------------------------------------------- #
# Scenario behaviour and direction
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("delay", [7, 21, 45])
def test_longer_supplier_delays_never_improve_service(planning, assumptions, delay):
    results = run_scenarios(
        planning.policy,
        planning.test,
        [baseline_scenario(), Scenario(f"d{delay}", supplier_delay_days=delay)],
        assumptions,
    )
    base, scen = results["baseline"].kpis, results[f"d{delay}"].kpis
    assert scen["fill_rate"] <= base["fill_rate"]
    assert scen["stockout_weeks"] >= base["stockout_weeks"]
    assert scen["otif"] <= base["otif"]


def test_supplier_delay_is_monotonic_in_service(planning, assumptions):
    results = run_scenarios(
        planning.policy,
        planning.test,
        [Scenario("d0"), Scenario("d14", supplier_delay_days=14),
         Scenario("d35", supplier_delay_days=35)],
        assumptions,
    )
    fills = [results[n].kpis["fill_rate"] for n in ("d0", "d14", "d35")]
    assert fills == sorted(fills, reverse=True), fills


def test_a_demand_spike_raises_demand_proportionally(planning, assumptions):
    results = run_scenarios(
        planning.policy, planning.test,
        [baseline_scenario(), Scenario("spike", demand_spike_pct=30.0)], assumptions,
    )
    base, spike = results["baseline"].kpis, results["spike"].kpis
    assert spike["demand_units"] == pytest.approx(base["demand_units"] * 1.30, rel=1e-9)
    assert spike["fill_rate"] <= base["fill_rate"]


def test_a_negative_demand_spike_reduces_demand(planning, assumptions):
    results = run_scenarios(
        planning.policy, planning.test,
        [baseline_scenario(), Scenario("slump", demand_spike_pct=-25.0)], assumptions,
    )
    assert results["slump"].kpis["demand_units"] == pytest.approx(
        results["baseline"].kpis["demand_units"] * 0.75, rel=1e-9
    )


def test_tariff_change_is_in_percentage_points_not_relative(planning, assumptions):
    """+10 takes a 25% duty to 35%. The two readings differ a lot on a 25% base."""
    policy = planning.policy
    cn = policy[policy["country"] == "CN"]
    assert len(cn) > 0
    base_rate = float(cn["tariff_rate"].iloc[0])
    assert base_rate == pytest.approx(0.25)

    results = run_scenarios(
        planning.policy, planning.test,
        [baseline_scenario(), Scenario("t", tariff_change_pct=10.0, country_scope=("CN",))],
        assumptions,
    )
    base_item = results["baseline"].per_item.set_index(["sku", "store"])
    scen_item = results["t"].per_item.set_index(["sku", "store"])
    key = (cn["sku"].iloc[0], cn["store"].iloc[0])
    unit_cost = float(cn["unit_cost"].iloc[0])
    delta = scen_item.loc[key, "landed_cost_per_unit"] - base_item.loc[key, "landed_cost_per_unit"]
    # percentage points: unit_cost * 0.10.  relative would be unit_cost * 0.025.
    assert delta == pytest.approx(unit_cost * 0.10, rel=1e-6)


def test_tariff_rate_is_floored_at_zero(planning, assumptions):
    results = run_scenarios(
        planning.policy, planning.test,
        [Scenario("cut", tariff_change_pct=-50.0)], assumptions,
    )
    assert (results["cut"].per_item["landed_cost_per_unit"] > 0).all()


def test_a_tariff_rise_raises_landed_cost_and_working_capital(planning, assumptions):
    results = run_scenarios(
        planning.policy, planning.test,
        [baseline_scenario(), Scenario("t", tariff_change_pct=10.0)], assumptions,
    )
    base, scen = results["baseline"].kpis, results["t"].kpis
    assert scen["landed_cost_per_unit"] > base["landed_cost_per_unit"]
    assert scen["working_capital"] > base["working_capital"]
    # A duty change is a price event, not a supply event.
    assert scen["fill_rate"] == pytest.approx(base["fill_rate"])
    assert scen["stockout_weeks"] == base["stockout_weeks"]


def test_country_scope_limits_the_tariff_to_that_origin(planning, assumptions):
    results = run_scenarios(
        planning.policy, planning.test,
        [baseline_scenario(), Scenario("cn", tariff_change_pct=15.0, country_scope=("CN",))],
        assumptions,
    )
    base = results["baseline"].per_item.set_index(["sku", "store"])["landed_cost_per_unit"]
    scen = results["cn"].per_item.set_index(["sku", "store"])["landed_cost_per_unit"]
    changed = (scen - base).abs() > 1e-9
    countries = planning.policy.set_index(["sku", "store"])["country"].reindex(changed.index)
    assert set(countries[changed].unique()) == {"CN"}
    assert changed.sum() > 0


def test_supplier_scope_limits_the_delay_to_that_supplier(planning, assumptions):
    target = planning.policy["supplier_id"].value_counts().index[0]
    results = run_scenarios(
        planning.policy, planning.test,
        [baseline_scenario(), Scenario("one", supplier_delay_days=40, supplier_scope=(target,))],
        assumptions,
    )
    base = results["baseline"].per_item.set_index(["sku", "store"])["short_units"]
    scen = results["one"].per_item.set_index(["sku", "store"])["short_units"]
    worse = (scen - base) > 1e-9
    suppliers = planning.policy.set_index(["sku", "store"])["supplier_id"].reindex(worse.index)
    assert set(suppliers[worse].unique()) <= {target}


def test_in_scope_mask_handles_each_scoping_mode():
    suppliers = pd.Series(["A", "B", "C"])
    countries = pd.Series(["CN", "VN", "CN"])
    assert in_scope(Scenario(), suppliers, countries).all()
    np.testing.assert_array_equal(
        in_scope(Scenario(supplier_scope=("B",)), suppliers, countries), [False, True, False]
    )
    np.testing.assert_array_equal(
        in_scope(Scenario(country_scope=("CN",)), suppliers, countries), [True, False, True]
    )


# --------------------------------------------------------------------------- #
# Cost model
# --------------------------------------------------------------------------- #
def test_cost_to_serve_excludes_the_price_of_the_goods(planning, assumptions):
    """Including goods made a delay look cheaper than no delay. See the docstring."""
    result = run(planning, a=assumptions)
    k = result.kpis
    assert k["cost_to_serve"] == pytest.approx(
        k["holding_cost"] + k["stockout_cost"] + k["ordering_cost"] + k["action_cost"]
    )
    assert k["total_cost"] == pytest.approx(k["cost_to_serve"] + k["cost_of_goods_ordered"])
    assert k["cost_of_goods_ordered"] > k["cost_to_serve"]


def test_every_disruption_costs_more_to_serve_than_the_baseline(planning, assumptions):
    """The property that forced the cost-to-serve definition to be fixed."""
    scenarios = [
        baseline_scenario(),
        Scenario("delay", supplier_delay_days=21),
        Scenario("spike", demand_spike_pct=30.0),
        Scenario("tariff", tariff_change_pct=10.0),
    ]
    results = run_scenarios(planning.policy, planning.test, scenarios, assumptions)
    base = results["baseline"].kpis["cost_to_serve"]
    for name in ("delay", "spike", "tariff"):
        assert results[name].kpis["cost_to_serve"] > base, name


def test_working_capital_includes_goods_in_transit(planning, assumptions):
    """A delay strands cash on a ship; it does not free it up."""
    result = run(planning, a=assumptions)
    k = result.kpis
    assert k["working_capital"] == pytest.approx(
        k["avg_inventory_value"] + k["avg_in_transit_value"]
    )
    assert k["avg_in_transit_value"] > 0


def test_a_supplier_delay_increases_in_transit_capital(planning, assumptions):
    results = run_scenarios(
        planning.policy, planning.test,
        [baseline_scenario(), Scenario("delay", supplier_delay_days=21)], assumptions,
    )
    assert (
        results["delay"].kpis["avg_in_transit_value"]
        > results["baseline"].kpis["avg_in_transit_value"]
    )


def test_landed_cost_follows_the_documented_formula(planning, assumptions):
    result = run(planning, a=assumptions)
    joined = result.per_item.merge(
        planning.policy[["sku", "store", "unit_cost", "freight_per_unit", "tariff_rate"]],
        on=["sku", "store"],
    )
    expected = (
        joined["unit_cost"] + joined["freight_per_unit"] + joined["unit_cost"] * joined["tariff_rate"]
    )
    np.testing.assert_allclose(joined["landed_cost_per_unit"], expected, rtol=1e-6)


def test_stockout_cost_uses_the_configured_penalty(planning, assumptions):
    result = run(planning, a=assumptions)
    penalty = float(assumptions["costs"]["stockout_penalty_per_unit"])
    np.testing.assert_allclose(
        result.per_item["stockout_cost"], result.per_item["short_units"] * penalty, rtol=1e-9
    )


# --------------------------------------------------------------------------- #
# OTIF
# --------------------------------------------------------------------------- #
def test_otif_is_judged_against_the_quoted_date_not_the_mean(planning, assumptions):
    """Judging against a supplier's own average makes half its POs late by construction."""
    result = run(planning, a=assumptions)
    assert result.kpis["otif"] > 0.90, result.kpis["otif"]


def test_otif_collapses_under_a_large_delay(planning, assumptions):
    results = run_scenarios(
        planning.policy, planning.test,
        [baseline_scenario(), Scenario("delay", supplier_delay_days=30)], assumptions,
    )
    assert results["delay"].kpis["otif"] < 0.5
    assert results["baseline"].kpis["otif"] > 0.9


# --------------------------------------------------------------------------- #
# Overrides — the hook the ACT module prices recommendations through
# --------------------------------------------------------------------------- #
def test_expediting_an_item_improves_its_service(planning, assumptions, draws):
    worst = (
        simulate(planning.policy, planning.test,
                 Scenario("delay", supplier_delay_days=28), assumptions, lead_time_draws=draws)
        .per_item.sort_values("short_units", ascending=False)
        .iloc[0]
    )
    key = (worst["sku"], worst["store"])
    scenario = Scenario("delay", supplier_delay_days=28)
    before = simulate(planning.policy, planning.test, scenario, assumptions,
                      lead_time_draws=draws).per_item.set_index(["sku", "store"]).loc[key]
    after = simulate(
        planning.policy, planning.test, scenario, assumptions,
        overrides=Overrides(lead_time_days={key: 3.0}), lead_time_draws=draws,
    ).per_item.set_index(["sku", "store"]).loc[key]
    assert after["short_units"] <= before["short_units"]
    assert after["fill_rate"] >= before["fill_rate"]


def test_an_override_touches_only_the_item_it_names(planning, assumptions, draws):
    key = (planning.policy["sku"].iloc[0], planning.policy["store"].iloc[0])
    base = simulate(planning.policy, planning.test, baseline_scenario(), assumptions,
                    lead_time_draws=draws).per_item.set_index(["sku", "store"])
    changed = simulate(
        planning.policy, planning.test, baseline_scenario(), assumptions,
        overrides=Overrides(order_quantity={key: 5.0}), lead_time_draws=draws,
    ).per_item.set_index(["sku", "store"])
    differing = (base["units_ordered"] - changed["units_ordered"]).abs() > 1e-9
    assert set(differing[differing].index) <= {key}


def test_extra_cost_override_flows_into_cost_to_serve(planning, assumptions, draws):
    key = (planning.policy["sku"].iloc[0], planning.policy["store"].iloc[0])
    base = simulate(planning.policy, planning.test, baseline_scenario(), assumptions,
                    lead_time_draws=draws)
    with_cost = simulate(
        planning.policy, planning.test, baseline_scenario(), assumptions,
        overrides=Overrides(extra_cost={key: 250.0}), lead_time_draws=draws,
    )
    assert with_cost.kpis["cost_to_serve"] == pytest.approx(
        base.kpis["cost_to_serve"] + 250.0
    )
    assert with_cost.kpis["action_cost"] == pytest.approx(250.0)


def test_empty_overrides_change_nothing(planning, assumptions, draws):
    base = simulate(planning.policy, planning.test, baseline_scenario(), assumptions,
                    lead_time_draws=draws)
    same = simulate(planning.policy, planning.test, baseline_scenario(), assumptions,
                    overrides=Overrides(), lead_time_draws=draws)
    pd.testing.assert_frame_equal(base.per_item, same.per_item)
    assert Overrides().is_empty()


# --------------------------------------------------------------------------- #
# Mechanics
# --------------------------------------------------------------------------- #
def test_lognormal_params_recover_the_requested_moments():
    mu, sigma = lognormal_params(42.0, 9.0)
    draws = np.random.default_rng(0).lognormal(mu, sigma, size=400_000)
    assert draws.mean() == pytest.approx(42.0, rel=0.02)
    assert draws.std() == pytest.approx(9.0, rel=0.05)


def test_lead_time_draws_are_reproducible(planning):
    weeks = planning.test["week_index"].nunique()
    first = draw_lead_times(planning.policy, weeks, seed=123)
    second = draw_lead_times(planning.policy, weeks, seed=123)
    np.testing.assert_array_equal(first, second)
    assert first.shape == (len(planning.policy), weeks)
    assert (first > 0).all()


def test_a_wrong_shaped_draw_matrix_is_rejected(planning, assumptions):
    with pytest.raises(ValueError, match="lead_time_draws shape"):
        simulate(planning.policy, planning.test, None, assumptions,
                 lead_time_draws=np.ones((3, 3)))


def test_simulating_an_item_without_a_policy_row_is_rejected(planning, assumptions):
    partial = planning.policy.iloc[5:].copy()
    with pytest.raises(ValueError, match="no policy row"):
        simulate(partial, planning.test, None, assumptions)


def test_ledger_rows_cover_every_item_and_week(planning, assumptions):
    result = run(planning, a=assumptions, keep_ledger=True)
    weeks = planning.test["week_index"].nunique()
    assert len(result.ledger) == len(planning.policy) * weeks


def test_roll_up_matches_the_per_item_table(planning, assumptions):
    result = run(planning, a=assumptions)
    recomputed = roll_up(result.per_item, result.kpis["weeks"])
    assert recomputed == result.kpis


def test_compare_to_reports_signed_deltas(planning, assumptions):
    results = run_scenarios(
        planning.policy, planning.test,
        [baseline_scenario(), Scenario("delay", supplier_delay_days=21)], assumptions,
    )
    delta = results["delay"].compare_to(results["baseline"])
    assert delta["stockout_weeks"] > 0
    assert delta["fill_rate"] < 0


def test_scenario_description_reads_like_a_planner_wrote_it():
    assert baseline_scenario().describe() == "No disruption"
    scenario = Scenario("s", supplier_delay_days=21, demand_spike_pct=30.0,
                        tariff_change_pct=10.0, country_scope=("CN",))
    text = scenario.describe()
    assert "+21d supplier delay" in text
    assert "+30% demand" in text
    assert "+10pp tariff" in text
    assert "CN" in text
    assert not scenario.is_baseline
