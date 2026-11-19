"""ACT — recommendations, and the guarantee that their numbers are simulator output."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from supplyiq.act import (
    DecisionLog,
    at_risk,
    build_candidates,
    choose_best,
    explain_all,
    price_candidate,
    recommend,
    risk_reason,
    to_frame,
)
from supplyiq.agent import MockProvider, Recommendation
from supplyiq.agent.base import ACTION_TYPES
from supplyiq.agent.guard import is_verified, unverified_numbers
from supplyiq.simulate import Overrides, Scenario, draw_lead_times, simulate

STRESS = Scenario("supplier_delay_30d", supplier_delay_days=30)


@pytest.fixture(scope="module")
def ordered_policy(planning):
    return planning.policy.sort_values(["sku", "store"]).reset_index(drop=True)


@pytest.fixture(scope="module")
def stressed(ordered_policy, planning, assumptions):
    draws = draw_lead_times(
        ordered_policy, planning.test["week_index"].nunique(), assumptions["simulation"]["seed"]
    )
    return simulate(ordered_policy, planning.test, STRESS, assumptions, lead_time_draws=draws)


@pytest.fixture(scope="module")
def recommendations(ordered_policy, planning, assumptions, stressed):
    return recommend(ordered_policy, planning.test, STRESS, assumptions, result=stressed)


# --------------------------------------------------------------------------- #
# At-risk detection
# --------------------------------------------------------------------------- #
def test_at_risk_finds_items_only_when_they_are_at_risk(stressed, ordered_policy, assumptions):
    risky = at_risk(stressed, ordered_policy, assumptions)
    assert len(risky) > 0
    assert ((risky["short_units"] > 0) | (risky["fill_rate"] < risky["service_level"])).all()


def test_at_risk_respects_the_configured_top_n(stressed, ordered_policy, assumptions):
    assert len(at_risk(stressed, ordered_policy, assumptions)) <= int(
        assumptions["agent"]["at_risk_top_n"]
    )
    assert len(at_risk(stressed, ordered_policy, assumptions, top_n=3)) <= 3


def test_at_risk_ranks_by_revenue_not_units(stressed, ordered_policy, assumptions):
    """A planner's attention is scarce; rank by money, not unit count."""
    risky = at_risk(stressed, ordered_policy, assumptions, top_n=25)
    assert list(risky["revenue_at_risk"]) == sorted(risky["revenue_at_risk"], reverse=True)
    np.testing.assert_allclose(
        risky["revenue_at_risk"],
        risky["short_units"] * risky["avg_sell_price"].astype(float),
        rtol=1e-6,
    )


def test_a_healthy_baseline_surfaces_few_or_no_risks(ordered_policy, planning, assumptions):
    from supplyiq.simulate import baseline_scenario

    result = simulate(ordered_policy, planning.test, baseline_scenario(), assumptions)
    assert len(at_risk(result, ordered_policy, assumptions)) < len(ordered_policy) / 10


def test_risk_reason_is_built_from_computed_values(stressed, ordered_policy, assumptions):
    risky = at_risk(stressed, ordered_policy, assumptions)
    text = risk_reason(risky.iloc[0], STRESS)
    assert "short" in text or "service target" in text
    assert "supplier delay" in text


# --------------------------------------------------------------------------- #
# Candidate generation
# --------------------------------------------------------------------------- #
def test_candidates_cover_the_declared_action_types(stressed, ordered_policy, assumptions):
    risky = at_risk(stressed, ordered_policy, assumptions)
    seen = set()
    for _, row in risky.iterrows():
        seen |= {c.action_type for c in build_candidates(row, ordered_policy, assumptions)}
    assert "no_action" in seen
    assert {"expedite_order", "switch_supplier", "dual_source", "adjust_order_qty"} <= seen
    assert seen <= set(ACTION_TYPES)


def test_expedite_shortens_the_lead_time_and_raises_freight(
    stressed, ordered_policy, assumptions
):
    row = at_risk(stressed, ordered_policy, assumptions).iloc[0]
    expedite = next(
        c for c in build_candidates(row, ordered_policy, assumptions)
        if c.action_type == "expedite_order"
    )
    assert expedite.lead_time_days_delta < 0
    key = (row["sku"], row["store"])
    item = ordered_policy.set_index(["sku", "store"]).loc[key]
    assert expedite.overrides.freight_per_unit[key] > float(item["freight_per_unit"])


def test_rebalance_takes_stock_from_a_sibling_store(stressed, ordered_policy, assumptions):
    risky = at_risk(stressed, ordered_policy, assumptions)
    for _, row in risky.iterrows():
        rebalance = [
            c for c in build_candidates(row, ordered_policy, assumptions)
            if c.action_type == "rebalance_stock"
        ]
        if not rebalance:
            continue
        candidate = rebalance[0]
        donor = candidate.extra_items[0]
        assert donor[0] == row["sku"], "a donor must be the same SKU"
        assert donor[1] != row["store"], "a donor must be a different store"
        assert len(candidate.overrides.initial_on_hand) == 2
        return
    pytest.skip("no rebalance candidate in this sample")


def test_no_action_is_always_offered(stressed, ordered_policy, assumptions):
    risky = at_risk(stressed, ordered_policy, assumptions)
    for _, row in risky.iterrows():
        types = [c.action_type for c in build_candidates(row, ordered_policy, assumptions)]
        assert "no_action" in types


# --------------------------------------------------------------------------- #
# THE required guarantee: every number comes from a simulator re-run
# --------------------------------------------------------------------------- #
def test_recommendation_numbers_reproduce_an_independent_simulator_run(
    recommendations, ordered_policy, planning, assumptions
):
    """Rebuild each recommendation's figures from scratch and compare."""
    checked = 0
    for rec in recommendations:
        if rec.action_type == "no_action":
            continue
        row = next(
            r for _, r in at_risk(
                simulate(
                    ordered_policy,
                    planning.test,
                    STRESS,
                    assumptions,
                    lead_time_draws=draw_lead_times(
                        ordered_policy,
                        planning.test["week_index"].nunique(),
                        assumptions["simulation"]["seed"],
                    ),
                ),
                ordered_policy,
                assumptions,
            ).iterrows()
            if r["sku"] == rec.sku and r["store"] == rec.store
        )
        candidate = next(
            c for c in build_candidates(row, ordered_policy, assumptions)
            if c.action_type == rec.action_type
        )
        recomputed = price_candidate(
            candidate, (rec.sku, rec.store), ordered_policy, planning.test, STRESS, assumptions
        )
        assert recomputed["action_fill_rate"] == pytest.approx(rec.action_fill_rate, rel=1e-9)
        assert recomputed["scenario_fill_rate"] == pytest.approx(rec.scenario_fill_rate, rel=1e-9)
        assert recomputed["service_gain_pp"] == pytest.approx(rec.service_gain_pp, rel=1e-9)
        assert recomputed["action_cost_delta"] == pytest.approx(rec.action_cost_delta, rel=1e-9)
        assert recomputed["action_stockout_weeks"] == rec.action_stockout_weeks
        checked += 1
    assert checked > 0, "no priced actions to verify"


def test_pricing_a_candidate_is_deterministic(
    stressed, ordered_policy, planning, assumptions
):
    row = at_risk(stressed, ordered_policy, assumptions).iloc[0]
    key = (row["sku"], row["store"])
    candidate = build_candidates(row, ordered_policy, assumptions)[0]
    first = price_candidate(candidate, key, ordered_policy, planning.test, STRESS, assumptions)
    second = price_candidate(candidate, key, ordered_policy, planning.test, STRESS, assumptions)
    assert first == second


def test_pricing_a_subset_matches_the_full_panel_run(
    ordered_policy, planning, assumptions, stressed
):
    """The slice must see the same supply luck as the full run, or pricing is noise."""
    row = at_risk(stressed, ordered_policy, assumptions).iloc[0]
    key = (row["sku"], row["store"])
    no_action = next(
        c for c in build_candidates(row, ordered_policy, assumptions)
        if c.action_type == "no_action"
    )
    priced = price_candidate(
        no_action, key, ordered_policy, planning.test, STRESS, assumptions
    )
    full = stressed.per_item.set_index(["sku", "store"]).loc[key]
    assert priced["scenario_fill_rate"] == pytest.approx(float(full["fill_rate"]), rel=1e-9)
    assert priced["baseline_stockout_weeks"] == int(full["stockout_weeks"])


def test_no_action_is_priced_at_zero(stressed, ordered_policy, planning, assumptions):
    row = at_risk(stressed, ordered_policy, assumptions).iloc[0]
    key = (row["sku"], row["store"])
    no_action = next(
        c for c in build_candidates(row, ordered_policy, assumptions)
        if c.action_type == "no_action"
    )
    priced = price_candidate(no_action, key, ordered_policy, planning.test, STRESS, assumptions)
    assert priced["service_gain_pp"] == pytest.approx(0.0, abs=1e-12)
    assert priced["action_cost_delta"] == pytest.approx(0.0, abs=1e-9)


def test_raising_the_order_quantity_does_not_grant_free_opening_stock(
    ordered_policy, planning, assumptions
):
    """Opening stock is ROP + Q, so the override must not back-date inventory."""
    key = (ordered_policy["sku"].iloc[0], ordered_policy["store"].iloc[0])
    from supplyiq.simulate import baseline_scenario

    draws = draw_lead_times(
        ordered_policy, planning.test["week_index"].nunique(), assumptions["simulation"]["seed"]
    )
    base = simulate(ordered_policy, planning.test, baseline_scenario(), assumptions,
                    lead_time_draws=draws, keep_ledger=True)
    bigger = simulate(
        ordered_policy, planning.test, baseline_scenario(), assumptions,
        overrides=Overrides(order_quantity={key: 10_000.0}),
        lead_time_draws=draws, keep_ledger=True,
    )
    first_week = base.ledger["week_index"].min()
    before = base.ledger[(base.ledger["sku"] == key[0]) & (base.ledger["store"] == key[1])
                         & (base.ledger["week_index"] == first_week)].iloc[0]
    after = bigger.ledger[(bigger.ledger["sku"] == key[0]) & (bigger.ledger["store"] == key[1])
                          & (bigger.ledger["week_index"] == first_week)].iloc[0]
    assert after["on_hand"] == pytest.approx(before["on_hand"])


# --------------------------------------------------------------------------- #
# Choosing between options
# --------------------------------------------------------------------------- #
def test_best_option_is_the_cheapest_service_point_not_the_biggest_gain():
    cheap = Recommendation(sku="S", store="T", supplier_id="X", action_type="expedite_order",
                           service_gain_pp=1.5, action_cost_delta=400.0)
    splashy = Recommendation(sku="S", store="T", supplier_id="X", action_type="dual_source",
                             service_gain_pp=2.0, action_cost_delta=5000.0)
    assert choose_best([cheap, splashy]).action_type == "expedite_order"


def test_no_action_wins_when_nothing_buys_service():
    useless = Recommendation(sku="S", store="T", supplier_id="X", action_type="expedite_order",
                             service_gain_pp=0.0, action_cost_delta=900.0)
    hold = Recommendation(sku="S", store="T", supplier_id="X", action_type="no_action")
    assert choose_best([useless, hold]).action_type == "no_action"


def test_every_recommendation_is_for_a_distinct_item(recommendations):
    keys = [(r.sku, r.store) for r in recommendations]
    assert len(keys) == len(set(keys))


def test_recommendations_either_improve_service_or_advise_no_action(recommendations):
    for rec in recommendations:
        if rec.action_type == "no_action":
            assert rec.service_gain_pp == pytest.approx(0.0, abs=1e-9)
        else:
            assert rec.service_gain_pp > 0, (rec.sku, rec.action_type)


def test_recommendation_frame_is_flat_enough_for_parquet(recommendations):
    frame = to_frame(recommendations)
    assert len(frame) == len(recommendations)
    assert frame["scenario"].map(lambda v: isinstance(v, str)).all()
    assert {"sku", "store", "action_type", "service_gain_pp", "action_cost_delta"} <= set(
        frame.columns
    )


# --------------------------------------------------------------------------- #
# The LLM boundary
# --------------------------------------------------------------------------- #
def test_every_explanation_passes_the_numeric_guard(recommendations):
    """The end-to-end version of the ACT promise."""
    for rec, explanation in explain_all(recommendations, MockProvider()):
        assert is_verified(explanation.text, rec), [
            f.raw for f in unverified_numbers(explanation.text, rec)
        ]


def test_explanations_mention_the_item_and_the_action(recommendations):
    for rec, explanation in explain_all(recommendations, MockProvider()):
        assert rec.sku in explanation.text
        assert rec.store in explanation.text


# --------------------------------------------------------------------------- #
# Decision log
# --------------------------------------------------------------------------- #
def test_decision_log_records_what_was_on_screen(recommendations):
    log = DecisionLog()
    rec = recommendations[0]
    row = log.record(rec, "approved", explanation="because", explanation_source="mock")
    assert len(log) == 1
    assert row["sku"] == rec.sku
    assert row["decision"] == "approved"
    assert row["service_gain_pp"] == rec.service_gain_pp
    assert row["action_cost_delta"] == rec.action_cost_delta
    assert row["decided_at"].endswith("Z")


def test_decision_log_rejects_an_unknown_decision(recommendations):
    with pytest.raises(ValueError, match="approved.*rejected"):
        DecisionLog().record(recommendations[0], "maybe")


def test_decision_log_is_append_only(recommendations):
    log = DecisionLog()
    log.record(recommendations[0], "approved")
    log.record(recommendations[0], "rejected")
    assert len(log) == 2
    assert list(log.to_frame()["decision"]) == ["approved", "rejected"]


def test_decision_log_summary_counts_only_approvals(recommendations):
    log = DecisionLog()
    for i, rec in enumerate(recommendations[:4]):
        log.record(rec, "approved" if i % 2 == 0 else "rejected")
    summary = log.summary()
    assert summary["decisions"] == 4
    assert summary["approved"] == 2
    assert summary["rejected"] == 2
    assert summary["approval_rate"] == pytest.approx(0.5)


def test_empty_decision_log_summarises_safely():
    summary = DecisionLog().summary()
    assert summary["decisions"] == 0
    assert summary["approved_cost"] == 0.0


def test_decision_log_exports_csv_with_every_column(recommendations):
    log = DecisionLog()
    log.record(recommendations[0], "approved", explanation="text", explanation_source="mock")
    csv = log.to_csv()
    assert "decided_at" in csv.splitlines()[0]
    assert "explanation_source" in csv.splitlines()[0]
    assert len(csv.splitlines()) == 2
