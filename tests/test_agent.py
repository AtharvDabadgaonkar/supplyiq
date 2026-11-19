"""The agent runs with no API key, and cannot put an invented number on screen."""
from __future__ import annotations

import pytest

from supplyiq.agent import (
    MockProvider,
    ProviderUnavailable,
    Recommendation,
    available_providers,
    get_provider,
    provider_status,
)
from supplyiq.agent.base import ACTION_TYPES, NUMERIC_FIELDS
from supplyiq.agent.guard import is_verified, mask_identifiers, unverified_numbers
from supplyiq.agent.llm import LLMProvider


def make_rec(action_type: str = "expedite_order", **kw) -> Recommendation:
    base = dict(
        sku="FOODS_1_004",
        store="CA_2",
        supplier_id="SUP-VN-02",
        action_type=action_type,
        baseline_fill_rate=0.93,
        scenario_fill_rate=0.812,
        action_fill_rate=0.964,
        baseline_stockout_weeks=4,
        action_stockout_weeks=1,
        baseline_inventory_value=48_000.0,
        action_inventory_value=52_500.0,
        baseline_landed_cost=18_240.0,
        action_landed_cost=21_310.0,
        action_cost_delta=3_070.0,
        service_gain_pp=15.2,
        units=1_800,
        lead_time_days_delta=-19.0,
        abc_class="A",
        xyz_class="Y",
        risk_reason="facing a 21-day supplier delay",
        scenario={"supplier_delay_days": 21, "demand_spike_pct": 0.0},
    )
    base.update(kw)
    return Recommendation(**base)


# --------------------------------------------------------------------------- #
# Runs with no key
# --------------------------------------------------------------------------- #
def test_default_provider_is_the_mock_without_a_key(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    provider = get_provider()
    assert provider.is_mock
    assert provider.name == "mock"
    assert provider.available()


def test_mock_is_always_in_the_available_list():
    assert "mock" in available_providers()


def test_status_badge_announces_demo_mode(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    status = provider_status()
    assert status["is_mock"] is True
    assert status["badge"] == "Demo mode: rule-based explanations"
    # A status dict must never carry a key value.
    assert set(status["keys_present"].values()) == {False}
    assert "sk-" not in str(status)


def test_requesting_a_live_provider_without_a_key_is_an_explicit_error(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(ProviderUnavailable, match="ANTHROPIC_API_KEY"):
        get_provider(force="anthropic")


def test_unknown_provider_is_rejected():
    with pytest.raises(ProviderUnavailable, match="unknown agent.provider"):
        get_provider(force="not-a-provider")


def test_mock_explains_every_action_type():
    provider = MockProvider()
    for action in ACTION_TYPES:
        rec = make_rec(action)
        explanation = provider.explain(rec)
        assert explanation.is_mock
        assert explanation.provider == "mock"
        assert len(explanation.text) > 60, action
        assert rec.sku in explanation.text


def test_explain_many_matches_explain():
    provider = MockProvider()
    recs = [make_rec(a) for a in ACTION_TYPES]
    assert [e.text for e in provider.explain_many(recs)] == [
        provider.explain(r).text for r in recs
    ]


# --------------------------------------------------------------------------- #
# The LLM cannot produce a number
# --------------------------------------------------------------------------- #
def test_assumptions_declare_that_the_llm_produces_no_numbers(assumptions):
    assert assumptions["agent"]["llm_may_produce_numbers"] is False


def test_every_numeric_field_is_covered_by_the_guard():
    """A new simulator output must be added to NUMERIC_FIELDS or the guard misses it."""
    rec = make_rec()
    payload = rec.as_payload()
    numeric = {
        k
        for k, v in payload.items()
        if isinstance(v, (int, float)) and not isinstance(v, bool)
    }
    assert numeric == set(NUMERIC_FIELDS), numeric.symmetric_difference(NUMERIC_FIELDS)


@pytest.mark.parametrize("action", ACTION_TYPES)
def test_mock_output_always_passes_the_guard(action):
    rec = make_rec(action)
    assert is_verified(MockProvider().explain(rec).text, rec)


def test_guard_rejects_an_invented_figure():
    rec = make_rec()
    bad = "Expediting lifts fill rate to 99.8% and costs about $12,400 over the quarter."
    findings = unverified_numbers(bad, rec)
    assert {f.value for f in findings} >= {99.8, 12_400.0}


def test_guard_accepts_simulator_figures_restated():
    rec = make_rec()
    good = (
        "Fill rate improves from 81.2% to 96.4%, a gain of 15.2pp, and stockout weeks "
        "fall from 4 to 1. The extra landed cost is $3,070 for 1,800 units, pulling the "
        "order in 19 days."
    )
    assert is_verified(good, rec), [f.raw for f in unverified_numbers(good, rec)]


def test_guard_ignores_identifiers():
    rec = make_rec()
    assert is_verified(f"{rec.sku} at {rec.store} from {rec.supplier_id} needs attention.", rec)
    masked = mask_identifiers("FOODS_1_004 at CA_2 via SUP-VN-02", rec)
    assert "004" not in masked and "CA_2" not in masked


def test_guard_allows_numbers_the_simulator_wrote_into_context():
    rec = make_rec(risk_reason="exposed to a 25% tariff and a 21-day delay")
    assert is_verified("A 25% tariff plus a 21-day delay is the root cause.", rec)


def test_llm_provider_falls_back_to_mock_without_a_key():
    provider = LLMProvider(kind="anthropic", model="claude-opus-5", api_key=None)
    assert not provider.available()
    explanation = provider.explain(make_rec())
    assert explanation.is_mock
    assert explanation.provider == "mock"
    assert "no API key" in (explanation.fallback_reason or "")


def test_llm_provider_falls_back_when_the_call_fails(monkeypatch):
    provider = LLMProvider(kind="anthropic", model="claude-opus-5", api_key="test-key-not-real")

    def boom(self, rec):
        raise TimeoutError("network down")

    monkeypatch.setattr(LLMProvider, "_call", boom)
    explanation = provider.explain(make_rec())
    assert explanation.is_mock
    assert "call failed: TimeoutError" in (explanation.fallback_reason or "")


def test_llm_text_with_an_invented_number_is_discarded(monkeypatch):
    """The whole point: unverifiable prose never reaches the planner."""
    provider = LLMProvider(kind="anthropic", model="claude-opus-5", api_key="test-key-not-real")
    monkeypatch.setattr(
        LLMProvider,
        "_call",
        lambda self, rec: "This saves roughly $87,500 and lifts fill rate to 99.9%.",
    )
    explanation = provider.explain(make_rec())
    assert explanation.is_mock
    assert "rejected LLM text" in (explanation.fallback_reason or "")
    assert "87,500" not in explanation.text


def test_llm_text_that_only_restates_simulator_numbers_is_kept(monkeypatch):
    provider = LLMProvider(kind="anthropic", model="claude-opus-5", api_key="test-key-not-real")
    monkeypatch.setattr(
        LLMProvider,
        "_call",
        lambda self, rec: (
            "Pulling the order in 19 days lifts fill rate from 81.2% to 96.4% for an extra "
            "$3,070. Worth approving on an A-class item."
        ),
    )
    explanation = provider.explain(make_rec())
    assert not explanation.is_mock
    assert explanation.provider == "anthropic"
    assert explanation.fallback_reason is None


def test_api_key_is_never_placed_in_the_request_url():
    provider = LLMProvider(kind="openai", model="gpt-x", api_key="super-secret")
    body, headers = provider._payload(make_rec())
    assert "super-secret" not in str(body)
    assert headers["authorization"].startswith("Bearer ")


def test_prompt_forbids_calculation():
    from supplyiq.agent.llm import SYSTEM_PROMPT

    assert "Do NOT calculate" in SYSTEM_PROMPT


# --------------------------------------------------------------------------- #
# Recommendation payload is read-only
# --------------------------------------------------------------------------- #
def test_recommendation_is_frozen():
    rec = make_rec()
    with pytest.raises(Exception):
        rec.action_cost_delta = 0.0  # type: ignore[misc]


def test_payload_mutation_does_not_touch_the_recommendation():
    rec = make_rec()
    payload = rec.as_payload()
    payload["action_cost_delta"] = -99_999.0
    assert rec.action_cost_delta == 3_070.0


def test_cost_per_service_point():
    rec = make_rec()
    assert rec.cost_per_service_point == pytest.approx(3_070.0 / 15.2)
    assert make_rec(service_gain_pp=0.0).cost_per_service_point is None
