"""Rule-based provider. The default, and the reason no API key is required.

It writes the same shape of explanation an LLM would, by selecting a template
from the action type and filling it from the simulator's numbers. Nothing here
calls out to a network, so tests and the deployed demo are deterministic.
"""
from __future__ import annotations

from supplyiq.agent.base import AgentProvider, Explanation, Recommendation


def _money(x: float) -> str:
    sign = "-" if x < 0 else ""
    v = abs(float(x))
    if v >= 1_000_000:
        return f"{sign}${v / 1_000_000:.2f}M"
    if v >= 1_000:
        return f"{sign}${v / 1_000:,.1f}k"
    return f"{sign}${v:,.0f}"


def _pp(x: float) -> str:
    return f"{float(x):+.1f}pp"


class MockProvider(AgentProvider):
    """Deterministic, offline explanation writer."""

    name = "mock"
    is_mock = True

    def available(self) -> bool:
        return True

    # ------------------------------------------------------------------ #
    def explain(self, rec: Recommendation) -> Explanation:
        p = rec.as_payload()
        head = self._headline(rec)
        risk = self._risk_clause(rec)
        trade = self._tradeoff_clause(rec)
        verdict = self._verdict_clause(rec)
        text = " ".join(part for part in (risk, head, trade, verdict) if part)
        return Explanation(text=text, provider=self.name, model=None, is_mock=True)

    # ------------------------------------------------------------------ #
    def _risk_clause(self, rec: Recommendation) -> str:
        where = f"{rec.sku} at {rec.store}"
        cls = f"{rec.abc_class}{rec.xyz_class}".strip()
        cls_txt = f" ({cls} item)" if cls else ""
        reason = rec.risk_reason or "projected to miss the service target"
        return f"{where}{cls_txt} is {reason}."

    def _headline(self, rec: Recommendation) -> str:
        a = rec.action_type
        if a == "expedite_order":
            return (
                f"Expediting {rec.units:,} units pulls the inbound order in by "
                f"{abs(rec.lead_time_days_delta):.0f} days."
            )
        if a == "rebalance_stock":
            return f"Rebalancing {rec.units:,} units in from another store covers the gap."
        if a == "switch_supplier":
            return (
                f"Switching this SKU to its alternate source changes the lead time by "
                f"{rec.lead_time_days_delta:+.0f} days."
            )
        if a == "dual_source":
            return (
                f"Dual-sourcing {rec.units:,} units splits the order across the primary "
                "and alternate supplier."
            )
        if a == "adjust_order_qty":
            direction = "raising" if rec.units >= 0 else "cutting"
            return f"{direction.capitalize()} the order quantity by {abs(rec.units):,} units."
        return "No action is recommended."

    def _tradeoff_clause(self, rec: Recommendation) -> str:
        if rec.action_type == "no_action":
            return (
                f"Fill rate is projected at {rec.scenario_fill_rate:.1%}, already at or above "
                "target under this scenario."
            )
        return (
            f"Fill rate moves {rec.scenario_fill_rate:.1%} -> {rec.action_fill_rate:.1%} "
            f"({_pp(rec.service_gain_pp)}), stockout weeks {rec.baseline_stockout_weeks} -> "
            f"{rec.action_stockout_weeks}, for a cost change of "
            f"{_money(rec.action_cost_delta)}."
        )

    def _verdict_clause(self, rec: Recommendation) -> str:
        if rec.action_type == "no_action":
            return "Hold the current plan."
        cps = rec.cost_per_service_point
        if cps is None:
            return (
                "This buys no measurable service improvement in the simulation, so it is "
                "only worth doing if you are protecting against a worse scenario."
            )
        if rec.action_cost_delta <= 0:
            return (
                f"It improves service and reduces cost, so it is worth doing on both counts "
                f"({_money(cps)} per service point)."
            )
        cheap = cps < 250
        return (
            f"That is {_money(cps)} per point of fill rate, which is "
            f"{'cheap enough to approve' if cheap else 'expensive -- worth it only for A-class or contracted service'}."
        )
