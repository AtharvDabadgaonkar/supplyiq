"""Numeric guard: proof that the LLM did not invent a number.

The ACT module's contract is that every figure a planner sees is computed by
the simulator. We cannot stop a language model from typing digits, so instead
we verify: every number in the generated prose must reconcile with a value on
the Recommendation. Anything that does not is an unverified number, and the
caller discards that explanation and uses the rule-based one instead.

This is what makes "the LLM does not produce numbers" an enforced property
rather than a promise in a prompt.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from supplyiq.agent.base import NUMERIC_FIELDS, Recommendation

# $1.2k / 95.3% / +1.4pp / 1,234 / 42 days / -0.5
_NUMBER_RE = re.compile(
    r"""
    (?P<currency>[$])?              # optional leading $
    (?P<sign>[-+])?
    (?P<value>\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?)
    \s*
    (?P<suffix>%|pp|[kKmM](?![a-zA-Z]))?
    """,
    re.VERBOSE,
)

# Years, ordinals and the like would otherwise read as unverified numbers.
_IGNORE_CONTEXT = re.compile(r"(week\s+\d+|\bQ[1-4]\b|\b(19|20)\d{2}\b)", re.IGNORECASE)

# Identifiers carry digits but are not quantities: FOODS_1_004, CA_2, SUP-VN-02,
# PO-000123. They are masked out before the numeric scan.
_IDENTIFIER_RE = re.compile(r"\b[A-Za-z][A-Za-z0-9]*(?:[_-][A-Za-z0-9]+)+\b")
_MASK = " \u00b7 "


@dataclass(frozen=True)
class Finding:
    raw: str
    value: float
    reason: str


def _scale(suffix: str | None) -> float:
    if suffix in ("k", "K"):
        return 1_000.0
    if suffix in ("m", "M"):
        return 1_000_000.0
    return 1.0


# Fields held as a 0-1 proportion, which a provider may legitimately render as a
# percentage. Percent scaling is applied ONLY to these: scaling a count such as
# `action_stockout_weeks=1` to 100 would silently whitelist "99.8%".
RATE_FIELDS: frozenset[str] = frozenset(
    {"baseline_fill_rate", "scenario_fill_rate", "action_fill_rate"}
)


def allowed_values(rec: Recommendation) -> set[float]:
    """Every number a provider is permitted to restate, in all plausible scales."""
    out: set[float] = set()
    payload = rec.as_payload()
    # (value, may_be_shown_as_a_percentage)
    raw: list[tuple[float, bool]] = [
        (float(payload[f]), f in RATE_FIELDS) for f in NUMERIC_FIELDS
    ]
    cps = rec.cost_per_service_point
    if cps is not None:
        raw.append((float(cps), False))
    # Deltas a provider might legitimately phrase itself.
    raw.append((float(rec.action_fill_rate - rec.scenario_fill_rate), True))
    raw.append((float(rec.baseline_stockout_weeks - rec.action_stockout_weeks), False))
    raw.append((float(rec.action_inventory_value - rec.baseline_inventory_value), False))
    raw.append((float(rec.action_landed_cost - rec.baseline_landed_cost), False))

    # Numbers the simulator itself wrote into the context fields (e.g. a
    # risk_reason of "facing a 21-day supplier delay", or a scenario setting)
    # are simulator output too, so a provider may restate them.
    context = [str(rec.risk_reason), str(rec.abc_class), str(rec.xyz_class)]
    context += [f"{k} {v}" for k, v in (rec.scenario or {}).items()]
    for blob in context:
        for m in _NUMBER_RE.finditer(_IDENTIFIER_RE.sub(_MASK, blob)):
            value = float(m.group("value").replace(",", "")) * _scale(m.group("suffix"))
            raw.append((value, False))

    for value, is_rate in raw:
        out.add(round(value, 6))
        out.add(round(abs(value), 6))
        if is_rate:
            out.add(round(value * 100.0, 6))
            out.add(round(abs(value) * 100.0, 6))
    return out


def _matches(value: float, allowed: set[float]) -> bool:
    for a in allowed:
        if abs(value - a) <= max(0.51, abs(a) * 0.02):
            return True
    return False


def mask_identifiers(text: str, rec: Recommendation | None = None) -> str:
    """Blank out SKU codes, store codes and supplier ids so the scan sees only quantities."""
    out = text
    if rec is not None:
        for ident in (rec.sku, rec.store, rec.supplier_id):
            if ident:
                out = out.replace(str(ident), _MASK)
    return _IDENTIFIER_RE.sub(_MASK, out)


def unverified_numbers(text: str, rec: Recommendation) -> list[Finding]:
    """Numbers in `text` that do not reconcile with any value on `rec`."""
    allowed = allowed_values(rec)
    text = mask_identifiers(text, rec)
    findings: list[Finding] = []
    for m in _NUMBER_RE.finditer(text):
        window = text[max(0, m.start() - 12) : m.end() + 6]
        if _IGNORE_CONTEXT.search(window):
            continue
        value = float(m.group("value").replace(",", "")) * _scale(m.group("suffix"))
        if m.group("sign") == "-":
            value = -value
        if not _matches(value, allowed) and not _matches(abs(value), allowed):
            findings.append(
                Finding(raw=m.group(0).strip(), value=value, reason="not derived from the simulator")
            )
    return findings


def is_verified(text: str, rec: Recommendation) -> bool:
    return not unverified_numbers(text, rec)
