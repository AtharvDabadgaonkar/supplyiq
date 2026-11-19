"""Loader for config/assumptions.yaml.

Rule enforced by tests: no module in supplyiq/ may hard-code a planning
parameter. Everything comes through `load_assumptions()`.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

from supplyiq.paths import ASSUMPTIONS_FILE


class AssumptionsError(RuntimeError):
    """Raised when the assumptions file is missing or internally inconsistent."""


REQUIRED_TOP_LEVEL = (
    "meta",
    "data_sources",
    "sample",
    "suppliers",
    "costs",
    "policy",
    "baseline_policy",
    "scenarios",
    "forecast",
    "agent",
)


def _validate(a: dict[str, Any]) -> None:
    missing = [k for k in REQUIRED_TOP_LEVEL if k not in a]
    if missing:
        raise AssumptionsError(f"assumptions.yaml is missing sections: {missing}")

    supplier_ids = {s["supplier_id"] for s in a["suppliers"]["network"]}

    # Every allocation row must sum to 1.0 and reference a known supplier.
    for category, alloc in a["suppliers"]["sku_allocation"].items():
        total = sum(alloc.values())
        if abs(total - 1.0) > 1e-6:
            raise AssumptionsError(
                f"suppliers.sku_allocation[{category}] sums to {total}, expected 1.0"
            )
        unknown = set(alloc) - supplier_ids
        if unknown:
            raise AssumptionsError(
                f"suppliers.sku_allocation[{category}] references unknown suppliers: {unknown}"
            )

    # Alternate sources must be real suppliers and never point at themselves.
    for src, alt in a["suppliers"]["alternate_source"].items():
        if src not in supplier_ids or alt not in supplier_ids:
            raise AssumptionsError(f"alternate_source {src}->{alt} references unknown supplier")
        if src == alt:
            raise AssumptionsError(f"alternate_source {src} points at itself")

    # Every supplier country needs a freight and a tariff entry.
    countries = {s["country"] for s in a["suppliers"]["network"]}
    for table in ("freight_per_unit_by_country", "tariff_rate_by_country"):
        missing_c = countries - set(a["costs"][table])
        if missing_c:
            raise AssumptionsError(f"costs.{table} is missing countries: {missing_c}")

    # Service levels must be valid probabilities.
    levels = [a["policy"]["service_level_target"]]
    levels += list(a["policy"]["service_level_options"])
    levels += list(a["policy"]["service_level_by_abc"].values())
    for lv in levels:
        if not 0.0 < lv < 1.0:
            raise AssumptionsError(f"service level {lv} must be strictly between 0 and 1")

    # The holdout must be shorter than the history.
    s = a["sample"]
    if s["holdout_weeks"] >= s["n_weeks"]:
        raise AssumptionsError("sample.holdout_weeks must be smaller than sample.n_weeks")
    if s["n_items"] * s["n_stores"] != s["n_planning_items"]:
        raise AssumptionsError(
            f"sample.n_planning_items={s['n_planning_items']} must equal "
            f"n_items({s['n_items']}) * n_stores({s['n_stores']})"
        )
    if s["n_stores"] < 2:
        raise AssumptionsError("sample.n_stores must be >= 2 so stock rebalancing is possible")

    # Scenario defaults must be the no-disruption case, so that the SIMULATE
    # module's "zero disruption reproduces baseline" guarantee holds.
    for name, spec in a["scenarios"].items():
        if spec["default"] != 0 and spec["default"] != 0.0:
            raise AssumptionsError(f"scenarios.{name}.default must be 0 (no disruption)")


@lru_cache(maxsize=4)
def load_assumptions(path: str | Path | None = None) -> dict[str, Any]:
    """Read, validate and cache the assumptions file."""
    p = Path(path) if path is not None else ASSUMPTIONS_FILE
    if not p.exists():
        raise AssumptionsError(f"assumptions file not found: {p}")
    with p.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    if not isinstance(data, dict):
        raise AssumptionsError(f"{p} did not parse to a mapping")
    _validate(data)
    return data


def simulated_sections(a: dict[str, Any] | None = None) -> list[str]:
    """Names of the assumption sections that are simulated, for UI labelling."""
    a = a or load_assumptions()
    out = []
    for key, value in a.items():
        if isinstance(value, dict) and value.get("simulated") is True:
            out.append(key)
    for key, value in a.get("data_sources", {}).items():
        if isinstance(value, dict) and value.get("simulated") is True:
            out.append(f"data_sources.{key}")
    return sorted(out)


def supplier_frame():
    """Supplier master as a DataFrame, with landed-cost inputs joined on."""
    import pandas as pd

    a = load_assumptions()
    rows = []
    for s in a["suppliers"]["network"]:
        c = s["country"]
        rows.append(
            {
                **s,
                "freight_per_unit": a["costs"]["freight_per_unit_by_country"][c],
                "tariff_rate": a["costs"]["tariff_rate_by_country"][c],
                "alternate_supplier_id": a["suppliers"]["alternate_source"].get(
                    s["supplier_id"]
                ),
            }
        )
    return pd.DataFrame(rows)
