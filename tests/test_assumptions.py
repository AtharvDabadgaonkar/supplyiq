"""The assumptions registry is the single source of planning parameters."""
from __future__ import annotations

import re
from pathlib import Path

import pytest

from supplyiq.config import (
    AssumptionsError,
    REQUIRED_TOP_LEVEL,
    load_assumptions,
    simulated_sections,
    supplier_frame,
)
from supplyiq.paths import ASSUMPTIONS_FILE, REPO_ROOT


def test_loads_and_has_every_section(assumptions):
    for section in REQUIRED_TOP_LEVEL:
        assert section in assumptions


def test_allocation_shares_sum_to_one(assumptions):
    for category, shares in assumptions["suppliers"]["sku_allocation"].items():
        assert abs(sum(shares.values()) - 1.0) < 1e-9, category


def test_every_supplier_has_a_distinct_alternate_source(assumptions):
    ids = {s["supplier_id"] for s in assumptions["suppliers"]["network"]}
    alt = assumptions["suppliers"]["alternate_source"]
    assert set(alt) == ids, "every supplier needs an alternate for dual-sourcing"
    for src, dst in alt.items():
        assert dst in ids and dst != src


def test_scenario_defaults_are_no_disruption(assumptions):
    """The SIMULATE guarantee depends on this: defaults must be the null scenario."""
    for name, spec in assumptions["scenarios"].items():
        assert float(spec["default"]) == 0.0, name
        assert spec["min"] <= spec["default"] <= spec["max"], name


def test_simulated_sections_are_labelled(assumptions):
    sections = simulated_sections(assumptions)
    assert "suppliers" in sections
    assert "costs" in sections
    # The two real datasets must NOT be marked simulated.
    for key in ("demand", "orders"):
        assert assumptions["data_sources"][key]["simulated"] is False


def test_supplier_frame_resolves_costs(assumptions):
    df = supplier_frame()
    assert len(df) == len(assumptions["suppliers"]["network"])
    assert df["freight_per_unit"].notna().all()
    assert df["tariff_rate"].notna().all()
    assert (df["mean_lead_time_days"] > 0).all()
    assert (df["lead_time_sd_days"] > 0).all()
    assert df["otif_target"].between(0, 1).all()


def test_validator_rejects_a_broken_allocation(tmp_path: Path):
    import yaml

    data = yaml.safe_load(ASSUMPTIONS_FILE.read_text())
    data["suppliers"]["sku_allocation"]["FOODS"] = {"SUP-US-01": 0.5}
    bad = tmp_path / "bad.yaml"
    bad.write_text(yaml.safe_dump(data))
    with pytest.raises(AssumptionsError, match="sums to"):
        load_assumptions(bad)


def test_validator_rejects_a_nonzero_scenario_default(tmp_path: Path):
    import yaml

    data = yaml.safe_load(ASSUMPTIONS_FILE.read_text())
    data["scenarios"]["demand_spike_pct"]["default"] = 10.0
    bad = tmp_path / "bad.yaml"
    bad.write_text(yaml.safe_dump(data))
    with pytest.raises(AssumptionsError, match="no disruption"):
        load_assumptions(bad)


# --------------------------------------------------------------------------- #
# "No made-up metrics": planning parameters must not be hard-coded in the code.
# --------------------------------------------------------------------------- #
PARAM_PATTERNS = [
    re.compile(r"service_level\s*=\s*0\.\d+"),
    re.compile(r"holding_cost\w*\s*=\s*0\.\d+"),
    re.compile(r"tariff\w*\s*=\s*0\.\d+"),
    re.compile(r"lead_time\w*\s*=\s*\d+\.?\d*\s*$", re.MULTILINE),
    re.compile(r"gross_margin\w*\s*=\s*0\.\d+"),
]


def test_no_hardcoded_planning_parameters():
    """Planning constants belong in assumptions.yaml, not in a literal."""
    offenders = []
    for path in (REPO_ROOT / "supplyiq").rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        for pattern in PARAM_PATTERNS:
            for match in pattern.finditer(text):
                line = text[: match.start()].count("\n") + 1
                offenders.append(f"{path.relative_to(REPO_ROOT)}:{line}: {match.group(0)!r}")
    assert not offenders, "hard-coded planning parameters found:\n" + "\n".join(offenders)
