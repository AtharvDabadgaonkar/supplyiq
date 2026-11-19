"""SENSE — supplier performance metrics."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from supplyiq.config import supplier_frame
from supplyiq.suppliers import (
    DAYS_PER_WEEK,
    item_lead_times,
    overall_performance,
    po_flags,
    supplier_performance,
)


def test_every_supplier_appears_once(performance, tables):
    assert len(performance) == len(tables["suppliers"])
    assert not performance["supplier_id"].duplicated().any()


def test_otif_is_on_time_and_in_full(tables):
    flagged = po_flags(tables["purchase_orders"])
    expected = ((flagged["late_delivery"] == 0) & (flagged["in_full"] == 1)).astype(int)
    np.testing.assert_array_equal(flagged["otif_flag"].to_numpy(), expected.to_numpy())
    # OTIF can never exceed either of its two components.
    assert flagged["otif_flag"].mean() <= flagged["on_time"].mean()
    assert flagged["otif_flag"].mean() <= flagged["in_full_flag"].mean()


def test_rates_are_valid_probabilities(performance):
    for column in ("otif", "on_time_rate", "in_full_rate", "late_delivery_rate", "mean_fill_ratio"):
        assert performance[column].between(0.0, 1.0).all(), column


def test_on_time_and_late_rates_are_complements(performance):
    np.testing.assert_allclose(
        performance["on_time_rate"] + performance["late_delivery_rate"], 1.0, rtol=1e-9
    )


def test_lead_time_weeks_are_days_over_seven(performance):
    np.testing.assert_allclose(
        performance["lead_time_mean_weeks"],
        performance["lead_time_mean_days"] / DAYS_PER_WEEK,
        rtol=1e-9,
    )
    np.testing.assert_allclose(
        performance["lead_time_sd_weeks"],
        performance["lead_time_sd_days"] / DAYS_PER_WEEK,
        rtol=1e-9,
    )


def test_measured_lead_times_recover_the_configured_distribution(performance):
    """Measured from receipts, not read from config -- so this proves the wiring."""
    configured = supplier_frame().set_index("supplier_id")
    for row in performance.itertuples(index=False):
        assert row.lead_time_mean_days == pytest.approx(
            float(configured.loc[row.supplier_id, "mean_lead_time_days"]), abs=2.0
        ), row.supplier_id
        assert row.lead_time_sd_days == pytest.approx(
            float(configured.loc[row.supplier_id, "lead_time_sd_days"]), abs=2.0
        ), row.supplier_id


def test_quoted_lead_time_exceeds_the_measured_mean(performance):
    assert (performance["quoted_lead_time_days"] > performance["lead_time_mean_days"]).all()


def test_p95_lead_time_is_above_the_mean(performance):
    assert (performance["lead_time_p95_days"] >= performance["lead_time_mean_days"]).all()


def test_overseas_suppliers_have_longer_and_more_variable_lead_times(performance):
    """The procurement argument the Supplier Performance page has to make."""
    by_country = performance.groupby("country")[["lead_time_mean_days", "lead_time_sd_days"]].mean()
    assert by_country.loc["CN", "lead_time_mean_days"] > by_country.loc["US", "lead_time_mean_days"]
    assert by_country.loc["CN", "lead_time_sd_days"] > by_country.loc["US", "lead_time_sd_days"]


def test_otif_lands_near_each_supplier_target(performance):
    assert performance["otif_vs_target_pp"].abs().max() < 5.0


def test_overall_performance_pools_every_po(tables):
    overall = overall_performance(tables["purchase_orders"])
    pos = tables["purchase_orders"]
    assert overall["pos"] == len(pos)
    assert overall["late_delivery_rate"] == pytest.approx(float(pos["late_delivery"].mean()))
    assert overall["unit_fill_rate"] == pytest.approx(
        float(pos["units_received"].sum() / pos["units_ordered"].sum())
    )
    assert 0 < overall["unit_fill_rate"] <= 1


def test_item_lead_times_cover_every_planning_item(tables, performance, assumptions):
    joined = item_lead_times(tables["sku_master"], performance, assumptions)
    assert len(joined) == len(tables["sku_master"])
    assert joined["lead_time_mean_weeks"].notna().all()
    assert joined["lead_time_sd_weeks"].notna().all()


def test_item_lead_times_raise_when_a_supplier_has_no_history(tables, performance, assumptions):
    """A silent NaN here would quietly zero out safety stock for those items."""
    partial = performance.iloc[1:].copy()
    with pytest.raises(ValueError, match="no PO history"):
        item_lead_times(tables["sku_master"], partial, assumptions)


def test_days_late_is_consistent_with_the_late_flag(tables):
    flagged = po_flags(tables["purchase_orders"])
    assert (flagged.loc[flagged["late_delivery"] == 1, "days_late"] > 0).all()
    assert (flagged.loc[flagged["late_delivery"] == 0, "days_late"] <= 0).all()
