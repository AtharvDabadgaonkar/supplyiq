"""Ingest: determinism, provenance honesty, and the business rules of the sample."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from supplyiq import ingest
from supplyiq.config import supplier_frame
from supplyiq.data import split_weeks, train_test
from supplyiq.schemas import TABLES_BY_NAME, validate


# --------------------------------------------------------------------------- #
# Determinism
# --------------------------------------------------------------------------- #
def test_synthetic_build_is_deterministic(assumptions):
    """Same seed, same bytes — otherwise `make ingest` would not be reproducible."""
    first, prov_a = ingest.build_synthetic(assumptions)
    second, prov_b = ingest.build_synthetic(assumptions)
    for name in TABLES_BY_NAME:
        pd.testing.assert_frame_equal(first[name], second[name])
    assert prov_a.row_counts == prov_b.row_counts
    assert prov_a.origin == prov_b.origin == "synthetic"


def test_synthetic_build_validates_against_every_spec(assumptions):
    tables, _ = ingest.build_synthetic(assumptions)
    for name, spec in TABLES_BY_NAME.items():
        validate(tables[name], spec)


def test_build_falls_back_to_synthetic_without_raw_files(tmp_path, assumptions):
    assert not ingest.have_kaggle(tmp_path)
    _, prov = ingest.build(prefer="auto", raw_dir=tmp_path, a=assumptions)
    assert prov.origin == "synthetic"


def test_build_from_kaggle_raises_when_raw_files_absent(tmp_path, assumptions):
    with pytest.raises(FileNotFoundError, match="missing raw files"):
        ingest.build_from_kaggle(tmp_path, assumptions)


# --------------------------------------------------------------------------- #
# Provenance honesty — the sample must never claim to be real when it is not
# --------------------------------------------------------------------------- #
def test_provenance_matches_whether_raw_files_exist(provenance):
    if provenance.origin == "kaggle":
        assert ingest.have_kaggle(), "sample claims Kaggle origin but data/raw is incomplete"
        assert provenance.is_real
    else:
        assert provenance.origin == "synthetic"
        assert not provenance.is_real
        assert "SYNTHETIC" in provenance.notes.upper()


def test_provenance_badge_states_what_is_simulated(provenance):
    assert "simulated" in provenance.badge.lower()
    assert "suppliers" in provenance.simulated_sections
    assert "costs" in provenance.simulated_sections


def test_row_counts_in_provenance_match_the_files(provenance, tables):
    for name, count in provenance.row_counts.items():
        assert len(tables[name]) == count, name


# --------------------------------------------------------------------------- #
# Sample scope matches what the config promises
# --------------------------------------------------------------------------- #
def test_sample_scope_matches_config(tables, assumptions):
    s = assumptions["sample"]
    demand, sku_master = tables["demand"], tables["sku_master"]
    assert len(sku_master) == s["n_planning_items"]
    assert demand["sku"].nunique() == s["n_items"]
    assert demand["store"].nunique() == s["n_stores"]
    assert int(demand["week_index"].max()) + 1 == s["n_weeks"]
    assert set(demand["category"]) <= set(s["categories"])
    assert set(demand["state"]) == {s["state"]}


def test_week_index_is_dense_and_aligned_to_week_start(tables):
    demand = tables["demand"]
    weeks = demand[["week_index", "week_start"]].drop_duplicates().sort_values("week_index")
    assert list(weeks["week_index"]) == list(range(len(weeks)))
    # one week_start per index, exactly 7 days apart
    gaps = weeks["week_start"].diff().dropna().unique()
    assert list(gaps) == [pd.Timedelta(days=7)]


def test_every_week_starts_on_the_configured_weekday(tables, assumptions):
    expected = assumptions["meta"]["week_starts_on"].upper()
    names = tables["demand"]["week_start"].dt.day_name().str.upper().str[:3].unique()
    assert list(names) == [expected]


def test_demand_is_a_complete_panel(tables, assumptions):
    """Every planning item has every week — the simulator assumes no gaps."""
    demand = tables["demand"]
    n_items, n_weeks = len(tables["sku_master"]), assumptions["sample"]["n_weeks"]
    assert len(demand) == n_items * n_weeks
    per_item = demand.groupby(["sku", "store"], observed=True).size()
    assert per_item.min() == per_item.max() == n_weeks


def test_demand_values_are_plausible(tables):
    demand = tables["demand"]
    assert (demand["units_sold"] >= 0).all()
    assert (demand["sell_price"] > 0).all()
    np.testing.assert_allclose(
        demand["revenue"].astype(float),
        demand["units_sold"].astype(float) * demand["sell_price"].astype(float),
        rtol=1e-3,
    )


def test_sample_includes_intermittent_demand(tables):
    """Slow movers are the hard case; the sample must contain some."""
    demand = tables["demand"]
    zero_share = (
        demand.groupby(["sku", "store"], observed=True)["units_sold"]
        .apply(lambda s: (s == 0).mean())
    )
    assert zero_share.max() > 0.25, "no intermittent items — XYZ and MAPE guards untested"


# --------------------------------------------------------------------------- #
# Supplier assignment and the cost stack
# --------------------------------------------------------------------------- #
def test_each_sku_has_exactly_one_supplier(tables):
    per_sku = tables["sku_master"].groupby("sku")["supplier_id"].nunique()
    assert per_sku.max() == 1, "a SKU must be sourced from one vendor across the network"


def test_supplier_mix_matches_configured_shares(tables, assumptions):
    items = tables["sku_master"][["sku", "category", "supplier_id"]].drop_duplicates()
    for category, shares in assumptions["suppliers"]["sku_allocation"].items():
        realised = items[items["category"] == category]["supplier_id"].value_counts(
            normalize=True
        )
        assert set(realised.index) <= set(shares)
        for supplier_id, share in shares.items():
            assert abs(float(realised.get(supplier_id, 0.0)) - share) < 0.05, (
                category,
                supplier_id,
            )


def test_landed_cost_follows_the_documented_formula(tables, assumptions):
    sm = tables["sku_master"].astype(
        {"unit_cost": "float64", "freight_per_unit": "float64",
         "tariff_rate": "float64", "landed_cost": "float64", "avg_sell_price": "float64"}
    )
    expected = sm["unit_cost"] + sm["freight_per_unit"] + sm["unit_cost"] * sm["tariff_rate"]
    np.testing.assert_allclose(sm["landed_cost"], expected, rtol=1e-4)


def test_unit_cost_follows_the_documented_formula(tables, assumptions):
    sup = supplier_frame().set_index("supplier_id")
    sm = tables["sku_master"]
    margin = assumptions["costs"]["gross_margin_target"]
    expected = (
        sm["avg_sell_price"].astype(float)
        * (1 - margin)
        * sm["supplier_id"].map(sup["unit_cost_index"]).astype(float)
    )
    np.testing.assert_allclose(sm["unit_cost"].astype(float), expected, rtol=1e-4)


def test_unit_cost_is_below_sell_price(tables):
    sm = tables["sku_master"]
    assert (sm["unit_cost"] < sm["avg_sell_price"]).all(), "negative gross margin"


def test_tariff_rates_match_the_supplier_country(tables, assumptions):
    sup = tables["suppliers"].set_index("supplier_id")
    sm = tables["sku_master"]
    expected = sm["supplier_id"].map(sup["country"]).map(
        assumptions["costs"]["tariff_rate_by_country"]
    )
    np.testing.assert_allclose(sm["tariff_rate"].astype(float), expected.astype(float))


# --------------------------------------------------------------------------- #
# Purchase orders / supplier performance
# --------------------------------------------------------------------------- #
def test_realised_lead_times_match_the_configured_distribution(tables):
    po = tables["purchase_orders"]
    cfg = supplier_frame().set_index("supplier_id")
    stats = po.groupby("supplier_id")["days_actual"].agg(["mean", "std"])
    for supplier_id, row in stats.iterrows():
        assert abs(row["mean"] - cfg.loc[supplier_id, "mean_lead_time_days"]) < 2.0, supplier_id
        assert abs(row["std"] - cfg.loc[supplier_id, "lead_time_sd_days"]) < 2.0, supplier_id


def test_realised_otif_lands_on_the_configured_target(tables):
    """OTIF is split evenly between on-time and in-full, so it must hit the target."""
    po = tables["purchase_orders"]
    cfg = supplier_frame().set_index("supplier_id")
    otif = po.assign(
        otif=((po["late_delivery"] == 0) & (po["in_full"] == 1)).astype(int)
    ).groupby("supplier_id")["otif"].mean()
    for supplier_id, value in otif.items():
        target = float(cfg.loc[supplier_id, "otif_target"])
        assert abs(value - target) < 0.05, (supplier_id, value, target)


def test_late_delivery_flag_is_consistent_with_the_dates(tables):
    po = tables["purchase_orders"]
    expected = (po["received_date"] > po["promised_date"]).astype(int)
    assert (po["late_delivery"].astype(int) == expected).all()


def test_units_received_never_exceeds_units_ordered(tables):
    po = tables["purchase_orders"]
    assert (po["units_received"] <= po["units_ordered"]).all()
    in_full = po["in_full"] == 1
    assert (po.loc[in_full, "units_received"] == po.loc[in_full, "units_ordered"]).all()
    assert (po.loc[~in_full, "units_received"] < po.loc[~in_full, "units_ordered"]).all()


def test_order_quantities_respect_supplier_moq(tables):
    po = tables["purchase_orders"]
    moq = supplier_frame().set_index("supplier_id")["moq_units"]
    assert (po["units_ordered"] >= po["supplier_id"].map(moq)).all()


def test_quoted_lead_time_buffers_the_mean(tables):
    """A quoted date at the mean would make half of all POs late by construction."""
    po = tables["purchase_orders"]
    cfg = supplier_frame().set_index("supplier_id")
    quoted = po.groupby("supplier_id")["days_scheduled"].mean()
    for supplier_id, value in quoted.items():
        assert value > float(cfg.loc[supplier_id, "mean_lead_time_days"]), supplier_id


def test_every_supplier_has_purchase_orders(tables):
    assert set(tables["purchase_orders"]["supplier_id"]) == set(
        tables["suppliers"]["supplier_id"]
    )


# --------------------------------------------------------------------------- #
# The no-leakage guarantee for forecast evaluation (SENSE module depends on it)
# --------------------------------------------------------------------------- #
def test_split_is_time_based_and_does_not_overlap(tables, assumptions):
    demand = tables["demand"]
    holdout = assumptions["sample"]["holdout_weeks"]
    last_train, first_test = split_weeks(demand, holdout)
    assert first_test == last_train + 1
    train, test = train_test(demand, holdout)
    assert train["week_index"].max() < test["week_index"].min()
    assert test["week_index"].nunique() == holdout
    assert len(train) + len(test) == len(demand)
    # No week, and no (item, week) pair, appears in both sides.
    assert not set(train["week_index"]) & set(test["week_index"])


def test_every_item_appears_in_both_split_sides(tables, assumptions):
    train, test = train_test(tables["demand"], assumptions["sample"]["holdout_weeks"])
    key = ["sku", "store"]
    assert set(map(tuple, train[key].drop_duplicates().to_numpy())) == set(
        map(tuple, test[key].drop_duplicates().to_numpy())
    )


def test_split_rejects_an_impossible_holdout(tables):
    with pytest.raises(ValueError, match="no training data"):
        split_weeks(tables["demand"], 10_000)


# --------------------------------------------------------------------------- #
# Deployment budget
# --------------------------------------------------------------------------- #
def test_committed_sample_fits_the_streamlit_cloud_budget():
    from supplyiq.paths import SAMPLE_DIR

    total = sum(p.stat().st_size for p in SAMPLE_DIR.glob("*.parquet"))
    assert total < 25e6, f"sample is {total / 1e6:.1f} MB; budget is 25 MB"
