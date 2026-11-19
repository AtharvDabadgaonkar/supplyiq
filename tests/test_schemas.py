"""The committed sample must match its schema contract exactly."""
from __future__ import annotations

import pandas as pd
import pytest

from supplyiq.schemas import ALL_TABLES, SchemaError, TABLES_BY_NAME, coerce, validate


def test_every_table_validates(tables):
    for name, spec in TABLES_BY_NAME.items():
        validate(tables[name], spec)


def test_column_order_is_the_contract(tables):
    for name, spec in TABLES_BY_NAME.items():
        assert list(tables[name].columns) == spec.column_names, name


def test_primary_keys_are_unique(tables):
    for name, spec in TABLES_BY_NAME.items():
        assert not tables[name].duplicated(subset=list(spec.primary_key)).any(), name


def test_validate_catches_a_missing_column(tables):
    spec = TABLES_BY_NAME["demand"]
    broken = tables["demand"].drop(columns=["revenue"])
    with pytest.raises(SchemaError, match="column mismatch"):
        validate(broken, spec)


def test_validate_catches_a_wrong_dtype(tables):
    spec = TABLES_BY_NAME["demand"]
    broken = tables["demand"].copy()
    broken["units_sold"] = broken["units_sold"].astype("float64")
    with pytest.raises(SchemaError, match="dtype"):
        validate(broken, spec)


def test_validate_catches_duplicate_keys(tables):
    spec = TABLES_BY_NAME["sku_master"]
    broken = pd.concat([tables["sku_master"], tables["sku_master"].head(1)], ignore_index=True)
    with pytest.raises(SchemaError, match="duplicate rows"):
        validate(broken, spec)


def test_validate_catches_nulls(tables):
    spec = TABLES_BY_NAME["sku_master"]
    broken = tables["sku_master"].copy()
    broken.loc[0, "landed_cost"] = None
    with pytest.raises(SchemaError, match="null values"):
        validate(broken, spec)


def test_coerce_reorders_and_casts(tables):
    spec = TABLES_BY_NAME["suppliers"]
    shuffled = tables["suppliers"][list(reversed(spec.column_names))]
    fixed = coerce(shuffled, spec)
    validate(fixed, spec)


def test_simulated_columns_are_declared():
    """Anything derived from assumptions.yaml must be flagged for the UI label."""
    sku = TABLES_BY_NAME["sku_master"]
    for col in ("supplier_id", "unit_cost", "freight_per_unit", "tariff_rate", "landed_cost"):
        assert col in sku.simulated_columns
    sup = TABLES_BY_NAME["suppliers"]
    for col in ("mean_lead_time_days", "lead_time_sd_days", "otif_target"):
        assert col in sup.simulated_columns
    # Real M5 fields must not be mislabelled as simulated.
    demand = TABLES_BY_NAME["demand"]
    assert demand.simulated_columns == ()


def test_specs_are_internally_consistent():
    for spec in ALL_TABLES:
        assert set(spec.primary_key) <= set(spec.columns), spec.name
        assert set(spec.simulated_columns) <= set(spec.columns), spec.name
        assert spec.filename.endswith(".parquet"), spec.name
