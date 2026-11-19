"""Schema contract for the processed sample in data/sample/.

Both ingest paths — real Kaggle files and the synthetic fallback — must emit
exactly these tables with exactly these columns and dtypes. Tests assert it,
so a later phase can rely on the shape without re-checking.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd


@dataclass(frozen=True)
class TableSpec:
    name: str
    filename: str
    # column -> pandas dtype string
    columns: dict[str, str]
    primary_key: tuple[str, ...]
    description: str
    # Columns whose values come from a simulated assumption rather than data.
    simulated_columns: tuple[str, ...] = field(default=())

    @property
    def column_names(self) -> list[str]:
        return list(self.columns)


DEMAND = TableSpec(
    name="demand",
    filename="demand_weekly.parquet",
    columns={
        "sku": "string",
        "category": "string",
        "department": "string",
        "store": "string",
        "state": "string",
        "week_start": "datetime64[ns]",
        "week_index": "int32",
        "units_sold": "int32",
        "sell_price": "float32",
        "revenue": "float32",
    },
    primary_key=("sku", "store", "week_start"),
    description=(
        "Weekly unit sales per SKU per store, aggregated from M5 daily sales. "
        "week_index is 0-based and dense, so time-based splits are exact."
    ),
)

SKU_MASTER = TableSpec(
    name="sku_master",
    filename="sku_master.parquet",
    columns={
        "sku": "string",
        "category": "string",
        "department": "string",
        "store": "string",
        "state": "string",
        "supplier_id": "string",
        "unit_cost": "float32",
        "freight_per_unit": "float32",
        "tariff_rate": "float32",
        "landed_cost": "float32",
        "avg_sell_price": "float32",
    },
    primary_key=("sku", "store"),
    description=(
        "One row per planning item. Supplier assignment and the whole cost "
        "stack are simulated from config/assumptions.yaml."
    ),
    simulated_columns=(
        "supplier_id",
        "unit_cost",
        "freight_per_unit",
        "tariff_rate",
        "landed_cost",
    ),
)

SUPPLIERS = TableSpec(
    name="suppliers",
    filename="suppliers.parquet",
    columns={
        "supplier_id": "string",
        "name": "string",
        "country": "string",
        "mean_lead_time_days": "float32",
        "lead_time_sd_days": "float32",
        "otif_target": "float32",
        "unit_cost_index": "float32",
        "moq_units": "int32",
        "order_multiple_units": "int32",
        "freight_per_unit": "float32",
        "tariff_rate": "float32",
        "alternate_supplier_id": "string",
    },
    primary_key=("supplier_id",),
    description="Simulated supplier master. Lead-time spread is calibrated to DataCo shipping.",
    simulated_columns=(
        "mean_lead_time_days",
        "lead_time_sd_days",
        "otif_target",
        "unit_cost_index",
        "moq_units",
        "order_multiple_units",
        "freight_per_unit",
        "tariff_rate",
        "alternate_supplier_id",
    ),
)

PURCHASE_ORDERS = TableSpec(
    name="purchase_orders",
    filename="purchase_orders.parquet",
    columns={
        "po_id": "string",
        "supplier_id": "string",
        "order_date": "datetime64[ns]",
        "promised_date": "datetime64[ns]",
        "received_date": "datetime64[ns]",
        "shipping_mode": "string",
        "days_scheduled": "float32",
        "days_actual": "float32",
        "late_delivery": "int8",
        "in_full": "int8",
        "units_ordered": "int32",
        "units_received": "int32",
        "customer_region": "string",
    },
    primary_key=("po_id",),
    description=(
        "Historical inbound order lines. Shipping timings and late-delivery "
        "flags come from DataCo; the supplier_id link is simulated."
    ),
    simulated_columns=("supplier_id", "units_ordered", "units_received", "in_full"),
)

ALL_TABLES: tuple[TableSpec, ...] = (DEMAND, SKU_MASTER, SUPPLIERS, PURCHASE_ORDERS)
TABLES_BY_NAME: dict[str, TableSpec] = {t.name: t for t in ALL_TABLES}


class SchemaError(AssertionError):
    """Raised when a processed table does not match its TableSpec."""


def coerce(df: pd.DataFrame, spec: TableSpec) -> pd.DataFrame:
    """Reorder and cast `df` to `spec`. Raises SchemaError on missing columns."""
    missing = [c for c in spec.columns if c not in df.columns]
    if missing:
        raise SchemaError(f"{spec.name}: missing columns {missing}")
    out = df.loc[:, spec.column_names].copy()
    for col, dtype in spec.columns.items():
        if dtype.startswith("datetime"):
            out[col] = pd.to_datetime(out[col])
        else:
            out[col] = out[col].astype(dtype)
    return out.reset_index(drop=True)


def validate(df: pd.DataFrame, spec: TableSpec) -> None:
    """Assert `df` matches `spec`: columns, order, dtypes, key uniqueness, no nulls."""
    if list(df.columns) != spec.column_names:
        raise SchemaError(
            f"{spec.name}: column mismatch\n  got      {list(df.columns)}\n"
            f"  expected {spec.column_names}"
        )
    for col, dtype in spec.columns.items():
        actual = str(df[col].dtype)
        if actual != dtype:
            raise SchemaError(f"{spec.name}.{col}: dtype {actual!r}, expected {dtype!r}")
    if df.empty:
        raise SchemaError(f"{spec.name}: table is empty")
    dupes = int(df.duplicated(subset=list(spec.primary_key)).sum())
    if dupes:
        raise SchemaError(f"{spec.name}: {dupes} duplicate rows on key {spec.primary_key}")
    # received_date is legitimately null for open POs; everything else must be filled.
    nullable = {"received_date", "alternate_supplier_id"}
    for col in spec.columns:
        if col in nullable:
            continue
        n_null = int(df[col].isna().sum())
        if n_null:
            raise SchemaError(f"{spec.name}.{col}: {n_null} null values")


def describe() -> list[dict[str, Any]]:
    """Machine-readable schema summary, used by the docs and the UI data page."""
    return [
        {
            "table": t.name,
            "file": t.filename,
            "primary_key": list(t.primary_key),
            "n_columns": len(t.columns),
            "simulated_columns": list(t.simulated_columns),
            "description": t.description,
        }
        for t in ALL_TABLES
    ]
