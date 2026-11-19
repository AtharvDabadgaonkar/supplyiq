"""SENSE — supplier performance.

Turns the inbound PO log into the numbers procurement actually argues about:
lead time and its variability, OTIF, late-delivery rate, and fill (in-full)
rate, per supplier.

The lead-time *variability* here is the input the safety-stock calculation
needs. A supplier with a 7-day average and a 6-day spread is more expensive to
hold stock against than one with a 30-day average and a 2-day spread, and that
is the point the Supplier Performance page exists to make.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from supplyiq.config import load_assumptions

DAYS_PER_WEEK = 7.0


def po_flags(pos: pd.DataFrame) -> pd.DataFrame:
    """Add the derived service flags used by every metric below."""
    out = pos.copy()
    out["on_time"] = (out["late_delivery"] == 0).astype(int)
    out["in_full_flag"] = (out["in_full"] == 1).astype(int)
    out["otif_flag"] = (out["on_time"] & out["in_full_flag"]).astype(int)
    out["days_late"] = (
        (out["received_date"] - out["promised_date"]).dt.total_seconds() / 86400.0
    )
    out["fill_ratio"] = np.where(
        out["units_ordered"] > 0, out["units_received"] / out["units_ordered"], np.nan
    )
    return out


def supplier_performance(pos: pd.DataFrame, suppliers: pd.DataFrame | None = None) -> pd.DataFrame:
    """One row per supplier: service, lead time, and lead-time variability.

    `lead_time_sd_weeks` is what the safety-stock formula consumes, so it is
    measured from realised receipts, not taken from the assumption that
    generated them. On real data those two would differ; here the test suite
    checks they agree, which is how we know the pipeline is wired up.
    """
    df = po_flags(pos)
    agg = (
        df.groupby("supplier_id", observed=True)
        .agg(
            pos=("po_id", "size"),
            otif=("otif_flag", "mean"),
            on_time_rate=("on_time", "mean"),
            in_full_rate=("in_full_flag", "mean"),
            late_delivery_rate=("late_delivery", "mean"),
            mean_fill_ratio=("fill_ratio", "mean"),
            lead_time_mean_days=("days_actual", "mean"),
            lead_time_sd_days=("days_actual", "std"),
            lead_time_p95_days=("days_actual", lambda s: float(np.percentile(s, 95))),
            quoted_lead_time_days=("days_scheduled", "mean"),
            mean_days_late=("days_late", "mean"),
            worst_days_late=("days_late", "max"),
            units_ordered=("units_ordered", "sum"),
            units_received=("units_received", "sum"),
        )
        .reset_index()
    )
    agg["lead_time_sd_days"] = agg["lead_time_sd_days"].fillna(0.0)
    agg["lead_time_mean_weeks"] = agg["lead_time_mean_days"] / DAYS_PER_WEEK
    agg["lead_time_sd_weeks"] = agg["lead_time_sd_days"] / DAYS_PER_WEEK
    agg["lead_time_cv"] = np.where(
        agg["lead_time_mean_days"] > 0,
        agg["lead_time_sd_days"] / agg["lead_time_mean_days"],
        0.0,
    )

    if suppliers is not None:
        cols = ["supplier_id", "name", "country", "otif_target", "moq_units",
                "order_multiple_units", "alternate_supplier_id"]
        agg = agg.merge(suppliers[cols], on="supplier_id", how="left")
        agg["otif_vs_target_pp"] = (agg["otif"] - agg["otif_target"]) * 100
    return agg.sort_values("otif", ascending=False).reset_index(drop=True)


def overall_performance(pos: pd.DataFrame) -> dict[str, float]:
    """Network-level service, pooled across every PO."""
    df = po_flags(pos)
    return {
        "pos": int(len(df)),
        "otif": float(df["otif_flag"].mean()),
        "on_time_rate": float(df["on_time"].mean()),
        "in_full_rate": float(df["in_full_flag"].mean()),
        "late_delivery_rate": float(df["late_delivery"].mean()),
        "lead_time_mean_days": float(df["days_actual"].mean()),
        "lead_time_sd_days": float(df["days_actual"].std(ddof=1)),
        "units_ordered": int(df["units_ordered"].sum()),
        "units_received": int(df["units_received"].sum()),
        "unit_fill_rate": float(df["units_received"].sum() / df["units_ordered"].sum()),
    }


def item_lead_times(
    sku_master: pd.DataFrame, performance: pd.DataFrame, a: dict[str, Any] | None = None
) -> pd.DataFrame:
    """Attach each planning item's supplier lead-time profile, in weeks.

    This is the join that lets the inventory policy account for lead-time
    variability per SKU rather than assuming one number for the network.
    """
    a = a or load_assumptions()
    cols = [
        "supplier_id",
        "lead_time_mean_days",
        "lead_time_sd_days",
        "lead_time_mean_weeks",
        "lead_time_sd_weeks",
        "lead_time_cv",
        "otif",
        "on_time_rate",
        "late_delivery_rate",
    ]
    out = sku_master.merge(performance[cols], on="supplier_id", how="left", validate="many_to_one")
    missing = out["lead_time_mean_weeks"].isna()
    if missing.any():
        raise ValueError(
            f"{int(missing.sum())} planning items have a supplier with no PO history: "
            f"{sorted(out.loc[missing, 'supplier_id'].unique())}"
        )
    return out
