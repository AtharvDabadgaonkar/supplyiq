#!/usr/bin/env python
"""Recompute every reported number into reports/metrics.json.

This file is the ONLY source for figures quoted in the README or the UI. If a
number is not in here, it does not get reported.

It also precomputes the planning artifacts into data/derived/ so the deployed
app loads Parquet instead of refitting models on the free Cloud tier.

Phases 1-3 are written. Each later phase adds its own block:
    phase 4 -> agent_recommendations, backtest
"""
from __future__ import annotations

import json
import platform
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import warnings  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from supplyiq import __version__, paths  # noqa: E402
from supplyiq.config import load_assumptions  # noqa: E402
from supplyiq.data import load_all, read_provenance, split_weeks, utc_now_iso  # noqa: E402
from supplyiq.inventory import (  # noqa: E402
    demand_sd_over_leadtime,
    policy_summary,
    segment_summary,
)
from supplyiq.pipeline import build as build_planning, write_artifacts  # noqa: E402


def data_block(tables: dict[str, pd.DataFrame], a: dict) -> dict:
    demand = tables["demand"]
    last_train, first_test = split_weeks(demand, int(a["sample"]["holdout_weeks"]))
    per_item = demand.groupby(["sku", "store"], observed=True)["units_sold"]
    mean_wk = per_item.mean()
    cv = (per_item.std() / mean_wk.replace(0, np.nan)).dropna()
    return {
        "planning_items": int(len(tables["sku_master"])),
        "distinct_skus": int(demand["sku"].nunique()),
        "stores": int(demand["store"].nunique()),
        "suppliers": int(len(tables["suppliers"])),
        "weeks_of_history": int(demand["week_index"].max()) + 1,
        "week_start_first": str(demand["week_start"].min().date()),
        "week_start_last": str(demand["week_start"].max().date()),
        "train_weeks": [0, int(last_train)],
        "holdout_weeks": [int(first_test), int(demand["week_index"].max())],
        "demand_rows": int(len(demand)),
        "purchase_order_rows": int(len(tables["purchase_orders"])),
        "total_units_sold": int(demand["units_sold"].sum()),
        "total_revenue": round(float(demand["revenue"].sum()), 2),
        "zero_demand_week_share": round(float((demand["units_sold"] == 0).mean()), 4),
        "median_weekly_units_per_item": round(float(mean_wk.median()), 2),
        "demand_cv_median": round(float(cv.median()), 4),
        "demand_cv_p90": round(float(cv.quantile(0.90)), 4),
    }


def supplier_block(tables: dict[str, pd.DataFrame]) -> dict:
    po = tables["purchase_orders"].copy()
    po["on_time"] = (po["late_delivery"] == 0).astype(int)
    po["otif"] = ((po["late_delivery"] == 0) & (po["in_full"] == 1)).astype(int)
    by_sup = (
        po.groupby("supplier_id")
        .agg(
            pos=("po_id", "size"),
            on_time_rate=("on_time", "mean"),
            in_full_rate=("in_full", "mean"),
            otif=("otif", "mean"),
            late_delivery_rate=("late_delivery", "mean"),
            lead_time_mean_days=("days_actual", "mean"),
            lead_time_sd_days=("days_actual", "std"),
            quoted_lead_time_days=("days_scheduled", "mean"),
        )
        .round(4)
    )
    sup = tables["suppliers"].set_index("supplier_id")
    by_sup["country"] = sup["country"]
    by_sup["otif_target"] = sup["otif_target"].round(4)
    by_sup["otif_vs_target_pp"] = ((by_sup["otif"] - by_sup["otif_target"]) * 100).round(2)
    return {
        "overall_otif": round(float(po["otif"].mean()), 4),
        "overall_on_time_rate": round(float(po["on_time"].mean()), 4),
        "overall_late_delivery_rate": round(float(po["late_delivery"].mean()), 4),
        "overall_in_full_rate": round(float(po["in_full"].mean()), 4),
        "lead_time_mean_days": round(float(po["days_actual"].mean()), 2),
        "max_abs_otif_error_vs_target_pp": round(
            float(by_sup["otif_vs_target_pp"].abs().max()), 2
        ),
        "by_supplier": json.loads(by_sup.reset_index().to_json(orient="records")),
    }


def cost_block(tables: dict[str, pd.DataFrame], a: dict) -> dict:
    sm = tables["sku_master"]
    return {
        "simulated": True,
        "formula_unit_cost": "avg_sell_price * (1 - gross_margin_target) * supplier.unit_cost_index",
        "formula_landed_cost": "unit_cost + freight_per_unit + unit_cost * tariff_rate",
        "gross_margin_target": float(a["costs"]["gross_margin_target"]),
        "mean_unit_cost": round(float(sm["unit_cost"].mean()), 4),
        "mean_landed_cost": round(float(sm["landed_cost"].mean()), 4),
        "mean_tariff_rate": round(float(sm["tariff_rate"].mean()), 4),
        "landed_vs_unit_cost_uplift_pct": round(
            float((sm["landed_cost"].sum() / sm["unit_cost"].sum() - 1) * 100), 2
        ),
        "items_exposed_to_tariff": int((sm["tariff_rate"] > 0).sum()),
        "tariff_exposed_share": round(float((sm["tariff_rate"] > 0).mean()), 4),
    }


def forecast_block(planning) -> dict:
    ev = planning.evaluation
    per_item = ev.per_item
    models = {}
    for model, acc in ev.overall.items():
        rows = per_item[per_item["model"] == model]
        models[model] = {
            **{k: (round(v, 6) if isinstance(v, float) else v) for k, v in acc.as_dict().items()},
            "median_item_wape": round(float(rows["wape"].median()), 6),
            "items_scored": int(len(rows)),
        }
    baseline = ev.overall[ev.baseline_model]
    candidate = ev.overall[ev.candidate_model]
    return {
        "method": (
            "Time-based holdout: fit on the training weeks, score on the final weeks, "
            "which the models never see. Pooled (volume-weighted), not a mean of per-item ratios."
        ),
        "baseline_model": ev.baseline_model,
        "candidate_model": ev.candidate_model,
        "winner": ev.winner,
        "train_weeks": list(ev.train_weeks),
        "holdout_weeks": list(ev.holdout_weeks),
        "wape_improvement_pp": round(ev.improvement_pp(), 4),
        "wape_improvement_pct_of_baseline": round(
            float((baseline.wape - candidate.wape) / baseline.wape * 100), 2
        ),
        "mape_note": (
            f"{candidate.n_zero_weeks} of {candidate.n_weeks} holdout weeks had zero sales and are "
            "excluded from MAPE; WAPE keeps every unit of demand in the denominator."
        ),
        "models": models,
    }


def segmentation_block(planning, a: dict) -> dict:
    policy = planning.policy
    grid = segment_summary(policy)
    by_abc = (
        policy.groupby("abc_class")
        .agg(
            items=("sku", "size"),
            revenue=("revenue", "sum"),
            inventory_value=("average_inventory_value", "sum"),
            median_demand_cv=("demand_cv", "median"),
            service_level=("service_level", "first"),
        )
        .reset_index()
    )
    by_abc["revenue_share"] = (by_abc["revenue"] / by_abc["revenue"].sum()).round(4)
    by_abc["inventory_value_share"] = (
        by_abc["inventory_value"] / by_abc["inventory_value"].sum()
    ).round(4)
    return {
        "method": (
            "ABC on cumulative revenue share (Pareto); XYZ on the coefficient of variation "
            "of weekly demand. Both computed on training weeks only."
        ),
        "abc_cutoffs": a["policy"]["abc_revenue_cutoffs"],
        "xyz_cutoffs": a["policy"]["xyz_cv_cutoffs"],
        "by_abc": json.loads(by_abc.round(4).to_json(orient="records")),
        "xyz_counts": {k: int(v) for k, v in policy["xyz_class"].value_counts().items()},
        "abc_xyz_grid": json.loads(grid.round(4).to_json(orient="records")),
    }


def inventory_block(planning, a: dict) -> dict:
    policy = planning.policy
    summary = policy_summary(policy)

    # What the lead-time term contributes: rebuild safety stock with sd_L = 0.
    demand_only = np.array(
        [
            demand_sd_over_leadtime(d, sd, lt, 0.0)
            for d, sd, lt in zip(
                policy["mean_weekly_demand"], policy["sd_weekly_demand"], policy["protection_weeks"]
            )
        ]
    )
    ss_demand_only = float((policy["z"].to_numpy() * demand_only).sum())
    ss_full = float(policy["safety_stock"].sum())

    return {
        "method": {
            "safety_stock": "z * sqrt(L * sd_demand^2 + mean_demand^2 * sd_leadtime^2)",
            "reorder_point": "mean_demand * (lead_time + review_period) + safety_stock",
            "eoq": "sqrt(2 * annual_demand * ordering_cost / (landed_cost * holding_rate))",
            "order_quantity": "EOQ, raised to the supplier MOQ and order multiple, then capped at max_weeks_of_supply",
            "fill_rate": "1 - sd_over_leadtime * G(z) / order_quantity, where G is the normal loss function",
        },
        "service_level_by_abc": a["policy"]["service_level_by_abc"],
        "review_period_weeks": a["policy"]["review_period_weeks"],
        "max_weeks_of_supply": a["policy"]["max_weeks_of_supply"],
        **{k: (round(v, 4) if isinstance(v, float) else v) for k, v in summary.items()},
        "safety_stock_units_demand_risk_only": round(ss_demand_only, 2),
        "safety_stock_units_with_leadtime_risk": round(ss_full, 2),
        "leadtime_risk_uplift_pct": round(float((ss_full / ss_demand_only - 1) * 100), 2),
        "items_capped_by_coverage": int(policy["eoq_capped_by_coverage"].sum()),
        "items_ordering_at_moq": int(policy["order_at_moq"].sum()),
        "median_lead_time_weeks": round(float(policy["lead_time_mean_weeks"].median()), 3),
    }


def scenario_block(planning, a: dict) -> dict:
    table = planning.scenario_table()
    base = planning.baseline.kpis
    rows = []
    for row in table.to_dict(orient="records"):
        rows.append(
            {
                "scenario": row["scenario"],
                "description": row["description"],
                "supplier_delay_days": row["supplier_delay_days"],
                "demand_spike_pct": row["demand_spike_pct"],
                "tariff_change_pct": row["tariff_change_pct"],
                "scope": row["country_scope"] or row["supplier_scope"] or "all",
                "fill_rate": round(row["fill_rate"], 5),
                "stockout_weeks": int(row["stockout_weeks"]),
                "items_with_a_stockout": int(row["items_with_a_stockout"]),
                "otif": round(row["otif"], 5),
                "working_capital": round(row["working_capital"], 2),
                "landed_cost_per_unit": round(row["landed_cost_per_unit"], 4),
                "cost_to_serve": round(row["cost_to_serve"], 2),
                "fill_rate_vs_baseline_pp": round(row["delta_fill_rate"] * 100, 3),
                "cost_to_serve_vs_baseline": round(row["delta_cost_to_serve"], 2),
                "working_capital_vs_baseline": round(row["delta_working_capital"], 2),
            }
        )
    worst = min(rows, key=lambda r: r["fill_rate"])
    return {
        "method": (
            "Weekly inventory twin over the holdout weeks, lost sales, periodic (s,S) "
            "review. Baseline and scenario runs share the same pre-drawn lead times "
            "(common random numbers), so a difference between them is the scenario and "
            "not simulation noise. Zero disruption reproduces the baseline exactly."
        ),
        "cost_to_serve_definition": (
            "holding + stockout + ordering + action costs. The price of the goods is "
            "excluded: over a finite window a supplier delay pushes reorders past the "
            "end of it, so including goods made a disruption look cheaper than no "
            "disruption."
        ),
        "tariff_units": "percentage points added to the ad-valorem rate, floored at zero",
        "weeks_simulated": int(planning.baseline.kpis["weeks"]),
        "baseline": {
            "fill_rate": round(base["fill_rate"], 5),
            "stockout_weeks": int(base["stockout_weeks"]),
            "otif": round(base["otif"], 5),
            "working_capital": round(base["working_capital"], 2),
            "cost_to_serve": round(base["cost_to_serve"], 2),
        },
        "worst_case": {
            "scenario": worst["scenario"],
            "fill_rate": worst["fill_rate"],
            "fill_rate_vs_baseline_pp": worst["fill_rate_vs_baseline_pp"],
            "stockout_weeks": worst["stockout_weeks"],
        },
        "scenarios": rows,
    }


def main() -> int:
    a = load_assumptions()
    tables = load_all(check=True)
    prov = read_provenance()

    print("building planning artifacts (fitting forecast models) ...")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        planning = build_planning(tables, a, with_evaluation=True, with_scenarios=True)
    written = write_artifacts(planning)
    for name, path in written.items():
        print(f"  {name:22s} {path.stat().st_size / 1e6:6.2f} MB  {path.name}")

    metrics = {
        "_about": (
            "Generated by `make metrics`. Every number reported in the README or the UI "
            "comes from this file. Do not edit by hand."
        ),
        "generated_at": utc_now_iso(),
        "supplyiq_version": __version__,
        "python_version": platform.python_version(),
        "phases_complete": [1, 2, 3],
        "data_provenance": {
            "origin": prov.origin,
            "is_real_data": prov.is_real,
            "label": prov.badge,
            "demand_source": prov.demand_source,
            "orders_source": prov.orders_source,
            "simulated_sections": prov.simulated_sections,
            "built_at": prov.built_at,
            "notes": prov.notes,
        },
        "sample_scope": prov.sample_scope,
        "data": data_block(tables, a),
        "supplier_performance": supplier_block(tables),
        "cost_model": cost_block(tables, a),
        "forecast_accuracy": forecast_block(planning),
        "segmentation": segmentation_block(planning, a),
        "inventory_policy": inventory_block(planning, a),
        "supplier_service": planning.overall,
        "scenarios": scenario_block(planning, a),
        "agent": {
            "provider_configured": str(a["agent"]["provider"]),
            "llm_may_produce_numbers": bool(a["agent"]["llm_may_produce_numbers"]),
            "note": (
                "Explanations are prose only. Any number in generated text is checked "
                "against the simulator's output by supplyiq.agent.guard; unverifiable "
                "text is discarded in favour of the rule-based explanation."
            ),
        },
        "pending_phases": {
            "4": ["agent_recommendations", "backtest"],
        },
    }

    paths.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    paths.METRICS_FILE.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")

    print(f"wrote {paths.METRICS_FILE.relative_to(paths.REPO_ROOT)}")
    print(f"  origin               {prov.origin}")
    print(f"  planning items       {metrics['data']['planning_items']:,}")
    print(f"  weeks of history     {metrics['data']['weeks_of_history']}")
    print(f"  overall OTIF         {metrics['supplier_performance']['overall_otif']:.1%}")
    print(f"  late delivery rate   {metrics['supplier_performance']['overall_late_delivery_rate']:.1%}")
    print(f"  mean landed cost     ${metrics['cost_model']['mean_landed_cost']:.2f}")
    print(f"  tariff-exposed items {metrics['cost_model']['tariff_exposed_share']:.1%}")
    fc = metrics["forecast_accuracy"]
    inv = metrics["inventory_policy"]
    print(f"  forecast WAPE        {fc['models'][fc['baseline_model']]['wape']:.3f} "
          f"({fc['baseline_model']}) -> {fc['models'][fc['candidate_model']]['wape']:.3f} "
          f"({fc['candidate_model']}), {fc['wape_improvement_pp']:+.1f} pp")
    print(f"  inventory turns      {inv['portfolio_inventory_turns']:.1f}")
    print(f"  weighted fill rate   {inv['weighted_fill_rate']:.3%}")
    print(f"  lead-time SS uplift  {inv['leadtime_risk_uplift_pct']:.1f}%")
    sc = metrics["scenarios"]
    print(f"  baseline fill rate   {sc['baseline']['fill_rate']:.3%} "
          f"({sc['baseline']['stockout_weeks']} stockout weeks)")
    print(f"  worst scenario       {sc['worst_case']['scenario']} "
          f"({sc['worst_case']['fill_rate_vs_baseline_pp']:+.2f} pp fill rate)")

    # Keep the README's metrics block in sync, so no figure is ever hand-typed.
    from scripts.render_readme import main as render_readme  # noqa: PLC0415

    render_readme()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
