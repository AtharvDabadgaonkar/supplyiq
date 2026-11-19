"""Build the processed sample in data/sample/.

Two paths, one output schema:

  kaggle     — reads raw M5 + DataCo CSVs from data/raw/ and processes them.
  synthetic  — a seeded generator that emits the same tables so the repo runs,
               and the app deploys, with no Kaggle download.

Which path ran is recorded in data/sample/provenance.json and surfaced as a
badge in the UI. Supplier, cost and tariff fields are simulated from
config/assumptions.yaml on BOTH paths — see `simulated_columns` in schemas.py.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd

from supplyiq import paths
from supplyiq.config import load_assumptions, simulated_sections, supplier_frame
from supplyiq.data import Origin, Provenance, utc_now_iso, write_table
from supplyiq.schemas import ALL_TABLES, coerce

Prefer = Literal["auto", "kaggle", "synthetic"]

# Raw filenames we look for, in data/raw/ (recursively).
M5_FILES = {
    "sales": ("sales_train_evaluation.csv", "sales_train_validation.csv"),
    "calendar": ("calendar.csv",),
    "prices": ("sell_prices.csv",),
}
DATACO_FILES = ("DataCoSupplyChainDataset.csv", "dataco.csv")


# --------------------------------------------------------------------------- #
# Raw file discovery + tolerant column matching
# --------------------------------------------------------------------------- #
def _norm(name: str) -> str:
    """Normalise a column name so 'Days for shipping (real)' == 'days_for_shipping_real'."""
    return re.sub(r"[^a-z0-9]+", "_", str(name).strip().lower()).strip("_")


def find_raw_file(candidates: tuple[str, ...], raw_dir: Path) -> Path | None:
    for name in candidates:
        direct = raw_dir / name
        if direct.exists():
            return direct
        hits = sorted(raw_dir.rglob(name))
        if hits:
            return hits[0]
    return None


def kaggle_files_present(raw_dir: Path | None = None) -> dict[str, Path | None]:
    raw_dir = raw_dir or paths.RAW_DIR
    found = {k: find_raw_file(v, raw_dir) for k, v in M5_FILES.items()}
    found["dataco"] = find_raw_file(DATACO_FILES, raw_dir)
    return found


def have_kaggle(raw_dir: Path | None = None) -> bool:
    return all(v is not None for v in kaggle_files_present(raw_dir).values())


def resolve_column(df: pd.DataFrame, candidates: list[str], table: str) -> str:
    """Find a column by normalised name. Raises with the real header on failure."""
    lookup = {_norm(c): c for c in df.columns}
    for cand in candidates:
        key = _norm(cand)
        if key in lookup:
            return lookup[key]
    # fall back to a substring match on the normalised names
    for cand in candidates:
        key = _norm(cand)
        for norm_name, original in lookup.items():
            if key in norm_name:
                return original
    raise KeyError(
        f"{table}: could not find any of {candidates}. "
        f"Columns present: {sorted(df.columns)[:40]}"
    )


def _z(p: float) -> float:
    """Standard normal quantile. Kept local so ingest needs no scipy import."""
    from statistics import NormalDist

    return float(NormalDist().inv_cdf(p))


# --------------------------------------------------------------------------- #
# Shared: supplier master, supplier assignment, cost stack
# --------------------------------------------------------------------------- #
def build_suppliers(a: dict[str, Any]) -> pd.DataFrame:
    df = supplier_frame()
    return coerce(df, _spec("suppliers"))


def _spec(name: str):
    return {t.name: t for t in ALL_TABLES}[name]


def assign_suppliers(sku_master: pd.DataFrame, a: dict[str, Any], rng: np.random.Generator) -> pd.Series:
    """Allocate suppliers per category share, at the ITEM level.

    A SKU is sourced from one vendor across the whole network, so every store
    carrying that SKU shares its supplier and therefore its landed cost. The
    allocation is quota-based on distinct SKUs and then shuffled with the seeded
    rng, so the realised mix matches the configured shares.
    """
    alloc = a["suppliers"]["sku_allocation"]
    item_cat = (
        sku_master[["sku", "category"]]
        .drop_duplicates()
        .sort_values("sku")
        .reset_index(drop=True)
    )
    assigned: dict[str, str] = {}
    for category, shares in alloc.items():
        idx = item_cat.index[item_cat["category"] == category]
        if len(idx) == 0:
            continue
        supplier_ids = list(shares)
        probs = np.array([shares[s] for s in supplier_ids], dtype=float)
        probs = probs / probs.sum()
        # Quota-based so the realised mix matches the configured shares closely,
        # then shuffled with the seeded rng so the assignment is not ordered.
        counts = np.floor(probs * len(idx)).astype(int)
        while counts.sum() < len(idx):
            counts[int(np.argmax(probs * len(idx) - counts))] += 1
        picks = np.concatenate([np.repeat(s, c) for s, c in zip(supplier_ids, counts)])
        rng.shuffle(picks)
        for sku, pick in zip(item_cat.loc[idx, "sku"], picks):
            assigned[sku] = str(pick)

    out = sku_master["sku"].map(assigned)
    if out.isna().any():
        unknown = sorted(sku_master.loc[out.isna(), "category"].unique())
        raise ValueError(f"no supplier allocation configured for categories: {unknown}")
    return out.astype("string")


def attach_costs(sku_master: pd.DataFrame, a: dict[str, Any]) -> pd.DataFrame:
    """Build the simulated cost stack.

        unit_cost   = avg_sell_price * (1 - gross_margin) * supplier.unit_cost_index
        landed_cost = unit_cost + freight_per_unit + unit_cost * tariff_rate
    """
    sup = supplier_frame().set_index("supplier_id")
    margin = float(a["costs"]["gross_margin_target"])
    df = sku_master.copy()
    df["unit_cost"] = (
        df["avg_sell_price"].astype(float)
        * (1.0 - margin)
        * df["supplier_id"].map(sup["unit_cost_index"]).astype(float)
    )
    df["freight_per_unit"] = df["supplier_id"].map(sup["freight_per_unit"]).astype(float)
    df["tariff_rate"] = df["supplier_id"].map(sup["tariff_rate"]).astype(float)
    df["landed_cost"] = (
        df["unit_cost"] + df["freight_per_unit"] + df["unit_cost"] * df["tariff_rate"]
    )
    return df


def simulate_purchase_orders(
    sku_master: pd.DataFrame,
    demand: pd.DataFrame,
    a: dict[str, Any],
    rng: np.random.Generator,
    dataco: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Build the historical inbound PO table.

    Inbound lead times are simulated (lognormal per supplier, from
    assumptions.yaml) because neither dataset contains purchase lead times.
    When DataCo is available its REAL shipping-mode mix, order regions and
    late-delivery rate are sampled into the table, so the realised OTIF and
    late-delivery rates per supplier are anchored to observed data rather than
    invented. Everything else on this table is simulated.
    """
    sup = supplier_frame().set_index("supplier_id")
    weeks = demand[["week_index", "week_start"]].drop_duplicates().sort_values("week_index")
    week_map = dict(zip(weeks["week_index"], weeks["week_start"]))
    n_weeks = len(week_map)

    if dataco is not None and len(dataco):
        modes = dataco["shipping_mode"].dropna().astype(str).to_numpy()
        regions = dataco["customer_region"].dropna().astype(str).to_numpy()
        observed_late_rate = float(dataco["late_delivery"].mean())
    else:
        modes = np.array(["Standard Class", "Second Class", "First Class", "Same Day"])
        regions = np.array(["West of USA", "US Center", "South of USA", "East of USA"])
        observed_late_rate = float("nan")

    # Average weekly demand per planning item drives order sizes.
    avg_wk = (
        demand.groupby(["sku", "store"], observed=True)["units_sold"].mean().rename("avg_weekly")
    )
    items = sku_master.merge(avg_wk, on=["sku", "store"], how="left")
    items["avg_weekly"] = items["avg_weekly"].fillna(0.0)

    rows: list[dict[str, Any]] = []
    po_seq = 0
    for item in items.itertuples(index=False):
        s = sup.loc[item.supplier_id]
        lt_mean = float(s.mean_lead_time_days)
        lt_sd = float(s.lead_time_sd_days)
        # Order roughly every lead time's worth of weeks, at least every 8 weeks.
        cadence = max(2, min(8, int(round(lt_mean / 7.0))))
        order_weeks = range(0, max(1, n_weeks - 2), cadence)
        # Lognormal parameters matched to the configured mean and sd.
        sigma = float(np.sqrt(np.log1p((lt_sd / lt_mean) ** 2)))
        mu = float(np.log(lt_mean) - 0.5 * sigma**2)
        # OTIF = P(on time) * P(in full). We split the supplier's otif_target
        # evenly between the two, so the realised OTIF in this table lands on
        # the configured target instead of drifting below it. The QUOTED lead
        # time is then the on-time quantile of the lead-time distribution --
        # i.e. the buffer a supplier actually commits to, not its mean.
        p_component = float(np.sqrt(float(s.otif_target)))
        quoted_days = float(np.exp(mu + sigma * _z(p_component)))
        for w in order_weeks:
            po_seq += 1
            order_date = pd.Timestamp(week_map[w])
            days_scheduled = quoted_days
            days_actual = float(rng.lognormal(mu, sigma))
            promised = order_date + pd.Timedelta(days=round(days_scheduled))
            received = order_date + pd.Timedelta(days=round(days_actual))
            units_ordered = int(
                max(
                    int(s.moq_units),
                    round(item.avg_weekly * cadence * float(rng.uniform(0.9, 1.3))),
                )
            )
            # Short shipments: 1 - otif_target of POs arrive incomplete.
            in_full = int(rng.random() < p_component)
            units_received = (
                units_ordered
                if in_full
                else int(round(units_ordered * float(rng.uniform(0.6, 0.97))))
            )
            rows.append(
                {
                    "po_id": f"PO-{po_seq:06d}",
                    "supplier_id": item.supplier_id,
                    "order_date": order_date,
                    "promised_date": promised,
                    "received_date": received,
                    "shipping_mode": str(rng.choice(modes)),
                    "days_scheduled": days_scheduled,
                    "days_actual": days_actual,
                    "late_delivery": int(received > promised),
                    "in_full": in_full,
                    "units_ordered": units_ordered,
                    "units_received": units_received,
                    "customer_region": str(rng.choice(regions)),
                }
            )
    po = pd.DataFrame(rows)
    po.attrs["observed_late_rate"] = observed_late_rate
    return coerce(po, _spec("purchase_orders"))


# --------------------------------------------------------------------------- #
# Path 1: real Kaggle data
# --------------------------------------------------------------------------- #
def build_from_kaggle(raw_dir: Path | None = None, a: dict[str, Any] | None = None):
    a = a or load_assumptions()
    raw_dir = raw_dir or paths.RAW_DIR
    found = kaggle_files_present(raw_dir)
    missing = [k for k, v in found.items() if v is None]
    if missing:
        raise FileNotFoundError(
            f"missing raw files for {missing} under {raw_dir}. See README > Data."
        )

    s = a["sample"]
    rng = np.random.default_rng(int(s["random_seed"]))

    # ---- M5 calendar: d_x -> date, wm_yr_wk -------------------------------
    cal = pd.read_csv(found["calendar"], usecols=lambda c: _norm(c) in
                      {"date", "wm_yr_wk", "d"})
    c_date = resolve_column(cal, ["date"], "calendar")
    c_wk = resolve_column(cal, ["wm_yr_wk"], "calendar")
    c_d = resolve_column(cal, ["d"], "calendar")
    cal = cal.rename(columns={c_date: "date", c_wk: "wm_yr_wk", c_d: "d"})
    cal["date"] = pd.to_datetime(cal["date"])

    # ---- M5 sales: filter state + categories, pick n_items ----------------
    header = pd.read_csv(found["sales"], nrows=0)
    id_cols = {
        "item_id": resolve_column(header, ["item_id"], "sales"),
        "dept_id": resolve_column(header, ["dept_id"], "sales"),
        "cat_id": resolve_column(header, ["cat_id"], "sales"),
        "store_id": resolve_column(header, ["store_id"], "sales"),
        "state_id": resolve_column(header, ["state_id"], "sales"),
    }
    d_cols = [c for c in header.columns if re.fullmatch(r"d_\d+", _norm(c))]
    if not d_cols:
        raise KeyError("sales file has no d_* day columns")

    sales = pd.read_csv(found["sales"], usecols=list(id_cols.values()) + d_cols)
    sales = sales.rename(columns={v: k for k, v in id_cols.items()})
    sales = sales[
        (sales["state_id"] == s["state"]) & (sales["cat_id"].isin(s["categories"]))
    ]
    if sales.empty:
        raise ValueError(
            f"no M5 rows for state={s['state']} categories={s['categories']}"
        )

    # Keep the n_items best-selling items so the sample is not all near-zero
    # series, and n_stores stores, then take the full cross product.
    stores = sorted(sales["store_id"].unique())[: int(s["n_stores"])]
    sales = sales[sales["store_id"].isin(stores)]
    totals = sales.groupby("item_id", observed=True)[d_cols].sum().sum(axis=1)
    keep_items = list(totals.sort_values(ascending=False).head(int(s["n_items"])).index)
    sales = sales[sales["item_id"].isin(keep_items)]

    # ---- daily -> weekly ---------------------------------------------------
    long = sales.melt(
        id_vars=["item_id", "dept_id", "cat_id", "store_id", "state_id"],
        value_vars=d_cols,
        var_name="d",
        value_name="units",
    )
    long = long.merge(cal[["d", "date", "wm_yr_wk"]], on="d", how="inner")
    weekly = (
        long.groupby(
            ["item_id", "store_id", "cat_id", "dept_id", "state_id", "wm_yr_wk"],
            observed=True,
        )
        .agg(units_sold=("units", "sum"), week_start=("date", "min"))
        .reset_index()
    )

    # Keep the last n_weeks complete weeks.
    last_weeks = sorted(weekly["wm_yr_wk"].unique())[-int(s["n_weeks"]) :]
    weekly = weekly[weekly["wm_yr_wk"].isin(last_weeks)]

    # ---- real sell prices --------------------------------------------------
    prices = pd.read_csv(found["prices"])
    p_item = resolve_column(prices, ["item_id"], "sell_prices")
    p_store = resolve_column(prices, ["store_id"], "sell_prices")
    p_wk = resolve_column(prices, ["wm_yr_wk"], "sell_prices")
    p_price = resolve_column(prices, ["sell_price"], "sell_prices")
    prices = prices.rename(
        columns={p_item: "item_id", p_store: "store_id", p_wk: "wm_yr_wk", p_price: "sell_price"}
    )[["item_id", "store_id", "wm_yr_wk", "sell_price"]]
    weekly = weekly.merge(prices, on=["item_id", "store_id", "wm_yr_wk"], how="left")
    # Items not on sale in a week have no price in M5; carry the item median.
    med = weekly.groupby("item_id", observed=True)["sell_price"].transform("median")
    weekly["sell_price"] = weekly["sell_price"].fillna(med)
    weekly = weekly.dropna(subset=["sell_price"])

    demand = _finalise_demand(
        weekly.rename(
            columns={
                "item_id": "sku",
                "store_id": "store",
                "cat_id": "category",
                "dept_id": "department",
                "state_id": "state",
            }
        )
    )

    # ---- DataCo: real shipping mix + late-delivery rate --------------------
    dataco = _read_dataco(found["dataco"])

    sku_master = _sku_master_from_demand(demand, a, rng)
    po = simulate_purchase_orders(sku_master, demand, a, rng, dataco=dataco)
    tables = {
        "demand": demand,
        "sku_master": sku_master,
        "suppliers": build_suppliers(a),
        "purchase_orders": po,
    }
    prov = _provenance(
        "kaggle",
        tables,
        a,
        notes=(
            "Demand, sell prices and the retail calendar are real M5 data. "
            "Shipping-mode mix, order regions and the late-delivery rate are real "
            f"DataCo data (observed late rate {po.attrs.get('observed_late_rate'):.3f}). "
            "Supplier master, inbound lead times, unit cost, freight and tariffs are "
            "simulated from config/assumptions.yaml."
        ),
    )
    return tables, prov


def _read_dataco(path: Path) -> pd.DataFrame:
    """Load just the DataCo columns we use, with tolerant header matching."""
    header = pd.read_csv(path, nrows=0, encoding_errors="replace")
    wanted = {
        "shipping_mode": ["Shipping Mode", "shipping_mode"],
        "customer_region": ["Order Region", "Customer Region", "Market", "order_region"],
        "days_actual": ["Days for shipping (real)", "days_for_shipping_real"],
        "days_scheduled": ["Days for shipment (scheduled)", "days_for_shipment_scheduled"],
        "late_delivery": ["Late_delivery_risk", "late_delivery_risk"],
    }
    resolved = {k: resolve_column(header, v, "dataco") for k, v in wanted.items()}
    df = pd.read_csv(path, usecols=list(resolved.values()), encoding_errors="replace")
    df = df.rename(columns={v: k for k, v in resolved.items()})
    df["late_delivery"] = pd.to_numeric(df["late_delivery"], errors="coerce").fillna(0).astype(int)
    return df


# --------------------------------------------------------------------------- #
# Path 2: synthetic fallback
# --------------------------------------------------------------------------- #
def build_synthetic(a: dict[str, Any] | None = None):
    """Seeded generator producing M5-shaped weekly demand and a DataCo-shaped PO log.

    Deliberately includes slow, intermittent items (many zero weeks) so that the
    MAPE guard, XYZ segmentation and safety-stock maths are exercised on the kind
    of series that actually break them.
    """
    a = a or load_assumptions()
    s = a["sample"]
    rng = np.random.default_rng(int(s["random_seed"]))

    n_items, n_stores, n_weeks = int(s["n_items"]), int(s["n_stores"]), int(s["n_weeks"])
    categories = list(s["categories"])
    state = str(s["state"])

    departments = {
        "FOODS": ["FOODS_1", "FOODS_2", "FOODS_3"],
        "HOUSEHOLD": ["HOUSEHOLD_1", "HOUSEHOLD_2"],
        "HOBBIES": ["HOBBIES_1", "HOBBIES_2"],
    }
    # Category mix roughly mirrors M5 (FOODS dominates the item count).
    cat_weights = {"FOODS": 0.46, "HOUSEHOLD": 0.35, "HOBBIES": 0.19}
    probs = np.array([cat_weights.get(c, 1.0 / len(categories)) for c in categories])
    probs /= probs.sum()
    item_cats = rng.choice(categories, size=n_items, p=probs)

    stores = [f"{state}_{i + 1}" for i in range(n_stores)]
    store_factor = rng.uniform(0.65, 1.35, size=n_stores)

    last_week = pd.Timestamp(str(s["synthetic_last_week_start"]))
    week_starts = pd.date_range(end=last_week, periods=n_weeks, freq="7D")
    t = np.arange(n_weeks)

    # Per-item latent parameters.
    base_rate = rng.lognormal(mean=1.5, sigma=1.15, size=n_items)   # units/week, long tail
    seasonal_amp = rng.uniform(0.05, 0.45, size=n_items)
    seasonal_phase = rng.uniform(0, 2 * np.pi, size=n_items)
    trend = rng.normal(0.0, 0.0022, size=n_items)                   # weekly drift
    intermittency = np.clip(rng.beta(1.3, 7.0, size=n_items), 0.0, 0.55)
    price_mu = {"FOODS": 3.2, "HOUSEHOLD": 6.5, "HOBBIES": 9.0}
    base_price = np.array(
        [rng.lognormal(np.log(price_mu[c]), 0.42) for c in item_cats]
    )
    elasticity = rng.uniform(1.2, 2.4, size=n_items)

    frames = []
    for si, store in enumerate(stores):
        for i in range(n_items):
            cat = item_cats[i]
            dept = departments[cat][i % len(departments[cat])]
            season = 1.0 + seasonal_amp[i] * np.sin(
                2 * np.pi * t / a["forecast"]["seasonal_period_weeks"] + seasonal_phase[i]
            )
            drift = np.exp(trend[i] * t)
            lam = base_rate[i] * store_factor[si] * season * drift

            # Promotions: ~6% of weeks at 15-30% off, with an elastic demand lift.
            promo = rng.random(n_weeks) < 0.06
            discount = np.where(promo, rng.uniform(0.15, 0.30, size=n_weeks), 0.0)
            price = base_price[i] * (1.0 - discount)
            lam = lam * (1.0 + elasticity[i] * discount)

            units = rng.poisson(np.maximum(lam, 1e-6))
            # Intermittency: independent chance of a no-sale week.
            units = np.where(rng.random(n_weeks) < intermittency[i], 0, units)

            frames.append(
                pd.DataFrame(
                    {
                        "sku": f"{dept}_{i + 1:03d}",
                        "category": cat,
                        "department": dept,
                        "store": store,
                        "state": state,
                        "week_start": week_starts,
                        "units_sold": units.astype(int),
                        "sell_price": price,
                    }
                )
            )
    demand = _finalise_demand(pd.concat(frames, ignore_index=True))
    sku_master = _sku_master_from_demand(demand, a, rng)
    po = simulate_purchase_orders(sku_master, demand, a, rng, dataco=None)
    tables = {
        "demand": demand,
        "sku_master": sku_master,
        "suppliers": build_suppliers(a),
        "purchase_orders": po,
    }
    prov = _provenance(
        "synthetic",
        tables,
        a,
        notes=(
            "No Kaggle files were present, so this sample is SYNTHETIC: a seeded "
            "generator shaped to look like M5 weekly demand (intermittent slow movers, "
            "annual seasonality, promotions) and a DataCo-like PO log. No field here is "
            "observed data. Drop the raw Kaggle files into data/raw/ and re-run "
            "`make ingest` to replace it with the real sample."
        ),
    )
    return tables, prov


# --------------------------------------------------------------------------- #
# Shared finalisation
# --------------------------------------------------------------------------- #
def _finalise_demand(df: pd.DataFrame) -> pd.DataFrame:
    """Add a dense week_index and revenue, then coerce to the demand spec."""
    out = df.copy()
    out["week_start"] = pd.to_datetime(out["week_start"])
    weeks = sorted(out["week_start"].unique())
    index_map = {w: i for i, w in enumerate(weeks)}
    out["week_index"] = out["week_start"].map(index_map)
    out["revenue"] = out["units_sold"].astype(float) * out["sell_price"].astype(float)
    out = out.sort_values(["sku", "store", "week_index"]).reset_index(drop=True)
    return coerce(out, _spec("demand"))


def _sku_master_from_demand(
    demand: pd.DataFrame, a: dict[str, Any], rng: np.random.Generator
) -> pd.DataFrame:
    base = (
        demand.groupby(["sku", "store"], observed=True)
        .agg(
            category=("category", "first"),
            department=("department", "first"),
            state=("state", "first"),
            avg_sell_price=("sell_price", "mean"),
        )
        .reset_index()
        .sort_values(["sku", "store"])
        .reset_index(drop=True)
    )
    base["supplier_id"] = assign_suppliers(base, a, rng)
    base = attach_costs(base, a)
    return coerce(base, _spec("sku_master"))


def _provenance(
    origin: Origin, tables: dict[str, pd.DataFrame], a: dict[str, Any], notes: str
) -> Provenance:
    ds = a["data_sources"]
    return Provenance(
        origin=origin,
        built_at=utc_now_iso(),
        demand_source=ds["demand"]["name"] if origin == "kaggle" else "synthetic generator (seeded)",
        orders_source=ds["orders"]["name"] if origin == "kaggle" else "synthetic generator (seeded)",
        simulated_sections=simulated_sections(a),
        sample_scope={
            k: a["sample"][k]
            for k in ("state", "categories", "n_items", "n_stores", "n_planning_items",
                      "n_weeks", "holdout_weeks", "random_seed")
        },
        row_counts={name: int(len(df)) for name, df in tables.items()},
        notes=notes,
    )


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #
def build(prefer: Prefer = "auto", raw_dir: Path | None = None, a: dict[str, Any] | None = None):
    """Build the tables. `auto` uses Kaggle data when it is present."""
    a = a or load_assumptions()
    raw_dir = raw_dir or paths.RAW_DIR
    if prefer == "kaggle" or (prefer == "auto" and have_kaggle(raw_dir)):
        return build_from_kaggle(raw_dir, a)
    return build_synthetic(a)


def write_sample(tables: dict[str, pd.DataFrame], directory: Path | None = None) -> dict[str, Path]:
    """Validate and write every table. Returns name -> written path."""
    return {name: write_table(df, name, directory) for name, df in tables.items()}
