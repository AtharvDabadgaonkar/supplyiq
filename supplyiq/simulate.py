"""SIMULATE — the weekly inventory twin and the disruption scenarios.

What it does
------------
For each planning item, step through the holdout weeks:

  1. receive anything arriving this week
  2. meet demand from stock; whatever is not met is LOST, not backordered
  3. review: if inventory position <= reorder point, order up to ROP + Q,
     then raise the order to the supplier's MOQ and order multiple
  4. the order arrives ceil(lead_time_days / 7) weeks later

Why the randomness is shared
----------------------------
Lead times are redrawn per order from each supplier's lognormal. If a baseline
run and a scenario run drew independently, the difference between them would
mix the scenario's effect with simulation noise, and a planner could not tell
which was which. So the draws are generated once, per item and per week, from a
fixed seed, and BOTH runs consume the same ones -- common random numbers. A
scenario's delay is then added on top of an identical draw. This is also what
makes `simulate(baseline) == simulate(zero_scenario)` exact rather than
approximate, which is the property the tests pin down.

Costs
-----
    landed cost / unit = unit_cost + freight + unit_cost * tariff   (SCENARIO tariff)
    cost of goods      = units ORDERED * landed cost per unit       (the commitment)
    holding cost       = average on-hand units * landed * holding_rate * weeks/52
    stockout cost      = units short * stockout_penalty_per_unit
    COST TO SERVE      = holding + stockout + ordering + action costs

Cost to serve deliberately EXCLUDES the price of the goods, which is the
standard meaning of the term and is also what keeps it honest over a finite
window. Including goods made a 21-day supplier delay look $2.7k CHEAPER than
the baseline: the delay pushed thirteen reorders past the end of the 12-week
window, so less was committed, and that fall swamped the stockout cost it
caused. That is an artifact of where the window closes, not economics. With
goods excluded, the same delay costs more than the baseline, as it should.

`cost_of_goods_ordered` and `cost_of_goods_sold` are both reported so the
commitment and the consumption views are available, and
`ending_position_value` exposes the window-edge effect rather than hiding it.

Working capital counts goods in transit as well as goods on hand. A delay does
not free up cash, it strands it on a ship.
"""
from __future__ import annotations

import math
import zlib
from dataclasses import dataclass, field, replace
from typing import Any, Iterable, Literal

import numpy as np
import pandas as pd

from supplyiq.config import load_assumptions
from supplyiq.inventory import apply_order_constraints

ITEM_KEYS = ["sku", "store"]
WEEKS_PER_YEAR = 52.0

ScopeKind = Literal["all", "supplier", "country"]


# --------------------------------------------------------------------------- #
# Scenario definition
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Scenario:
    """A disruption. All three levers default to the no-disruption case.

    `supplier_scope` / `country_scope` limit the delay and the tariff change to
    part of the network -- "what if China goes to 35% duty" is a different
    question from "what if every duty moves".
    """

    name: str = "baseline"
    supplier_delay_days: float = 0.0
    demand_spike_pct: float = 0.0
    tariff_change_pct: float = 0.0
    supplier_scope: tuple[str, ...] | None = None
    country_scope: tuple[str, ...] | None = None

    @property
    def is_baseline(self) -> bool:
        return (
            self.supplier_delay_days == 0
            and self.demand_spike_pct == 0
            and self.tariff_change_pct == 0
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "supplier_delay_days": self.supplier_delay_days,
            "demand_spike_pct": self.demand_spike_pct,
            "tariff_change_pct": self.tariff_change_pct,
            "supplier_scope": list(self.supplier_scope) if self.supplier_scope else None,
            "country_scope": list(self.country_scope) if self.country_scope else None,
        }

    def describe(self) -> str:
        if self.is_baseline:
            return "No disruption"
        parts = []
        if self.supplier_delay_days:
            parts.append(f"+{self.supplier_delay_days:.0f}d supplier delay")
        if self.demand_spike_pct:
            parts.append(f"{self.demand_spike_pct:+.0f}% demand")
        if self.tariff_change_pct:
            parts.append(f"{self.tariff_change_pct:+.0f}pp tariff")
        scope = self.scope_label()
        return ", ".join(parts) + (f" ({scope})" if scope != "all suppliers" else "")

    def scope_label(self) -> str:
        if self.supplier_scope:
            return ", ".join(self.supplier_scope)
        if self.country_scope:
            return ", ".join(self.country_scope)
        return "all suppliers"


def baseline_scenario() -> Scenario:
    return Scenario(name="baseline")


def scenario_from_config(a: dict[str, Any] | None = None, **overrides: Any) -> Scenario:
    """Build a scenario from the configured defaults, which are the null case."""
    a = a or load_assumptions()
    cfg = a["scenarios"]
    base = dict(
        supplier_delay_days=float(cfg["supplier_delay_days"]["default"]),
        demand_spike_pct=float(cfg["demand_spike_pct"]["default"]),
        tariff_change_pct=float(cfg["tariff_change_pct"]["default"]),
    )
    base.update(overrides)
    return Scenario(**base)


def in_scope(scenario: Scenario, supplier_ids: pd.Series, countries: pd.Series) -> np.ndarray:
    """Boolean mask of the items a scenario's delay and tariff apply to."""
    if scenario.supplier_scope:
        return supplier_ids.isin(scenario.supplier_scope).to_numpy()
    if scenario.country_scope:
        return countries.isin(scenario.country_scope).to_numpy()
    return np.ones(len(supplier_ids), dtype=bool)


# --------------------------------------------------------------------------- #
# Per-item action overrides, used by the ACT module in Phase 4
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Overrides:
    """Per-item changes an action applies before a re-run.

    Each maps (sku, store) -> value. The ACT module prices a recommendation by
    re-running this simulator with the relevant override set, so the numbers a
    planner sees come from the same engine as the baseline.
    """

    lead_time_days: dict[tuple[str, str], float] = field(default_factory=dict)
    freight_per_unit: dict[tuple[str, str], float] = field(default_factory=dict)
    order_quantity: dict[tuple[str, str], float] = field(default_factory=dict)
    initial_on_hand: dict[tuple[str, str], float] = field(default_factory=dict)
    extra_cost: dict[tuple[str, str], float] = field(default_factory=dict)

    def is_empty(self) -> bool:
        return not any(
            (
                self.lead_time_days,
                self.freight_per_unit,
                self.order_quantity,
                self.initial_on_hand,
                self.extra_cost,
            )
        )


# --------------------------------------------------------------------------- #
# Lead-time draws: generated once, shared by every run
# --------------------------------------------------------------------------- #
def lognormal_params(mean_days: float, sd_days: float) -> tuple[float, float]:
    """mu and sigma of a lognormal with the given mean and standard deviation."""
    if mean_days <= 0:
        return 0.0, 0.0
    sigma = math.sqrt(math.log1p((sd_days / mean_days) ** 2))
    mu = math.log(mean_days) - 0.5 * sigma**2
    return mu, sigma


def item_seed(seed: int, sku: str, store: str) -> int:
    """A stable per-item seed, independent of where the item sits in the table.

    Seeding one generator and walking it row by row would tie an item's draws to
    its position, so simulating a SUBSET of items -- which is exactly what the
    ACT module does when it prices one recommendation -- would give that item
    different supply luck than the full run, and the comparison would be
    meaningless. Keying the stream to the item's own identity makes any subset
    reproduce the full run exactly.
    """
    key = f"{sku}|{store}".encode("utf-8")
    return (int(seed) ^ zlib.crc32(key)) & 0xFFFFFFFF


def draw_lead_times(
    policy: pd.DataFrame, n_weeks: int, seed: int, days_per_week: float = 7.0
) -> np.ndarray:
    """One lead-time draw per item per week, in DAYS. Shape (items, weeks).

    Row order follows `policy`. Each row's stream is keyed to the item, so the
    same item always draws the same values whatever else is in the table.
    """
    out = np.zeros((len(policy), n_weeks), dtype=float)
    means = policy["lead_time_mean_weeks"].to_numpy(dtype=float) * days_per_week
    sds = policy["lead_time_sd_weeks"].to_numpy(dtype=float) * days_per_week
    skus = policy["sku"].to_numpy()
    stores = policy["store"].to_numpy()
    for i, (mean_days, sd_days) in enumerate(zip(means, sds)):
        mu, sigma = lognormal_params(mean_days, sd_days)
        if sigma <= 0:
            out[i] = mean_days
            continue
        rng = np.random.default_rng(item_seed(seed, skus[i], stores[i]))
        out[i] = rng.lognormal(mu, sigma, size=n_weeks)
    return out


# --------------------------------------------------------------------------- #
# The simulation
# --------------------------------------------------------------------------- #
@dataclass
class SimulationResult:
    """Per-item outcomes plus the roll-up a planner actually reads."""

    scenario: Scenario
    per_item: pd.DataFrame
    ledger: pd.DataFrame | None
    kpis: dict[str, float]

    def compare_to(self, other: "SimulationResult") -> dict[str, float]:
        """This result minus `other`, for every KPI they share."""
        return {
            key: float(self.kpis[key] - other.kpis[key])
            for key in self.kpis
            if isinstance(self.kpis[key], (int, float))
            and isinstance(other.kpis.get(key), (int, float))
        }


def simulate(
    policy: pd.DataFrame,
    demand: pd.DataFrame,
    scenario: Scenario | None = None,
    a: dict[str, Any] | None = None,
    overrides: Overrides | None = None,
    lead_time_draws: np.ndarray | None = None,
    keep_ledger: bool = False,
) -> SimulationResult:
    """Run the weekly inventory twin over `demand`'s weeks.

    `policy` must carry: order_quantity, reorder_point, lead_time_mean_weeks,
    lead_time_sd_weeks, unit_cost, freight_per_unit, tariff_rate, moq_units,
    order_multiple_units, mean_weekly_demand, supplier_id, country.
    """
    a = a or load_assumptions()
    scenario = scenario or baseline_scenario()
    overrides = overrides or Overrides()
    sim_cfg = a["simulation"]
    costs = a["costs"]

    weeks = sorted(demand["week_index"].unique())
    n_weeks = len(weeks)
    wide = (
        demand.pivot_table(
            index=ITEM_KEYS, columns="week_index", values="units_sold", aggfunc="sum"
        )
        .reindex(columns=weeks)
        .fillna(0.0)
    )
    items = policy.set_index(ITEM_KEYS).reindex(wide.index)
    if items["order_quantity"].isna().any():
        missing = int(items["order_quantity"].isna().sum())
        raise ValueError(f"{missing} items in the demand window have no policy row")
    items = items.reset_index()

    demand_matrix = wide.to_numpy(dtype=float)
    if scenario.demand_spike_pct:
        demand_matrix = demand_matrix * (1.0 + scenario.demand_spike_pct / 100.0)

    scoped = in_scope(scenario, items["supplier_id"], items.get("country", items["supplier_id"]))

    # --- scenario economics ------------------------------------------------
    unit_cost = items["unit_cost"].to_numpy(dtype=float)
    freight = items["freight_per_unit"].to_numpy(dtype=float)
    tariff = items["tariff_rate"].to_numpy(dtype=float).copy()
    if scenario.tariff_change_pct:
        # Percentage POINTS on the ad-valorem rate, floored at zero.
        tariff[scoped] = np.maximum(0.0, tariff[scoped] + scenario.tariff_change_pct / 100.0)

    keys = list(zip(items["sku"], items["store"]))
    if overrides.freight_per_unit:
        for i, key in enumerate(keys):
            if key in overrides.freight_per_unit:
                freight[i] = overrides.freight_per_unit[key]
    landed = unit_cost + freight + unit_cost * tariff

    # --- supply parameters -------------------------------------------------
    base_lead_days = items["lead_time_mean_weeks"].to_numpy(dtype=float) * 7.0
    order_qty = items["order_quantity"].to_numpy(dtype=float).copy()
    rop = items["reorder_point"].to_numpy(dtype=float)
    moq = items["moq_units"].to_numpy(dtype=float)
    multiple = items["order_multiple_units"].to_numpy(dtype=float)
    mean_demand = items["mean_weekly_demand"].to_numpy(dtype=float)

    # NOTE: the order-quantity override is applied AFTER opening stock is set,
    # further down. Opening stock is ROP + order quantity, so applying it here
    # would let "raise the order quantity" retroactively grant the item free
    # inventory on week 0 -- which made that action look like it improved fill
    # rate at negative cost. A planner's current stock is what it is; changing
    # the order quantity only affects what is ordered from now on.

    if lead_time_draws is None:
        lead_time_draws = draw_lead_times(items, n_weeks, int(sim_cfg["seed"]))
    if lead_time_draws.shape != (len(items), n_weeks):
        raise ValueError(
            f"lead_time_draws shape {lead_time_draws.shape} != {(len(items), n_weeks)}"
        )

    lead_days = lead_time_draws.copy()
    if overrides.lead_time_days:
        for i, key in enumerate(keys):
            if key in overrides.lead_time_days:
                lead_days[i, :] = overrides.lead_time_days[key]
    if scenario.supplier_delay_days:
        lead_days[scoped, :] = lead_days[scoped, :] + scenario.supplier_delay_days

    # --- opening inventory, from the UNCHANGED policy ------------------------
    if a["policy"]["initial_stock_policy"] == "order_up_to":
        on_hand = (rop + order_qty).astype(float).copy()
    else:
        on_hand = (mean_demand * float(a["policy"]["initial_stock_weeks_of_cover"])).copy()
    if overrides.initial_on_hand:
        for i, key in enumerate(keys):
            if key in overrides.initial_on_hand:
                on_hand[i] = overrides.initial_on_hand[key]

    # Now the override may change what gets ORDERED from week 0 onwards.
    if overrides.order_quantity:
        for i, key in enumerate(keys):
            if key in overrides.order_quantity:
                order_qty[i] = overrides.order_quantity[key]

    n_items = len(items)
    pipeline = np.zeros((n_items, n_weeks + 64), dtype=float)   # arrivals by week offset
    po_late = np.zeros(n_items, dtype=float)
    po_count = np.zeros(n_items, dtype=float)

    in_transit_sum = np.zeros(n_items)
    demanded = np.zeros(n_items)
    shipped = np.zeros(n_items)
    short = np.zeros(n_items)
    stockout_weeks = np.zeros(n_items)
    on_hand_sum = np.zeros(n_items)
    units_ordered = np.zeros(n_items)
    units_received = np.zeros(n_items)
    orders_placed = np.zeros(n_items)

    # On-time is judged against the supplier's QUOTED date -- the commitment it
    # made -- not against its own average lead time. Judging against the mean
    # would make roughly half of all POs late by construction.
    if "quoted_lead_time_days" in items.columns:
        quoted_days = items["quoted_lead_time_days"].to_numpy(dtype=float)
    else:
        quoted_days = base_lead_days
    quoted_weeks = np.ceil(quoted_days / 7.0)
    ledger_rows: list[dict[str, Any]] = []

    for w in range(n_weeks):
        arrivals = pipeline[:, w].copy()
        on_hand += arrivals
        units_received += arrivals

        week_demand = demand_matrix[:, w]
        sold = np.minimum(on_hand, week_demand)
        unmet = week_demand - sold
        on_hand -= sold

        demanded += week_demand
        shipped += sold
        short += unmet
        stockout_weeks += (unmet > 1e-9).astype(float)
        on_hand_sum += on_hand

        # Periodic review: inventory position includes what is already on order.
        on_order = pipeline[:, w + 1 :].sum(axis=1)
        in_transit_sum += on_order
        position = on_hand + on_order
        needs_order = position <= rop
        if needs_order.any():
            raw = (rop + order_qty) - position
            for i in np.flatnonzero(needs_order):
                qty = apply_order_constraints(max(0.0, raw[i]), moq[i], multiple[i])
                if qty <= 0:
                    continue
                offset = int(math.ceil(lead_days[i, w] / 7.0))
                arrival = w + max(1, offset)
                if arrival < pipeline.shape[1]:
                    pipeline[i, arrival] += qty
                units_ordered[i] += qty
                orders_placed[i] += 1
                po_count[i] += 1
                if offset > quoted_weeks[i]:
                    po_late[i] += 1

        if keep_ledger:
            for i in range(n_items):
                ledger_rows.append(
                    {
                        "sku": items["sku"].iloc[i],
                        "store": items["store"].iloc[i],
                        "week_index": weeks[w],
                        "demand": float(week_demand[i]),
                        "shipped": float(sold[i]),
                        "short": float(unmet[i]),
                        "on_hand": float(on_hand[i]),
                        "on_order": float(on_order[i]),
                        "received": float(arrivals[i]),
                    }
                )

    avg_on_hand = on_hand_sum / n_weeks
    avg_in_transit = in_transit_sum / n_weeks
    weeks_fraction = n_weeks / WEEKS_PER_YEAR

    per_item = items[ITEM_KEYS + ["supplier_id", "abc_class", "xyz_class"]].copy()
    per_item["demand_units"] = demanded
    per_item["shipped_units"] = shipped
    per_item["short_units"] = short
    # np.where would still evaluate shipped/demanded everywhere first, which
    # divides by zero on items with no demand in the window. An item that was
    # never asked for was never short, so its fill rate is 1.
    per_item["fill_rate"] = np.divide(
        shipped, demanded, out=np.ones_like(shipped, dtype=float), where=demanded > 0
    )
    per_item["stockout_weeks"] = stockout_weeks
    per_item["avg_on_hand_units"] = avg_on_hand
    per_item["avg_in_transit_units"] = avg_in_transit
    per_item["ending_on_hand_units"] = on_hand
    per_item["landed_cost_per_unit"] = landed
    per_item["avg_inventory_value"] = avg_on_hand * landed
    per_item["avg_in_transit_value"] = avg_in_transit * landed
    per_item["working_capital"] = (avg_on_hand + avg_in_transit) * landed
    per_item["units_ordered"] = units_ordered
    per_item["units_received"] = units_received
    per_item["orders_placed"] = orders_placed
    per_item["pos_late"] = po_late
    per_item["otif"] = np.divide(
        po_count - po_late,
        po_count,
        out=np.full_like(po_count, np.nan, dtype=float),
        where=po_count > 0,
    )
    per_item["cost_of_goods_ordered"] = units_ordered * landed
    per_item["cost_of_goods_received"] = units_received * landed
    per_item["cost_of_goods_sold"] = shipped * landed
    per_item["ending_position_value"] = (on_hand + pipeline[:, n_weeks:].sum(axis=1)) * landed
    per_item["holding_cost"] = (
        avg_on_hand * landed * float(costs["holding_cost_rate_annual"]) * weeks_fraction
    )
    per_item["stockout_cost"] = short * float(costs["stockout_penalty_per_unit"])
    per_item["ordering_cost"] = orders_placed * float(costs["ordering_cost_per_po"])
    extra = np.zeros(n_items)
    if overrides.extra_cost:
        for i, key in enumerate(keys):
            extra[i] = overrides.extra_cost.get(key, 0.0)
    per_item["action_cost"] = extra
    # Goods are deliberately not in here -- see the module docstring.
    per_item["cost_to_serve"] = (
        per_item["holding_cost"]
        + per_item["stockout_cost"]
        + per_item["ordering_cost"]
        + per_item["action_cost"]
    )
    per_item["total_cost"] = per_item["cost_to_serve"] + per_item["cost_of_goods_ordered"]

    ledger = pd.DataFrame(ledger_rows) if keep_ledger else None
    return SimulationResult(
        scenario=scenario, per_item=per_item, ledger=ledger, kpis=roll_up(per_item, n_weeks)
    )


def roll_up(per_item: pd.DataFrame, n_weeks: int) -> dict[str, float]:
    """Network KPIs. Fill rate is pooled on units, never a mean of per-item rates."""
    total_demand = float(per_item["demand_units"].sum())
    total_shipped = float(per_item["shipped_units"].sum())
    po_total = float(per_item["orders_placed"].sum())
    po_late = float(per_item["pos_late"].sum())
    return {
        "items": int(len(per_item)),
        "weeks": int(n_weeks),
        "demand_units": total_demand,
        "shipped_units": total_shipped,
        "short_units": float(per_item["short_units"].sum()),
        "fill_rate": (total_shipped / total_demand) if total_demand > 0 else 1.0,
        "stockout_weeks": float(per_item["stockout_weeks"].sum()),
        "items_with_a_stockout": int((per_item["stockout_weeks"] > 0).sum()),
        "in_stock_rate": float(
            1.0 - per_item["stockout_weeks"].sum() / max(1, len(per_item) * n_weeks)
        ),
        "otif": (1.0 - po_late / po_total) if po_total > 0 else float("nan"),
        "orders_placed": po_total,
        "avg_inventory_units": float(per_item["avg_on_hand_units"].sum()),
        "avg_inventory_value": float(per_item["avg_inventory_value"].sum()),
        "avg_in_transit_value": float(per_item["avg_in_transit_value"].sum()),
        "working_capital": float(per_item["working_capital"].sum()),
        "units_ordered": float(per_item["units_ordered"].sum()),
        "units_received": float(per_item["units_received"].sum()),
        "cost_of_goods_ordered": float(per_item["cost_of_goods_ordered"].sum()),
        "cost_of_goods_received": float(per_item["cost_of_goods_received"].sum()),
        "cost_of_goods_sold": float(per_item["cost_of_goods_sold"].sum()),
        "ending_position_value": float(per_item["ending_position_value"].sum()),
        "landed_cost_per_unit": (
            float(per_item["cost_of_goods_ordered"].sum() / per_item["units_ordered"].sum())
            if per_item["units_ordered"].sum() > 0
            else float("nan")
        ),
        "holding_cost": float(per_item["holding_cost"].sum()),
        "stockout_cost": float(per_item["stockout_cost"].sum()),
        "ordering_cost": float(per_item["ordering_cost"].sum()),
        "action_cost": float(per_item["action_cost"].sum()),
        "cost_to_serve": float(per_item["cost_to_serve"].sum()),
        "total_cost": float(per_item["total_cost"].sum()),
    }


def run_scenarios(
    policy: pd.DataFrame,
    demand: pd.DataFrame,
    scenarios: Iterable[Scenario],
    a: dict[str, Any] | None = None,
) -> dict[str, SimulationResult]:
    """Run several scenarios against one shared set of lead-time draws."""
    a = a or load_assumptions()
    weeks = sorted(demand["week_index"].unique())
    items = policy.set_index(ITEM_KEYS).reindex(
        demand.pivot_table(index=ITEM_KEYS, columns="week_index", values="units_sold").index
    ).reset_index()
    draws = draw_lead_times(items, len(weeks), int(a["simulation"]["seed"]))
    return {
        s.name: simulate(policy, demand, s, a, lead_time_draws=draws) for s in scenarios
    }
