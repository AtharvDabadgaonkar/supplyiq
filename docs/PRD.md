# SupplyIQ — Product Requirements

**Status:** Phase 1 of 5. Sections marked _(Phase 5)_ are filled in once the modules they describe exist.

---

## 1. Problem

A planner is measured on three things that pull against each other: hit the service level, hold less
inventory, and absorb disruption without an expedite bill. Existing tools report the past — last
month's fill rate, current stock on hand — so the trade between the three is made on judgement and
spreadsheets. When a supplier slips three weeks or a tariff moves, there is no fast way to see which
SKUs break, what each possible response costs, and what service it actually buys.

SupplyIQ makes that trade explicit and reproducible: forecast demand, set a policy per SKU, simulate
the disruption, price each response against the same simulator, and show the service-versus-capital
outcome before anyone commits.

## 2. Users

| User | What they need | Primary screen |
| --- | --- | --- |
| Demand planner | Weekly forecast per SKU with honest accuracy, and to know which items are unforecastable | Demand Forecast |
| Inventory planner | Safety stock, reorder point and order quantity that reflect real variability, not a flat weeks-of-cover rule | Inventory Policy |
| Procurement manager | Supplier lead time, variability and OTIF; the cost of switching, dual-sourcing or expediting | Supplier Performance, Planning Agent |
| Operations leader | One screen: current risk, what it costs, what is being done about it | Control Tower |

## 3. Goals

1. A planner can see, for any SKU, the service level and inventory cost implied by the current policy.
2. A planner can apply a disruption (supplier delay, demand spike, tariff change) and see which SKUs
   break and by how much.
3. For each at-risk SKU, the planner is offered concrete actions with the cost and service impact of
   each **computed by the simulator**, and can approve or reject.
4. Every number is reproducible from the repo: one config file for assumptions, one command to
   rebuild every reported figure.

## 4. Non-goals

- Multi-echelon inventory optimisation across warehouse tiers. Single echelon (store level) only.
- A live ERP or planning-system integration. No SAP connection, no purchase orders actually issued.
- Production-grade authentication, multi-tenancy or role-based access.
- Beating a state-of-the-art forecasting benchmark. Two models only — a seasonal-naive baseline and
  one better model — because the point is the decision downstream, not the leaderboard.
- Optimising price or promotions.
- Permanent storage of planner decisions. The deployed demo keeps approve/reject in session state
  with a CSV download; the filesystem resets on redeploy.

## 5. MVP scope

Five modules, in order of dependency:

1. **Sense** — weekly demand forecast per SKU over an 8–12 week horizon, evaluated on a time-based
   holdout against a seasonal-naive baseline. Supplier lead time, variability, OTIF and
   late-delivery rate.
2. **Plan** — ABC/XYZ segmentation; safety stock accounting for demand and lead-time variability;
   reorder point; EOQ; weeks of supply; projected fill rate at a chosen service-level target.
3. **Simulate** — a weekly inventory simulation over the holdout window, with three disruption
   sliders, reporting baseline versus scenario.
4. **Act** — recommendations drawn from five action types, each priced by re-running the simulator
   with the action applied. Approve/reject, logged.
5. **Measure** — a backtest of the SupplyIQ policy against a naive current policy, and a Control
   Tower overview.

Out of MVP: anything in Non-goals, and anything in the Phase 2 roadmap below.

## 6. KPIs

| Tier | Metric | Why |
| --- | --- | --- |
| **Primary** | Fill rate, OTIF | What the business actually asks the planner for |
| Secondary | Inventory turns, forecast MAPE/WAPE, landed cost per unit | The levers that move the primary metrics |
| **Guardrail** | Working capital (inventory value) | Stops "improve service" becoming "hold more of everything" |

Reported in `reports/metrics.json`, written by `make metrics`. The success condition for the MVP is
the backtest in Phase 4: a measurable fill-rate improvement at equal-or-lower inventory value
against the naive baseline policy, on held-out weeks.

## 7. Risks

| Risk | Mitigation |
| --- | --- |
| Simulated supplier and cost data gets mistaken for real data | Labelled in the UI badge, `provenance.json`, the metrics file and the README; a test asserts the labels |
| Intermittent demand breaks MAPE (division by zero on zero-sales weeks) | WAPE reported alongside; MAPE excludes zero-actual weeks by configuration; the sample deliberately contains intermittent items |
| Forecast accuracy looks good because of leakage | One time-based split definition in `supplyiq/data.py`, asserted by test to be non-overlapping |
| An LLM invents a cost or service number | The agent writes prose only; generated text is verified against simulator output and discarded if it contains an unverifiable figure |
| The deployed app is slow or runs out of memory on the free tier | Forecasts, policies and backtests precomputed and committed; only the simulator runs live; column-projected Parquet reads; `@st.cache_data` on every load |
| Scope creep turns a 1.5-day build into two weeks | Five fixed phases; two forecast models only; single echelon |

## 8. Phase 2 roadmap

- **Multi-echelon inventory** across DC and store tiers, with risk-pooling at the upstream level.
- **S&OP workflow** — a monthly consensus cycle: demand review, supply review, scenario sign-off,
  and a plan-versus-actual review with variance attribution.
- **ERP integration** — SAP-style material master and purchase orders, so an approved
  recommendation can be issued as a PO rather than logged.
- **Supplier scorecards** with contracted service levels and automated exception alerting.
- **Persistent decision log** behind a real database, enabling decision-quality analysis over time.

---

_Charts and KPI definitions per screen: (Phase 5)._
