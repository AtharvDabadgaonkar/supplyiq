# SupplyIQ — Product Requirements

**Status:** complete — all five modules built, tested and deployable.

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

## 9. Screens

| Screen | Audience | What it answers |
| --- | --- | --- |
| **Control Tower** | Operations leader | Where does the network stand, what is at risk, what is being done, did it work? Carries the one-screen story: risk → scenario → action → impact. |
| **Demand Forecast** | Demand planner | How accurate is the forecast, on which items is it unreliable, and what does next quarter look like? |
| **Inventory Policy** | Inventory planner | What should each item's safety stock, reorder point and order quantity be, and what does a service-level target cost? |
| **Supplier Performance** | Procurement manager | Which lanes are slow, which are erratic, and what working capital does each one force us to carry? |
| **Scenario Planner** | All | If a supplier slips, demand spikes or a duty moves, which items break and by how much? |
| **Planning Agent** | Inventory planner | For each at-risk item, what are the options, what does each cost, and what service does it buy? |
| **Backtest Results** | Operations leader | Does this policy actually beat what we do today, like for like? |
| **Data & Assumptions** | Anyone auditing it | Which numbers are real, which are simulated, and how do I reproduce them? |

Every chart carries a one-line "how it's calculated" caption. A test asserts that each
page has at least one.

## 10. What the MVP found

Results from the committed sample, all reproducible with `make metrics`:

- **Forecasting.** ETS beats a seasonal-naive baseline by 9.0 pp of WAPE (0.550 → 0.461),
  a 16% relative improvement, with near-zero bias. 22% of holdout weeks have zero sales,
  so MAPE is reported second and its exclusions are disclosed.
- **Inventory.** Accounting for lead-time variability adds 26.6% to total safety stock
  versus a demand-only formula, concentrated on the long overseas lanes.
- **Minimum order quantities, not policy, drive slow-mover stock.** 275 of 400 items order
  at their supplier's MOQ, which is why C items sit above the coverage cap. The lever
  there is renegotiating MOQs, not tuning service levels.
- **Backtest.** +4.50 pp fill rate and 90% of stockout weeks avoided against a naive
  flat-cover policy. At their default settings that costs 47% more working capital; at
  **equal service** the SupplyIQ policy needs **22% less** (range 8.7–37.0%).
- **The headline finding.** Driving the service level to 0.50 removes safety stock
  entirely and the policy still beats the flat rule. The advantage is not the buffer — it
  is reordering against each item's own lead time, which a flat weeks-of-cover rule
  cannot express.

## 11. Known limitations

Stated here rather than discovered by a reviewer:

- **Single echelon.** Store level only. Risk pooling across a DC tier would change the
  safety-stock maths materially, and is the first item on the Phase 2 roadmap.
- **A 12-week holdout has window edges.** Orders placed near the end arrive after it
  closes. Cost to serve excludes the price of goods for exactly this reason, and the
  action pricing credits back closing inventory — but a longer horizon would be cleaner.
- **Dual sourcing is approximated** as a 50/50 blend of two lanes' lead time and freight.
  A true two-lane model needs a second pipeline per item; the UI labels it an approximation.
- **The decision log is not durable.** Streamlit Community Cloud resets its filesystem,
  so approvals live in session state with a CSV export.
- **Supplier and cost data are simulated.** Neither source dataset has a vendor dimension
  or inbound lead times. Everything derived from them is labelled simulated throughout.
- **One holdout, not rolling origin.** A rolling-origin evaluation would give a more
  stable accuracy estimate than a single 12-week split.
