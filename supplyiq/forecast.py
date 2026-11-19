"""SENSE — weekly demand forecasting and honest accuracy measurement.

Two models only, by design:

  seasonal_naive  the baseline. Last year's same week. Falls back to the last
                  observed value when there is under a year of history.
  ets             the candidate. statsmodels Holt-Winters with a damped trend
                  and additive annual seasonality.

Both are fitted on the training weeks alone and asked for the holdout horizon
in one shot. Nothing in this module may read a week at or beyond the split --
`tests/test_forecast.py` asserts it by corrupting the holdout and checking the
forecasts do not move.

Accuracy is reported as WAPE first, MAPE second. Retail demand is intermittent:
a week with zero sales makes the percentage error undefined, so MAPE silently
drops those weeks (see `forecast.mape_exclude_zero_actuals`) while WAPE keeps
every unit of demand in the denominator. Quoting MAPE alone on this kind of
series flatters the model.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any, Callable, Iterable

import numpy as np
import pandas as pd

from supplyiq.config import load_assumptions
from supplyiq.data import train_test

ITEM_KEYS = ["sku", "store"]


# --------------------------------------------------------------------------- #
# Accuracy metrics
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Accuracy:
    """Forecast error for one series, or for a pooled group of series."""

    wape: float
    mape: float
    bias_pct: float
    rmse: float
    mae: float
    n_weeks: int
    n_zero_weeks: int
    total_actual: float
    total_forecast: float

    def as_dict(self) -> dict[str, float]:
        return {
            "wape": self.wape,
            "mape": self.mape,
            "bias_pct": self.bias_pct,
            "rmse": self.rmse,
            "mae": self.mae,
            "n_weeks": self.n_weeks,
            "n_zero_weeks": self.n_zero_weeks,
            "total_actual": self.total_actual,
            "total_forecast": self.total_forecast,
        }


def accuracy(
    actual: Iterable[float], forecast: Iterable[float], exclude_zero_actuals: bool = True
) -> Accuracy:
    """Compute every error metric for one aligned actual/forecast pair.

    WAPE     sum|a - f| / sum(a)        -- volume weighted, defined whenever any
                                           demand occurred
    MAPE     mean(|a - f| / a)          -- over non-zero weeks only, when
                                           exclude_zero_actuals is set
    bias_pct sum(f - a) / sum(a)        -- positive means over-forecasting
    """
    a = np.asarray(list(actual), dtype=float)
    f = np.asarray(list(forecast), dtype=float)
    if a.shape != f.shape:
        raise ValueError(f"actual {a.shape} and forecast {f.shape} are not aligned")
    if a.size == 0:
        raise ValueError("cannot score an empty series")

    err = f - a
    abs_err = np.abs(err)
    total_actual = float(a.sum())
    n_zero = int((a == 0).sum())

    wape = float(abs_err.sum() / total_actual) if total_actual > 0 else float("nan")
    bias = float(err.sum() / total_actual) if total_actual > 0 else float("nan")

    mask = a != 0 if exclude_zero_actuals else np.ones_like(a, dtype=bool)
    mape = float(np.mean(abs_err[mask] / a[mask])) if mask.any() else float("nan")

    return Accuracy(
        wape=wape,
        mape=mape,
        bias_pct=bias,
        rmse=float(np.sqrt(np.mean(err**2))),
        mae=float(abs_err.mean()),
        n_weeks=int(a.size),
        n_zero_weeks=n_zero,
        total_actual=total_actual,
        total_forecast=float(f.sum()),
    )


# --------------------------------------------------------------------------- #
# Models. Each takes the training history and returns `horizon` values.
# --------------------------------------------------------------------------- #
def naive(history: np.ndarray, horizon: int) -> np.ndarray:
    """Repeat the last observed value."""
    if history.size == 0:
        return np.zeros(horizon)
    return np.repeat(float(history[-1]), horizon)


def seasonal_naive(history: np.ndarray, horizon: int, season: int = 52) -> np.ndarray:
    """Last season's same week. Falls back to `naive` below one full season."""
    if history.size < season:
        return naive(history, horizon)
    tail = history[-season:]
    return np.array([float(tail[h % season]) for h in range(horizon)])


def ets(
    history: np.ndarray,
    horizon: int,
    season: int = 52,
    min_weeks_for_seasonal: int | None = None,
) -> np.ndarray:
    """Holt-Winters: damped additive trend, additive annual seasonality.

    Degrades in steps rather than failing: seasonal -> trend-only -> seasonal
    naive. Statsmodels needs two full cycles to identify annual seasonality.
    """
    from statsmodels.tsa.holtwinters import ExponentialSmoothing

    need = min_weeks_for_seasonal if min_weeks_for_seasonal is not None else 2 * season
    attempts: list[dict[str, Any]] = []
    if history.size >= need:
        attempts.append(
            dict(trend="add", damped_trend=True, seasonal="add", seasonal_periods=season)
        )
    attempts.append(dict(trend="add", damped_trend=True, seasonal=None))

    for kwargs in attempts:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                fit = ExponentialSmoothing(
                    history, initialization_method="estimated", **kwargs
                ).fit(optimized=True)
                out = np.asarray(fit.forecast(horizon), dtype=float)
            if np.all(np.isfinite(out)):
                return out
        except Exception:
            continue
    return seasonal_naive(history, horizon, season)


MODELS: dict[str, Callable[..., np.ndarray]] = {
    "naive": lambda h, n, season: naive(h, n),
    "seasonal_naive": seasonal_naive,
    "ets": ets,
}


def clip_non_negative(values: np.ndarray) -> np.ndarray:
    """Unit sales cannot be negative; ETS can extrapolate below zero."""
    return np.maximum(values, 0.0)


# --------------------------------------------------------------------------- #
# Running a model across the whole panel
# --------------------------------------------------------------------------- #
def forecast_panel(
    train: pd.DataFrame,
    model: str,
    horizon: int,
    season: int = 52,
    value_col: str = "units_sold",
) -> pd.DataFrame:
    """Fit `model` per planning item on `train` and forecast `horizon` weeks.

    Returns one row per (sku, store, step) with `week_index` continuing from the
    end of training, so it joins straight onto the holdout actuals.
    """
    if model not in MODELS:
        raise KeyError(f"unknown model {model!r}; expected one of {sorted(MODELS)}")
    fn = MODELS[model]
    last_train_week = int(train["week_index"].max())

    rows: list[pd.DataFrame] = []
    for (sku, store), group in train.sort_values("week_index").groupby(
        ITEM_KEYS, observed=True
    ):
        history = group[value_col].to_numpy(dtype=float)
        values = clip_non_negative(np.asarray(fn(history, horizon, season), dtype=float))
        rows.append(
            pd.DataFrame(
                {
                    "sku": sku,
                    "store": store,
                    "step": np.arange(1, horizon + 1),
                    "week_index": last_train_week + np.arange(1, horizon + 1),
                    "forecast": values,
                    "model": model,
                }
            )
        )
    return pd.concat(rows, ignore_index=True)


def score_panel(
    forecasts: pd.DataFrame, actuals: pd.DataFrame, exclude_zero_actuals: bool = True
) -> tuple[pd.DataFrame, Accuracy]:
    """Join forecasts to holdout actuals and score per item and overall.

    The overall figure is pooled, not an average of per-item percentages: a
    mean of ratios lets a tiny slow-moving SKU swamp the number.
    """
    joined = forecasts.merge(
        actuals[ITEM_KEYS + ["week_index", "units_sold"]],
        on=ITEM_KEYS + ["week_index"],
        how="inner",
        validate="one_to_one",
    )
    if len(joined) != len(forecasts):
        missing = len(forecasts) - len(joined)
        raise ValueError(f"{missing} forecast rows had no matching actual week")

    per_item = (
        joined.groupby(ITEM_KEYS, observed=True)
        .apply(
            lambda g: pd.Series(
                accuracy(g["units_sold"], g["forecast"], exclude_zero_actuals).as_dict()
            ),
            include_groups=False,
        )
        .reset_index()
    )
    per_item["model"] = forecasts["model"].iloc[0]
    overall = accuracy(joined["units_sold"], joined["forecast"], exclude_zero_actuals)
    return per_item, overall


# --------------------------------------------------------------------------- #
# The full evaluation: baseline vs candidate on a time-based holdout
# --------------------------------------------------------------------------- #
@dataclass
class Evaluation:
    forecasts: pd.DataFrame            # every model's forecast on the holdout
    per_item: pd.DataFrame             # per (sku, store, model) accuracy
    overall: dict[str, Accuracy]       # model -> pooled accuracy
    baseline_model: str
    candidate_model: str
    train_weeks: tuple[int, int]
    holdout_weeks: tuple[int, int]

    @property
    def winner(self) -> str:
        """Lower pooled WAPE wins. WAPE, not MAPE — see the module docstring."""
        return min(self.overall, key=lambda m: self.overall[m].wape)

    def improvement_pp(self) -> float:
        """Percentage points of WAPE the candidate takes off the baseline."""
        base = self.overall[self.baseline_model].wape
        cand = self.overall[self.candidate_model].wape
        return float((base - cand) * 100)


def evaluate(
    demand: pd.DataFrame, a: dict[str, Any] | None = None, models: list[str] | None = None
) -> Evaluation:
    """Fit on the training weeks, score on the untouched holdout."""
    a = a or load_assumptions()
    holdout = int(a["sample"]["holdout_weeks"])
    season = int(a["forecast"]["seasonal_period_weeks"])
    exclude_zero = bool(a["forecast"]["mape_exclude_zero_actuals"])
    baseline = str(a["forecast"]["baseline_model"])
    candidate = str(a["forecast"]["candidate_model"])
    models = models or [baseline, candidate]

    train, test = train_test(demand, holdout)
    horizon = int(test["week_index"].nunique())

    all_forecasts, all_per_item, overall = [], [], {}
    for model in models:
        fc = forecast_panel(train, model, horizon, season)
        per_item, pooled = score_panel(fc, test, exclude_zero)
        all_forecasts.append(fc)
        all_per_item.append(per_item)
        overall[model] = pooled

    return Evaluation(
        forecasts=pd.concat(all_forecasts, ignore_index=True),
        per_item=pd.concat(all_per_item, ignore_index=True),
        overall=overall,
        baseline_model=baseline,
        candidate_model=candidate,
        train_weeks=(int(train["week_index"].min()), int(train["week_index"].max())),
        holdout_weeks=(int(test["week_index"].min()), int(test["week_index"].max())),
    )


def future_forecast(
    demand: pd.DataFrame, model: str, a: dict[str, Any] | None = None
) -> pd.DataFrame:
    """Forecast forward from the END of all history, for the planning horizon.

    `evaluate` measures the model; this is what the inventory policy consumes.
    """
    a = a or load_assumptions()
    horizon = int(a["sample"]["forecast_horizon_weeks"])
    season = int(a["forecast"]["seasonal_period_weeks"])
    return forecast_panel(demand, model, horizon, season)


def demand_stats(demand: pd.DataFrame, value_col: str = "units_sold") -> pd.DataFrame:
    """Mean, standard deviation and CV of weekly demand per planning item.

    These feed XYZ segmentation and the safety-stock calculation, so they are
    computed once, here, from the training weeks only.
    """
    grouped = demand.groupby(ITEM_KEYS, observed=True)[value_col]
    stats = grouped.agg(
        mean_weekly_demand="mean",
        sd_weekly_demand=lambda s: float(s.std(ddof=1)),
        total_demand="sum",
        weeks_observed="size",
        zero_weeks=lambda s: int((s == 0).sum()),
    ).reset_index()
    stats["sd_weekly_demand"] = stats["sd_weekly_demand"].fillna(0.0)
    stats["demand_cv"] = np.where(
        stats["mean_weekly_demand"] > 0,
        stats["sd_weekly_demand"] / stats["mean_weekly_demand"],
        np.nan,
    )
    stats["zero_week_share"] = stats["zero_weeks"] / stats["weeks_observed"]
    return stats
