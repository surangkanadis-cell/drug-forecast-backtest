"""Build the input a model sees at one rolling origin.

Rule: at origin O the model sees drug data only up to month O-1.
  * context  : target for months [max(context_start, first observed month), O-1]
               + past_only covariates for the same months
               + known_future covariates for the same months
  * future   : known_future covariates for the target months O .. O+horizon-1
               (holidays counted only if announced before O)
Actual values for the target months are NOT part of the task. The backtest reads
them separately, after the forecast is made.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .covariates import CovariateSource
from .io import DRUG, MONTH, TARGET


@dataclass
class ForecastTask:
    origin: pd.Timestamp
    horizon: int
    target_months: pd.DatetimeIndex
    context: pd.DataFrame                # drug_id, month, quantity, <past_cols>, <future_cols>
    future: pd.DataFrame                 # drug_id, month, <future_cols>  (one row per drug x target month)
    past_cols: list = field(default_factory=list)
    future_cols: list = field(default_factory=list)

    @property
    def drug_ids(self) -> list:
        return list(pd.unique(self.context[DRUG]))


def naive_scale(history: pd.Series, lag: int) -> float:
    """MASE denominator: mean |y_t - y_{t-lag}| over the monthly grid of `history`.

    Only pairs where both months are observed count. NaN if no pair exists or the
    mean is 0 (the scaled error would be undefined).
    """
    v = np.asarray(history, float)
    if len(v) <= lag:
        return np.nan
    d = np.abs(v[lag:] - v[:-lag])
    d = d[~np.isnan(d)]
    if d.size == 0:
        return np.nan
    m = float(d.mean())
    return m if m > 0 else np.nan


def build_task(panel: pd.DataFrame, origin, *, horizon: int = 1, min_history: int = 1,
               context_start=None, covariates: CovariateSource | None = None,
               use_covariates: bool = False, drug_ids=None):
    """Returns (task or None, meta DataFrame, skipped list).

    meta has one row per drug in the task: history_months, scale_lag1, scale_m12, hist_mean
    (all computed from data before the origin only).
    """
    origin = pd.Timestamp(origin)
    target_months = pd.date_range(origin, periods=horizon, freq="MS")
    hist = panel.loc[panel.index < origin]
    if context_start is not None:
        hist = hist.loc[hist.index >= pd.Timestamp(context_start)]

    parts, meta, skipped = [], [], []
    for d in (drug_ids if drug_ids is not None else panel.columns):
        s = hist[d] if d in hist.columns else pd.Series(dtype=float)
        first = s.first_valid_index()
        if first is None:
            skipped.append({DRUG: d, "reason": "no_data_before_origin", "history_months": 0})
            continue
        s = s.loc[first:]
        n_obs = int(s.notna().sum())
        if n_obs < min_history:
            skipped.append({DRUG: d, "reason": f"history_lt_min_history({min_history})",
                            "history_months": n_obs})
            continue
        parts.append(pd.DataFrame({DRUG: d, MONTH: s.index, TARGET: s.to_numpy(float)}))
        mean = float(np.nanmean(s.to_numpy(float)))
        meta.append({DRUG: d, "history_months": n_obs, "first_month": first,
                     "scale_lag1": naive_scale(s, 1), "scale_m12": naive_scale(s, 12),
                     "hist_mean": mean if mean != 0 else np.nan})

    meta = pd.DataFrame(meta, columns=[DRUG, "history_months", "first_month",
                                       "scale_lag1", "scale_m12", "hist_mean"])
    if not parts:
        return None, meta, skipped

    context = pd.concat(parts, ignore_index=True)
    drugs = list(pd.unique(context[DRUG]))
    future = pd.DataFrame({DRUG: np.repeat(drugs, horizon),
                           MONTH: np.tile(target_months, len(drugs))})
    past_cols, future_cols = [], []
    if use_covariates:
        if covariates is None:
            raise ValueError("use_covariates=True แต่ไม่ได้ระบุ covariate_config")
        past_cols, future_cols = covariates.past_cols, covariates.future_cols
        months = pd.date_range(context[MONTH].min(), target_months[-1], freq="MS")
        cov = covariates.frame(months, origin).rename_axis(MONTH).reset_index()
        context = context.merge(cov[[MONTH] + past_cols + future_cols], on=MONTH, how="left")
        future = future.merge(cov[[MONTH] + future_cols], on=MONTH, how="left")

    task = ForecastTask(origin=origin, horizon=horizon, target_months=target_months,
                        context=context, future=future,
                        past_cols=list(past_cols), future_cols=list(future_cols))
    check_no_leakage(task)
    return task, meta, skipped


def check_no_leakage(task: ForecastTask) -> None:
    """Raise if anything from the origin month onward could reach the model."""
    if (task.context[MONTH] >= task.origin).any():
        raise AssertionError("context contains months >= origin")
    if TARGET in task.future.columns:
        raise AssertionError("future frame contains the target")
    leaked = [c for c in task.past_cols if c in task.future.columns]
    if leaked:
        raise AssertionError(f"past_only covariates in future frame: {leaked}")
    if not set(task.future[MONTH]).issubset(set(task.target_months)):
        raise AssertionError("future frame has months outside the target months")
