"""Row-level metrics and summaries.

Per row (one drug x origin x horizon x model x covariate setting), with median = q0.5:
  mae         |y - q0.5|
  wape        |y - q0.5| / |y|           (NaN when y = 0). The pooled WAPE in the
                                         summaries is sum|e| / sum|y|.
  mase_lag1   |y - q0.5| / mean|y_t - y_{t-1}|    over history before the origin
  mase_m12    |y - q0.5| / mean|y_t - y_{t-12}|   over history before the origin
              (NaN when no pair of observed months exists, or the scale is 0)
  crps        2 * mean_k pinball_{tau_k}(y, q_tau_k), k = the 9 levels 0.1..0.9
              (quantile approximation of CRPS = 2 * integral of pinball loss over tau)
  crps_scaled crps / mean(y before origin)   (NaN when that mean is 0)
  covered_80  1 if q0.1 <= y <= q0.9 else 0
  crps_skill  left empty; filled when a Seasonal Naive reference is added
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .models.base import Q_COLS, QUANTILES

CRPS_FACTOR = 2.0
METRICS = ["mae", "wape", "mase_lag1", "mase_m12", "crps", "crps_scaled", "covered_80"]


def pinball(y, q, tau):
    """rho_tau(y, q) = (tau - 1{y < q}) (y - q) >= 0. Broadcasts."""
    y, q, tau = np.asarray(y, float), np.asarray(q, float), np.asarray(tau, float)
    diff = y - q
    return np.where(diff >= 0, tau * diff, (tau - 1.0) * diff)


def add_row_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """df needs actual, q0.1..q0.9, scale_lag1, scale_m12, hist_mean."""
    out = df.copy()
    y = out["actual"].to_numpy(float)
    q = out[Q_COLS].to_numpy(float)
    med = out["q0.5"].to_numpy(float)
    ae = np.abs(y - med)
    with np.errstate(divide="ignore", invalid="ignore"):
        out["mae"] = ae
        out["wape"] = np.where(y != 0, ae / np.abs(y), np.nan)
        out["mase_lag1"] = ae / out["scale_lag1"].to_numpy(float)
        out["mase_m12"] = ae / out["scale_m12"].to_numpy(float)
        crps = CRPS_FACTOR * pinball(y[:, None], q, np.asarray(QUANTILES)[None, :]).mean(axis=1)
        out["crps"] = crps
        hm = out["hist_mean"].to_numpy(float)
        out["crps_scaled"] = np.where(np.isfinite(hm) & (hm != 0), crps / hm, np.nan)
    covered = (y >= out["q0.1"].to_numpy(float)) & (y <= out["q0.9"].to_numpy(float))
    out["covered_80"] = np.where(np.isnan(y), np.nan, covered.astype(float))
    out["crps_skill"] = np.nan
    return out


def _agg(g: pd.DataFrame) -> dict:
    row = {"n_rows": len(g)}
    for m in METRICS:
        v = g[m]
        row[f"{m}_mean"] = v.mean()
        row[f"{m}_sd"] = v.std(ddof=1)
        row[f"{m}_n"] = int(v.notna().sum())
    sum_y = g["actual"].abs().sum()
    row["wape_pooled"] = g["mae"].sum() / sum_y if sum_y > 0 else np.nan
    row["coverage_80"] = g["covered_80"].mean()
    row["crps_skill_mean"] = g["crps_skill"].mean()
    return row


def summarize(res: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    """Mean and SD of each metric per group. Rows without an actual value are left out."""
    r = res.dropna(subset=["actual"])
    rows = []
    for key, g in r.groupby(by, dropna=False, sort=True):
        key = key if isinstance(key, tuple) else (key,)
        rows.append(dict(zip(by, key)) | _agg(g))
    return pd.DataFrame(rows)
