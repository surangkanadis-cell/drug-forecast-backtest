"""Rolling-origin (expanding window) backtest.

For k = 0 .. n_origins-1: origin = start_origin + k*step months.
At each origin all eligible drugs are forecast in one batch call to the model.
"""
from __future__ import annotations

import random
import time

import numpy as np
import pandas as pd

from .io import DRUG, MONTH
from .metrics import add_row_metrics
from .models.base import Q_COLS, ForecastModel
from .tasks import build_task

RESULT_COLUMNS = (
    [DRUG, "origin", "target_month", "horizon", "model", "covariate_setting",
     "history_months", "context", "actual"] + Q_COLS +
    ["mae", "wape", "mase_lag1", "mase_m12", "crps", "crps_scaled", "covered_80", "crps_skill"]
)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
    except ImportError:
        pass


def origin_months(start_origin, n_origins: int, step: int) -> list[pd.Timestamp]:
    start = pd.Timestamp(start_origin)
    return [start + pd.DateOffset(months=k * step) for k in range(n_origins)]


def cov_label(use_cov: bool) -> str:
    return "with_cov" if use_cov else "no_cov"


def _actual(panel: pd.DataFrame, drug, month) -> float:
    if month in panel.index and drug in panel.columns:
        return float(panel.at[month, drug])
    return np.nan


def run_backtest(panel: pd.DataFrame, model: ForecastModel, *, start_origin, n_origins: int,
                 horizon: int = 1, step: int = 1, min_history: int = 1, context_start=None,
                 covariates=None, use_covariates=(False,), drug_ids=None, context_label="all",
                 seed: int = 42, progress=print):
    """Returns (results DataFrame, skipped DataFrame, timing DataFrame)."""
    results, skipped, timing = [], [], []
    for origin in origin_months(start_origin, n_origins, step):
        for i, use_cov in enumerate(use_covariates):
            task, meta, sk = build_task(panel, origin, horizon=horizon, min_history=min_history,
                                        context_start=context_start, covariates=covariates,
                                        use_covariates=use_cov, drug_ids=drug_ids)
            if i == 0:  # eligibility does not depend on covariates; record it once per origin
                skipped += [s | {"origin": origin, "context": context_label} for s in sk]
            if task is None:
                progress(f"  origin {origin:%Y-%m} [{cov_label(use_cov)}]: ไม่มียาที่เข้าเงื่อนไข")
                continue
            set_seed(seed)
            t0 = time.perf_counter()
            pred = model.forecast(task)
            secs = time.perf_counter() - t0
            _check_prediction(pred, task)
            pred = pred.merge(meta, on=DRUG, how="left")
            pred["origin"] = origin
            pred["target_month"] = pd.to_datetime(pred[MONTH])
            pred["horizon"] = ((pred["target_month"].dt.year - origin.year) * 12
                               + pred["target_month"].dt.month - origin.month + 1)
            pred["model"] = model.name
            pred["covariate_setting"] = cov_label(use_cov)
            pred["context"] = context_label
            pred["actual"] = [_actual(panel, d, m) for d, m in zip(pred[DRUG], pred["target_month"])]
            results.append(add_row_metrics(pred))
            n = len(task.drug_ids)
            timing.append({"origin": origin, "covariate_setting": cov_label(use_cov),
                           "context": context_label, "n_drugs": n, "seconds": round(secs, 2)})
            progress(f"  origin {origin:%Y-%m} [{cov_label(use_cov)}]: {n} ยา, {secs:.1f} s, "
                     f"ข้าม {len(sk)} ยา")
    res = pd.concat(results, ignore_index=True)[RESULT_COLUMNS] if results else pd.DataFrame(columns=RESULT_COLUMNS)
    return res, pd.DataFrame(skipped), pd.DataFrame(timing)


def _check_prediction(pred: pd.DataFrame, task) -> None:
    expected = len(task.future)
    if len(pred) != expected:
        raise RuntimeError(f"โมเดลคืน {len(pred)} แถว แต่ต้องการ {expected}")
    keys = set(zip(pred[DRUG].astype(str), pd.to_datetime(pred[MONTH])))
    want = set(zip(task.future[DRUG].astype(str), pd.to_datetime(task.future[MONTH])))
    if keys != want:
        raise RuntimeError("drug_id/month ที่โมเดลคืนไม่ตรงกับที่ขอ")
    if pred[Q_COLS].isna().any().any():
        raise RuntimeError("โมเดลคืนค่าควอนไทล์เป็น NaN")
