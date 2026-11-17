"""Common model interface.

To add a model (TimesFM, Seasonal Naive, ETS ...):
  1. subclass ForecastModel and implement forecast(task)
  2. register the class in drugfc/models/__init__.py (MODEL_REGISTRY)
  3. set `model: {name: <key>}` in the experiment config

forecast(task) returns a DataFrame with one row per (drug_id, month) in task.future:
  drug_id, month, q0.1, q0.2, ..., q0.9
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import pandas as pd

QUANTILES = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]


def qcol(tau: float) -> str:
    """0.1 -> 'q0.1', 0.5 -> 'q0.5'."""
    text = f"{float(tau):.2f}".rstrip("0").rstrip(".")
    return f"q{text}"


Q_COLS = [qcol(t) for t in QUANTILES]


def clean_quantiles(values: np.ndarray) -> np.ndarray:
    """Rows = forecasts, columns = quantile levels in ascending order.

    Demand cannot be negative, so values are clipped at 0, and quantiles are made
    non-decreasing across levels (no crossing).
    """
    v = np.clip(np.asarray(values, float), 0, None)
    return np.maximum.accumulate(v, axis=1)


class ForecastModel(ABC):
    name: str = "base"
    supports_covariates: bool = False

    @abstractmethod
    def forecast(self, task) -> pd.DataFrame:
        ...

    def info(self) -> dict:
        """Model and package versions for the run log."""
        return {"model": self.name}
