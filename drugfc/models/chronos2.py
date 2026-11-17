"""Chronos-2 (amazon/chronos-2), zero-shot, CPU.

API used, from chronos-forecasting 2.x (README "Forecasting" example and
Chronos2Pipeline.predict_df docstring):

    pipeline = Chronos2Pipeline.from_pretrained("amazon/chronos-2", device_map="cpu")
    pred_df = pipeline.predict_df(
        context_df, future_df=future_df, prediction_length=h,
        quantile_levels=[...], id_column=..., timestamp_column=..., target=..., freq="MS")

* Columns in context_df other than id/timestamp/target are past-only covariates,
  unless they also appear in future_df, in which case they are known-future covariates.
* Timestamps must be regular (no gaps); missing values are written as NaN rows.
* The output has one column per quantile level, named str(level), e.g. "0.1".
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..io import DRUG, MONTH, TARGET
from .base import Q_COLS, QUANTILES, ForecastModel, clean_quantiles


class Chronos2Model(ForecastModel):
    name = "Chronos-2"
    supports_covariates = True

    def __init__(self, model_id: str = "amazon/chronos-2", device: str = "cpu",
                 torch_dtype: str = "float32", batch_size: int = 256,
                 cross_learning: bool = False, revision: str | None = None):
        import torch
        from chronos import Chronos2Pipeline

        self.model_id, self.device = model_id, device
        self.batch_size, self.cross_learning = batch_size, cross_learning
        self.torch_dtype = torch_dtype
        kwargs = {"device_map": device, "dtype": getattr(torch, torch_dtype)}
        if revision:
            kwargs["revision"] = revision
        self.pipeline = Chronos2Pipeline.from_pretrained(model_id, **kwargs)
        self.pipeline.model.eval()

    def forecast(self, task) -> pd.DataFrame:
        cov = list(task.past_cols) + list(task.future_cols)
        ctx = (task.context[[DRUG, MONTH, TARGET] + cov]
               .rename(columns={DRUG: "item_id", MONTH: "timestamp"})
               .sort_values(["item_id", "timestamp"]))
        future_df = None
        if task.future_cols:
            future_df = (task.future[[DRUG, MONTH] + list(task.future_cols)]
                         .rename(columns={DRUG: "item_id", MONTH: "timestamp"})
                         .sort_values(["item_id", "timestamp"]))
        pred = self.pipeline.predict_df(
            ctx, future_df=future_df, prediction_length=task.horizon,
            quantile_levels=QUANTILES, id_column="item_id", timestamp_column="timestamp",
            target=TARGET, freq="MS", batch_size=self.batch_size,
            cross_learning=self.cross_learning)
        qvals = clean_quantiles(pred[[str(t) for t in QUANTILES]].to_numpy(float))
        out = pd.DataFrame(qvals, columns=Q_COLS)
        out.insert(0, MONTH, pd.to_datetime(pred["timestamp"]).to_numpy())
        out.insert(0, DRUG, pred["item_id"].astype(str).to_numpy())
        return out

    def info(self) -> dict:
        import chronos
        import torch
        import transformers
        cfg = self.pipeline.model.config
        return {
            "model": self.name,
            "model_id": self.model_id,
            "model_commit": getattr(cfg, "_commit_hash", None),
            "device": self.device,
            "torch_dtype": self.torch_dtype,
            "batch_size": self.batch_size,
            "cross_learning": self.cross_learning,
            "chronos_forecasting": getattr(chronos, "__version__", None),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
        }
