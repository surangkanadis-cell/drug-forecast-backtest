"""Experiment config (YAML).

    name: exp1_old_drugs
    seed: 42
    output_dir: outputs/exp1_old_drugs
    data:
      drug_file: data/drugs.csv
      covariate_config: configs/covariates.yaml
    drug_filter:                 # optional
      ids_file: null             # CSV with a drug_id column (or one id per line)
      first_obs_by: null         # keep drugs whose first observed month is <= this month
    model:
      name: chronos2
      params: {model_id: amazon/chronos-2, device: cpu, torch_dtype: float32}
    backtest:
      start_origin: 2025-10
      n_origins: 12
      horizon: 1
      step: 1
      min_history: 13
      context_start: null
      use_covariates: [false, true]
    runs:                        # optional; each entry overrides backtest keys
      - {context: 1, context_start: 2024-10, min_history: 12}

Months may be B.E. or C.E. Relative paths are resolved from the working directory.
"""
from __future__ import annotations

import copy

import pandas as pd
import yaml

from .io import parse_month

BACKTEST_DEFAULTS = {"horizon": 1, "step": 1, "min_history": 1, "context_start": None,
                     "use_covariates": [False]}
BACKTEST_KEYS = {"start_origin", "n_origins", "horizon", "step", "min_history",
                 "context_start", "use_covariates"}


def load_config(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    for key in ("data", "model", "backtest"):
        if key not in cfg:
            raise ValueError(f"config ขาดหัวข้อ {key}")
    cfg.setdefault("name", path)
    cfg.setdefault("seed", 42)
    cfg.setdefault("output_dir", f"outputs/{cfg['name']}")
    cfg.setdefault("drug_filter", {})
    cfg["model"].setdefault("params", {})
    cfg["runs"] = expand_runs(cfg)
    return cfg


def expand_runs(cfg: dict) -> list[dict]:
    """One dict per run with every backtest key resolved."""
    base = BACKTEST_DEFAULTS | cfg["backtest"]
    runs = cfg.get("runs") or [{}]
    out = []
    for r in runs:
        unknown = set(r) - BACKTEST_KEYS - {"context"}
        if unknown:
            raise ValueError(f"runs มีคีย์ที่ไม่รู้จัก: {sorted(unknown)}")
        run = copy.deepcopy(base) | r
        run.setdefault("context", "all")
        run["start_origin"] = parse_month(run["start_origin"])
        run["context_start"] = parse_month(run["context_start"]) if run["context_start"] else None
        uc = run["use_covariates"]
        run["use_covariates"] = [bool(x) for x in (uc if isinstance(uc, list) else [uc])]
        for k in ("n_origins", "horizon", "step", "min_history"):
            run[k] = int(run[k])
            if run[k] < 1:
                raise ValueError(f"{k} ต้อง >= 1")
        out.append(run)
    return out


def load_drug_ids(path: str) -> list[str]:
    df = pd.read_csv(path, dtype=str, encoding="utf-8-sig")
    if "drug_id" in df.columns:
        col = df["drug_id"]
    else:  # no header: one id per line
        col = pd.read_csv(path, dtype=str, encoding="utf-8-sig", header=None).iloc[:, 0]
    return [s.strip() for s in col.dropna().astype(str) if s.strip()]
