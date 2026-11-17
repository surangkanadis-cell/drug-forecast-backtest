"""Run one whole experiment from a prepared config. Used by run.py and the web app."""
from __future__ import annotations

import json
import os
import platform
import sys
import time
from datetime import datetime

import numpy as np
import pandas as pd

from .backtest import run_backtest, set_seed
from .config import load_drug_ids
from .covariates import load_covariate_config
from .io import DRUG, load_drugs, parse_month
from .metrics import summarize
from .models import get_model

SUMMARY_KEYS = ["model", "covariate_setting", "context", "horizon"]


def select_drugs(panel: pd.DataFrame, flt: dict, log: dict) -> list[str]:
    drugs = list(panel.columns)
    if flt.get("ids_file"):
        wanted = load_drug_ids(flt["ids_file"])
        missing = [d for d in wanted if d not in panel.columns]
        if missing:
            log["warnings"].append(f"ids_file มี {len(missing)} รหัสที่ไม่มีในไฟล์ยา: {missing[:10]}")
        drugs = [d for d in wanted if d in panel.columns]
    if flt.get("first_obs_by"):
        cutoff = parse_month(flt["first_obs_by"])
        first = panel[drugs].apply(lambda s: s.first_valid_index())
        drugs = [d for d in drugs if pd.notna(first[d]) and first[d] <= cutoff]
    log["drug_filter"] = {"n_in_file": panel.shape[1], "n_selected": len(drugs), **flt}
    return drugs


def fmt_month(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s).dt.strftime("%Y-%m")


def versions() -> dict:
    out = {"python": sys.version.split()[0], "platform": platform.platform(),
           "numpy": np.__version__, "pandas": pd.__version__}
    for mod in ("torch", "transformers", "chronos", "yaml"):
        try:
            m = __import__(mod)
            out[mod if mod != "chronos" else "chronos_forecasting"] = getattr(m, "__version__", "?")
        except ImportError:
            pass
    return out


def count_steps(cfg: dict) -> int:
    return sum(r["n_origins"] * len(r["use_covariates"]) for r in cfg["runs"])


def run_experiment(cfg: dict, *, config_file: str | None = None, max_drugs: int | None = None,
                   model=None, progress=print, on_step=None) -> dict:
    """cfg must come from prepare_config/load_config.

    model    : an already-loaded ForecastModel to reuse (else one is loaded here)
    progress : called with a text line for each step
    on_step  : called with (done, total) after each origin x covariate setting
    Returns {"out_dir", "results", "per_drug", "overall", "log", "model"}.
    """
    t_start = time.perf_counter()
    out_dir = cfg["output_dir"]
    os.makedirs(out_dir, exist_ok=True)
    log = {"config_file": os.path.abspath(config_file) if config_file else "(web app)",
           "config": cfg | {"runs": None},
           "started": datetime.now().isoformat(timespec="seconds"), "seed": cfg["seed"],
           "warnings": []}
    set_seed(cfg["seed"])

    panel = load_drugs(cfg["data"]["drug_file"])
    progress(f"ไฟล์ยา: {panel.shape[1]} ยา, {panel.index.min():%Y-%m} ถึง {panel.index.max():%Y-%m}")
    log["drug_data"] = {"file": os.path.abspath(cfg["data"]["drug_file"]),
                        "n_drugs": panel.shape[1], "first_month": f"{panel.index.min():%Y-%m}",
                        "last_month": f"{panel.index.max():%Y-%m}",
                        "n_observed_cells": int(panel.notna().sum().sum())}
    neg = int((panel < 0).sum().sum())
    if neg:
        log["warnings"].append(f"มีปริมาณติดลบ {neg} ค่า (ใช้ตามจริง ไม่ได้แก้)")

    covariates = load_covariate_config(cfg["data"].get("covariate_config"))
    if covariates is not None:
        log["covariates"] = covariates.kinds
        log["warnings"] += covariates.warnings
    if any(True in r["use_covariates"] for r in cfg["runs"]) and covariates is None:
        raise ValueError("use_covariates มี true แต่ไม่ได้ระบุ data.covariate_config")

    drugs = select_drugs(panel, cfg["drug_filter"], log)
    if max_drugs:
        drugs = drugs[:max_drugs]
        log["warnings"].append(f"จำกัดจำนวนยา {max_drugs} ตัว: ผลนี้ใช้ทดสอบเท่านั้น")
    if not drugs:
        raise ValueError("ไม่มียาที่ผ่านเงื่อนไขการเลือกยา")
    progress(f"ยาที่เลือก: {len(drugs)}")

    t0 = time.perf_counter()
    if model is None:
        model = get_model(cfg["model"]["name"], **cfg["model"]["params"])
    log["model_load_seconds"] = round(time.perf_counter() - t0, 1)
    log["model"] = model.info()
    progress(f"โมเดล {model.name} พร้อม ({log['model_load_seconds']} s)")

    total, done = count_steps(cfg), 0

    def step_progress(msg):
        nonlocal done
        progress(msg)
        if msg.lstrip().startswith("origin"):
            done += 1
            if on_step:
                on_step(done, total)

    all_res, all_skip, all_time, run_logs = [], [], [], []
    for run in cfg["runs"]:
        cs = run["context_start"]
        progress(f"รอบ context={run['context']} "
                 f"context_start={'ทั้งหมด' if cs is None else f'{cs:%Y-%m}'} "
                 f"min_history={run['min_history']}")
        t_run = time.perf_counter()
        res, skipped, timing = run_backtest(
            panel, model, start_origin=run["start_origin"], n_origins=run["n_origins"],
            horizon=run["horizon"], step=run["step"], min_history=run["min_history"],
            context_start=run["context_start"], covariates=covariates,
            use_covariates=run["use_covariates"], drug_ids=drugs,
            context_label=run["context"], seed=cfg["seed"], progress=step_progress)
        all_res.append(res)
        all_skip.append(skipped)
        all_time.append(timing)
        reasons = skipped.groupby("reason").size().to_dict() if len(skipped) else {}
        run_logs.append({
            **{k: (f"{v:%Y-%m}" if isinstance(v, pd.Timestamp) else v) for k, v in run.items()},
            "seconds": round(time.perf_counter() - t_run, 1),
            "n_result_rows": len(res),
            "n_rows_without_actual": int(res["actual"].isna().sum()),
            "skipped_drug_origin_pairs": len(skipped),
            "skipped_by_reason": reasons,
        })

    res = pd.concat(all_res, ignore_index=True)
    skipped = pd.concat(all_skip, ignore_index=True) if any(len(s) for s in all_skip) else pd.DataFrame()
    timing = pd.concat(all_time, ignore_index=True) if any(len(t) for t in all_time) else pd.DataFrame()

    per_drug = summarize(res, [DRUG] + SUMMARY_KEYS)
    overall = summarize(res, SUMMARY_KEYS)

    out = res.copy()
    out["origin"] = fmt_month(out["origin"])
    out["target_month"] = fmt_month(out["target_month"])
    out.to_csv(os.path.join(out_dir, "results_long.csv"), index=False, encoding="utf-8-sig")
    per_drug.to_csv(os.path.join(out_dir, "summary_per_drug.csv"), index=False, encoding="utf-8-sig")
    overall.to_csv(os.path.join(out_dir, "summary_overall.csv"), index=False, encoding="utf-8-sig")
    skip_path = os.path.join(out_dir, "skipped_drugs.csv")
    if len(skipped):
        skipped["origin"] = fmt_month(skipped["origin"])
        skipped.to_csv(skip_path, index=False, encoding="utf-8-sig")
    elif os.path.exists(skip_path):
        os.remove(skip_path)
    if len(timing):
        timing["origin"] = fmt_month(timing["origin"])

    log["runs"] = run_logs
    log["forecast_timing"] = timing.to_dict("records")
    log["versions"] = versions()
    log["finished"] = datetime.now().isoformat(timespec="seconds")
    log["total_seconds"] = round(time.perf_counter() - t_start, 1)
    with open(os.path.join(out_dir, "run_log.json"), "w", encoding="utf-8") as f:
        json.dump(log, f, ensure_ascii=False, indent=2, default=str)
    write_text_log(os.path.join(out_dir, "run_log.txt"), log)
    progress(f"เสร็จใน {log['total_seconds']} s -> {out_dir}")
    return {"out_dir": out_dir, "results": out, "per_drug": per_drug, "overall": overall,
            "log": log, "model": model}


def write_text_log(path: str, log: dict) -> None:
    lines = [
        f"config      : {log['config_file']}",
        f"started     : {log['started']}",
        f"finished    : {log['finished']}",
        f"total time  : {log['total_seconds']} s (model load {log['model_load_seconds']} s)",
        f"seed        : {log['seed']}",
        "model       : " + ", ".join(f"{k}={v}" for k, v in log["model"].items()),
        "versions    : " + ", ".join(f"{k}={v}" for k, v in log["versions"].items()),
        f"drugs       : {log['drug_filter']}",
        "",
    ]
    for r in log["runs"]:
        lines.append(f"run context={r['context']} context_start={r['context_start']} "
                     f"start_origin={r['start_origin']} n_origins={r['n_origins']} "
                     f"min_history={r['min_history']} use_covariates={r['use_covariates']}")
        lines.append(f"  time {r['seconds']} s, rows {r['n_result_rows']}, "
                     f"rows without actual {r['n_rows_without_actual']}")
        lines.append(f"  skipped (drug x origin) {r['skipped_drug_origin_pairs']}: {r['skipped_by_reason']}")
    if log["warnings"]:
        lines += ["", "warnings:"] + [f"  - {w}" for w in log["warnings"]]
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
