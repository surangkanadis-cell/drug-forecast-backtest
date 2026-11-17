"""Run one experiment:  python run.py --config configs/exp1_old_drugs.yaml

Outputs in output_dir:
  results_long.csv       one row = drug x origin x horizon x model x covariate_setting (x context)
  summary_per_drug.csv   mean / SD of each metric per drug
  summary_overall.csv    mean / SD per model, covariate setting and context; pooled WAPE; 80% coverage
  skipped_drugs.csv      drugs left out at each origin, with the reason
  run_log.json / run_log.txt

Web app: streamlit run app.py
"""
from __future__ import annotations

import argparse
import os
import sys

os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")

from drugfc.config import load_config
from drugfc.pipeline import SUMMARY_KEYS, run_experiment


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Rolling-origin backtest of drug demand forecasts")
    ap.add_argument("--config", required=True)
    ap.add_argument("--max-drugs", type=int, default=None, help="ใช้ยาแค่ N ตัวแรก (ทดสอบเร็ว)")
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    out = run_experiment(cfg, config_file=args.config, max_drugs=args.max_drugs)
    overall = out["overall"]
    print(overall[SUMMARY_KEYS + ["n_rows", "mae_mean", "wape_pooled", "mase_lag1_mean",
                                  "crps_mean", "coverage_80"]].to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
