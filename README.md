# Drug demand forecasting: Chronos-2 zero-shot (rolling origin)

Forecasts monthly drug quantities with `amazon/chronos-2` (no fine-tuning, CPU only)
and evaluates them with a rolling-origin (expanding window) backtest.

## Installation (Python 3.11, CPU)

```bash
python -m venv .venv
.venv\Scripts\activate                # Windows  (Linux/mac: source .venv/bin/activate)
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```
On first run the model is downloaded from Hugging Face (about 480 MB) and cached.

## Data to place in `data/`

| File | Columns | Notes |
|---|---|---|
| `drugs.csv` | `drug_id, month, quantity` | month as `2568-10` (B.E.) or `2025-10`; months with no data have an empty quantity or no row at all → NaN (never 0) |
| `covariates.csv` | `month, opd, ipd, temperature, ...` | the covariate type for each column is set in `configs/covariates.yaml` |
| `holidays.csv` | `date, announced_date, name` | date as `2568-10-23`; an empty `announced_date` means a holiday known in advance |
| `new_drugs.csv` | `drug_id` | the 30 new drugs for experiment 3 |

`month_of_year`, `n_holidays` and `working_days` are computed automatically.
`n_holidays` and `working_days` count only holidays announced **before the 1st day of the origin month**.
`working_days` = Mon–Fri days − holidays that fall on Mon–Fri.

## Running

```bash
python run.py --config configs/exp1_old_drugs.yaml       # Experiment 1: existing drugs, with and without covariates
python run.py --config configs/exp2_context_length.yaml  # Experiment 2: context = 1/3/5 years
python run.py --config configs/exp3_new_drugs.yaml       # Experiment 3: new drugs
python run.py --config configs/exp1_old_drugs.yaml --max-drugs 20   # quick test
pytest                                                    # unit tests (including the leakage tests)
```

Trying it on synthetic data (not hospital data):
```bash
python examples/make_example_data.py
python run.py --config examples/smoke.yaml
```

## Outputs (`outputs/<name>/`)

- `results_long.csv`: 1 row = drug × origin × horizon × model × covariate_setting (× context)
- `summary_per_drug.csv`, `summary_overall.csv`: `<metric>_mean`, `<metric>_sd`, `<metric>_n`,
  plus `wape_pooled` (= Σ|e| / Σ|y|) and `coverage_80`
- `skipped_drugs.csv`: drugs skipped at each origin, with the reason
- `run_log.txt` / `run_log.json`: time taken, seed, model version (HF commit) and packages, warnings

Months in the output files are in C.E. (`2025-10`).

## Key definitions

- Origin O: the model sees drug data up to month O−1 only; past_only covariates up to O−1 only
- `history_months` = number of **months with data** (non-NaN) from the first month the drug appears (or `context_start`) to O−1;
  drugs with `history_months < min_history` are skipped
- Point forecast = q0.5
- `mase_lag1`, `mase_m12`: denominator = mean |y_t − y_{t−lag}| over adjacent month pairs both observed, using data before O only (within `context_start`);
  NaN if there is no pair or the denominator = 0
- `crps` = 2 × mean pinball loss over q0.1…q0.9 (the quantile approximation of CRPS)
- `crps_scaled` = crps ÷ the drug's mean before O (NaN if the mean = 0)
- `covered_80` = 1 if q0.1 ≤ actual ≤ q0.9
- `crps_skill`: left empty for now (to be filled once Seasonal Naïve is added)
- Quantiles are clipped at ≥ 0 and adjusted to be non-decreasing (no crossing)

## Adding a model

Write a subclass of `drugfc.models.base.ForecastModel` with a `forecast(task)` method that returns
`drug_id, month, q0.1..q0.9`, register it in `MODEL_REGISTRY` in `drugfc/models/__init__.py`,
and set `model.name` in the config.

## Structure

```
run.py                      entry point
drugfc/io.py                file reading, B.E. → C.E. conversion
drugfc/covariates.py        past_only / known_future, holidays as known at the origin
drugfc/tasks.py             builds the input at each origin + leakage check
drugfc/backtest.py          rolling origin, batch forecasting per origin
drugfc/metrics.py           metrics and summaries
drugfc/models/              ForecastModel interface + Chronos-2
configs/                    experiment configs
tests/                      unit tests
```
