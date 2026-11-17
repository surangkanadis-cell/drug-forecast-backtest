"""Generate SYNTHETIC data in examples/data/ for a smoke test of the pipeline.

Not hospital data. Only used to check that run.py works end to end:
    python examples/make_example_data.py
    python run.py --config examples/smoke.yaml
"""
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path(__file__).parent / "data"


def be(ts: pd.Timestamp) -> str:
    return f"{ts.year + 543}-{ts.month:02d}"


def main():
    OUT.mkdir(exist_ok=True)
    rng = np.random.default_rng(1)
    months = pd.date_range("2020-10-01", "2026-09-01", freq="MS")
    rows = []
    for i in range(12):
        start = months[0] if i < 8 else months[rng.integers(20, 60)]
        base = rng.uniform(50, 500)
        for t, m in enumerate(months):
            if m < start or rng.random() < 0.03:       # not started yet / missing month
                continue
            season = 1 + 0.2 * np.sin(2 * np.pi * (m.month - 1) / 12)
            rows.append({"drug_id": f"D{i:03d}", "month": be(m),
                         "quantity": round(max(0, base * season * (1 + 0.004 * t) + rng.normal(0, base * 0.1)))})
    pd.DataFrame(rows).to_csv(OUT / "drugs.csv", index=False)

    cov_months = pd.date_range("2020-10-01", "2027-06-01", freq="MS")
    cov = pd.DataFrame({
        "month": [be(m) for m in cov_months],
        "opd": rng.integers(20000, 30000, len(cov_months)),
        "ipd": rng.integers(1500, 2500, len(cov_months)),
        "temperature": np.round(28 + 3 * np.sin(2 * np.pi * (cov_months.month - 4) / 12), 1),
    })
    cov.loc[cov_months >= "2026-10-01", ["opd", "ipd", "temperature"]] = np.nan
    cov.to_csv(OUT / "covariates.csv", index=False)

    hol = []
    for y in range(2020, 2028):
        for md, name in [("01-01", "new year"), ("04-13", "songkran"), ("04-14", "songkran"),
                         ("04-15", "songkran"), ("12-05", "father's day"), ("12-31", "new year eve")]:
            hol.append({"date": f"{y + 543}-{md}", "announced_date": "", "name": name})
    hol.append({"date": "2568-10-24", "announced_date": "2568-10-05", "name": "special (late announcement)"})
    pd.DataFrame(hol).to_csv(OUT / "holidays.csv", index=False)
    pd.DataFrame({"drug_id": ["D008", "D009", "D010", "D011"]}).to_csv(OUT / "new_drugs.csv", index=False)
    print(f"wrote synthetic data to {OUT}")


if __name__ == "__main__":
    main()
