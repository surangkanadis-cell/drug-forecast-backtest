import numpy as np
import pandas as pd
import pytest

from drugfc.io import load_drugs, parse_month
from drugfc.metrics import add_row_metrics, pinball
from drugfc.models.base import Q_COLS, QUANTILES, clean_quantiles
from drugfc.tasks import naive_scale


@pytest.mark.parametrize("raw,expected", [
    ("2568-10", "2025-10-01"), ("2568/1", "2025-01-01"), ("10/2568", "2025-10-01"),
    ("2025-10", "2025-10-01"), ("2025-10-01", "2025-10-01"), ("2563-10-01", "2020-10-01"),
])
def test_parse_month_buddhist_era(raw, expected):
    assert parse_month(raw) == pd.Timestamp(expected)


def test_load_drugs_keeps_missing_as_nan(tmp_path):
    f = tmp_path / "d.csv"
    f.write_text("drug_id,month,quantity\nA,2568-08,10\nA,2568-10,\nB,2568-10,0\n", encoding="utf-8")
    p = load_drugs(str(f))
    assert list(p.index) == list(pd.date_range("2025-08-01", "2025-10-01", freq="MS"))
    assert np.isnan(p.loc["2025-09-01", "A"]) and np.isnan(p.loc["2025-10-01", "A"])
    assert np.isnan(p.loc["2025-08-01", "B"])     # before the drug started
    assert p.loc["2025-10-01", "B"] == 0          # a real zero stays zero


def test_load_drugs_rejects_duplicates(tmp_path):
    f = tmp_path / "d.csv"
    f.write_text("drug_id,month,quantity\nA,2568-08,10\nA,2025-08,3\n", encoding="utf-8")
    with pytest.raises(ValueError):
        load_drugs(str(f))


def test_naive_scale_uses_adjacent_months_only():
    s = pd.Series([10, 12, np.nan, 20, 21], dtype=float)
    assert naive_scale(s, 1) == pytest.approx((2 + 1) / 2)
    assert np.isnan(naive_scale(s, 12))            # fewer than 13 months
    assert np.isnan(naive_scale(pd.Series([5.0, 5.0, 5.0]), 1))  # zero scale


def test_row_metrics_known_values():
    q = np.array([[60, 70, 80, 90, 100, 110, 120, 130, 140]], float)
    df = pd.DataFrame(q, columns=Q_COLS)
    df["actual"] = 150.0
    df["scale_lag1"], df["scale_m12"], df["hist_mean"] = 25.0, np.nan, 100.0
    r = add_row_metrics(df).iloc[0]
    assert r["mae"] == 50 and r["wape"] == pytest.approx(50 / 150)
    assert r["mase_lag1"] == 2 and np.isnan(r["mase_m12"])
    expected = 2 * np.mean([t * (150 - v) for t, v in zip(QUANTILES, q[0])])
    assert r["crps"] == pytest.approx(expected)
    assert r["crps_scaled"] == pytest.approx(expected / 100)
    assert r["covered_80"] == 0 and np.isnan(r["crps_skill"])


def test_pinball_and_quantile_cleaning():
    assert pinball(10, 12, 0.1) == pytest.approx(0.9 * 2)
    assert pinball(12, 10, 0.1) == pytest.approx(0.1 * 2)
    c = clean_quantiles(np.array([[-1, 5, 4, 6, 7, 8, 9, 10, 11]], float))
    assert c[0, 0] == 0 and (np.diff(c[0]) >= 0).all()


def test_zero_history_mean_gives_nan_scaled_crps():
    df = pd.DataFrame([[0.0] * 9], columns=Q_COLS)
    df["actual"], df["scale_lag1"], df["scale_m12"], df["hist_mean"] = 3.0, np.nan, np.nan, np.nan
    assert np.isnan(add_row_metrics(df)["crps_scaled"].iloc[0])
