"""No information from the origin month onward may reach the model."""
import numpy as np
import pandas as pd
import pytest

from drugfc.backtest import run_backtest
from drugfc.covariates import CovariateSource
from drugfc.io import DRUG, MONTH, TARGET
from drugfc.models.base import Q_COLS, ForecastModel
from drugfc.tasks import build_task

SENTINEL = 999_999.0
MONTHS = pd.date_range("2023-01-01", "2026-12-01", freq="MS")


def make_panel():
    rng = np.random.default_rng(0)
    p = pd.DataFrame(rng.integers(50, 150, size=(len(MONTHS), 3)).astype(float),
                     index=MONTHS, columns=["A", "B", "NEW"])
    p.loc[:"2025-06-01", "NEW"] = np.nan          # NEW starts 2025-07
    p.loc["2024-03-01", "A"] = np.nan             # a gap inside the history
    return p


def make_covariates():
    table = pd.DataFrame({"opd": np.arange(len(MONTHS), dtype=float) + 1000},
                         index=MONTHS)
    holidays = pd.DataFrame({
        "date": pd.to_datetime(["2025-10-13", "2025-10-23", "2025-10-24"]),
        # 2025-10-24 announced on 2025-10-05, i.e. after the 2025-10 origin
        "announced_date": pd.to_datetime([None, "2025-01-01", "2025-10-05"]),
        "name": ["a", "b", "c"]})
    return CovariateSource(kinds={"month_of_year": "known_future", "n_holidays": "known_future",
                                  "working_days": "known_future", "opd": "past_only"},
                           table=table, holidays=holidays)


def poison(panel, cov, origin):
    """Overwrite every value from `origin` onward with a sentinel."""
    p = panel.copy()
    p.loc[p.index >= origin] = SENTINEL
    c = CovariateSource(kinds=cov.kinds, table=cov.table.copy(), holidays=cov.holidays)
    c.table.loc[c.table.index >= origin] = SENTINEL
    return p, c


@pytest.mark.parametrize("origin", ["2025-10-01", "2026-01-01", "2026-06-01"])
@pytest.mark.parametrize("use_cov", [False, True])
def test_context_has_nothing_from_origin_onward(origin, use_cov):
    origin = pd.Timestamp(origin)
    panel, cov = poison(make_panel(), make_covariates(), origin)
    task, meta, _ = build_task(panel, origin, horizon=1, min_history=1,
                               covariates=cov, use_covariates=use_cov)
    assert task.context[MONTH].max() < origin
    assert not (task.context.select_dtypes("number") == SENTINEL).any().any()
    assert not (task.future.select_dtypes("number") == SENTINEL).any().any()
    assert TARGET not in task.future.columns
    assert "opd" not in task.future.columns
    assert set(task.future[MONTH]) == {origin}
    # scales and means come from history only
    assert (meta[["scale_lag1", "hist_mean"]].fillna(0) < SENTINEL / 10).all().all()


def test_past_only_covariate_is_nan_from_origin():
    cov = make_covariates()
    origin = pd.Timestamp("2025-10-01")
    f = cov.frame(pd.date_range("2025-08-01", "2025-11-01", freq="MS"), origin)
    assert f["opd"].notna().tolist() == [True, True, False, False]


def test_holiday_announced_after_origin_not_counted():
    cov = make_covariates()
    m = pd.DatetimeIndex([pd.Timestamp("2025-10-01")])
    at_origin = cov.frame(m, pd.Timestamp("2025-10-01"))
    later = cov.frame(m, pd.Timestamp("2025-11-01"))
    assert at_origin["n_holidays"].iloc[0] == 2      # 10-24 announced 10-05 -> unknown on 10-01
    assert later["n_holidays"].iloc[0] == 3
    # Oct 2025 has 23 weekdays; 10-13 (Mon), 10-23 (Thu), 10-24 (Fri)
    assert at_origin["working_days"].iloc[0] == 21
    assert later["working_days"].iloc[0] == 20


class SpyModel(ForecastModel):
    """Forecast = mean of the visible history; records every task it sees."""
    name = "spy"

    def __init__(self):
        self.tasks = []

    def forecast(self, task):
        self.tasks.append(task)
        means = task.context.groupby(DRUG)[TARGET].mean()
        out = task.future[[DRUG, MONTH]].copy()
        for i, q in enumerate(Q_COLS):
            out[q] = out[DRUG].map(means).to_numpy() * (0.6 + 0.1 * i)
        return out


def test_backtest_forecasts_ignore_future_values():
    """Changing every value at or after an origin must not change that origin's forecast."""
    panel, cov = make_panel(), make_covariates()
    kw = dict(start_origin="2025-10-01", n_origins=1, min_history=1,
              covariates=cov, use_covariates=[False, True])
    clean, _, _ = run_backtest(panel, SpyModel(), progress=lambda *_: None, **kw)
    p2, c2 = poison(panel, cov, pd.Timestamp("2025-10-01"))
    spy = SpyModel()
    dirty, _, _ = run_backtest(p2, spy, progress=lambda *_: None, **kw)
    pd.testing.assert_frame_equal(clean[Q_COLS], dirty[Q_COLS])
    for t in spy.tasks:
        assert t.context[MONTH].max() < t.origin
        assert not (t.context.select_dtypes("number") == SENTINEL).any().any()


def test_rolling_origins_expand_and_skip_short_history():
    panel = make_panel()
    spy = SpyModel()
    res, skipped, _ = run_backtest(panel, spy, start_origin="2025-10-01", n_origins=3,
                                   min_history=4, progress=lambda *_: None)
    assert [t.origin for t in spy.tasks] == list(pd.date_range("2025-10-01", periods=3, freq="MS"))
    for t in spy.tasks:
        assert t.context[MONTH].max() == t.origin - pd.DateOffset(months=1)
    # NEW has 3 months (Jul-Sep 2025) before 2025-10 -> skipped there, kept from 2025-11
    first = skipped[skipped["origin"] == pd.Timestamp("2025-10-01")]
    assert first[DRUG].tolist() == ["NEW"]
    assert set(res.loc[res["origin"] == pd.Timestamp("2025-11-01"), DRUG]) == {"A", "B", "NEW"}


def test_context_start_cuts_history_and_missing_stays_nan():
    panel = make_panel()
    task, meta, _ = build_task(panel, pd.Timestamp("2025-10-01"), min_history=1,
                               context_start=pd.Timestamp("2024-01-01"))
    a = task.context[task.context[DRUG] == "A"].set_index(MONTH)[TARGET]
    assert a.index.min() == pd.Timestamp("2024-01-01")
    assert np.isnan(a.loc["2024-03-01"])          # gap kept as NaN, not 0
    assert meta.set_index(DRUG).loc["A", "history_months"] == 20  # 21 months, 1 missing
