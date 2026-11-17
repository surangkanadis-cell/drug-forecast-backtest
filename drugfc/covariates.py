"""Covariates as seen from a given origin.

Two kinds, set per column in the covariate config:
  past_only    : OPD, IPD, temperature ... Used only for months before the origin.
  known_future : calendar month, working days, holidays ... May be used for the target month.

Derived columns (computed, not read from the file):
  month_of_year : 1..12
  n_holidays    : public holidays in the month (needs holidays_file)
  working_days  : Mon-Fri days minus holidays that fall on Mon-Fri (needs holidays_file)

For n_holidays and working_days, a holiday counts only if it was announced before
the first day of the origin month (announced_date < origin). A holiday with no
announced_date is treated as known in advance.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import yaml

from .io import load_covariate_table, load_holidays

PAST_ONLY, KNOWN_FUTURE = "past_only", "known_future"
DERIVED_CALENDAR = "month_of_year"
DERIVED_HOLIDAY = ("n_holidays", "working_days")


@dataclass
class CovariateSource:
    kinds: dict                      # column -> past_only | known_future
    table: pd.DataFrame | None       # values read from the covariate file, indexed by month
    holidays: pd.DataFrame | None    # date, announced_date
    warnings: list = field(default_factory=list)

    @property
    def past_cols(self) -> list:
        return [c for c, k in self.kinds.items() if k == PAST_ONLY]

    @property
    def future_cols(self) -> list:
        return [c for c, k in self.kinds.items() if k == KNOWN_FUTURE]

    def _derived_holiday(self, c: str) -> bool:
        return c in DERIVED_HOLIDAY and self.holidays is not None

    def frame(self, months: pd.DatetimeIndex, origin: pd.Timestamp) -> pd.DataFrame:
        """Covariate values for `months`, as known at the start of `origin`.

        past_only columns are NaN for every month >= origin, so they can never leak.
        """
        out = pd.DataFrame(index=months)
        for c, kind in self.kinds.items():
            if c == DERIVED_CALENDAR and (self.table is None or c not in self.table.columns):
                vals = pd.Series(months.month.astype(float), index=months)
            elif self._derived_holiday(c):
                vals = holiday_counts(self.holidays, months, origin)[c]
            else:
                vals = self.table[c].reindex(months)
            if kind == PAST_ONLY:
                vals = vals.where(months < origin)
            out[c] = vals.astype(float).values
        return out


def holiday_counts(holidays: pd.DataFrame, months: pd.DatetimeIndex, origin: pd.Timestamp) -> pd.DataFrame:
    """n_holidays and working_days for each month, using only holidays announced before `origin`."""
    known = holidays[holidays["announced_date"].isna() | (holidays["announced_date"] < origin)]
    dates = pd.DatetimeIndex(known["date"])
    rows = []
    for m in months:
        end = m + pd.offsets.MonthBegin(1)
        in_month = dates[(dates >= m) & (dates < end)]
        weekdays = np.busday_count(m.date(), end.date())
        rows.append({
            "n_holidays": float(len(in_month)),
            "working_days": float(weekdays - int((in_month.dayofweek < 5).sum())),
        })
    return pd.DataFrame(rows, index=months)


def load_covariate_config(path: str | None) -> CovariateSource | None:
    """YAML:
        file: data/covariates.csv          # month + covariate columns
        holidays_file: data/holidays.csv   # optional
        columns:
          month_of_year: known_future
          working_days: known_future
          n_holidays: known_future
          opd: past_only
          ipd: past_only
          temperature: past_only
    """
    if not path:
        return None
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    kinds = dict(cfg.get("columns") or {})
    for c, k in kinds.items():
        if k not in (PAST_ONLY, KNOWN_FUTURE):
            raise ValueError(f"ชนิดตัวแปร {c} ต้องเป็น past_only หรือ known_future (ได้ {k!r})")
    holidays = load_holidays(cfg["holidays_file"]) if cfg.get("holidays_file") else None
    warnings = []
    from_file = [c for c in kinds
                 if not (c == DERIVED_CALENDAR)
                 and not (c in DERIVED_HOLIDAY and holidays is not None)]
    table = None
    if cfg.get("file"):
        header = pd.read_csv(cfg["file"], nrows=0, encoding="utf-8-sig").columns
        wanted = from_file + ([DERIVED_CALENDAR] if DERIVED_CALENDAR in kinds and DERIVED_CALENDAR in header else [])
        table = load_covariate_table(cfg["file"], wanted)
    elif from_file:
        raise ValueError(f"ต้องระบุ file สำหรับตัวแปร {from_file}")
    for c in DERIVED_HOLIDAY:
        if c in kinds and holidays is None:
            warnings.append(
                f"{c}: อ่านจากไฟล์ตัวแปรร่วมโดยตรง ไม่มี holidays_file จึงบังคับกฎ"
                " 'วันหยุดต้องประกาศก่อนวันที่ 1 ของเดือน origin' ไม่ได้")
    if holidays is not None and holidays["announced_date"].isna().all():
        warnings.append("holidays_file ไม่มี announced_date: ถือว่าทุกวันหยุดประกาศล่วงหน้าแล้ว")
    return CovariateSource(kinds=kinds, table=table, holidays=holidays, warnings=warnings)
