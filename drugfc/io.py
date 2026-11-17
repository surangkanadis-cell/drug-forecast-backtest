"""Read the drug file and the covariate file.

Months may be written in the Buddhist era (2568-10) or the Common era (2025-10).
Any year >= 2400 is treated as B.E. and converted by subtracting 543.
Internally every month is a pandas Timestamp on the first day of the month (C.E.).

Missing quantities stay NaN. They are never replaced by 0.
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

BE_OFFSET = 543
BE_THRESHOLD = 2400

DRUG, MONTH, TARGET = "drug_id", "month", "quantity"

_YM = re.compile(r"^\s*(\d{4})[-/.](\d{1,2})(?:[-/.](\d{1,2}))?(?:[ T].*)?\s*$")
_MY = re.compile(r"^\s*(\d{1,2})[-/.](\d{4})\s*$")


def _year_ce(year: int) -> int:
    return year - BE_OFFSET if year >= BE_THRESHOLD else year


def parse_month(value) -> pd.Timestamp:
    """'2568-10', '2568/10', '10/2568', '2025-10', '2025-10-01' -> Timestamp('2025-10-01')."""
    if isinstance(value, pd.Timestamp):
        return pd.Timestamp(year=_year_ce(value.year), month=value.month, day=1)
    s = str(value).strip()
    m = _YM.match(s)
    if m:
        year, month = int(m.group(1)), int(m.group(2))
    else:
        m = _MY.match(s)
        if not m:
            raise ValueError(f"อ่านค่าเดือนไม่ได้: {value!r} (ใช้รูปแบบ YYYY-MM เช่น 2568-10 หรือ 2025-10)")
        month, year = int(m.group(1)), int(m.group(2))
    if not 1 <= month <= 12:
        raise ValueError(f"เดือนไม่ถูกต้อง: {value!r}")
    return pd.Timestamp(year=_year_ce(year), month=month, day=1)


def parse_date(value) -> pd.Timestamp:
    """A full date. B.E. years are converted. Empty -> NaT."""
    if value is None or (isinstance(value, float) and np.isnan(value)) or str(value).strip() == "":
        return pd.NaT
    m = _YM.match(str(value))
    if not m or m.group(3) is None:
        raise ValueError(f"อ่านวันที่ไม่ได้: {value!r} (ใช้รูปแบบ YYYY-MM-DD)")
    return pd.Timestamp(year=_year_ce(int(m.group(1))), month=int(m.group(2)), day=int(m.group(3)))


def parse_months(values: pd.Series) -> pd.Series:
    uniq = pd.unique(values.astype(str))
    lookup = {u: parse_month(u) for u in uniq}
    return values.astype(str).map(lookup)


def _numeric(col: pd.Series, name: str) -> pd.Series:
    raw = col.astype(str).str.strip().str.replace(",", "", regex=False)
    blank = col.isna() | raw.isin(["", "nan", "NaN", "None", "NA"])
    num = pd.to_numeric(raw.where(~blank), errors="coerce")
    bad = num.isna() & ~blank
    if bad.any():
        examples = col[bad].astype(str).unique()[:5].tolist()
        raise ValueError(f"คอลัมน์ {name} มีค่าที่ไม่ใช่ตัวเลข: {examples}")
    return num.astype(float)


def load_drugs(path: str) -> pd.DataFrame:
    """Long CSV (drug_id, month, quantity) -> panel: index = every month in range, columns = drug_id.

    Cells for months without a record are NaN.
    """
    df = pd.read_csv(path, dtype={DRUG: str}, encoding="utf-8-sig")
    missing = {DRUG, MONTH, TARGET} - set(df.columns)
    if missing:
        raise ValueError(f"ไฟล์ยาขาดคอลัมน์: {sorted(missing)}")
    df = df[[DRUG, MONTH, TARGET]].copy()
    df[DRUG] = df[DRUG].astype(str).str.strip()
    df[MONTH] = parse_months(df[MONTH])
    df[TARGET] = _numeric(df[TARGET], TARGET)
    dup = df.duplicated([DRUG, MONTH], keep=False)
    if dup.any():
        ex = df.loc[dup, [DRUG, MONTH]].head(5).to_dict("records")
        raise ValueError(f"มีแถวซ้ำ (drug_id, month) {int(dup.sum())} แถว เช่น {ex}")
    panel = df.pivot(index=MONTH, columns=DRUG, values=TARGET)
    full = pd.date_range(panel.index.min(), panel.index.max(), freq="MS")
    panel = panel.reindex(full)
    panel.index.name = MONTH
    panel.columns.name = DRUG
    return panel.astype(float)


def load_covariate_table(path: str, columns: list[str]) -> pd.DataFrame:
    """CSV with a month column plus covariate columns -> DataFrame indexed by month (C.E.)."""
    df = pd.read_csv(path, encoding="utf-8-sig")
    if MONTH not in df.columns:
        raise ValueError("ไฟล์ตัวแปรร่วมต้องมีคอลัมน์ month")
    absent = [c for c in columns if c not in df.columns]
    if absent:
        raise ValueError(f"ไฟล์ตัวแปรร่วมขาดคอลัมน์: {absent}")
    df[MONTH] = parse_months(df[MONTH])
    if df[MONTH].duplicated().any():
        raise ValueError("ไฟล์ตัวแปรร่วมมีเดือนซ้ำ")
    out = df.set_index(MONTH)[columns].copy()
    for c in columns:
        out[c] = _numeric(out[c], c)
    return out.sort_index()


def load_holidays(path: str) -> pd.DataFrame:
    """CSV: date, announced_date (optional), name (optional).

    announced_date is the date the holiday was officially announced. Empty means it
    is a fixed holiday known long in advance.
    """
    df = pd.read_csv(path, dtype=str, encoding="utf-8-sig")
    if "date" not in df.columns:
        raise ValueError("ไฟล์วันหยุดต้องมีคอลัมน์ date")
    out = pd.DataFrame({"date": df["date"].map(parse_date)})
    if "announced_date" in df.columns:
        out["announced_date"] = df["announced_date"].map(parse_date)
    else:
        out["announced_date"] = pd.NaT
    out["name"] = df["name"] if "name" in df.columns else ""
    if out["date"].isna().any():
        raise ValueError("ไฟล์วันหยุดมีแถวที่ไม่มีวันที่")
    return out.drop_duplicates("date").sort_values("date").reset_index(drop=True)
