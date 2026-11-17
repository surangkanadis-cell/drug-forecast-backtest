"""Web app (runs on this computer):  streamlit run app.py

Tabs: 1) data upload  2) configure and run an experiment  3) results, charts, downloads
"""
from __future__ import annotations

import io
import json
import os
import shutil
import tempfile
from datetime import datetime
from pathlib import Path

os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")
ROOT = Path(__file__).resolve().parent
os.chdir(ROOT)

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yaml

from drugfc.config import prepare_config
from drugfc.covariates import load_covariate_config
from drugfc.io import DRUG, MONTH, TARGET, load_drugs, load_holidays, parse_month
from drugfc.models import get_model
from drugfc.pipeline import count_steps, run_experiment

DATA = ROOT / "data"
OUTPUTS = ROOT / "outputs"
CONFIGS = ROOT / "configs"
DRUGS_FILE = DATA / "drugs.csv"
NEW_DRUGS_FILE = DATA / "new_drugs.csv"
COV_CONFIG = CONFIGS / "covariates.yaml"

# Series colors (validated categorical slots 1-2); actual values in neutral ink.
COLOR = {"no_cov": "#2a78d6", "with_cov": "#1baf7a"}
BAND = {"no_cov": "rgba(42,120,214,0.14)", "with_cov": "rgba(27,175,122,0.14)"}
INK = "#6b7a8c"
GRID = "rgba(29,43,58,0.08)"
COV_TH = {"no_cov": "ไม่มีตัวแปรร่วม", "with_cov": "มีตัวแปรร่วม"}
METRIC_TH = {
    "mae_mean": "MAE",
    "wape_pooled": "WAPE (รวมทุกแถว)",
    "mase_lag1_mean": "MASE (lag 1)",
    "mase_m12_mean": "MASE (m = 12)",
    "crps_mean": "CRPS",
    "crps_scaled_mean": "CRPS ปรับสเกล",
    "coverage_80": "ความครอบคลุมช่วง 80%",
}
PRESETS = {
    "การทดลอง 1: ยาเก่า": "exp1_old_drugs.yaml",
    "การทดลอง 2: ความยาวข้อมูล 1/3/5 ปี": "exp2_context_length.yaml",
    "การทดลอง 3: ยาใหม่": "exp3_new_drugs.yaml",
}

st.set_page_config(page_title="พยากรณ์การใช้ยา", page_icon=":material/medication:", layout="wide")


# ----------------------------------------------------------------- helpers
def read_upload(upload) -> pd.DataFrame:
    """CSV (UTF-8 or Thai Windows encoding) or Excel -> DataFrame of strings."""
    raw = upload.getvalue()
    if upload.name.lower().endswith((".xlsx", ".xls")):
        sheets = pd.read_excel(io.BytesIO(raw), sheet_name=None, dtype=str)
        if len(sheets) > 1:
            name = st.selectbox("เลือกชีต", list(sheets), key=f"sheet_{upload.name}")
            return sheets[name]
        return next(iter(sheets.values()))
    for enc in ("utf-8-sig", "cp874"):
        try:
            return pd.read_csv(io.BytesIO(raw), dtype=str, encoding=enc)
        except UnicodeDecodeError:
            continue
    raise ValueError("อ่านไฟล์ไม่ได้: ลองบันทึกเป็น CSV UTF-8")


def try_month(value) -> pd.Timestamp | None:
    try:
        return parse_month(value)
    except (ValueError, TypeError):
        return None


def backup(path: Path) -> Path | None:
    if not path.exists():
        return None
    dest = path.with_name(f"{path.stem}_backup_{datetime.now():%Y%m%d_%H%M%S}{path.suffix}")
    shutil.copy2(path, dest)
    return dest


def validate_drug_long(df: pd.DataFrame) -> pd.DataFrame:
    """Write to a temp file and read it back with the same loader run.py uses."""
    with tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False, encoding="utf-8") as f:
        df.to_csv(f, index=False)
        tmp = f.name
    try:
        return load_drugs(tmp)
    finally:
        os.remove(tmp)


def file_status(path: Path) -> str:
    if not path.exists():
        return "ยังไม่มีไฟล์"
    return f"{datetime.fromtimestamp(path.stat().st_mtime):%d/%m/%Y %H:%M}"


@st.cache_resource(show_spinner="กำลังโหลดโมเดล (ครั้งแรกอาจใช้เวลาสักครู่)…")
def cached_model(name: str, params_json: str):
    return get_model(name, **json.loads(params_json))


@st.cache_data
def cached_panel(path: str, mtime: float) -> pd.DataFrame:
    return load_drugs(path)


def list_outputs() -> list[Path]:
    if not OUTPUTS.exists():
        return []
    dirs = [p for p in OUTPUTS.iterdir() if (p / "results_long.csv").exists()]
    return sorted(dirs, key=lambda p: (p / "results_long.csv").stat().st_mtime, reverse=True)


# ----------------------------------------------------------------- tab 1: data
def tab_data():
    st.markdown("#### สถานะไฟล์ข้อมูล")
    cols = st.columns(4)
    for col, (label, path) in zip(cols, [("ข้อมูลการใช้ยา", DRUGS_FILE), ("รายชื่อยาใหม่", NEW_DRUGS_FILE),
                                         ("ตัวแปรร่วม", DATA / "covariates.csv"),
                                         ("วันหยุด", DATA / "holidays.csv")]):
        with col.container(border=True):
            st.markdown(f"**{label}**")
            st.caption(f"`data/{path.name}`")
            if path.exists():
                st.markdown(f":blue-badge[พร้อม] {file_status(path)}")
            else:
                st.markdown(":gray-badge[ยังไม่มีไฟล์]")

    if DRUGS_FILE.exists():
        try:
            panel = cached_panel(str(DRUGS_FILE), DRUGS_FILE.stat().st_mtime)
            n_obs = int(panel.notna().sum().sum())
            st.info(f"ข้อมูลยาปัจจุบัน: **{panel.shape[1]:,} ยา**, เดือน {panel.index.min():%Y-%m} "
                    f"ถึง {panel.index.max():%Y-%m} (ค.ศ.), มีค่า {n_obs:,} ช่อง, "
                    f"ค่าว่าง {panel.size - n_obs:,} ช่อง")
        except Exception as e:
            st.error(f"ไฟล์ data/drugs.csv อ่านไม่ได้: {e}")

    st.divider()
    upload_drugs()
    st.divider()
    upload_new_drugs()
    st.divider()
    with st.expander("แทนที่ไฟล์ตัวแปรร่วม หรือไฟล์วันหยุด"):
        upload_simple("ตัวแปรร่วม (month, opd, ipd, temperature)", DATA / "covariates.csv", check_covariates)
        upload_simple("วันหยุด (date, announced_date, name)", DATA / "holidays.csv", check_holidays)


def upload_drugs():
    st.markdown("#### อัปโหลดข้อมูลการใช้ยา")
    st.caption("รับไฟล์ CSV หรือ Excel เดือนเป็น พ.ศ. หรือ ค.ศ. ก็ได้ เดือนที่ไม่มีข้อมูลให้เว้นว่าง (ไม่ใส่ 0)")
    up = st.file_uploader("ไฟล์การใช้ยา", type=["csv", "xlsx", "xls"], key="up_drugs")
    if up is None:
        return
    try:
        df = read_upload(up)
    except Exception as e:
        st.error(str(e))
        return
    st.write(f"อ่านได้ {len(df):,} แถว, {df.shape[1]} คอลัมน์")
    st.dataframe(df.head(8), width="stretch")

    shape = st.radio("รูปแบบตาราง", ["แบบยาว: 1 แถว = ยา 1 ตัว × 1 เดือน",
                                      "แบบกว้าง: 1 แถว = ยา 1 ตัว, คอลัมน์ = เดือน"], horizontal=True)
    columns = list(df.columns)

    def guess(names, fallback):
        for n in names:
            if n in columns:
                return columns.index(n)
        return min(fallback, len(columns) - 1)

    if shape.startswith("แบบยาว"):
        c1, c2, c3 = st.columns(3)
        id_col = c1.selectbox("คอลัมน์รหัส/ชื่อยา", columns, index=guess(["drug_id", "icode", "drugname"], 0))
        m_col = c2.selectbox("คอลัมน์เดือน", columns, index=guess(["month", "year_month"], 1))
        q_col = c3.selectbox("คอลัมน์ปริมาณ", columns, index=guess(["quantity", "qty"], 2))
        long = pd.DataFrame({DRUG: df[id_col], MONTH: df[m_col], TARGET: df[q_col]})
    else:
        id_col = st.selectbox("คอลัมน์รหัส/ชื่อยา", columns, index=guess(["drug_id", "icode", "drugname"], 0))
        month_cols = [c for c in columns if c != id_col and try_month(c) is not None]
        other = [c for c in columns if c != id_col and c not in month_cols]
        st.write(f"พบคอลัมน์เดือน {len(month_cols)} คอลัมน์"
                 + (f" (ไม่ใช้: {', '.join(map(str, other[:8]))})" if other else ""))
        if not month_cols:
            st.error("ไม่พบคอลัมน์ที่เป็นเดือน (ชื่อคอลัมน์ควรเป็น เช่น 2568-10)")
            return
        long = df.melt(id_vars=[id_col], value_vars=month_cols, var_name=MONTH, value_name=TARGET)
        long = long.rename(columns={id_col: DRUG})

    long[DRUG] = long[DRUG].astype(str).str.strip()
    long = long[long[DRUG].ne("") & long[DRUG].ne("nan")]
    bad_months = [m for m in pd.unique(long[MONTH].astype(str)) if try_month(m) is None]
    if bad_months:
        st.error(f"อ่านเดือนไม่ได้ {len(bad_months)} ค่า เช่น {bad_months[:5]}")
        return
    long[MONTH] = long[MONTH].map(lambda v: f"{parse_month(v):%Y-%m}")
    dup = long.duplicated([DRUG, MONTH], keep=False)
    if dup.any():
        st.warning(f"มีแถวซ้ำ (ยาเดียวกัน เดือนเดียวกัน) {int(dup.sum())} แถว")
        if not st.checkbox("รวมแถวซ้ำด้วยการบวกปริมาณ"):
            st.dataframe(long[dup].sort_values([DRUG, MONTH]).head(20), width="stretch")
            return
        q = pd.to_numeric(long[TARGET].astype(str).str.replace(",", ""), errors="coerce")
        long = (long.assign(**{TARGET: q}).groupby([DRUG, MONTH], as_index=False)[TARGET]
                .sum(min_count=1))
    try:
        panel = validate_drug_long(long)
    except Exception as e:
        st.error(f"ข้อมูลไม่ผ่านการตรวจ: {e}")
        return
    n_obs = int(panel.notna().sum().sum())
    zeros = int((panel == 0).sum().sum())
    st.success(f"ตรวจผ่าน: {panel.shape[1]:,} ยา, {panel.index.min():%Y-%m} ถึง {panel.index.max():%Y-%m} "
               f"(ค.ศ.), มีค่า {n_obs:,} ช่อง, ค่าว่าง {panel.size - n_obs:,} ช่อง, ค่าเป็น 0 {zeros:,} ช่อง")
    label = "บันทึกทับ data/drugs.csv (สำรองไฟล์เดิมไว้ให้)" if DRUGS_FILE.exists() else "บันทึกเป็น data/drugs.csv"
    if st.button(label, type="primary"):
        b = backup(DRUGS_FILE)
        long.to_csv(DRUGS_FILE, index=False, encoding="utf-8-sig")
        st.success("บันทึกแล้ว" + (f" (ไฟล์เดิมสำรองไว้ที่ data/{b.name})" if b else ""))
        st.cache_data.clear()


def upload_new_drugs():
    st.markdown("#### อัปโหลดรายชื่อยาใหม่ (การทดลอง 3)")
    up = st.file_uploader("ไฟล์รายชื่อยา", type=["csv", "xlsx", "xls", "txt"], key="up_new")
    if up is None:
        if NEW_DRUGS_FILE.exists():
            ids = pd.read_csv(NEW_DRUGS_FILE, dtype=str, encoding="utf-8-sig")["drug_id"].dropna().tolist()
            st.caption(f"ไฟล์ปัจจุบันมี {len(ids)} ยา")
        return
    try:
        df = read_upload(up)
    except Exception as e:
        st.error(str(e))
        return
    col = st.selectbox("คอลัมน์รหัส/ชื่อยา", list(df.columns), key="new_col")
    ids = [s.strip() for s in df[col].dropna().astype(str) if s.strip()]
    ids = list(dict.fromkeys(ids))
    st.write(f"{len(ids)} ยา")
    if DRUGS_FILE.exists():
        panel = cached_panel(str(DRUGS_FILE), DRUGS_FILE.stat().st_mtime)
        missing = [d for d in ids if d not in panel.columns]
        if missing:
            st.warning(f"{len(missing)} รหัสไม่มีในข้อมูลการใช้ยา: {missing[:10]}")
        else:
            st.success("ทุกรหัสมีในข้อมูลการใช้ยา")
    if st.button("บันทึกเป็น data/new_drugs.csv", key="save_new"):
        b = backup(NEW_DRUGS_FILE)
        pd.DataFrame({"drug_id": ids}).to_csv(NEW_DRUGS_FILE, index=False, encoding="utf-8-sig")
        st.success("บันทึกแล้ว" + (f" (ไฟล์เดิมสำรองไว้ที่ data/{b.name})" if b else ""))


def check_covariates(path: str) -> str:
    cfg = load_covariate_config(str(COV_CONFIG))
    return f"ตัวแปร: {', '.join(cfg.kinds)}"


def check_holidays(path: str) -> str:
    h = load_holidays(path)
    return f"{len(h)} วันหยุด ถึง {h['date'].max():%Y-%m-%d}"


def upload_simple(label: str, dest: Path, check):
    up = st.file_uploader(label, type=["csv"], key=f"up_{dest.name}")
    if up is None:
        return
    if st.button(f"บันทึกทับ data/{dest.name}", key=f"save_{dest.name}"):
        b = backup(dest)
        dest.write_bytes(up.getvalue())
        try:
            st.success(f"บันทึกแล้ว: {check(str(dest))}")
        except Exception as e:
            if b:
                shutil.copy2(b, dest)
            st.error(f"ไฟล์ไม่ผ่านการตรวจ จึงคืนไฟล์เดิม: {e}")


# ----------------------------------------------------------------- tab 2: run
def tab_run():
    st.markdown("#### ตั้งค่าการทดลอง")
    preset = st.selectbox("เริ่มจากค่าตั้งของ", list(PRESETS))
    with open(CONFIGS / PRESETS[preset], encoding="utf-8") as f:
        base = yaml.safe_load(f)
    bt = base["backtest"]
    key = PRESETS[preset]

    c1, c2, c3, c4 = st.columns(4)
    start_origin = c1.text_input("origin แรก (ปี-เดือน)", str(bt["start_origin"]), key=f"so_{key}",
                                 help="พ.ศ. หรือ ค.ศ. ก็ได้ เช่น 2568-10")
    n_origins = c2.number_input("จำนวน origin", 1, 60, int(bt["n_origins"]), key=f"no_{key}")
    horizon = c3.number_input("horizon (เดือน)", 1, 12, int(bt.get("horizon", 1)), key=f"h_{key}")
    step = c4.number_input("step (เดือน)", 1, 12, int(bt.get("step", 1)), key=f"st_{key}")
    c1, c2, c3 = st.columns(3)
    min_history = c1.number_input("min_history (เดือนที่มีข้อมูล)", 1, 120, int(bt["min_history"]),
                                  key=f"mh_{key}")
    context_start = c2.text_input("context_start (ว่าง = ใช้ข้อมูลทั้งหมด)",
                                  "" if not bt.get("context_start") else str(bt["context_start"]),
                                  key=f"cs_{key}")
    cov_choice = c3.multiselect("ตัวแปรร่วม", ["ไม่มี", "มี"], default=["ไม่มี", "มี"], key=f"cov_{key}")

    flt = base.get("drug_filter") or {}
    c1, c2, c3 = st.columns(3)
    first_obs_by = c1.text_input("เลือกเฉพาะยาที่เริ่มมีข้อมูลไม่หลังเดือน (ว่าง = ไม่กรอง)",
                                 str(flt.get("first_obs_by") or ""), key=f"fo_{key}")
    use_new = c2.checkbox("ใช้เฉพาะยาใน data/new_drugs.csv", bool(flt.get("ids_file")), key=f"un_{key}")
    max_drugs = c3.number_input("จำกัดจำนวนยา (0 = ทั้งหมด)", 0, 100000, 0, key=f"md_{key}",
                                help="ใช้ทดสอบเร็ว ๆ ผลที่จำกัดจำนวนยาไม่ควรนำไปใช้ในวิทยานิพนธ์")

    runs = base.get("runs") or [{"context": "all"}]
    if len(runs) > 1 or any("context_start" in r for r in runs):
        st.markdown("**รอบย่อย** (แต่ละแถวแทนค่า context_start / min_history ด้านบน)")
        rdf = pd.DataFrame([{"context": str(r.get("context", "all")),
                             "context_start": str(r.get("context_start") or ""),
                             "min_history": r.get("min_history")} for r in runs])
        rdf = st.data_editor(rdf, num_rows="dynamic", width="stretch", key=f"runs_{key}")
        runs = []
        for r in rdf.to_dict("records"):
            item = {"context": r["context"] or "all"}
            if str(r.get("context_start") or "").strip():
                item["context_start"] = str(r["context_start"]).strip()
            if r.get("min_history") is not None and not pd.isna(r["min_history"]):
                item["min_history"] = int(r["min_history"])
            runs.append(item)

    out_name = st.text_input("ชื่อโฟลเดอร์ผลลัพธ์ (ใน outputs/)", base["name"], key=f"out_{key}")

    cfg_raw = {
        "name": out_name, "seed": base.get("seed", 42), "output_dir": str(OUTPUTS / out_name),
        "data": {"drug_file": str(DRUGS_FILE),
                 "covariate_config": str(COV_CONFIG) if "มี" in cov_choice else None},
        "drug_filter": {k: v for k, v in {"first_obs_by": first_obs_by.strip() or None,
                                          "ids_file": str(NEW_DRUGS_FILE) if use_new else None}.items() if v},
        "model": base["model"],
        "backtest": {"start_origin": start_origin.strip(), "n_origins": int(n_origins),
                     "horizon": int(horizon), "step": int(step), "min_history": int(min_history),
                     "context_start": context_start.strip() or None,
                     "use_covariates": [c == "มี" for c in ["ไม่มี", "มี"] if c in cov_choice]},
        "runs": runs,
    }
    problems = []
    if not cov_choice:
        problems.append("เลือกตัวแปรร่วมอย่างน้อย 1 แบบ")
    if not DRUGS_FILE.exists():
        problems.append("ยังไม่มี data/drugs.csv (อัปโหลดในแท็บข้อมูล)")
    if use_new and not NEW_DRUGS_FILE.exists():
        problems.append("ยังไม่มี data/new_drugs.csv")
    cfg = None
    if not problems:
        try:
            cfg = prepare_config(cfg_raw)
        except Exception as e:
            problems.append(f"ค่าตั้งไม่ถูกต้อง: {e}")
    if cfg is not None:
        origins = [cfg["runs"][0]["start_origin"] + pd.DateOffset(months=k * cfg["runs"][0]["step"])
                   for k in range(cfg["runs"][0]["n_origins"])]
        st.caption(f"origin: {origins[0]:%Y-%m} ถึง {origins[-1]:%Y-%m} (ค.ศ.) · "
                   f"รอบย่อย {len(cfg['runs'])} · ต้องพยากรณ์ทั้งหมด {count_steps(cfg)} ครั้ง (ครั้งละทุกยา)")
        if (OUTPUTS / out_name / "results_long.csv").exists():
            st.warning(f"โฟลเดอร์ outputs/{out_name} มีผลเดิมอยู่ จะถูกเขียนทับ")
    for p in problems:
        st.error(p)

    with st.expander("ดูค่าตั้งเป็น YAML (บันทึกไว้ใช้กับ run.py ได้)"):
        st.code(yaml.safe_dump(cfg_raw, allow_unicode=True, sort_keys=False), language="yaml")

    if st.button("รันการทดลอง", type="primary", icon=":material/play_arrow:", disabled=bool(problems)):
        execute(cfg, int(max_drugs) or None)


def execute(cfg: dict, max_drugs):
    model = cached_model(cfg["model"]["name"], json.dumps(cfg["model"]["params"], sort_keys=True))
    bar = st.progress(0.0, text="เริ่ม…")
    lines = []
    with st.status("กำลังรัน… อย่าปิดหรือรีเฟรชหน้านี้", expanded=True) as status:
        box = st.empty()

        def progress(msg):
            lines.append(msg)
            box.code("\n".join(lines[-25:]))

        def on_step(done, total):
            bar.progress(done / total, text=f"{done}/{total}")

        try:
            out = run_experiment(cfg, max_drugs=max_drugs, model=model,
                                 progress=progress, on_step=on_step)
        except Exception as e:
            status.update(label="รันไม่สำเร็จ", state="error")
            st.error(f"{type(e).__name__}: {e}")
            return
        status.update(label=f"เสร็จใน {out['log']['total_seconds']} วินาที", state="complete", expanded=False)
    st.success(f"บันทึกผลไว้ที่ outputs/{Path(out['out_dir']).name} · ไปที่แท็บ “ผลลัพธ์” เพื่อดูกราฟและดาวน์โหลด")
    st.session_state["last_output"] = Path(out["out_dir"]).name
    if out["log"]["warnings"]:
        for w in out["log"]["warnings"]:
            st.warning(w)


# ----------------------------------------------------------------- tab 3: results
def tab_results():
    dirs = list_outputs()
    if not dirs:
        st.info("ยังไม่มีผลลัพธ์ รันการทดลองในแท็บ “รันการทดลอง” ก่อน")
        return
    names = [d.name for d in dirs]
    last = st.session_state.get("last_output")
    folder = OUTPUTS / st.selectbox("ผลการทดลอง", names, index=names.index(last) if last in names else 0)

    res = pd.read_csv(folder / "results_long.csv", dtype={DRUG: str, "context": str}, encoding="utf-8-sig")
    overall = pd.read_csv(folder / "summary_overall.csv", dtype={"context": str}, encoding="utf-8-sig")
    log = json.loads((folder / "run_log.json").read_text(encoding="utf-8"))

    c = st.columns(4)
    c[0].metric("ยาที่พยากรณ์", f"{res[DRUG].nunique():,}")
    c[1].metric("แถวผลลัพธ์", f"{len(res):,}")
    c[2].metric("แถวที่ยังไม่มีค่าจริง", f"{int(res['actual'].isna().sum()):,}")
    c[3].metric("เวลาที่ใช้", f"{log['total_seconds']:.0f} วินาที")
    st.caption(f"รันเมื่อ {log['started']} · โมเดล {log['model'].get('model_id')} · seed {log['seed']}")
    for w in log.get("warnings", []):
        st.warning(w)

    st.markdown("#### สรุปรวม")
    show = overall.copy()
    show["covariate_setting"] = show["covariate_setting"].map(COV_TH).fillna(show["covariate_setting"])
    cols = ["covariate_setting", "context", "horizon", "n_rows"] + list(METRIC_TH)
    table = show[cols].rename(columns={"covariate_setting": "ตัวแปรร่วม", "n_rows": "จำนวนแถว", **METRIC_TH})
    st.dataframe(table.style.format(precision=3), width="stretch", hide_index=True)

    metric = st.selectbox("ตัวชี้วัดในกราฟ", list(METRIC_TH), format_func=METRIC_TH.get)
    st.plotly_chart(overall_chart(overall, metric), width="stretch")

    st.markdown("#### รายยา")
    drug = st.selectbox("เลือกยา", sorted(res[DRUG].unique()))
    contexts = sorted(res["context"].unique())
    context = st.radio("context", contexts, horizontal=True) if len(contexts) > 1 else contexts[0]
    sub = res[(res[DRUG] == drug) & (res["context"] == context)]
    history = None
    drug_file = log.get("drug_data", {}).get("file")
    if drug_file and os.path.exists(drug_file):
        panel = cached_panel(drug_file, os.path.getmtime(drug_file))
        if drug in panel.columns:
            history = panel[drug]
    st.plotly_chart(drug_chart(sub, history, drug), width="stretch")
    with st.expander("ตารางผลของยานี้"):
        st.dataframe(sub, width="stretch", hide_index=True)

    st.markdown("#### ดาวน์โหลด")
    files = ["results_long.csv", "summary_per_drug.csv", "summary_overall.csv",
             "skipped_drugs.csv", "run_log.txt"]
    cols = st.columns(len(files) + 1)
    for col, name in zip(cols, files):
        p = folder / name
        if p.exists():
            col.download_button(name, p.read_bytes(), file_name=f"{folder.name}_{name}")
    cols[-1].download_button("ทั้งหมดเป็น Excel", excel_bytes(folder),
                             file_name=f"{folder.name}.xlsx", type="primary")


def base_figure() -> go.Figure:
    fig = go.Figure()
    fig.update_layout(font=dict(family="Sarabun, sans-serif", color="#1d2b3a", size=13),
                      paper_bgcolor="#ffffff", plot_bgcolor="#ffffff")
    return fig


def overall_chart(overall: pd.DataFrame, metric: str) -> go.Figure:
    fig = base_figure()
    for setting in ["no_cov", "with_cov"]:
        d = overall[overall["covariate_setting"] == setting]
        if d.empty:
            continue
        x = [f"context {c}" for c in d["context"]]
        fig.add_bar(x=x, y=d[metric], name=COV_TH[setting], marker_color=COLOR[setting],
                    marker_line_width=0, text=[f"{v:.3f}" for v in d[metric]], textposition="outside",
                    hovertemplate="%{x}<br>" + METRIC_TH[metric] + ": %{y:.3f}<extra>" + COV_TH[setting] + "</extra>")
    fig.update_layout(barmode="group", bargap=0.5, bargroupgap=0.08, height=360, barcornerradius=4,
                      margin=dict(l=10, r=10, t=30, b=10), yaxis_title=METRIC_TH[metric],
                      legend=dict(orientation="h", y=1.12, x=0))
    fig.update_yaxes(rangemode="tozero", gridcolor=GRID)
    return fig


def drug_chart(sub: pd.DataFrame, history: pd.Series | None, drug: str) -> go.Figure:
    fig = base_figure()
    if history is not None and len(sub):
        months = pd.to_datetime(sub["target_month"])
        h = history.loc[months.min() - pd.DateOffset(months=36): months.max()]
        fig.add_scatter(x=h.index, y=h.values, name="ค่าจริง", mode="lines+markers",
                        line=dict(color=INK, width=2), marker=dict(size=5),
                        hovertemplate="%{x|%Y-%m}<br>ค่าจริง: %{y:,.0f}<extra></extra>")
    for setting in ["no_cov", "with_cov"]:
        d = sub[sub["covariate_setting"] == setting].sort_values("target_month")
        if d.empty:
            continue
        x = pd.to_datetime(d["target_month"])
        fig.add_scatter(x=list(x) + list(x[::-1]), y=list(d["q0.9"]) + list(d["q0.1"][::-1]),
                        fill="toself", fillcolor=BAND[setting], line=dict(width=0), mode="lines",
                        hoverinfo="skip", showlegend=False, name=f"ช่วง 80% {COV_TH[setting]}")
        fig.add_scatter(x=x, y=d["q0.5"], name=f"พยากรณ์ ({COV_TH[setting]})", mode="lines+markers",
                        line=dict(color=COLOR[setting], width=2), marker=dict(size=8),
                        customdata=np.stack([d["q0.1"], d["q0.9"]], axis=1),
                        hovertemplate="%{x|%Y-%m}<br>มัธยฐาน: %{y:,.0f}<br>"
                                      "ช่วง 80%: %{customdata[0]:,.0f} – %{customdata[1]:,.0f}<extra></extra>")
    fig.update_layout(height=420, margin=dict(l=10, r=10, t=40, b=10), hovermode="x unified",
                      title=dict(text=f"{drug}: ค่าจริงเทียบค่าพยากรณ์ (แถบ = ช่วง q0.1–q0.9)", font_size=15),
                      legend=dict(orientation="h", y=-0.15, x=0), yaxis_title="ปริมาณ")
    fig.update_yaxes(rangemode="tozero", gridcolor=GRID)
    fig.update_xaxes(gridcolor=GRID)
    return fig


def excel_bytes(folder: Path) -> bytes:
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as xw:
        for name, sheet in [("summary_overall.csv", "summary_overall"), ("summary_per_drug.csv", "summary_per_drug"),
                            ("results_long.csv", "results_long"), ("skipped_drugs.csv", "skipped_drugs")]:
            p = folder / name
            if p.exists():
                pd.read_csv(p, encoding="utf-8-sig").to_excel(xw, sheet_name=sheet, index=False)
        log = (folder / "run_log.txt").read_text(encoding="utf-8").splitlines()
        pd.DataFrame({"run_log": log}).to_excel(xw, sheet_name="run_log", index=False)
    return buf.getvalue()


# ----------------------------------------------------------------- page
st.markdown("### พยากรณ์ความต้องการใช้ยารายเดือน")
st.caption("Chronos-2 (amazon/chronos-2) แบบ zero-shot · ทดสอบแบบ rolling origin · รันบนเครื่องนี้ ข้อมูลไม่ถูกส่งออกไปภายนอก")
t1, t2, t3 = st.tabs(["ข้อมูล", "รันการทดลอง", "ผลลัพธ์"])
with t1:
    tab_data()
with t2:
    tab_run()
with t3:
    tab_results()
