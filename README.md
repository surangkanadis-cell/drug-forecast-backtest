---
title: Drug Forecast Backtest
emoji: 💊
colorFrom: blue
colorTo: blue
sdk: docker
app_port: 7860
pinned: false
---

# พยากรณ์ความต้องการใช้ยารายเดือน: Chronos-2 แบบ zero-shot (rolling origin)

พยากรณ์ปริมาณการใช้ยารายเดือนด้วย `amazon/chronos-2` โดยไม่ fine-tune และรันบน CPU
แล้วประเมินผลด้วยการทดสอบแบบ rolling origin (expanding window)

## ติดตั้ง (Python 3.11, CPU)

```bash
python -m venv .venv
.venv\Scripts\activate                # Windows  (Linux/mac: source .venv/bin/activate)
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```
ครั้งแรกที่รัน โปรแกรมจะดาวน์โหลดโมเดลจาก Hugging Face (ประมาณ 480 MB) แล้วเก็บไว้ใช้ครั้งต่อไป

## เว็บแอป (รันบนเครื่องตัวเอง)

ดับเบิลคลิก `start_app.bat` หรือพิมพ์
```bash
streamlit run app.py
```
เบราว์เซอร์จะเปิดหน้า http://localhost:8501 ให้เอง มี 3 แท็บ
1. **ข้อมูล**: อัปโหลดไฟล์การใช้ยา (CSV/Excel ได้ทั้งแบบยาวและแบบกว้าง เลือกคอลัมน์เองได้) และรายชื่อยาใหม่
   ระบบตรวจข้อมูลก่อนบันทึกลง `data/` และสำรองไฟล์เดิมไว้ให้ทุกครั้ง
2. **รันการทดลอง**: เลือกการทดลอง 1/2/3 ปรับค่าได้ แล้วกดรัน มีแถบแสดงความคืบหน้า
3. **ผลลัพธ์**: ตารางสรุป กราฟเปรียบเทียบ กราฟรายยา (ค่าจริงเทียบค่าพยากรณ์ พร้อมช่วง 80%)
   และปุ่มดาวน์โหลด CSV/Excel

ข้อมูลอยู่ในเครื่องนี้เท่านั้น ไม่ถูกส่งออกไปภายนอก (ใช้อินเทอร์เน็ตแค่ตอนดาวน์โหลดโมเดลครั้งแรก)

## ขึ้นเว็บด้วย Hugging Face Space

ไฟล์ `Dockerfile` และ `requirements-space.txt` ใช้สำหรับ Space แบบ Docker (Python 3.12, CPU)
ไฟล์ที่อัปโหลดและผลลัพธ์บน Space จะหายเมื่อ Space รีสตาร์ท ให้ดาวน์โหลดผลเก็บไว้ทุกครั้ง

## ไฟล์ข้อมูลใน `data/`

| ไฟล์ | คอลัมน์ | หมายเหตุ |
|---|---|---|
| `drugs.csv` | `drug_id, month, quantity` | เดือนเขียนแบบ `2568-10` (พ.ศ.) หรือ `2025-10` ก็ได้ เดือนที่ไม่มีข้อมูลให้เว้นว่างหรือไม่ต้องมีแถว ระบบจะถือเป็น NaN (ไม่แทนด้วย 0) |
| `covariates.csv` | `month, opd, ipd, temperature, ...` | ชนิดของแต่ละตัวแปรกำหนดใน `configs/covariates.yaml` |
| `holidays.csv` | `date, announced_date, name` | วันที่แบบ `2568-10-23` ถ้า `announced_date` ว่าง ถือว่าเป็นวันหยุดที่รู้ล่วงหน้า |
| `new_drugs.csv` | `drug_id` | ยาใหม่ 30 ตัวสำหรับการทดลองที่ 3 |

ระบบคำนวณ `month_of_year`, `n_holidays` และ `working_days` ให้เอง
โดย `n_holidays` และ `working_days` นับเฉพาะวันหยุดที่ **ประกาศก่อนวันที่ 1 ของเดือน origin**
และ `working_days` = จำนวนวันจันทร์–ศุกร์ ลบวันหยุดที่ตรงกับวันจันทร์–ศุกร์

## รันผ่านคำสั่ง

```bash
python run.py --config configs/exp1_old_drugs.yaml       # การทดลองที่ 1: ยาเก่า ทั้งมีและไม่มีตัวแปรร่วม
python run.py --config configs/exp2_context_length.yaml  # การทดลองที่ 2: ความยาวข้อมูล 1/3/5 ปี
python run.py --config configs/exp3_new_drugs.yaml       # การทดลองที่ 3: ยาใหม่
python run.py --config configs/exp1_old_drugs.yaml --max-drugs 20   # ลองเร็ว ๆ กับยา 20 ตัว
pytest                                                    # เทสต์ทั้งหมด รวมเทสต์ป้องกันข้อมูลรั่ว
```

ลองกับข้อมูลสังเคราะห์ (ไม่ใช่ข้อมูลโรงพยาบาล)
```bash
python examples/make_example_data.py
python run.py --config examples/smoke.yaml
```

## ไฟล์ผลลัพธ์ (`outputs/<ชื่อการทดลอง>/`)

- `results_long.csv`: 1 แถว = ยา × origin × horizon × โมเดล × การตั้งค่าตัวแปรร่วม (× context)
- `summary_per_drug.csv`, `summary_overall.csv`: `<ตัวชี้วัด>_mean`, `<ตัวชี้วัด>_sd`, `<ตัวชี้วัด>_n`
  พร้อม `wape_pooled` (= Σ|e| / Σ|y|) และ `coverage_80`
- `skipped_drugs.csv`: ยาที่ถูกข้ามในแต่ละ origin พร้อมเหตุผล
- `run_log.txt` / `run_log.json`: เวลาที่ใช้, seed, เวอร์ชันโมเดล (HF commit) และแพ็กเกจ, คำเตือน

เดือนในไฟล์ผลลัพธ์เป็น ค.ศ. (`2025-10`)

## นิยามสำคัญ

- origin O: โมเดลเห็นข้อมูลยาถึงเดือน O−1 เท่านั้น และตัวแปรร่วมแบบ past_only ก็ใช้ได้ถึง O−1 เท่านั้น
- `history_months` = จำนวน **เดือนที่มีข้อมูล** (ไม่ใช่ NaN) ตั้งแต่เดือนแรกที่ยาเข้า (หรือ `context_start`) ถึง O−1
  ยาที่มี `history_months < min_history` จะถูกข้าม
- ค่าพยากรณ์จุด = q0.5
- `mase_lag1`, `mase_m12`: ตัวหาร = ค่าเฉลี่ย |y_t − y_{t−lag}| จากคู่เดือนที่มีข้อมูลทั้งสองเดือน ใช้เฉพาะข้อมูลก่อน O (ภายใน `context_start`)
  ถ้าไม่มีคู่เดือนเลย หรือตัวหาร = 0 จะได้ NaN
- `crps` = 2 × ค่าเฉลี่ย pinball loss ของ q0.1…q0.9 (ค่าประมาณ CRPS จากควอนไทล์)
- `crps_scaled` = crps ÷ ค่าเฉลี่ยของยานั้นก่อน O (ถ้าค่าเฉลี่ย = 0 จะได้ NaN)
- `covered_80` = 1 ถ้า q0.1 ≤ ค่าจริง ≤ q0.9
- `crps_skill`: เว้นว่างไว้ก่อน (จะคำนวณเมื่อเพิ่ม Seasonal Naïve)
- ควอนไทล์ถูกตัดไม่ให้ติดลบ และปรับให้ไม่ลดลงตามระดับ (ไม่ไขว้กัน)

## การเพิ่มโมเดล

เขียนคลาสลูกของ `drugfc.models.base.ForecastModel` ที่มีเมธอด `forecast(task)`
คืนค่า `drug_id, month, q0.1..q0.9` แล้วลงทะเบียนใน `MODEL_REGISTRY` ที่ `drugfc/models/__init__.py`
และตั้ง `model.name` ใน config

## โครงสร้าง

```
app.py                      เว็บแอป (Streamlit)
start_app.bat               ดับเบิลคลิกเพื่อเปิดเว็บแอป
run.py                      รันผ่านคำสั่ง
drugfc/io.py                อ่านไฟล์ แปลง พ.ศ. เป็น ค.ศ.
drugfc/covariates.py        past_only / known_future และวันหยุดตามที่รู้ ณ origin
drugfc/tasks.py             สร้างข้อมูลนำเข้าในแต่ละ origin + ตรวจข้อมูลรั่ว
drugfc/backtest.py          rolling origin และพยากรณ์ทุกยาเป็น batch ต่อ origin
drugfc/metrics.py           ตัวชี้วัดและสรุป
drugfc/pipeline.py          รันการทดลองทั้งชุด (ใช้ร่วมกันระหว่าง run.py และเว็บแอป)
drugfc/models/              interface ForecastModel + Chronos-2
configs/                    config ของแต่ละการทดลอง
tests/                      เทสต์
```
