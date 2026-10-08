# PM·TWIN — Predictive Maintenance เครื่องล้างหม้อหุงและฝา

ระบบนี้ใช้ Digital Twin ร่วมกับ AI เพื่อทำนายความล้มเหลวของเครื่องจักร โดยใช้วิธีตาม Paper *"Development of a Digital Twin and AI System for Predictive Maintenance"* (Vichaidit & Muangprathub, Rev0)

## โครงสร้างไฟล์

| ไฟล์ | หน้าที่ |
|---|---|
| `pdm_core.py` | Logic หลัก: ผังอุปกรณ์ (12 Part → 16 Channel), จำแนกโหมด run/ramp-up/stop, mode-wise normalisation, sliding window, LSTM-Autoencoder, Eq.2/Eq.3, persistence rule, RUL |
| `train.py` | Train โมเดลด้วยข้อมูลสภาวะปกติเท่านั้น โดยแบ่งข้อมูลตามเวลาเป็น **Train 60% / Validation 20% / Test 20%** |
| `evaluate.py` | ประเมินผลแบบ Lead-time-oriented ตาม Section IV ได้แก่ AP, Event F0.5, FA/week, Lead time และ Localization พร้อม Baseline (PCA, Isolation Forest, Moving std) โดยไม่ใช้ point adjustment |
| `stability.py` | สรุปความเสถียรของผลจากหลาย Random seed |
| `export_ui.py` | สร้าง `ui_data.json` (Data contract ของหน้าเว็บ) |
| `engine.py` | Digital Twin engine แบบ Online: รับข้อมูลทีละนาที แล้วให้ผล Health, HI, RUL และ Decision |
| `api.py` | FastAPI สำหรับ Real-time: REST + WebSocket + ตัวดึงข้อมูลจาก SQL Server (ใช้หรือไม่ใช้ก็ได้) |
| `nodered_flow.json` | ตัวอย่าง Flow ใน Node-RED สำหรับส่งข้อมูลเข้า `/api/v1/ingest` |
| `artifacts/` | โมเดลที่ Train แล้ว (`.keras`), scaler, threshold, ผลประเมิน และ `ui_data.json` |

## วิธีใช้งาน

```bash
pip install -r requirements.txt
export KERAS_BACKEND=jax          # ใช้ tensorflow หรือ torch แทนได้
python train.py    --csv data/raw.csv --out artifacts      # ~7 นาทีต่อโมเดลบน CPU 2 core
python evaluate.py --csv data/raw.csv --art artifacts
python export_ui.py --csv data/raw.csv --art artifacts
uvicorn api:app --host 0.0.0.0 --port 8000                 # เปิดดูเอกสาร API ได้ที่ http://localhost:8000/docs
```

ถ้าจะต่อ Real-time ให้ Node-RED ส่ง JSON ทุก 1 นาทีไปที่ `POST /api/v1/ingest` ตามตัวอย่างนี้:
`{"dt":"2026-07-16 18:40:00","Conveyor1_Load":0.52, ..., "Valve4_TempVal":55}`
ส่วนหน้าเว็บให้รับสถานะผ่าน `WS /ws/stream` ถ้าต้องการให้ API ดึงข้อมูลจาก SQL Server เอง ให้ตั้งค่า `MSSQL_CONN` และ `MSSQL_TABLE`

## ข้อมูลและการเตรียมข้อมูล

- ข้อมูล 250,000 แถว ช่วง 01/02/2569 09:20 – 24/07/2569 23:59 เก็บทุก 1 นาที ไม่มีค่าที่หายไปและไม่มีช่วงเวลาขาดหาย
- **Train**: 01/02 – 16/05/2569 (150,000 แถว) · **Validation**: 16/05 – 20/06/2569 (50,000 แถว) · **Test**: 20/06 – 24/07/2569 (50,000 แถว)
- โหมดการทำงาน: run 226,603 / ramp-up 10,440 (30 นาทีแรกหลังเริ่มเดินเครื่อง) / stop 12,957 นาที
- ช่วงพักตามแผนที่พบในข้อมูลคือ 05:30, 17:30 และบางวัน 15:30
- ในข้อมูลมีเหตุการณ์ความเสียหายจริง 1 ครั้ง (label `E`) คือ **16/07/2569 เวลา 19:48 น.** เครื่องหยุด 148 นาที
- Channel ที่ใช้มี 16 ช่อง ส่วน `Valve*_Load` และ `Valve*_TempSet` ไม่ได้นำมาใช้ เพราะมีค่าคงที่ตลอดช่วงที่เครื่องทำงาน

## โมเดล (Table IV)

ใช้ LSTM-Autoencoder ขนาด window 30 นาที, encoder 2 ชั้น (32 → latent 8), Adam lr 0.001, batch 128, loss เป็น MSE, early stopping patience 10 และ train สูงสุด 40 epoch (โมเดลใช้ครบ 40 epoch โดย loss ยังลดลงอยู่ จึงอาจเพิ่มจำนวน epoch ได้อีก)
Threshold ตั้งที่เปอร์เซ็นไทล์ 99.9 ของ error ในช่วง Train แยกตามโหมด เปอร์เซ็นไทล์นี้เลือกจาก Validation โดยกำหนดให้ FA ไม่เกิน 1 ครั้งต่อสัปดาห์ และกำหนด persistence k = 5 นาที

## ผลบน Test set (Table VI)

| Condition | AP | Event F0.5 | FA/week | Lead time | First alarm |
|---|---|---|---|---|---|
| **LSTM-AE (mode-aware)** | **0.850** | 0.714 | 0.22 | **9.38 h** | 16/07 10:25 |
| LSTM-AE (mode-blind) | 0.915 | 0.652 | 0.43 | 11.30 h | 16/07 08:30 |
| PCA | 0.015 | 0.000 | 0.43 | ตรวจไม่พบ | – |
| Isolation Forest | 0.068 | 0.000 | 0.00 | ตรวจไม่พบ | – |
| Moving std rule | 0.522 | 0.789 | 0.22 | 8.93 h | 16/07 10:52 |

- **Stability** จาก 4 seeds ของ mode-aware: AP 85.9% ± 3.7% (CV 4.2%) · Lead time 9.86 ± 0.94 h
- **Localization** (Table VII): Channel ที่ error สูงสุดคือ `RevArmOut_NoLoad` และ `RevArmOut_Load` ซึ่งชี้ไปที่ **Reversing Arm – Outlet** (Hit@1 = 1/1) ข้อสรุปนี้ได้จากรูปแบบของข้อมูล และยังต้องยืนยันกับบันทึกซ่อมบำรุงจริง
- **RUL** คำนวณด้วย log-linear extrapolation ของ HI ไปจนถึงระดับความเสียหาย ซึ่งปรับเทียบจากเหตุการณ์เดียวกันนี้ ผลจึงเป็น in-sample ที่ 1 ชม. ก่อนเสียทำนายได้ 87 นาที (ค่าจริง 59) และที่ 30 นาทีก่อนเสียทำนายได้ 19 นาที (ค่าจริง 29)

## ข้อจำกัดที่ควรระบุใน Paper

1. Test set มีเหตุการณ์ความเสียหายเพียง 1 ครั้ง ผลที่ได้จึงแสดงความเป็นไปได้ของวิธี (feasibility) แต่ยังสรุปเชิงสถิติไม่ได้
2. แบบ mode-blind ได้ AP และ lead time สูงกว่า แต่มี false alarm มากกว่า 2 เท่า จึงยังไม่ได้ผลยืนยันชัดเจนว่า mode-aware ดีกว่า
3. Moving-std rule ได้ Event F0.5 สูงกว่า แสดงว่าความผิดปกติครั้งนี้ค่อนข้างแยกออกได้ง่าย ซึ่งเป็นประเด็นที่ Paper ต้องอภิปราย
4. หน่วยของแต่ละ Channel, รหัสอะไหล่ และรอบ PM ยังเป็นข้อมูลที่ต้องเพิ่มเติม [TO BE PROVIDED]
