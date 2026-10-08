# PM·TWIN — Predictive Maintenance · Digital Twin

ระบบนี้ทำนายความล้มเหลวของอุปกรณ์แต่ละ Part ใน **เครื่องล้างหม้อหุงและฝา** ของสายการผลิตข้าว โดยใช้ Digital Twin ร่วมกับ AI (LSTM-Autoencoder แบบ mode-aware)

**เปิด Web App:** https://predictive-pm.github.io/

## ความสามารถ
- ประสิทธิภาพและ Health ของอุปกรณ์ทั้ง 12 Part พร้อมภาพตัวอย่างอะไหล่
- คาดการณ์ความล้มเหลวและ RUL เช่น "อีก 1 ชม. Part นี้จะเสียหาย ต้องหยุดเครื่องทันที"
- ผังตำแหน่ง Part บนเครื่องจักร (Digital Twin Layout)
- แจ้งเตือน และวิเคราะห์สาเหตุอัตโนมัติจาก per-channel attribution
- Timeline การทำงาน, ระบบเก็บชั่วโมงทำงาน และรายงานความเคลื่อนไหว
- เปอร์เซ็นต์ประสิทธิภาพของโมเดล (AP, Recall, Precision, F0.5, Hit@1, Lead time)
- โหมด LIVE ตามเวลาประเทศไทย และโหมด Replay

## โครงสร้าง Repository
| Path | รายละเอียด |
|---|---|
| `index.html` | Web App แบบ static (Vue 3) ไม่ต้อง build |
| `data/pm_twin_data.json` | ผลของโมเดลที่หน้าเว็บนำมาแสดง |
| `backend/` | โค้ด Train/Evaluate (Train 60% / Val 20% / Test 20%), โมเดลที่ Train แล้ว และ FastAPI สำหรับต่อ Real-time |
| `tools/build_webapp.py` | สคริปต์แปลงหน้า Design ให้เป็น `index.html` |

## อัปเดตข้อมูล
1. วางไฟล์ CSV ใหม่ไว้ที่ `backend/data/raw.csv` แล้วรัน `train.py` → `evaluate.py` → `export_ui.py`
2. คัดลอกไฟล์ `backend/artifacts/ui_data.json` ไปทับ `data/pm_twin_data.json` แล้ว commit และ push เมื่อ push แล้ว GitHub Pages จะอัปเดตหน้าเว็บให้เอง

Framework: Phongsakonvic · Digital Twin + AI for Predictive Maintenance
