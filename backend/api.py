"""
api.py — PM·TWIN real-time service (FastAPI)

Run:   KERAS_BACKEND=jax uvicorn api:app --host 0.0.0.0 --port 8000
Docs:  http://<host>:8000/docs

Endpoints (the web UI uses the same JSON contract as ui_data.json/replay):
  POST /api/v1/ingest        one row or {"rows":[...]} from Node-RED (1-min data)
  GET  /api/v1/state         latest Digital Twin state (health, HI, RUL, action)
  GET  /api/v1/alerts        alert episodes since start-up
  GET  /api/v1/ui-data       full offline result bundle (artifacts/ui_data.json)
  POST /api/v1/replay        replay a time range of the CSV through the engine
  POST /api/v1/retrain       retrain + re-evaluate (60/20/20) in the background
  WS   /ws/stream            pushes every new state to connected dashboards

Optional SQL Server polling (set env vars, requires `pyodbc`):
  MSSQL_CONN   e.g. "DRIVER={ODBC Driver 18 for SQL Server};SERVER=...;DATABASE=...;UID=...;PWD=...;TrustServerCertificate=yes"
  MSSQL_TABLE  default "dbo.WasherSensor"   (one row per minute, columns = CSV columns + Datetime)
  MSSQL_POLL_S default 60
"""
import os, json, asyncio, subprocess, sys
os.environ.setdefault("KERAS_BACKEND", "jax")
from typing import Any
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, BackgroundTasks, Body
from fastapi.middleware.cors import CORSMiddleware
from engine import TwinEngine, EngineConfig
from pdm_core import load_csv, CHANNELS

ART = os.environ.get("PM_ART", "artifacts")
CSV = os.environ.get("PM_CSV", "data/raw.csv")
app = FastAPI(title="PM·TWIN API", version="1.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
engine = TwinEngine(EngineConfig(art_dir=ART))
clients: set[WebSocket] = set()


async def broadcast(state: dict):
    dead = []
    for ws in clients:
        try:
            await ws.send_json(state)
        except Exception:
            dead.append(ws)
    for ws in dead:
        clients.discard(ws)


@app.post("/api/v1/ingest")
async def ingest(payload: Any = Body(...)):
    rows = payload.get("rows", [payload]) if isinstance(payload, dict) else payload
    st = None
    for r in rows:
        if "dt" not in r and "Datetime" in r:
            r["dt"] = r["Datetime"]
        st = engine.ingest(r)
    if st:
        await broadcast(st)
    return st or {}


@app.get("/api/v1/state")
def state():
    return engine.state()


@app.get("/api/v1/alerts")
def alerts():
    return engine.alerts


@app.get("/api/v1/ui-data")
def ui_data():
    return json.load(open(f"{ART}/ui_data.json"))


@app.post("/api/v1/replay")
async def replay(start: str = "2026-07-16 18:00", end: str = "2026-07-16 20:00"):
    df = load_csv(CSV)
    sel = df[(df.dt >= start) & (df.dt <= end)]
    engine.reset()
    # warm-up with the 30 minutes before the range
    warm = df[(df.dt < start)].tail(30)
    for _, r in warm.iterrows():
        engine.ingest({"dt": str(r["dt"]), **{c: r[c] for c in CHANNELS}})
    out = []
    for _, r in sel.iterrows():
        st = engine.ingest({"dt": str(r["dt"]), **{c: r[c] for c in CHANNELS}})
        out.append({k: st[k] for k in ("time", "mode", "score", "alarm", "action", "rul_device", "rul_min")})
    return out


def _retrain():
    subprocess.run([sys.executable, "train.py", "--csv", CSV, "--out", ART], check=False)
    subprocess.run([sys.executable, "evaluate.py", "--csv", CSV, "--art", ART], check=False)
    subprocess.run([sys.executable, "export_ui.py", "--csv", CSV, "--art", ART], check=False)


@app.post("/api/v1/retrain")
def retrain(bg: BackgroundTasks):
    bg.add_task(_retrain)
    return {"status": "started", "split": "train 60% / val 20% / test 20% (temporal)"}


@app.websocket("/ws/stream")
async def stream(ws: WebSocket):
    await ws.accept(); clients.add(ws)
    try:
        if engine.state():
            await ws.send_json(engine.state())
        while True:
            await ws.receive_text()
    except WebSocketDisconnect:
        clients.discard(ws)


# ---------------- optional SQL Server poller ----------------------------
async def mssql_poller():
    import pyodbc
    conn_str = os.environ["MSSQL_CONN"]
    table = os.environ.get("MSSQL_TABLE", "dbo.WasherSensor")
    every = int(os.environ.get("MSSQL_POLL_S", "60"))
    last = None
    while True:
        try:
            with pyodbc.connect(conn_str, timeout=10) as cn:
                q = f"SELECT * FROM {table} WHERE Datetime > ? ORDER BY Datetime" if last else \
                    f"SELECT TOP 30 * FROM {table} ORDER BY Datetime DESC"
                cur = cn.cursor().execute(q, last) if last else cn.cursor().execute(q)
                cols = [c[0] for c in cur.description]
                rows = [dict(zip(cols, r)) for r in cur.fetchall()]
                if not last:
                    rows = rows[::-1]
                for r in rows:
                    r["dt"] = str(r["Datetime"])
                    st = engine.ingest(r)
                    last = r["Datetime"]
                    await broadcast(st)
        except Exception as e:  # keep polling
            print("MSSQL poll error:", e)
        await asyncio.sleep(every)


@app.on_event("startup")
async def _startup():
    if os.environ.get("MSSQL_CONN"):
        asyncio.create_task(mssql_poller())
