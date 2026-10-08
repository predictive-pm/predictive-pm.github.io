"""
pdm_core.py — shared logic for the Digital Twin Predictive-Maintenance system
(Pot & Lid Washing Machine, rice production line).

Implements the method of "Development of a Digital Twin and AI System for
Predictive Maintenance" (Vichaidit & Muangprathub, Rev0):
  * operating-mode labelling (run / ramp-up / stop; idle is reserved)
  * mode-wise normalisation estimated on the training partition only
  * sliding windows X(t) = [x(t-w+1) .. x(t)]  (Eq. 1)
  * LSTM-Autoencoder trained on normal-condition data only
  * total score e(t)  = 1/(w d) sum_ij (x - x_hat)^2   (Eq. 2)
  * channel error e_j(t) = 1/w sum_i (x_ij - x_hat_ij)^2 (Eq. 3)
  * mode-specific threshold + persistence rule (k consecutive minutes)
  * device-level Digital Twin state (per-channel error mapped to topology)
"""
from __future__ import annotations

import os
import re
import json
import numpy as np
import pandas as pd

os.environ.setdefault("KERAS_BACKEND", "jax")

# --------------------------------------------------------------------------
# Machine topology (Digital Twin): device -> channels
# --------------------------------------------------------------------------
DEVICES = [
    # id, name (EN), name (TH), subsystem, channels
    ("CV1", "Conveyor 1 Drive Motor",   "มอเตอร์สายพาน 1",           "Mechanical", ["Conveyor1_Load", "Conveyor1_Speed"]),
    ("CV2", "Conveyor 2 Drive Motor",   "มอเตอร์สายพาน 2",           "Mechanical", ["Conveyor2_Load", "Conveyor2_NoLoad"]),
    ("RAI", "Reversing Arm – Inlet",    "แขนกลับหม้อ ขาเข้า",         "Mechanical", ["RevArmIn_Load", "RevArmIn_NoLoad"]),
    ("RAO", "Reversing Arm – Outlet",   "แขนกลับหม้อ ขาออก",          "Mechanical", ["RevArmOut_Load", "RevArmOut_NoLoad"]),
    ("P1",  "Wash Pump 1",              "ปั๊มน้ำล้าง 1",               "Fluid",      ["Pump1_Load"]),
    ("P2",  "Wash Pump 2",              "ปั๊มน้ำล้าง 2",               "Fluid",      ["Pump2_Load"]),
    ("P3",  "Wash Pump 3",              "ปั๊มน้ำล้าง 3",               "Fluid",      ["Pump3_Load"]),
    ("P4",  "Wash Pump 4",              "ปั๊มน้ำล้าง 4",               "Fluid",      ["Pump4_Load"]),
    ("SV1", "Steam Valve 1",            "วาล์วไอน้ำ 1",               "Thermal",    ["Valve1_TempVal"]),
    ("SV2", "Steam Valve 2",            "วาล์วไอน้ำ 2",               "Thermal",    ["Valve2_TempVal"]),
    ("SV3", "Steam Valve 3",            "วาล์วไอน้ำ 3",               "Thermal",    ["Valve3_TempVal"]),
    ("SV4", "Steam Valve 4",            "วาล์วไอน้ำ 4",               "Thermal",    ["Valve4_TempVal"]),
]
CHANNELS = [c for d in DEVICES for c in d[4]]          # 16 condition channels
CH2DEV = {c: d[0] for d in DEVICES for c in d[4]}

MODES = {"stop": 0, "ramp": 1, "run": 2, "idle": 3}
RAMP_MINUTES = 30          # minutes after a restart treated as ramp-up
SCHEDULED_BREAK_HOURS = (5, 15, 17)  # planned breaks observed in the record (05:30, 15:30, 17:30)


# --------------------------------------------------------------------------
# Loading & operating-mode labelling (Table III steps 1-3)
# --------------------------------------------------------------------------
def _be_to_ce(date_str: pd.Series) -> pd.Series:
    """Thai Buddhist-era dates (d/m/2569) -> Gregorian (d/m/2026)."""
    return date_str.str.replace(r"/(25\d\d)$", lambda m: "/" + str(int(m.group(1)) - 543), regex=True)


def load_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8-sig")
    df["dt"] = pd.to_datetime(_be_to_ce(df["Date"]) + " " + df["Time"], format="%d/%m/%Y %H:%M:%S")
    df = df.sort_values("dt").drop_duplicates("dt").reset_index(drop=True)
    return df


def label_modes(df: pd.DataFrame) -> np.ndarray:
    """stop: drive speed 0 and all loads 0; ramp: first RAMP_MINUTES after a
    restart; run: otherwise. ('idle' is kept in the scheme for future data.)"""
    stop = (df["Conveyor1_Speed"].values <= 0.5)
    mode = np.full(len(df), MODES["run"], dtype=np.int8)
    mode[stop] = MODES["stop"]
    # minutes since last stop
    since = np.zeros(len(df), dtype=np.int64)
    c = 10**6
    for i, s in enumerate(stop):
        c = 0 if s else c + 1
        since[i] = c
    mode[(~stop) & (since <= RAMP_MINUTES)] = MODES["ramp"]
    return mode


def quality_audit(df: pd.DataFrame) -> dict:
    gaps = df["dt"].diff().dt.total_seconds().fillna(60)
    return {
        "rows": int(len(df)),
        "start": str(df["dt"].iloc[0]),
        "end": str(df["dt"].iloc[-1]),
        "missing_values": int(df[CHANNELS].isna().sum().sum()),
        "sampling_gaps": int((gaps != 60).sum()),
        "duplicates_removed": 0,
        "monotonic": bool(df["dt"].is_monotonic_increasing),
    }


# --------------------------------------------------------------------------
# Mode-wise normalisation (Table III step 4)
# --------------------------------------------------------------------------
class ModeScaler:
    def __init__(self):
        self.stats: dict[int, tuple[np.ndarray, np.ndarray]] = {}

    def fit(self, X: np.ndarray, mode: np.ndarray, mask: np.ndarray, mode_aware=True):
        for m in (MODES["ramp"], MODES["run"]):
            sel = mask & ((mode == m) if mode_aware else np.isin(mode, [1, 2]))
            mu = X[sel].mean(0)
            sd = X[sel].std(0)
            sd = np.where(sd < 1e-3, 1e-3, sd)
            self.stats[m] = (mu, sd)
        return self

    def transform(self, X: np.ndarray, mode: np.ndarray) -> np.ndarray:
        Z = np.zeros_like(X, dtype=np.float32)
        for m, (mu, sd) in self.stats.items():
            sel = mode == m
            Z[sel] = (X[sel] - mu) / sd
        return np.clip(Z, -50, 50)

    def to_json(self):
        return {str(k): {"mean": v[0].tolist(), "std": v[1].tolist()} for k, v in self.stats.items()}

    @classmethod
    def from_json(cls, d):
        s = cls()
        s.stats = {int(k): (np.array(v["mean"]), np.array(v["std"])) for k, v in d.items()}
        return s


# --------------------------------------------------------------------------
# Windows (Table III step 5)
# --------------------------------------------------------------------------
def window_end_index(mode: np.ndarray, w: int) -> np.ndarray:
    """Indices t whose window [t-w+1, t] contains no stop rows."""
    stop = (mode == MODES["stop"]).astype(np.int32)
    cs = np.r_[0, np.cumsum(stop)]
    t = np.arange(w - 1, len(mode))
    ok = (cs[t + 1] - cs[t + 1 - w]) == 0
    return t[ok]


def make_windows(Z: np.ndarray, ends: np.ndarray, w: int) -> np.ndarray:
    idx = ends[:, None] + np.arange(-w + 1, 1)[None, :]
    return Z[idx]


# --------------------------------------------------------------------------
# LSTM-Autoencoder (Section III-F, Table IV)
# --------------------------------------------------------------------------
def build_lstm_ae(w: int, d: int, latent: int = 8, hidden: int = 32, depth: int = 2, lr: float = 1e-3):
    import keras
    from keras import layers
    inp = keras.Input((w, d))
    x = inp
    sizes = [hidden] * (depth - 1)
    for h in sizes:
        x = layers.LSTM(h, return_sequences=True)(x)
    z = layers.LSTM(latent, name="latent")(x)
    x = layers.RepeatVector(w)(z)
    for h in [latent] + sizes[::-1]:
        x = layers.LSTM(h, return_sequences=True)(x)
    out = layers.TimeDistributed(layers.Dense(d))(x)
    m = keras.Model(inp, out, name="lstm_autoencoder")
    m.compile(optimizer=keras.optimizers.Adam(lr), loss="mse")
    return m


def reconstruction_errors(model, Xw: np.ndarray, batch: int = 1024):
    """Return total score e(t) (Eq.2) and per-channel e_j(t) (Eq.3)."""
    rec = model.predict(Xw, batch_size=batch, verbose=0)
    sq = (Xw - rec) ** 2
    ej = sq.mean(axis=1)            # (n, d)   Eq.3
    e = ej.mean(axis=1)             # (n,)     Eq.2
    return e, ej


# --------------------------------------------------------------------------
# Decision layer: mode-specific threshold + persistence k (Section III-F)
# --------------------------------------------------------------------------
def persistent_alarm(flag: np.ndarray, k: int) -> np.ndarray:
    """True from the k-th consecutive exceedance onward."""
    run = np.zeros(len(flag), dtype=np.int32)
    c = 0
    for i, f in enumerate(flag):
        c = c + 1 if f else 0
        run[i] = c
    return run >= k


def episodes(alarm: np.ndarray, gap: int = 30) -> list[tuple[int, int]]:
    """Group alarm points into episodes (merge if separated by < gap samples)."""
    idx = np.where(alarm)[0]
    if len(idx) == 0:
        return []
    out, s, p = [], idx[0], idx[0]
    for i in idx[1:]:
        if i - p > gap:
            out.append((s, p)); s = i
        p = i
    out.append((s, p))
    return out


def device_scores(ej_row: np.ndarray) -> dict:
    """Map per-channel error onto devices (Digital Twin state, Fig. 4)."""
    res = {}
    for d in DEVICES:
        ids = [CHANNELS.index(c) for c in d[4]]
        res[d[0]] = float(np.max(ej_row[ids]))
    return res


def estimate_rul(t_minutes: np.ndarray, hi: np.ndarray, critical: float, horizon: int = 120):
    """Remaining-useful-life (minutes of operation) by log-linear extrapolation
    of the device health index toward the critical level:
        ln HI(t) ~ a + b t  ->  RUL = (ln HI_crit - ln HI_fit(now)) / b
    t_minutes must be OPERATING minutes (stops excluded). Returns None if the
    index is not rising."""
    ok = ~np.isnan(hi) & (hi > 0)
    t = t_minutes[ok][-horizon:]; y = np.log(hi[ok][-horizon:])
    if len(t) < 20:
        return None
    A = np.vstack([t - t[-1], np.ones_like(t)]).T
    slope, icpt = np.linalg.lstsq(A, y, rcond=None)[0]
    if slope <= 1e-5:
        return None
    rul = (np.log(critical) - icpt) / slope
    return float(max(rul, 0.0))


def next_planned_break(ts: "pd.Timestamp", starts=((5, 30), (17, 30))):
    """Next planned production break (observed schedule: ~05:30 and ~17:30)."""
    cands = []
    for day in (0, 1):
        for h, m in starts:
            c = ts.normalize() + pd.Timedelta(days=day, hours=h, minutes=m)
            if c > ts:
                cands.append(c)
    return min(cands)
