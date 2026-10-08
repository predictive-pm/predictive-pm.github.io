"""
engine.py — Online Digital Twin engine (paper layers L3–L7) for real-time use.

Feed one sensor row per minute (from Node-RED, SQL Server or a CSV replay);
the engine keeps the sliding window, labels the operating mode, scores the
window with the trained LSTM-Autoencoder, applies the mode-specific threshold
with the persistence rule, maps per-channel error to devices (Digital Twin
state), estimates RUL and produces the lead-time-aware decision.

The JSON returned by `TwinEngine.state()` is the contract consumed by the web UI.
"""
from __future__ import annotations

import os, json
from collections import deque
from dataclasses import dataclass, field
os.environ.setdefault("KERAS_BACKEND", "jax")

import numpy as np
import pandas as pd
from pdm_core import (CHANNELS, DEVICES, MODES, RAMP_MINUTES, ModeScaler, estimate_rul, next_planned_break)

MODE_NAME = {v: k for k, v in MODES.items()}


@dataclass
class EngineConfig:
    art_dir: str = "artifacts"
    window: int = 30
    k: int = 5                  # persistence (minutes)
    smooth: int = 15            # HI smoothing (minutes)
    rul_horizon: int = 120      # operating minutes used for the RUL fit
    critical_hi: float | None = None   # read from ui_data/meta if None
    break_times: tuple = ((5, 30), (17, 30))


class TwinEngine:
    def __init__(self, cfg: EngineConfig = EngineConfig()):
        import keras
        self.cfg = cfg
        a = cfg.art_dir
        self.model = keras.models.load_model(f"{a}/lstm_ae_mode_aware.keras", compile=False)
        self.scaler = ModeScaler.from_json(json.load(open(f"{a}/scaler_mode_aware.json")))
        th = json.load(open(f"{a}/thresholds.json"))
        self.thr = {MODES[m]: v for m, v in th["mode_aware"].items()}
        self.ch_ref = np.array(th["channel_ref_p999"])
        self.k = th.get("k", cfg.k)
        crit = cfg.critical_hi
        if crit is None and os.path.exists(f"{a}/ui_data.json"):
            crit = json.load(open(f"{a}/ui_data.json"))["meta"]["critical_hi"]
        self.crit = float(crit or 60.0)
        self.reset()

    # ------------------------------------------------------------------
    def reset(self):
        w = self.cfg.window
        self.rows: deque = deque(maxlen=w)          # raw channel vectors
        self.modes: deque = deque(maxlen=w)
        self.since_stop = 10**6
        self.run_len = 0
        self.op_min = 0
        self.hi_hist = {d[0]: deque(maxlen=600) for d in DEVICES}     # (op_min, smoothed HI)
        self.hi_raw = {d[0]: deque(maxlen=self.cfg.smooth) for d in DEVICES}
        self.last: dict = {}
        self.alerts: list = []
        self._open_alert = None

    # ------------------------------------------------------------------
    def _mode(self, row: dict) -> int:
        if float(row.get("Conveyor1_Speed", 0)) <= 0.5:
            self.since_stop = 0
            return MODES["stop"]
        self.since_stop += 1
        return MODES["ramp"] if self.since_stop <= RAMP_MINUTES else MODES["run"]

    def _health(self, hi):
        if hi is None:
            return None
        if hi <= 1.0:
            return float(100 - 30 * np.clip(hi - 0.5, 0, 0.5))
        return float(85 * (1 - np.sqrt(np.clip((hi - 1) / (self.crit - 1), 0, 1))))

    @staticmethod
    def _status(h):
        if h is None:
            return "off"
        return "ok" if h >= 80 else ("warn" if h >= 45 else "crit")

    # ------------------------------------------------------------------
    def ingest(self, row: dict) -> dict:
        """row: {'dt': 'YYYY-MM-DD HH:MM[:SS]', <channel>: value, ...}"""
        ts = pd.Timestamp(row["dt"])
        m = self._mode(row)
        x = np.array([float(row.get(c, 0.0)) for c in CHANNELS], dtype=np.float32)
        self.rows.append(x); self.modes.append(m)
        score = alarm_flag = None
        ej = None
        if m != MODES["stop"] and len(self.rows) == self.cfg.window and MODES["stop"] not in self.modes:
            X = np.stack(self.rows)
            Z = self.scaler.transform(X, np.array(self.modes))
            rec = self.model.predict(Z[None], verbose=0)[0]
            sq = (Z - rec) ** 2
            ej = sq.mean(0); e = float(ej.mean())
            score = e / self.thr[m]
            alarm_flag = score > 1.0
            self.op_min += 1
        self.run_len = self.run_len + 1 if alarm_flag else 0
        alarm = self.run_len >= self.k

        devices = []
        for d in DEVICES:
            did = d[0]
            hi = None
            if ej is not None:
                ids = [CHANNELS.index(c) for c in d[4]]
                self.hi_raw[did].append(float(np.max(ej[ids] / self.ch_ref[ids])))
                hi = float(np.mean(self.hi_raw[did]))
                self.hi_hist[did].append((self.op_min, hi))
            elif self.hi_hist[did]:
                hi = self.hi_hist[did][-1][1]
            h = self._health(hi)
            rul = None
            if ej is not None and hi is not None and hi > 1.0 and alarm:
                hist = np.array(self.hi_hist[did])
                rul = estimate_rul(hist[:, 0], hist[:, 1], self.crit, horizon=self.cfg.rul_horizon)
            devices.append({"id": did, "name": d[1], "th": d[2], "subsystem": d[3],
                            "hi": None if hi is None else round(hi, 3),
                            "health": None if h is None else round(h, 1),
                            "status": self._status(h), "rul_min": None if rul is None else round(rul),
                            "values": {c: float(row.get(c, 0.0)) for c in d[4]}})

        # decision layer (L7)
        nb = next_planned_break(ts, self.cfg.break_times)
        to_break = (nb - ts).total_seconds() / 60
        cands = [dv for dv in devices if dv["rul_min"] is not None]
        top = min(cands, key=lambda dv: dv["rul_min"]) if cands else None
        if top is None:
            action = "watch" if alarm else "none"
        elif top["rul_min"] <= 60:
            action = "stop_now"
        elif top["rul_min"] < to_break:
            action = "stop_before_break"
        else:
            action = "plan_break"

        # alert episodes
        if alarm and self._open_alert is None:
            rank = sorted(devices, key=lambda dv: -(dv["hi"] or 0))
            self._open_alert = {"t": str(ts), "device": rank[0]["id"], "peak": score, "dur_min": 0}
            self.alerts.append(self._open_alert)
        if self._open_alert is not None:
            if alarm:
                self._open_alert["dur_min"] += 1
                self._open_alert["peak"] = max(self._open_alert["peak"], score or 0)
                self._open_alert["end"] = str(ts)
            else:
                self._open_alert = None

        self.last = {
            "time": str(ts), "mode": MODE_NAME[m], "score": None if score is None else round(score, 3),
            "alarm": int(alarm), "action": action, "next_break": nb.strftime("%H:%M"),
            "rul_device": None if top is None else top["id"],
            "rul_min": None if top is None else top["rul_min"],
            "critical_hi": self.crit, "devices": devices,
        }
        return self.last

    def state(self) -> dict:
        return self.last
