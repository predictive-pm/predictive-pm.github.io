"""
export_ui.py — Build the data contract consumed by the web application
(artifacts/ui_data.json). The FastAPI service (api.py) emits the SAME schema
for live operation, so the UI can switch from replay to real-time unchanged.
"""
import os, json, argparse
import numpy as np
import pandas as pd
from pdm_core import (load_csv, label_modes, CHANNELS, DEVICES, MODES, estimate_rul, SCHEDULED_BREAK_HOURS, next_planned_break)

ap = argparse.ArgumentParser()
ap.add_argument("--csv", default="data/raw.csv")
ap.add_argument("--art", default="artifacts")
ap.add_argument("--replay_from", default="2026-07-16 05:00")
ap.add_argument("--replay_to", default="2026-07-16 22:40")
args = ap.parse_args()

df = load_csv(args.csv)
mode = label_modes(df)
n = len(df); i_tr, i_va = int(0.6 * n), int(0.8 * n)
S = np.load(f"{args.art}/scores.npz"); E = np.load(f"{args.art}/eval_arrays.npz")
R = json.load(open(f"{args.art}/results.json")); M = json.load(open(f"{args.art}/train_meta.json"))
TH = json.load(open(f"{args.art}/thresholds.json"))
ends = S["ends"]
scored = np.zeros(n, bool); scored[ends] = True
e = np.full(n, np.nan); e[ends] = S["e_mode_aware"]
ej = np.full((n, len(CHANNELS)), np.nan); ej[ends] = S["ej_mode_aware"]
thr = E["thr"]; alarm = E["alarm"]; ch_ref = E["ch_ref"]
mode_name = {v: k for k, v in MODES.items()}

# ---- device health index HI_d(t) = max_j e_j / ref_j  (smoothed 15 min) ---
HI = {}
for d in DEVICES:
    ids = [CHANNELS.index(c) for c in d[4]]
    hi = np.nanmax(ej[:, ids] / ch_ref[ids], axis=1)
    sm = pd.Series(hi).rolling(15, min_periods=1).mean().to_numpy(copy=True)
    sm[~scored] = np.nan                     # no state while stopped / window incomplete
    HI[d[0]] = sm
ev = R["_events"][0]; t_fail = ev["t_fail"]
top_dev = R["_localization"][0]["top_devices"][0] if R["_localization"] else "RAO"
# critical level: smoothed HI of the attributed device just before the recorded stoppage
crit = float(np.nanmax(HI[top_dev][t_fail - 10: t_fail]))
crit = round(crit, 2)

def health_pct(hi):
    """100 % at HI<=0.5, 85 % at HI=1 (alarm threshold), 0 % at the critical level
    (square-root mapping so early degradation is visible)."""
    hi = np.asarray(hi, float)
    out = np.where(hi <= 1.0, 100 - 30 * np.clip(hi - 0.5, 0, 0.5),
                   85 * (1 - np.sqrt(np.clip((hi - 1.0) / (crit - 1.0), 0, 1))))
    return np.where(np.isnan(hi), np.nan, out)

# ---- replay window -----------------------------------------------------
a = int(df.index[df.dt == pd.Timestamp(args.replay_from)][0])
b = int(df.index[df.dt == pd.Timestamp(args.replay_to)][0])
op_min = np.cumsum(scored).astype(float)          # operating-minute clock (stops excluded)
rul = np.full(n, np.nan)
action = ["none"] * n
nb = [""] * n
for t in range(a, b + 1):
    if not scored[t]:
        continue
    h = HI[top_dev][max(a, t - 400): t + 1]
    if np.isnan(h[-1]) or h[-1] <= 1.0 or not alarm[max(a, t - 30): t + 1].any():
        continue
    r = estimate_rul(op_min[max(a, t - 400): t + 1], h, crit, horizon=120)
    if r is None:
        continue
    rul[t] = r
rul_s = pd.Series(rul).rolling(5, min_periods=1).median().to_numpy(copy=True)
rul = np.where(np.isnan(rul), np.nan, rul_s)
for t in range(a, b + 1):
    nbt = next_planned_break(df.dt[t]); nb[t] = nbt.strftime("%H:%M")
    to_break = (nbt - df.dt[t]).total_seconds() / 60
    if np.isnan(rul[t]):
        action[t] = "watch" if alarm[t] else "none"
    elif rul[t] <= 60:
        action[t] = "stop_now"          # failure within 1 h: stop the machine immediately
    elif rul[t] < to_break:
        action[t] = "stop_before_break" # failure expected before the next planned break
    else:
        action[t] = "plan_break"        # intervene in the next planned break
# RUL accuracy at checkpoints (calibrated on this single event -> in-sample)
rul_eval = []
for hrs in [6, 4, 3, 2, 1, 0.5]:
    t = t_fail - int(hrs * 60)
    while t < t_fail and np.isnan(rul[t]):
        t += 1
    if t < t_fail:
        true_op = float(op_min[t_fail - 1] - op_min[t])
        rul_eval.append({"checkpoint_h_before": hrs, "t": str(df.dt[t]), "pred_min": round(float(rul[t]), 0),
                         "true_op_min": true_op, "true_clock_min": int(t_fail - t)})
print(rul_eval)

def r2(x, nd=2):
    return [None if (v is None or (isinstance(v, float) and np.isnan(v))) else round(float(v), nd) for v in x]

replay = {
    "start": args.replay_from, "step_min": 1, "n": b - a + 1,
    "mode": [mode_name[int(m)] for m in mode[a:b + 1]],
    "score": r2(e[a:b + 1] / np.where(np.isfinite(thr[a:b + 1]), thr[a:b + 1], np.nan), 3),
    "alarm": [int(x) for x in alarm[a:b + 1]],
    "rul_min": r2(rul[a:b + 1], 0),
    "rul_device": top_dev, "action": action[a:b + 1], "next_break": nb[a:b + 1],
    "health": {d[0]: r2(health_pct(HI[d[0]][a:b + 1]), 1) for d in DEVICES},
    "hi": {d[0]: r2(HI[d[0]][a:b + 1], 2) for d in DEVICES},
    "ch": {c: r2(df[c].values[a:b + 1], 2) for c in CHANNELS},
    "failure_at": str(df.dt[t_fail]),
}

# ---- per-device static info ---------------------------------------------
run = mode != MODES["stop"]
devices = []
for d in DEVICES:
    load_ch = d[4][0]
    on = df[load_ch].values > 0
    runtime_h = on.sum() / 60
    runtime_h_at_start = on[:a].sum() / 60
    starts = int(((np.diff(on.astype(int)) == 1)).sum())
    nominal = {c: round(float(df.loc[run & (np.arange(n) < i_tr), c].mean()), 3) for c in d[4]}
    # daily median health (all record)
    dd = pd.DataFrame({"day": df.dt.dt.date, "h": health_pct(HI[d[0]])})
    daily = dd.groupby("day")["h"].min().round(1)
    devices.append({"id": d[0], "name": d[1], "th": d[2], "subsystem": d[3], "channels": d[4],
                    "nominal": nominal, "runtime_h": round(runtime_h, 1), "runtime_h_at_start": round(runtime_h_at_start, 2), "starts": starts,
                    "starts_at_start": int(((np.diff(on[:a].astype(int)) == 1)).sum()),
                    "health_daily_min": [None if np.isnan(v) else float(v) for v in daily.values]})
days = [str(x) for x in pd.Series(df.dt.dt.date.unique())]

# ---- daily operations ----------------------------------------------------
g = pd.DataFrame({"day": df.dt.dt.date, "run": run, "alarm": alarm, "e": e / np.where(np.isfinite(thr), thr, np.nan)})
daily = g.groupby("day").agg(run_h=("run", lambda x: round(x.sum() / 60, 2)),
                            stop_h=("run", lambda x: round((~x).sum() / 60, 2)),
                            alarm_min=("alarm", "sum"), max_score=("e", "max")).reset_index()
daily_ops = [{"d": str(r.day), "run_h": r.run_h, "stop_h": r.stop_h, "alarm_min": int(r.alarm_min),
              "max_score": None if np.isnan(r.max_score) else round(float(r.max_score), 2)} for r in daily.itertuples()]

# ---- timeline segments (last 10 days) -----------------------------------
def segments(i0, i1):
    segs, s = [], i0
    lab = df["label"].values
    for i in range(i0 + 1, i1 + 1):
        if i == i1 or mode[i] != mode[s] or (lab[i] != lab[s]):
            kind = mode_name[int(mode[s])]
            if lab[s] == "E" and kind == "stop":
                kind = "fault"
            elif kind == "stop":
                kind = "planned_stop" if df.dt[s].hour in SCHEDULED_BREAK_HOURS else "stop"
            segs.append([int((df.dt[s] - df.dt[i0].normalize()).total_seconds() // 60),
                         int((df.dt[i - 1] - df.dt[i0].normalize()).total_seconds() // 60) + 1, kind])
            s = i
    return segs
timeline = []
rday = args.replay_from[:10]
ri = days.index(rday)
for day in days[max(0, ri - 9): ri + 1]:
    idx = np.where(df.dt.dt.date.astype(str).values == day)[0]
    timeline.append({"d": day, "segments": segments(int(idx[0]), int(idx[-1]) + 1)})

# ---- alerts & activity --------------------------------------------------
from pdm_core import episodes
alerts = []
for (s, t) in episodes(alarm):
    sc = float(np.nanmax(e[s:t + 1] / thr[s:t + 1]))
    span = np.arange(s, t + 1)
    agg = np.nanmean(ej[span] / ch_ref, axis=0)
    j = int(np.nanargmax(agg))
    dev = [d[0] for d in DEVICES if CHANNELS[j] in d[4]][0]
    alerts.append({"t": str(df.dt[s]), "end": str(df.dt[t]), "dur_min": int(t - s + 1),
                   "device": dev, "channel": CHANNELS[j], "peak": round(sc, 2),
                   "part": "train" if s < i_tr else ("val" if s < i_va else "test")})
meta = M["split"]
out = {
    "meta": {"machine": "Pot & Lid Washing Machine", "machine_th": "เครื่องล้างหม้อหุงและฝา",
             "rows": int(n), "period": [str(df.dt[0]), str(df.dt[n - 1])], "split": meta,
             "sampling": "1 min", "critical_hi": crit},
    "devices": devices, "days": days,
    "model": {"name": "LSTM-Autoencoder (mode-aware)", "window": 30, "latent": 8, "depth": 2,
              "channels": len(CHANNELS), "params": M["models"]["mode_aware"]["params"],
              "epochs": M["models"]["mode_aware"]["epochs_run"],
              "train_windows": M["models"]["mode_aware"]["train_windows"],
              "loss": M["models"]["mode_aware"]["loss"], "val_loss": M["models"]["mode_aware"]["val_loss"],
              "thresholds": TH["mode_aware"], "k": TH["k"],
              "table": [{k: v for k, v in r.items() if k not in ("first_alarm_idx",)} for k_, r in R.items() if not k_.startswith("_")],
              "localization": R["_localization"], "rul_eval": rul_eval,
              "stability": json.load(open(f"{args.art}/stability.json")) if os.path.exists(f"{args.art}/stability.json") else None, "events": R["_events"], "eval": R["_config"]},
    "replay": replay, "daily": daily_ops, "timeline": timeline, "alerts": alerts,
}
json.dump(out, open(f"{args.art}/ui_data.json", "w"), ensure_ascii=False, separators=(",", ":"))
print("crit", crit, "top", top_dev, "alerts", len(alerts), "size", os.path.getsize(f"{args.art}/ui_data.json"))
