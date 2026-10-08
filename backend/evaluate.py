"""
evaluate.py — Lead-time-oriented evaluation (paper Section IV-E, Table V/VI/VII)
No point adjustment is applied anywhere.

  * mode-specific threshold = percentile of TRAIN scores per operating mode,
    percentile level selected on VALIDATION (target false-alarm rate)
  * persistence rule: alarm after k consecutive exceedances
  * metrics on TEST: AP (threshold-free), event-level F0.5, FA/week,
    detection lead time LT = t_fail - t_alarm (Eq. 4), localization hit@1/@3
  * baselines: PCA, Isolation Forest, moving standard deviation
Outputs artifacts/results.json and artifacts/thresholds.json
"""
import os, json, argparse
os.environ.setdefault("KERAS_BACKEND", "jax")
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.metrics import average_precision_score
from pdm_core import (load_csv, label_modes, ModeScaler, persistent_alarm, episodes,
                      CHANNELS, DEVICES, CH2DEV, MODES, SCHEDULED_BREAK_HOURS)

ap = argparse.ArgumentParser()
ap.add_argument("--csv", default="data/raw.csv")
ap.add_argument("--art", default="artifacts")
ap.add_argument("--k", type=int, default=5, help="persistence (minutes)")
ap.add_argument("--horizon_h", type=float, default=12.0, help="pre-failure window used as positive class")
ap.add_argument("--fa_target", type=float, default=1.0, help="max false alarms / week on validation")
args = ap.parse_args()

df = load_csv(args.csv)
mode = label_modes(df)
n = len(df); i_tr, i_va = int(0.6 * n), int(0.8 * n)
part = np.zeros(n, np.int8); part[i_tr:i_va] = 1; part[i_va:] = 2
S = np.load(f"{args.art}/scores.npz")
ends = S["ends"]
scored = np.zeros(n, bool); scored[ends] = True

# ---- failure events from the record (label 'E' = recorded stoppage) -------
lab = df["label"].values
e_idx = np.where(lab == "E")[0]
ev_groups = np.split(e_idx, np.where(np.diff(e_idx) > 1)[0] + 1) if len(e_idx) else []
events = []
for g in ev_groups:
    # failure time = first stop minute inside the labelled stoppage
    stop_in = g[mode[g] == MODES["stop"]]
    t_fail = int(stop_in[0]) if len(stop_in) else int(g[0])
    events.append({"start": int(g[0]), "t_fail": t_fail, "end": int(g[-1]),
                   "t_fail_dt": str(df.dt[t_fail]), "duration_min": int(len(g))})
H = int(args.horizon_h * 60)
pos = np.zeros(n, bool)
for ev in events:
    pos[max(0, ev["t_fail"] - H): ev["t_fail"]] = True
    pos[ev["start"]: ev["end"] + 1] = True

def full(arr, fill=np.nan):
    out = np.full(n, fill, np.float64); out[ends] = arr; return out

# ---- baselines: pointwise scores on mode-wise normalised data -------------
X = df[CHANNELS].values.astype(np.float32)
trn = (part == 0) & (lab == "N") & (mode != MODES["stop"])
Z = ModeScaler().fit(X, mode, trn).transform(X, mode)
rng = np.random.default_rng(0)
sub = rng.choice(np.where(trn)[0], 60000, replace=False)
pca = PCA(n_components=0.90, random_state=0).fit(Z[sub])
pca_s = ((Z - pca.inverse_transform(pca.transform(Z))) ** 2).mean(1)
pca_s = pd.Series(pca_s).rolling(30, min_periods=1).mean().values      # same 30-min support
iso = IsolationForest(n_estimators=200, random_state=0).fit(Z[sub])
iso_s = pd.Series(-iso.score_samples(Z)).rolling(30, min_periods=1).mean().values
mstd = pd.DataFrame(Z).rolling(30, min_periods=10).std().values
ref = np.nanpercentile(mstd[trn], 99, axis=0)
mstd_s = np.nanmax(mstd / ref, axis=1)

detectors = {
    "LSTM-AE (mode-aware)": (full(S["e_mode_aware"]), True),
    "LSTM-AE (mode-blind)": (full(S["e_mode_blind"]), False),
    "PCA": (np.where(scored, pca_s, np.nan), True),
    "Isolation Forest": (np.where(scored, iso_s, np.nan), True),
    "Moving std rule": (np.where(scored, mstd_s, np.nan), True),
}

def weeks(mask):
    return mask.sum() / (60 * 24 * 7)

def evaluate(score, mode_aware, name):
    valid = ~np.isnan(score)
    m_groups = [MODES["ramp"], MODES["run"]] if mode_aware else [None]
    best = None
    for p in [99.0, 99.5, 99.9, 99.95, 99.99, 99.995, 100.0]:
        thr = np.full(n, np.inf)
        thr_tab = {}
        for m in m_groups:
            sel = valid & (part == 0) & (lab == "N") & ((mode == m) if m is not None else True)
            t = np.percentile(score[sel], p) * (1.0 if p < 100 else 1.05)
            thr_tab["all" if m is None else [k for k, v in MODES.items() if v == m][0]] = float(t)
            if m is None:
                thr[valid] = t
            else:
                thr[valid & (mode == m)] = t
        flag = valid & (score > thr)
        alarm = persistent_alarm(flag, args.k)
        va_eps = [e for e in episodes(alarm & (part == 1))]
        fa_va = len(va_eps) / weeks((part == 1) & (mode != MODES["stop"]))
        if fa_va <= args.fa_target:
            best = (p, thr, thr_tab, alarm, fa_va); break
        best = (p, thr, thr_tab, alarm, fa_va)
    p, thr, thr_tab, alarm, fa_va = best
    te = (part == 2) & valid
    norm = np.where(valid, score / np.where(np.isfinite(thr), thr, 1), np.nan)
    ap_ = float(average_precision_score(pos[te], norm[te])) if pos[te].any() else None
    eps = episodes(alarm & (part == 2))
    tp_eps = [e for e in eps if pos[e[0]: e[1] + 1].any()]
    fp_eps = [e for e in eps if not pos[e[0]: e[1] + 1].any()]
    test_events = [ev for ev in events if part[ev["t_fail"]] == 2]
    detected, leads = 0, []
    first_alarm = None
    for ev in test_events:
        win = np.arange(max(0, ev["t_fail"] - H), ev["t_fail"])
        hits = win[alarm[win]]
        if len(hits):
            detected += 1
            leads.append((ev["t_fail"] - hits[0]) / 60.0)
            first_alarm = int(hits[0])
    prec = len(tp_eps) / len(eps) if eps else 0.0
    rec = detected / len(test_events) if test_events else 0.0
    b2 = 0.25
    f05 = (1 + b2) * prec * rec / (b2 * prec + rec) if (prec + rec) else 0.0
    fa_wk = len(fp_eps) / weeks((part == 2) & (mode != MODES["stop"]) & ~pos)
    return {"name": name, "percentile": p, "thresholds": thr_tab, "k": args.k,
            "val_fa_per_week": round(fa_va, 3), "AP": None if ap_ is None else round(ap_, 4),
            "event_precision": round(prec, 3), "event_recall": round(rec, 3),
            "event_F0.5": round(f05, 3), "FA_per_week": round(fa_wk, 3),
            "lead_time_h": [round(x, 2) for x in leads],
            "first_alarm_idx": first_alarm,
            "first_alarm_dt": None if first_alarm is None else str(df.dt[first_alarm]),
            "test_alarm_episodes": len(eps)}, alarm, thr

results, alarms = {}, {}
for name, (sc, ma) in detectors.items():
    r, alarm, thr = evaluate(sc, ma, name)
    results[name] = r; alarms[name] = (alarm, thr)
    print(json.dumps(r, ensure_ascii=False))

# ---- fault localization (Table VII) using Eq.3 ----------------------------
main = "LSTM-AE (mode-aware)"
ej = np.full((n, len(CHANNELS)), np.nan); ej[ends] = S["ej_mode_aware"]
# per-channel normalisation by its own train-partition 99.9th percentile (run mode)
sel = scored & (part == 0) & (lab == "N") & (mode == MODES["run"])
ch_ref = np.percentile(ej[sel], 99.9, axis=0)
loc = []
for ev in events:
    fa_i = results[main]["first_alarm_idx"]
    if fa_i is None:
        continue
    span = np.arange(fa_i, ev["t_fail"])
    span = span[scored[span]]
    agg = np.nanmean(ej[span] / ch_ref, axis=0)
    order = np.argsort(-agg)
    ranked_dev = []
    for j in order:
        d = CH2DEV[CHANNELS[j]]
        if d not in ranked_dev:
            ranked_dev.append(d)
    loc.append({"event": ev["t_fail_dt"], "top_channels": [CHANNELS[j] for j in order[:3]],
                "top_devices": ranked_dev[:3],
                "channel_ratio": {CHANNELS[j]: round(float(agg[j]), 2) for j in order}})
results["_localization"] = loc
results["_events"] = events
results["_config"] = {"k": args.k, "horizon_h": args.horizon_h, "fa_target": args.fa_target,
                      "test_weeks": round(float(weeks(part == 2)), 2)}
json.dump(results, open(f"{args.art}/results.json", "w"), indent=2, ensure_ascii=False)
json.dump({"mode_aware": results[main]["thresholds"], "k": args.k, "channel_ref_p999": ch_ref.tolist(),
           "channels": CHANNELS}, open(f"{args.art}/thresholds.json", "w"), indent=2)
np.savez_compressed(f"{args.art}/eval_arrays.npz", pos=pos, alarm=alarms[main][0], thr=alarms[main][1],
                    ch_ref=ch_ref)
print(json.dumps(loc, indent=1))
