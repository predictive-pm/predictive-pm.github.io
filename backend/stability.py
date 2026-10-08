"""stability.py — aggregate mode-aware LSTM-AE metrics across random seeds (Table V 'Stability')."""
import json, glob, numpy as np
rows = [json.load(open("artifacts/results.json"))["LSTM-AE (mode-aware)"]]
for f in sorted(glob.glob("artifacts_seed*/results.json")):
    rows.append(json.load(open(f))["LSTM-AE (mode-aware)"])
def agg(vals):
    v = np.array(vals, float); m, s = v.mean(), v.std(ddof=1)
    return m, s, (s / m * 100 if m else 0)
out = {"n": len(rows), "rows": []}
for key, lab, fmt in [("AP", "AP (mean ± SD, CV)", "{:.1%}"), ("event_F0.5", "Event F0.5", "{:.3f}"),
                      ("FA_per_week", "FA / week", "{:.2f}"), ("lead", "Lead time (h)", "{:.2f}")]:
    vals = [r["lead_time_h"][0] if key == "lead" else r[key] for r in rows]
    m, s, cv = agg(vals)
    out["rows"].append({"k": lab, "v": f"{fmt.format(m)} ± {fmt.format(s)} · CV {cv:.1f}%"})
json.dump(out, open("artifacts/stability.json", "w"), indent=2, ensure_ascii=False)
print(json.dumps(out, indent=1, ensure_ascii=False))
