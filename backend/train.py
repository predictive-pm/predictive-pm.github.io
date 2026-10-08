"""
train.py — Train the LSTM-Autoencoder (paper Section III-F / Table IV)

Partitioning (strictly temporal, Section IV-B):
    Train 60 %  | Validation 20 % | Test 20 %
Training uses normal-condition data only (label == 'N', modes run/ramp-up).

Usage:
    python train.py --csv data/raw.csv --out artifacts [--epochs 40] [--seed 42]
Outputs:
    artifacts/lstm_ae_mode_aware.keras, artifacts/lstm_ae_mode_blind.keras
    artifacts/scaler_mode_aware.json,   artifacts/scaler_mode_blind.json
    artifacts/scores.npz   (e(t), e_j(t) for every valid window, both models)
    artifacts/train_meta.json
"""
import os, json, time, argparse
os.environ.setdefault("KERAS_BACKEND", "jax")
import numpy as np
import keras
from pdm_core import (load_csv, label_modes, quality_audit, ModeScaler, window_end_index,
                      make_windows, build_lstm_ae, reconstruction_errors, CHANNELS, MODES)

ap = argparse.ArgumentParser()
ap.add_argument("--csv", default="data/raw.csv")
ap.add_argument("--out", default="artifacts")
ap.add_argument("--window", type=int, default=30)
ap.add_argument("--latent", type=int, default=8)
ap.add_argument("--epochs", type=int, default=40)
ap.add_argument("--batch", type=int, default=128)
ap.add_argument("--lr", type=float, default=1e-3)
ap.add_argument("--patience", type=int, default=10)
ap.add_argument("--stride", type=int, default=2, help="training-window stride")
ap.add_argument("--seed", type=int, default=42)
ap.add_argument("--variants", default="mode_aware,mode_blind")
args = ap.parse_args()
os.makedirs(args.out, exist_ok=True)
keras.utils.set_random_seed(args.seed)

df = load_csv(args.csv)
mode = label_modes(df)
X = df[CHANNELS].values.astype(np.float32)
n = len(df)
i_tr, i_va = int(0.6 * n), int(0.8 * n)          # 60 / 20 / 20
normal = df["label"].values == "N"
w = args.window

ends = window_end_index(mode, w)
meta = {"audit": quality_audit(df), "split": {
    "train": [str(df.dt[0]), str(df.dt[i_tr - 1]), i_tr],
    "val": [str(df.dt[i_tr]), str(df.dt[i_va - 1]), i_va - i_tr],
    "test": [str(df.dt[i_va]), str(df.dt[n - 1]), n - i_va]},
    "mode_counts": {k: int((mode == v).sum()) for k, v in MODES.items()},
    "config": vars(args), "channels": CHANNELS, "models": {}}

all_scores = {"ends": ends}
for variant in args.variants.split(","):
    t0 = time.time()
    tr_mask = np.zeros(n, bool); tr_mask[:i_tr] = True; tr_mask &= normal
    sc = ModeScaler().fit(X, mode, tr_mask, mode_aware=(variant == "mode_aware"))
    Z = sc.transform(X, mode)
    # windows that are fully normal inside each partition
    lab_ok = np.r_[0, np.cumsum(~normal)]
    clean = (lab_ok[ends + 1] - lab_ok[ends + 1 - w]) == 0
    e_tr = ends[(ends < i_tr) & clean][:: args.stride]
    e_va = ends[(ends >= i_tr + w) & (ends < i_va) & clean][:: args.stride * 2]
    Wtr, Wva = make_windows(Z, e_tr, w), make_windows(Z, e_va, w)
    model = build_lstm_ae(w, len(CHANNELS), latent=args.latent, lr=args.lr)
    cb = [keras.callbacks.EarlyStopping(monitor="val_loss", patience=args.patience, restore_best_weights=True)]
    hist = model.fit(Wtr, Wtr, validation_data=(Wva, Wva), epochs=args.epochs,
                     batch_size=args.batch, callbacks=cb, verbose=2, shuffle=True)
    model.save(f"{args.out}/lstm_ae_{variant}.keras")
    json.dump(sc.to_json(), open(f"{args.out}/scaler_{variant}.json", "w"))
    # score every valid window of the whole record (chunked)
    E, EJ = [], []
    for s in range(0, len(ends), 20000):
        e, ej = reconstruction_errors(model, make_windows(Z, ends[s:s + 20000], w))
        E.append(e); EJ.append(ej)
    all_scores[f"e_{variant}"] = np.concatenate(E).astype(np.float32)
    all_scores[f"ej_{variant}"] = np.concatenate(EJ).astype(np.float32)
    meta["models"][variant] = {
        "train_windows": int(len(Wtr)), "val_windows": int(len(Wva)),
        "epochs_run": len(hist.history["loss"]),
        "best_val_loss": float(min(hist.history["val_loss"])),
        "loss": [float(v) for v in hist.history["loss"]],
        "val_loss": [float(v) for v in hist.history["val_loss"]],
        "params": int(model.count_params()), "train_seconds": round(time.time() - t0, 1)}
    print(variant, meta["models"][variant]["best_val_loss"], flush=True)

np.savez_compressed(f"{args.out}/scores.npz", mode=mode, **all_scores)
json.dump(meta, open(f"{args.out}/train_meta.json", "w"), indent=2, ensure_ascii=False)
print("done")
