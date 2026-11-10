"""Does a little labelled target data fix the shift? Fine-tune TrashNet models on part of RealWaste.

For each split seed s, RealWaste is divided once (stratified on its nine classes) into
  eval 60%  - fixed test set for every setting,
  calib 10% - fits the temperature (thresholds are fixed at k/(k + 1), not tuned),
  pool 30%  - fine-tuning data; the first third (10% of RealWaste) and first two thirds (20%) are used.
The TrashNet model of split s (last.pt after 30 epochs) is fine-tuned for a fixed number of epochs on
each fine-tuning set and the final weights are used. The calibration part is passed to Ultralytics
as its validation folder only for monitoring; no weights are selected on it (last.pt is used). Predictions are saved for calib, eval and the
TrashNet test split of seed s (to see what fine-tuning does to the original domain).

    python experiments/adapt_realwaste.py --preds experiments/preds --work experiments/work \
        --data-root DATA --out experiments/adapt
    python experiments/adapt_realwaste.py ... --analyze-only
"""
import argparse
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

CLASSES = ["general", "paper", "plastic"]
FRACS = {"10": 1 / 3, "20": 2 / 3}  # share of the 30% pool -> 10% and 20% of RealWaste

ap = argparse.ArgumentParser()
ap.add_argument("--preds", required=True)
ap.add_argument("--work", required=True)
ap.add_argument("--data-root", default="")
ap.add_argument("--out", required=True)
ap.add_argument("--seeds", default="0-9")
ap.add_argument("--epochs", type=int, default=15)
ap.add_argument("--imgsz", type=int, default=224)
ap.add_argument("--analyze-only", action="store_true")
args = ap.parse_args()
a, b = (int(x) for x in args.seeds.split("-"))
SEEDS = list(range(a, b + 1))
PRED, OUT = Path(args.preds), Path(args.out)
OUT.mkdir(parents=True, exist_ok=True)
root = Path(args.data_root)
rw = pd.read_csv(PRED / "realwaste_index.csv")
rw["abs"] = [str((root / p).resolve()) for p in rw.path]
missing = [p for p in rw["abs"][:20] if not os.path.exists(p)]
if missing:
    sys.exit(f"RealWaste images not found, e.g. {missing[0]}; check --data-root")
rw["y"] = [CLASSES.index(c) for c in rw.cls]


def partition(seed):
    idx = np.arange(len(rw))
    ev, rest = train_test_split(idx, test_size=0.40, stratify=rw.orig, random_state=2000 + seed)
    cal, pool = train_test_split(rest, test_size=0.75, stratify=rw.orig.values[rest], random_state=2000 + seed)
    rng = np.random.default_rng(3000 + seed)
    pool = rng.permutation(pool)
    sets = {k: pool[: int(round(f * len(pool)))] for k, f in FRACS.items()}
    return ev, cal, sets


def build_folder(dst, train_idx, val_idx):
    if dst.exists():
        shutil.rmtree(dst)
    for name, ids in (("train", train_idx), ("val", val_idx)):
        for i in ids:
            d = dst / name / rw.cls.iloc[i]
            d.mkdir(parents=True, exist_ok=True)
            src = rw["abs"].iloc[i]
            os.symlink(src, d / f"{rw.orig.iloc[i].replace(' ', '_')}_{Path(src).name}")


def predict(model, paths):
    names = model.names
    order = [next(k for k, v in names.items() if v == c) for c in CLASSES]
    P = []
    for i in range(0, len(paths), 64):
        for r in model.predict(paths[i:i + 64], imgsz=args.imgsz, verbose=False, device=dev):
            P.append(r.probs.data.cpu().numpy()[order])
    P = np.asarray(P, dtype=np.float64)
    if P.shape != (len(paths), 3):
        raise RuntimeError(f"bad prediction shape {P.shape}")
    return np.log(np.clip(P, 1e-12, 1.0))


if not args.analyze_only:
    import torch
    from ultralytics import YOLO
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    for s in SEEDS:
        ev, cal, sets = partition(s)
        tn_test = pd.read_csv(PRED / f"split_seed{s}_test.csv")
        tn_paths = [str(root / p) for p in tn_test.path]
        base_w = Path(args.work) / "runs" / f"seed{s}_m0" / "weights" / "last.pt"
        if not base_w.exists():
            sys.exit(f"missing {base_w}")
        for key, tr in sets.items():
            f = OUT / f"adapt_seed{s}_f{key}.npz"
            if f.exists():
                print("skip", f.name, flush=True)
                continue
            data = OUT / "work" / f"data_seed{s}_f{key}"
            build_folder(data, tr, cal)
            m = YOLO(str(base_w))
            m.train(data=str(data), epochs=args.epochs, imgsz=args.imgsz, batch=64, seed=4000 + s,
                    deterministic=True, device=dev, workers=4, project=str((OUT / "work" / "runs").resolve()),
                    name=f"seed{s}_f{key}", exist_ok=True, verbose=False, plots=False)
            w = OUT / "work" / "runs" / f"seed{s}_f{key}" / "weights" / "last.pt"
            m = YOLO(str(w))
            res = {"logp_cal": predict(m, list(rw["abs"].values[cal])),
                   "logp_eval": predict(m, list(rw["abs"].values[ev])),
                   "logp_tn_test": predict(m, tn_paths)}
            np.savez_compressed(f, **res, idx_eval=ev, idx_cal=cal, idx_train=tr,
                                y_tn_test=np.array([CLASSES.index(c) for c in tn_test.cls]),
                                epochs=args.epochs)
            acc = (res["logp_eval"].argmax(1) == rw.y.values[ev]).mean()
            print(f"seed {s} fine-tune {key}% ({len(tr)} images): RealWaste eval acc {acc:.3f}", flush=True)
        # the un-adapted model on the same partition
        f0 = OUT / f"adapt_seed{s}_f0.npz"
        if not f0.exists():
            with np.load(PRED / f"pred_seed{s}_m0.npz") as z:
                lr, lt = z["logp_rw"], z["logp_test"]
                y_tn = z["y_test"]
            np.savez_compressed(f0, logp_cal=lr[cal], logp_eval=lr[ev], logp_tn_test=lt, idx_eval=ev,
                                idx_cal=cal, idx_train=np.array([], int), y_tn_test=y_tn, epochs=0)

# ------------------------------------------------------------------ analysis
sys.path.insert(0, str(Path(__file__).parent))
from scipy.optimize import minimize_scalar  # noqa: E402
from scipy.stats import t as tdist  # noqa: E402


def softmax_T(l, T=1.0):
    z = l / T
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def nll(P, y):
    return float(-np.mean(np.log(np.clip(P[np.arange(len(y)), y], 1e-12, 1))))


def fitT(l, y):
    r = minimize_scalar(lambda lt: nll(softmax_T(l, np.exp(lt)), y), bounds=(-3, 3), method="bounded")
    return float(np.exp(r.x))


def ece(P, y, bins=15):
    conf, pred = P.max(1), P.argmax(1)
    e = 0.0
    for lo, hi in zip(np.linspace(0, 1, bins + 1)[:-1], np.linspace(0, 1, bins + 1)[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            e += m.mean() * abs((pred[m] == y[m]).mean() - conf[m].mean())
    return e


def route(P, tau):
    pred, c = P.argmax(1), P.max(1)
    r = pred.copy()
    r[np.isin(pred, [1, 2]) & (c < tau)] = 0
    return r


def outcome(r, y, k):
    adm = np.isin(r, [1, 2])
    cont = adm & (r != y)
    lost = np.isin(y, [1, 2]) & (r == 0)
    return {"adm": 100 * adm.mean(), "cont": 100 * cont.sum() / adm.sum() if adm.sum() else np.nan,
            "cost": k * cont.sum() + lost.sum()}


rows = []
for s in SEEDS:
    for key in ("0", "10", "20"):
        f = OUT / f"adapt_seed{s}_f{key}.npz"
        if not f.exists():
            continue
        with np.load(f) as z:
            d = {k: z[k] for k in z.files}
        yc, ye = rw.y.values[d["idx_cal"]], rw.y.values[d["idx_eval"]]
        matched = rw.matched.values[d["idx_eval"]].astype(bool)
        T = fitT(d["logp_cal"], yc)
        for subset, mask in (("matched", matched), ("all", np.ones(len(ye), bool))):
            P = softmax_T(d["logp_eval"][mask], T)
            y = ye[mask]
            row = {"seed": s, "finetune_%": int(key), "subset": subset, "T": T,
                   "accuracy_%": 100 * (P.argmax(1) == y).mean(), "ECE_%": 100 * ece(P, y), "NLL": nll(P, y)}
            u = {k: outcome(P.argmax(1), y, k) for k in (3, 9)}
            row["ungated_contamination_%"] = u[9]["cont"]
            row["ungated_admitted_%"] = u[9]["adm"]
            for k in (3, 9):
                g = outcome(route(P, k / (k + 1)), y, k)
                row[f"k{k}_contamination_%"] = g["cont"]
                row[f"k{k}_admitted_%"] = g["adm"]
                row[f"k{k}_rel_cost"] = g["cost"] / u[k]["cost"] if u[k]["cost"] else np.nan
            if subset == "all":
                Pt = softmax_T(d["logp_tn_test"], T)
                row["trashnet_test_accuracy_%"] = 100 * (Pt.argmax(1) == d["y_tn_test"]).mean()
            rows.append(row)
df = pd.DataFrame(rows)
df.to_csv(OUT / "per_seed_adapt.csv", index=False)


def pm(v, d=1):
    v = np.asarray(v, float)
    return f"{np.nanmean(v):.{d}f} ± {np.nanstd(v, ddof=1):.{d}f}"


n_eval = len(partition(SEEDS[0])[0])
n_train_tn = len(pd.read_csv(PRED / f"split_seed{SEEDS[0]}_train.csv"))


def nb(diffs, ratio):
    """Corrected resampled t-test across overlapping RealWaste partitions (Nadeau & Bengio);
    ratio = evaluation items / RealWaste items used for fitting (fine-tuning + calibration)."""
    d = np.asarray(diffs, float)
    J = len(d)
    se = np.sqrt(np.var(d, ddof=1) * (1 / J + ratio))
    q = tdist.ppf(0.975, J - 1)
    return {"mean": float(d.mean()), "ci95": [float(d.mean() - q * se), float(d.mean() + q * se)],
            "p": float(2 * tdist.sf(abs(d.mean() / se), J - 1))}


agg, stats = [], {}
for (subset, ft), g in df.groupby(["subset", "finetune_%"]):
    agg.append({"subset": subset, "fine-tuning data (% of RealWaste)": ft, "splits": len(g),
                "T": pm(g["T"], 2), "accuracy (%)": pm(g["accuracy_%"]), "ECE (%)": pm(g["ECE_%"]),
                "NLL": pm(g["NLL"], 3),
                "ungated contamination (%)": pm(g["ungated_contamination_%"]),
                "k=3 contamination (%)": pm(g["k3_contamination_%"]), "k=3 admitted (%)": pm(g["k3_admitted_%"]),
                "k=3 cost / ungated": pm(g["k3_rel_cost"], 2),
                "k=9 contamination (%)": pm(g["k9_contamination_%"]), "k=9 admitted (%)": pm(g["k9_admitted_%"]),
                "k=9 cost / ungated": pm(g["k9_rel_cost"], 2),
                "TrashNet test accuracy (%)": pm(g["trashnet_test_accuracy_%"]) if subset == "all" else ""})
    if ft == 0:
        continue
    # paired against the un-adapted model on the same partition; conservative ratio
    # n_eval / n_fine-tune (exploratory: 8 comparisons, no multiplicity correction)
    base = df[(df.subset == subset) & (df["finetune_%"] == 0)].set_index("seed")
    gg = g.set_index("seed").loc[base.index]
    ratio = 0.6 / (ft / 100)
    stats[f"{subset}_f{ft}"] = {c: nb((gg[c] - base[c]).values, ratio)
                                for c in ("accuracy_%", "ungated_contamination_%", "k9_contamination_%", "ECE_%")}
A = pd.DataFrame(agg)
A.to_csv(OUT / "table_adapt.csv", index=False)
(OUT / "table_adapt.md").write_text(A.to_markdown(index=False))
summary = {"seeds": SEEDS, "epochs": args.epochs, "n_eval": n_eval,
           "n_cal": len(partition(SEEDS[0])[1]), "n_train": {k: len(v) for k, v in partition(SEEDS[0])[2].items()},
           "stats": stats, "table": agg}
(OUT / "summary_adapt.json").write_text(json.dumps(summary, indent=2))
print(A.to_string())
