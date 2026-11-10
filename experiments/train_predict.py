"""Train YOLOv8n-cls on TrashNet splits and save predictions for later analysis.

For every split seed s this script
  1. makes the stratified 70/15/15 split used in the paper (seed s),
  2. fine-tunes one or more YOLOv8n-cls models on it (member m uses training seed 100*s + m),
  3. saves log-probabilities for the validation and test splits and for every RealWaste image.

Training and prediction are kept apart from the analysis (analyze.py), so the analysis can be
re-run without retraining. Usage:

    python experiments/train_predict.py --trashnet DATA/trashnet/dataset-resized \
        --realwaste DATA/realwaste/realwaste-main/RealWaste --out experiments/preds \
        --seeds 0-9 --members 1
"""
import argparse
import json
import os
import random
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split

CLASSES = ["general", "paper", "plastic"]  # alphabetical = Ultralytics folder order
MERGE_TRASHNET = {"cardboard": "paper", "paper": "paper", "plastic": "plastic",
                  "glass": "general", "metal": "general", "trash": "general"}
# RealWaste classes. "matched" marks classes that also exist in TrashNet; the others
# (food, textile, vegetation) are material the TrashNet model has never seen.
MERGE_REALWASTE = {"Cardboard": ("paper", True), "Paper": ("paper", True),
                   "Plastic": ("plastic", True), "Glass": ("general", True),
                   "Metal": ("general", True), "Miscellaneous Trash": ("general", True),
                   "Food Organics": ("general", False), "Textile Trash": ("general", False),
                   "Vegetation": ("general", False)}
IMG_EXT = {".jpg", ".jpeg", ".png"}


def parse_seeds(text):
    if "-" in text:
        a, b = text.split("-")
        return list(range(int(a), int(b) + 1))
    return [int(t) for t in text.split(",")]


def list_images(root, merge):
    rows = []
    for c in sorted(os.listdir(root)):
        d = Path(root) / c
        if not d.is_dir():
            continue
        if c not in merge:
            sys.exit(f"Unexpected class folder {c!r} in {root}")
        for p in sorted(d.iterdir()):
            if p.suffix.lower() in IMG_EXT:
                rows.append((str(p.resolve()), c))
    if not rows:
        sys.exit(f"No images found in {root}")
    return pd.DataFrame(rows, columns=["path", "orig"])


def make_split(df, seed, work):
    tr, rest = train_test_split(df, test_size=0.30, stratify=df.orig, random_state=seed)
    va, te = train_test_split(rest, test_size=0.50, stratify=rest.orig, random_state=seed)
    root = work / f"data_seed{seed}"
    if root.exists():
        shutil.rmtree(root)
    for name, part in (("train", tr), ("val", va), ("test", te)):
        for _, r in part.iterrows():
            d = root / name / r.cls
            d.mkdir(parents=True, exist_ok=True)
            os.symlink(r.path, d / Path(r.path).name)
    return root, {"train": tr, "val": va, "test": te}


def predict_logp(model, paths):
    names = model.names
    order = [next(k for k, v in names.items() if v == c) for c in CLASSES]
    probs = []
    batch = 64
    for i in range(0, len(paths), batch):
        for r in model.predict(paths[i:i + batch], imgsz=args.imgsz, verbose=False, device=args.device):
            probs.append(r.probs.data.cpu().numpy()[order])
    P = np.asarray(probs, dtype=np.float64)
    if P.shape != (len(paths), len(CLASSES)):
        raise RuntimeError(f"prediction shape {P.shape} does not match {len(paths)} images")
    if not np.allclose(P.sum(1), 1.0, atol=1e-3):
        raise RuntimeError("predicted probabilities do not sum to 1")
    return np.log(np.clip(P, 1e-12, 1.0))


def time_inference(model, paths, n=100):
    """CPU latency per image, batch size 1, after warm-up (a laptop proxy, not the Pi)."""
    for p in paths[:5]:
        model.predict(p, imgsz=args.imgsz, verbose=False, device="cpu")
    t0 = time.perf_counter()
    for p in paths[:n]:
        model.predict(p, imgsz=args.imgsz, verbose=False, device="cpu")
    return (time.perf_counter() - t0) / min(n, len(paths))


ap = argparse.ArgumentParser()
ap.add_argument("--trashnet", required=True)
ap.add_argument("--realwaste", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--work", default=None, help="folder for splits and training runs")
ap.add_argument("--seeds", default="0-9")
ap.add_argument("--members", type=int, default=1, help="models per split (deep ensemble)")
ap.add_argument("--epochs", type=int, default=30)
ap.add_argument("--imgsz", type=int, default=224)
ap.add_argument("--batch", type=int, default=64)
ap.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
args = ap.parse_args()

from ultralytics import YOLO  # noqa: E402  (import after argparse so --help is fast)
import ultralytics  # noqa: E402

out = Path(args.out)
out.mkdir(parents=True, exist_ok=True)
work = Path(args.work or out.parent / "work")
work.mkdir(parents=True, exist_ok=True)

tn = list_images(args.trashnet, MERGE_TRASHNET)
tn["cls"] = tn.orig.map(MERGE_TRASHNET)
rw = list_images(args.realwaste, MERGE_REALWASTE)
rw["cls"] = rw.orig.map(lambda c: MERGE_REALWASTE[c][0])
rw["matched"] = rw.orig.map(lambda c: MERGE_REALWASTE[c][1])
print(f"TrashNet {len(tn)} images, RealWaste {len(rw)} images", flush=True)
rw[["path", "orig", "cls", "matched"]].to_csv(out / "realwaste_index.csv", index=False)

env = {"python": sys.version.split()[0], "torch": torch.__version__, "ultralytics": ultralytics.__version__,
       "device": args.device, "epochs": args.epochs, "imgsz": args.imgsz, "batch": args.batch,
       "platform": sys.platform}
(out / "environment.json").write_text(json.dumps(env, indent=2))

for seed in parse_seeds(args.seeds):
    root, parts = make_split(tn, seed, work)
    for part, df in parts.items():
        df[["path", "orig", "cls"]].to_csv(out / f"split_seed{seed}_{part}.csv", index=False)
    y = {p: np.array([CLASSES.index(c) for c in parts[p].cls]) for p in ("val", "test")}
    for m in range(args.members):
        f = out / f"pred_seed{seed}_m{m}.npz"
        if f.exists():
            with np.load(f) as z:
                saved = (int(z["epochs"]), int(z["imgsz"])) if "epochs" in z.files else None
            # files written before these fields were added carry no record; accept them
            if saved is not None and saved != (args.epochs, args.imgsz):
                sys.exit(f"{f.name} exists but was made with epochs/imgsz {saved}; move it away first")
            print(f"skip {f.name} (exists)", flush=True)
            continue
        tseed = 100 * seed + m
        random.seed(tseed)
        np.random.seed(tseed)
        torch.manual_seed(tseed)
        t0 = time.time()
        model = YOLO("yolov8n-cls.pt")
        model.train(data=str(root), epochs=args.epochs, imgsz=args.imgsz, batch=args.batch,
                    seed=tseed, deterministic=True, device=args.device, workers=4,
                    project=str((work / "runs").resolve()), name=f"seed{seed}_m{m}",
                    exist_ok=True, verbose=False, plots=False)
        best = work / "runs" / f"seed{seed}_m{m}" / "weights" / "best.pt"
        if not best.exists():
            raise RuntimeError(f"training finished but {best} is missing")
        model = YOLO(str(best))
        res = {"val": predict_logp(model, list(parts["val"].path)),
               "test": predict_logp(model, list(parts["test"].path)),
               "realwaste": predict_logp(model, list(rw.path))}
        lat = time_inference(model, list(parts["test"].path)) if (seed == 0 and m == 0) else np.nan
        np.savez_compressed(f, logp_val=res["val"], y_val=y["val"], logp_test=res["test"], y_test=y["test"],
                            logp_rw=res["realwaste"], train_seed=tseed, cpu_latency_s=lat,
                            epochs=args.epochs, imgsz=args.imgsz,
                            train_minutes=(time.time() - t0) / 60)
        acc = (res["test"].argmax(1) == y["test"]).mean()
        print(f"seed {seed} member {m}: test acc {acc:.3f}, {(time.time() - t0) / 60:.1f} min", flush=True)
        if m == 0 and seed == 0:
            shutil.copy(best, out / "econexus_yolov8n_cls_seed0.pt")
print("done", flush=True)
