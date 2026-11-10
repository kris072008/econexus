"""Re-predict every trained model with its final checkpoint (last.pt, after the fixed 30 epochs).

Ultralytics' best.pt is chosen by accuracy on the validation split, which is also the split used to
fit the temperature and choose thresholds. Using last.pt keeps validation data for calibration only.

    python experiments/predict_last.py --preds experiments/preds --work experiments/work \
        --out experiments/preds_last
"""
import argparse
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from ultralytics import YOLO

CLASSES = ["general", "paper", "plastic"]
ap = argparse.ArgumentParser()
ap.add_argument("--preds", required=True, help="folder written by train_predict.py")
ap.add_argument("--work", required=True, help="its --work folder (holds runs/seed*_m*/weights)")
ap.add_argument("--out", required=True)
ap.add_argument("--imgsz", type=int, default=224)
ap.add_argument("--data-root", default="", help="prefix for relative image paths in the split files")
ap.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
args = ap.parse_args()
src, out = Path(args.preds), Path(args.out)
out.mkdir(parents=True, exist_ok=True)
for f in list(src.glob("split_seed*.csv")) + [src / "realwaste_index.csv", src / "environment.json"]:
    shutil.copy(f, out / f.name)
def paths(df):
    return [str(Path(args.data_root) / p) for p in df.path]


rw = pd.read_csv(src / "realwaste_index.csv")


def predict(model, paths):
    names = model.names
    order = [next(k for k, v in names.items() if v == c) for c in CLASSES]
    P = []
    for i in range(0, len(paths), 64):
        for r in model.predict(paths[i:i + 64], imgsz=args.imgsz, verbose=False, device=args.device):
            P.append(r.probs.data.cpu().numpy()[order])
    P = np.asarray(P, dtype=np.float64)
    if P.shape != (len(paths), 3):
        raise RuntimeError(f"bad prediction shape {P.shape}")
    return np.log(np.clip(P, 1e-12, 1.0))


for f in sorted(src.glob("pred_seed*_m*.npz")):
    s, m = f.stem.replace("pred_seed", "").split("_m")
    dst = out / f.name
    if dst.exists():
        continue
    w = Path(args.work) / "runs" / f"seed{s}_m{m}" / "weights" / "last.pt"
    if not w.exists():
        raise SystemExit(f"missing {w}")
    with np.load(f) as z:
        old = {k: z[k] for k in z.files}
    model = YOLO(str(w))
    val = pd.read_csv(src / f"split_seed{s}_val.csv")
    test = pd.read_csv(src / f"split_seed{s}_test.csv")
    old["logp_val"] = predict(model, paths(val))
    old["logp_test"] = predict(model, paths(test))
    old["logp_rw"] = predict(model, paths(rw))
    old["checkpoint"] = "last"
    np.savez_compressed(dst, **old)
    acc = (old["logp_test"].argmax(1) == old["y_test"]).mean()
    print(f"seed {s} member {m}: last.pt test acc {acc:.3f}", flush=True)
print("done")
