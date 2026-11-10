"""Near-duplicate checks for the RealWaste experiments (difference hash, Hamming distance <= 4).

1. RealWaste images that are near-duplicates of any TrashNet image (would blur the shift test).
2. For each split seed, RealWaste evaluation images that are near-duplicates of that seed's
   fine-tuning images (would inflate the fine-tuning result).

    python experiments/dupcheck_realwaste.py --preds experiments/preds --adapt experiments/adapt --data-root DATA
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image


def dhash(path, size=8):
    im = Image.open(path).convert("L").resize((size + 1, size), Image.Resampling.LANCZOS)
    a = np.asarray(im, dtype=np.int16)
    return (a[:, 1:] > a[:, :-1]).flatten()


def min_dist(A, B, chunk=500):
    out = np.empty(len(A), int)
    for i in range(0, len(A), chunk):
        out[i:i + chunk] = (A[i:i + chunk, None, :] != B[None, :, :]).sum(2).min(1)
    return out


ap = argparse.ArgumentParser()
ap.add_argument("--preds", required=True)
ap.add_argument("--adapt", required=True)
ap.add_argument("--data-root", default="")
ap.add_argument("--max-dist", type=int, default=4)
args = ap.parse_args()
P, root = Path(args.preds), Path(args.data_root)
rw = pd.read_csv(P / "realwaste_index.csv")
H_rw = np.stack([dhash(root / p) for p in rw.path])
tn_paths = sorted({p for f in P.glob("split_seed0_*.csv") for p in pd.read_csv(f).path})
H_tn = np.stack([dhash(root / p) for p in tn_paths])
cross = min_dist(H_rw, H_tn)
res = {"realwaste_vs_trashnet": {"realwaste_images": len(rw), "trashnet_images": len(tn_paths),
                                 "near_duplicates": int((cross <= args.max_dist).sum())}, "within_realwaste": {}}
for f in sorted(Path(args.adapt).glob("adapt_seed*_f20.npz")):
    with np.load(f) as z:
        ev, tr = z["idx_eval"], z["idx_train"]
    d = min_dist(H_rw[ev], H_rw[tr])
    res["within_realwaste"][f.stem] = {"eval": len(ev), "train": len(tr),
                                       "near_duplicates": int((d <= args.max_dist).sum())}
tot = sum(v["near_duplicates"] for v in res["within_realwaste"].values())
res["within_realwaste_total"] = {"near_duplicates": tot,
                                 "eval_images": sum(v["eval"] for v in res["within_realwaste"].values())}
(Path(args.adapt) / "dupcheck_realwaste.json").write_text(json.dumps(res, indent=2))
print(json.dumps({k: v for k, v in res.items() if k != "within_realwaste"}, indent=1))
