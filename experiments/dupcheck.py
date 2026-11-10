"""Near-duplicate check between training and test images (TrashNet) using a difference hash.

Counts test images whose 64-bit dHash is within a small Hamming distance of some training image in
the same split. Reported as a leakage check; it does not catch re-photographed objects at new angles.

    python experiments/dupcheck.py --preds experiments/preds --seeds 0-9 --max-dist 4
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
    return np.packbits((a[:, 1:] > a[:, :-1]).flatten())


ap = argparse.ArgumentParser()
ap.add_argument("--preds", required=True)
ap.add_argument("--seeds", default="0-9")
ap.add_argument("--max-dist", type=int, default=4)
ap.add_argument("--data-root", default="", help="prefix for relative image paths in the split files")
args = ap.parse_args()
a, b = (int(x) for x in args.seeds.split("-"))
P = Path(args.preds)
cache = {}
out = {}
for s in range(a, b + 1):
    tr = pd.read_csv(P / f"split_seed{s}_train.csv")
    te = pd.read_csv(P / f"split_seed{s}_test.csv")
    for df in (tr, te):
        df["path"] = [str(Path(args.data_root) / p) for p in df.path]
    for p in list(tr.path) + list(te.path):
        if p not in cache:
            cache[p] = dhash(p)
    H_tr = np.unpackbits(np.stack([cache[p] for p in tr.path]), axis=1).astype(bool)
    H_te = np.unpackbits(np.stack([cache[p] for p in te.path]), axis=1).astype(bool)
    d = (H_te[:, None, :] != H_tr[None, :, :]).sum(2).min(1)
    out[s] = {"test": len(te), "near_duplicates": int((d <= args.max_dist).sum()),
              "exact_hash_matches": int((d == 0).sum()),
              "flagged_test_rows": [int(i) for i in np.where(d <= args.max_dist)[0]]}
    print(s, out[s], flush=True)
tot = {k: sum(v[k] for v in out.values()) for k in ("test", "near_duplicates", "exact_hash_matches")}
print(json.dumps({"total": tot, "max_dist": args.max_dist}))
(P / "dupcheck.json").write_text(json.dumps({"per_seed": out, "total": tot, "max_dist": args.max_dist}, indent=2))
