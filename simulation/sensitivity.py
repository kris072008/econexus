"""One-at-a-time sensitivity of the collection results to the assumed simulation inputs.

For each assumed parameter, the low and high values below replace the base value; the fixed
three-times-weekly schedule and fill-based collection with co-collection are re-run on the same
generated waste, and the two headline comparisons are recorded:
  * overflow reduction of fill-based collection relative to the fixed schedule (%)
  * distance of fill-based collection relative to the fixed schedule (ratio)

    python simulation/sensitivity.py --seeds 5 --out simulation/results
"""
import argparse
import copy
import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import district_sim as ds  # noqa: E402

RANGES = {  # parameter: (low, high, label)
    "share_to_recycling_bin": (0.15, 0.35, "Share of waste placed in the bin"),
    "hub_daily_cv": (0.10, 0.40, "Day-to-day variation per hub (σ)"),
    "surge_prob_per_day": (0.0, 0.10, "Share of surge days"),
    "surge_multiplier": (1.4, 2.2, "Surge-day multiplier"),
    "district_side_km": (1.0, 2.0, "District side (km)"),
    "depot_offset_km": (1.0, 3.0, "Station distance from centre (km)"),
    "circuity_factor": (1.2, 1.4, "Road / straight-line distance"),
    "truck_speed_kmh": (15.0, 25.0, "Truck speed (km/h)"),
    "service_min_per_hub": (3.0, 7.0, "Service time per bin (min)"),
    "fill_threshold": (0.50, 0.70, "Fill trigger"),
    "co_collect_threshold": (0.20, 0.40, "Co-collection level"),
}


def headline(P, seeds):
    pts, D = ds.layout(P)
    router = ds.Router(D, P["truck_capacity_L"])
    fx, fb = [], []
    for s in range(seeds):
        dep, _, _ = ds.generate_deposits(P, s)
        fx.append(ds.run(P, s, "fixed_3x", router, D, dep))
        fb.append(ds.run(P, s, "fill_based", router, D, dep))
    of = np.mean([r["overflow_hubdays_per30d"] for r in fx])
    ob = np.mean([r["overflow_hubdays_per30d"] for r in fb])
    kf = np.mean([r["distance_km_per30d"] for r in fx])
    kb = np.mean([r["distance_km_per30d"] for r in fb])
    return {"overflow_fixed": of, "overflow_fill": ob,
            "overflow_reduction_%": 100 * (1 - ob / of) if of > 0 else np.nan,
            "distance_ratio": kb / kf}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "results"))
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    t0 = time.time()
    base = headline(copy.deepcopy(ds.PARAMS), a.seeds)
    rows = [{"parameter": "base", "label": "Base case", "value": "", **base}]
    for p, (lo, hi, lab) in RANGES.items():
        for v in (lo, hi):
            if p not in ds.PARAMS:
                raise SystemExit(f"unknown parameter {p}")
            P = copy.deepcopy(ds.PARAMS)
            P[p] = v
            rows.append({"parameter": p, "label": lab, "value": v, **headline(P, a.seeds)})
        print(f"{p} done ({time.time() - t0:.0f}s)", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(f"{a.out}/table_sensitivity.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.8), sharey=True)
    labels = [RANGES[p][2] for p in RANGES]
    for ax, col, ref, xl in ((axes[0], "overflow_reduction_%", base["overflow_reduction_%"],
                              "Overflow reduction vs. fixed schedule (%)"),
                             (axes[1], "distance_ratio", base["distance_ratio"],
                              "Distance relative to fixed schedule")):
        for i, p in enumerate(RANGES):
            lo, hi = df[(df.parameter == p)][col].values
            ax.plot([lo, hi], [i, i], color="#3B6EA5", lw=6, solid_capstyle="butt")
            ax.plot([lo], [i], "o", color="#C8553D", ms=4)
        ax.axvline(ref, color="black", lw=0.8)
        ax.set_xlabel(xl)
    axes[0].set_yticks(range(len(labels)), labels, fontsize=8)
    axes[0].invert_yaxis()
    fig.tight_layout()
    fig.savefig(f"{a.out}/fig_sensitivity.png", dpi=300)
    print(df.round(3).to_string())


if __name__ == "__main__":
    main()
