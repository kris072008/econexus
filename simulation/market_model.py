"""Bale-level market model: does the gate pay once buyers reject contaminated bales?

The district simulation (district_sim.py) gives the number of items the sorting station processes
per 30 days. Here each recycling stream (paper, plastic) is baled; a buyer rejects a bale whose
contamination exceeds a limit, and a rejected bale is disposed of and charged a penalty. Items the
gate diverts are disposed of with general waste. The model compares the net value of the three
sorting policies on the same processed items, and estimates the effective cost ratio
k_eff = (marginal cost of one more contaminant) / (cost of one landfilled recyclable).

Every parameter is in PARAMS with its source or marked ASSUMED. Prices are market snapshots.

    python simulation/market_model.py --policies experiments/results/policy_rates.json \
        --runs simulation/results/runs.csv --out simulation/results
"""
import argparse
import json
import os

import numpy as np
import pandas as pd
from scipy.stats import binom

ST = 1.10231  # short tons per metric tonne; US prices are quoted per short ton

PARAMS = {
    "item_mass_kg": 0.040,             # ASSUMED, same as district_sim.py
    "bale_mass_kg": 500.0,             # ASSUMED: typical baler output
    "contamination_limit": 0.015,      # buyer specification for recovered plastic, 1.5% (Lubongo & Alexandridis 2022)
    "variability_sigma": 0.5,          # ASSUMED: lognormal spread of contamination between bales
    "disposal_usd_per_t": 89.60 * 0.77,  # NEA Tuas WTE disposal fee S$89.60/t (2026); S$1 = US$0.77 ASSUMED
    # A rejected bale loses its value and is disposed of. Any further charge is ASSUMED: 0 in the base
    # case (conservative), US$100/t in the sensitivity runs.
    "rejection_penalty_usd_per_t": 0.0,
    "price_scenarios": {               # USD per metric tonne of bale
        # US national averages, 31 Dec 2025 (RecyclingMarkets.net, reported by NERC, Jan 2026):
        # residential mixed paper US$20.31/short ton, OCC US$46.88/short ton,
        # PET 5.38 US cents/lb, natural HDPE 46.81 US cents/lb
        "low":  {"paper": 20.31 * ST, "plastic": 0.0538 * 2000 * ST},   # mixed paper; PET
        "high": {"paper": 46.88 * ST, "plastic": 0.4681 * 2000 * ST},   # OCC; natural HDPE
    },
    "months": 4000,                    # Monte Carlo repetitions of one 30-day period
}
POLICIES = ["ungated", "gated_k3", "gated_k9"]
STREAMS = ["paper", "plastic"]


MAX_BALES = 400


def draws(seed, stream, months):
    """Common random numbers: the same normal and uniform draws for every policy and every bump,
    indexed by bale position, so differences between policies are not Monte Carlo noise."""
    rng = np.random.default_rng([seed, STREAMS.index(stream)])
    return rng.standard_normal((months, MAX_BALES)), rng.random((months, MAX_BALES))


def simulate_stream(items, contamination, value, P, Z, U):
    """Return per-month arrays: net value of the stream's bales, rejected bales; and bale count."""
    per_bale = int(P["bale_mass_kg"] / P["item_mass_kg"])
    n_full = int(items // per_bale)
    rest = int(round(items - n_full * per_bale))  # last, partly filled bale
    sizes = np.array([per_bale] * n_full + ([rest] if rest > 0 else []))
    n = len(sizes)
    if n == 0:
        return np.zeros(Z.shape[0]), np.zeros(Z.shape[0]), 0.0
    if n > MAX_BALES:
        raise SystemExit(f"{n} bales exceed MAX_BALES={MAX_BALES}")
    sig = P["variability_sigma"]
    p = np.clip(contamination * np.exp(sig * Z[:, :n] - sig ** 2 / 2), 0, 1)
    cont = binom.ppf(U[:, :n], sizes, p)
    rejected = cont / sizes > P["contamination_limit"]
    t = sizes * P["item_mass_kg"] / 1000.0      # tonnes per bale
    good = value * t
    bad = -(P["disposal_usd_per_t"] + P["rejection_penalty_usd_per_t"]) * t
    net = np.where(rejected, bad, good).sum(1)
    w = sizes / per_bale
    return net, (rejected * w).sum(1), float(w.sum())


def evaluate(N, rates, prices, P, seed=0, months=None, bump=None):
    """Net value per 30 days for each policy. bump=(policy, stream, delta) adds delta to that
    stream's contamination, used for the marginal-cost estimate."""
    months = months or P["months"]
    out = {}
    D = {s: draws(seed, s, months) for s in STREAMS}
    for pol in POLICIES:
        total = np.zeros(months)
        rej, bales = 0.0, 0.0
        diverted_or_general = N
        for s in STREAMS:
            items = N * rates[pol][f"{s}_share"]
            c = rates[pol][f"{s}_contamination"]
            if bump and bump[0] == pol and bump[1] == s:
                c += bump[2]
            net, r, b = simulate_stream(items, c, prices[s], P, *D[s])
            total += net
            rej += r.mean()
            bales += b
            diverted_or_general -= items
        # everything not admitted is disposed of with general waste
        total -= diverted_or_general * P["item_mass_kg"] / 1000 * P["disposal_usd_per_t"]
        out[pol] = {"net_usd": float(total.mean()), "net_sd": float(total.std()),
                    "rejected_bales": float(rej), "bales": float(bales)}
    return out


def k_effective(N, rates, prices, P, pol, s):
    """Marginal loss of one extra contaminant in stream s, over the loss of one landfilled recyclable."""
    items = N * rates[pol][f"{s}_share"]
    if items <= 0:
        return np.nan, np.nan, np.nan, np.nan
    delta = 0.001
    base = evaluate(N, rates, prices, P, seed=7)[pol]["net_usd"]
    up = evaluate(N, rates, prices, P, seed=7, bump=(pol, s, delta))[pol]["net_usd"]
    lc = (base - up) / (delta * items)          # USD lost per extra contaminant (marginal)
    c = rates[pol][f"{s}_contamination"]
    clean = evaluate(N, rates, prices, P, seed=7, bump=(pol, s, -c))[pol]["net_usd"]
    lc_avg = (clean - base) / (c * items) if c > 0 else np.nan  # average USD lost per contaminant
    lr = P["item_mass_kg"] / 1000 * (prices[s] + P["disposal_usd_per_t"])
    return lc / lr, lc, lr, lc_avg / lr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policies", required=True)
    ap.add_argument("--runs", required=True, help="runs.csv from district_sim.py")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    rates = json.load(open(a.policies))
    runs = pd.read_csv(a.runs)
    fb = runs[runs.strategy == "fill_based"]
    if fb.empty:
        raise SystemExit(f"no fill_based rows in {a.runs}")
    N = float(fb.items_processed_per30d.mean())
    for pol in POLICIES:
        r = rates[pol]
        if abs(r["paper_share"] + r["plastic_share"] - r["admitted"]) > 1e-9:
            raise SystemExit(f"{pol}: stream shares do not add up to the admitted share")
    P = PARAMS
    rows, keff = [], []
    for scen, prices in P["price_scenarios"].items():
        res = evaluate(N, rates, prices, P)
        for pol in POLICIES:
            r = res[pol]
            rows.append({"prices": scen, "policy": pol, "bales_per30d": r["bales"],
                         "rejected_bales_per30d": r["rejected_bales"],
                         "rejected_%": 100 * r["rejected_bales"] / r["bales"],
                         "net_value_usd_per30d": r["net_usd"],
                         "vs_ungated_usd": r["net_usd"] - res["ungated"]["net_usd"]})
            for s in STREAMS:
                k, lc, lr, kavg = k_effective(N, rates, prices, P, pol, s)
                keff.append({"prices": scen, "policy": pol, "stream": s,
                             "contamination_%": 100 * rates[pol][f"{s}_contamination"],
                             "Lc_marginal_usd_per_item": lc, "Lr_usd_per_item": lr,
                             "k_marginal": k, "k_average": kavg})
    t = pd.DataFrame(rows)
    t.to_csv(f"{a.out}/table_market.csv", index=False)
    kt = pd.DataFrame(keff)
    kt.to_csv(f"{a.out}/table_k_effective.csv", index=False)

    # sensitivity of the market result to its two most uncertain inputs
    sens = []
    for limit in (0.005, 0.015, 0.03):
        for sig in (0.25, 0.5, 1.0):
          for pen in (0.0, 100.0):
            Q = dict(P, contamination_limit=limit, variability_sigma=sig, months=1500,
                     rejection_penalty_usd_per_t=pen)
            for scen, prices in P["price_scenarios"].items():
                res = evaluate(N, rates, prices, Q)
                for pol in POLICIES:
                    sens.append({"limit_%": 100 * limit, "sigma": sig, "penalty_usd_per_t": pen,
                                 "prices": scen, "policy": pol,
                                 "rejected_%": 100 * res[pol]["rejected_bales"] / res[pol]["bales"],
                                 "vs_ungated_usd": res[pol]["net_usd"] - res["ungated"]["net_usd"]})
    pd.DataFrame(sens).to_csv(f"{a.out}/table_market_sensitivity.csv", index=False)
    info = {"items_processed_per30d": N, "params": P}
    json.dump(info, open(f"{a.out}/market_info.json", "w"), indent=2)
    print(t.round(2).to_string())
    print(kt.round(3).to_string())


if __name__ == "__main__":
    main()
