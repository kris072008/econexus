"""Analyse saved predictions: gate vs baselines, corrected resampled t-tests, cost-ratio sweep, shift test.

Reads the .npz files written by train_predict.py and writes tables (CSV + Markdown), figures
(PNG, 300 dpi) and summary.json with every number quoted in the paper.

    python experiments/analyze.py --preds experiments/preds --out experiments/results
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.optimize import minimize_scalar  # noqa: E402
from scipy.stats import beta  # noqa: E402

CLASSES = ["general", "paper", "plastic"]
GEN, PAPER, PLASTIC = 0, 1, 2
RIDX = [PAPER, PLASTIC]
# 0.00-0.99 in steps of 0.01, then finer steps up to 0.9999 so saturated raw scores are not cut off
GRID = np.unique(np.concatenate([np.round(np.arange(0.0, 0.991, 0.01), 2),
                                 np.round(1 - np.logspace(-2, -4, 9), 6)]))
KS = [1, 3, 9]

ap = argparse.ArgumentParser()
ap.add_argument("--preds", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--rw-calib-frac", type=float, default=0.2)
ap.add_argument("--exclude-dups", default=None,
                help="dupcheck.json: drop flagged near-duplicate test images (robustness check)")
args = ap.parse_args()
PRED, OUT = Path(args.preds), Path(args.out)
OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"savefig.dpi": 300, "font.size": 10})
summary = {}


# ---------------------------------------------------------------- basic functions
def softmax_T(logp, T=1.0):
    z = logp / T
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def nll(P, y):
    return float(-np.mean(np.log(np.clip(P[np.arange(len(y)), y], 1e-12, 1))))


def fit_temperature(logp, y):
    r = minimize_scalar(lambda lt: nll(softmax_T(logp, np.exp(lt)), y), bounds=(-3, 3), method="bounded")
    T = float(np.exp(r.x))
    if not 0.05 < T < 20:
        raise RuntimeError(f"temperature {T} hit the search bound")
    return T


def ece(P, y, bins=15):
    conf, pred = P.max(1), P.argmax(1)
    edges = np.linspace(0, 1, bins + 1)
    e, rows = 0.0, []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            acc, c = (pred[m] == y[m]).mean(), conf[m].mean()
            e += m.mean() * abs(acc - c)
            rows.append((c, acc, int(m.sum())))
    return float(e), rows


def brier(P, y):
    Y = np.eye(len(CLASSES))[y]
    return float(np.mean(np.sum((P - Y) ** 2, 1)))


# A routing method maps (pred, score, threshold) to routed classes. "score" is a confidence-like
# number in [0, 1]; paper/plastic predictions with score below the threshold go to general.
def scores(P, kind):
    s = np.sort(P, 1)
    if kind == "msp":
        return s[:, -1]
    if kind == "margin":
        return s[:, -1] - s[:, -2]
    if kind == "entropy":  # 1 - normalised entropy, so that higher = more confident
        H = -np.sum(P * np.log(np.clip(P, 1e-12, 1)), 1)
        return 1 - H / np.log(P.shape[1])
    raise ValueError(kind)


def route(pred, score, tau):
    """tau is a float, or a dict {class index: threshold} for per-stream thresholds."""
    routed = pred.copy()
    for c in RIDX:
        t = tau[c] if isinstance(tau, dict) else tau
        routed[(pred == c) & (score < t)] = GEN
    return routed


def outcome(routed, y):
    adm = np.isin(routed, RIDX)
    cont = adm & (routed != y)
    lost = np.isin(y, RIDX) & (routed == GEN)
    return adm, cont, lost


def metrics(routed, y, pred=None, k=None):
    adm, cont, lost = outcome(routed, y)
    d = {"items": len(y), "admitted": int(adm.sum()), "admitted_%": 100 * adm.mean(),
         "contaminants": int(cont.sum()),
         "contamination_%": 100 * cont.sum() / adm.sum() if adm.sum() else np.nan,
         "recyclables_lost": int(lost.sum())}
    if pred is not None:
        u_adm, u_cont, u_lost = outcome(pred, y)
        d["intercepted"] = int(u_cont.sum() - cont.sum())
        d["lost_to_gate"] = int((np.isin(y, RIDX) & (pred == y) & (routed == GEN)).sum())
    for c, name in ((PAPER, "paper"), (PLASTIC, "plastic")):
        s = routed == c
        d[f"{name}_purity_%"] = 100 * (y[s] == c).mean() if s.any() else np.nan
    for kk in ([k] if k else KS):
        d[f"cost_k{kk}"] = kk * int(cont.sum()) + int(lost.sum())
    return d


def cost(routed, y, k):
    _, cont, lost = outcome(routed, y)
    return k * cont.sum() + lost.sum()


def plateau_mid(costs, grid):
    """Validation cost is a step function, so several thresholds tie. Take the middle of the
    first run of grid points that reach the minimum, not its edge."""
    costs = np.asarray(costs, float)
    i = int(np.argmin(costs))
    j = i
    while j + 1 < len(costs) and costs[j + 1] == costs[i]:
        j += 1
    return float(grid[(i + j) // 2])


def choose_tau(pred, score, y, k, grid=GRID):
    return plateau_mid([cost(route(pred, score, t), y, k) for t in grid], grid)


def choose_tau_per_stream(pred, score, y, k, grid=GRID):
    """Separate thresholds for the paper and plastic streams; each stream's cost is separable."""
    taus = {}
    for c in RIDX:
        mask = pred == c  # only items predicted c change with this stream's threshold
        cs = []
        for t in grid:
            routed = pred[mask].copy()
            routed[score[mask] < t] = GEN
            cs.append(cost(routed, y[mask], k))
        taus[c] = plateau_mid(cs, grid)
    return taus


def choose_tau_risk(pred, score, y, target=0.02, delta=0.05, grid=GRID):
    """Fixed-sequence test (Angelopoulos et al., Learn then Test): start at the most conservative
    threshold and lower tau while the one-sided (1 - delta) Clopper-Pearson upper bound on
    contamination among admitted validation items stays <= target; stop at the first failure.
    Under exchangeability, P(true contamination at the chosen tau > target) <= delta. It does not
    hold under dataset shift. Returns (tau, feasible); if even the highest tau fails, nothing is
    admitted (tau = 1.01)."""
    chosen = None
    for t in grid[::-1]:
        adm, cont, _ = outcome(route(pred, score, t), y)
        n, x = int(adm.sum()), int(cont.sum())
        ub = beta.ppf(1 - delta, x + 1, n - x) if 0 < n and x < n else 1.0
        if n == 0 or ub > target:
            break
        chosen = float(t)
    return (chosen, True) if chosen is not None else (1.01, False)


def conformal_qhat(P_cal, y_cal, alpha):
    """Split-conformal LAC threshold: prediction set = {c : p_c >= 1 - qhat}."""
    s = 1 - P_cal[np.arange(len(y_cal)), y_cal]
    n = len(s)
    q = np.ceil((n + 1) * (1 - alpha)) / n
    return float(np.quantile(s, min(q, 1.0), method="higher"))


def conformal_route(P, qhat):
    """Admit to a stream only if the conformal set is exactly that single recyclable class."""
    in_set = P >= 1 - qhat
    single = in_set.sum(1) == 1
    pred = P.argmax(1)
    routed = np.full(len(P), GEN)
    ok = single & np.isin(pred, RIDX) & in_set[np.arange(len(P)), pred]
    routed[ok] = pred[ok]
    return routed


# ---------------------------------------------------------------- load predictions
files = sorted(PRED.glob("pred_seed*_m*.npz"))
if not files:
    raise SystemExit(f"no prediction files in {PRED}")
runs = {}
for f in files:
    s, m = f.stem.replace("pred_seed", "").split("_m")
    with np.load(f) as z:
        runs[(int(s), int(m))] = {k: z[k] for k in z.files}
SEEDS = sorted({s for s, m in runs if m == 0})
ENS_SEEDS = sorted(s for s in SEEDS if all((s, m) in runs for m in range(5)))
rw_index = pd.read_csv(PRED / "realwaste_index.csv")
y_rw = np.array([CLASSES.index(c) for c in rw_index.cls])
rw_matched = rw_index.matched.values.astype(bool)
# consistency checks: predictions must line up with the saved splits and the RealWaste index
for (s, m), d in runs.items():
    for part, key in (("val", "y_val"), ("test", "y_test")):
        split = pd.read_csv(PRED / f"split_seed{s}_{part}.csv")
        y_split = np.array([CLASSES.index(c) for c in split.cls])
        if not np.array_equal(y_split, d[key]):
            raise SystemExit(f"seed {s} member {m}: {part} labels do not match split_seed{s}_{part}.csv")
    if d["logp_rw"].shape[0] != len(rw_index):
        raise SystemExit(f"seed {s} member {m}: RealWaste predictions do not match realwaste_index.csv")
if args.exclude_dups:
    flagged = json.load(open(args.exclude_dups))["per_seed"]
    for (s, m), d in runs.items():
        keep = np.ones(len(d["y_test"]), bool)
        keep[flagged[str(s)]["flagged_test_rows"]] = False
        d["y_test"], d["logp_test"] = d["y_test"][keep], d["logp_test"][keep]
    summary["excluded_near_duplicates"] = {s: len(v["flagged_test_rows"]) for s, v in flagged.items()}
for s in ENS_SEEDS:
    tseeds = {int(runs[(s, m)]["train_seed"]) for m in range(5)}
    if len(tseeds) != 5:
        raise SystemExit(f"ensemble for split {s} does not have 5 distinct training seeds")
    for a in range(5):
        for b in range(a + 1, 5):
            if np.allclose(runs[(s, a)]["logp_test"], runs[(s, b)]["logp_test"]):
                raise SystemExit(f"ensemble members {a} and {b} of split {s} are identical")
print(f"seeds {SEEDS}; ensemble seeds {ENS_SEEDS}")
summary["seeds"] = SEEDS
summary["ensemble_seeds"] = ENS_SEEDS

R = []  # per-seed results for the main model (member 0)
for s in SEEDS:
    d = runs[(s, 0)]
    T = fit_temperature(d["logp_val"], d["y_val"])
    r = {"seed": s, "T": T, "yv": d["y_val"], "yt": d["y_test"],
         "Pv": {"raw": softmax_T(d["logp_val"]), "cal": softmax_T(d["logp_val"], T)},
         "Pt": {"raw": softmax_T(d["logp_test"]), "cal": softmax_T(d["logp_test"], T)},
         "Prw": {"raw": softmax_T(d["logp_rw"]), "cal": softmax_T(d["logp_rw"], T)},
         "logp_rw": d["logp_rw"], "lat": float(d["cpu_latency_s"])}
    R.append(r)


def pm(vals, f="{:.1f}"):
    a = np.asarray(vals, dtype=float)
    return f.format(np.nanmean(a)) + " ± " + f.format(np.nanstd(a, ddof=1) if len(a) > 1 else 0.0)


def save(df, name, index=True):
    df.to_csv(OUT / f"{name}.csv", index=index)
    (OUT / f"{name}.md").write_text(df.to_markdown(index=index))


# ---------------------------------------------------------------- classifier and calibration
cls_rows = {}
for c, name in enumerate(CLASSES):
    P_, R_, F_ = [], [], []
    for r in R:
        pred = r["Pt"]["raw"].argmax(1)
        tp = ((pred == c) & (r["yt"] == c)).sum()
        p = tp / max((pred == c).sum(), 1)
        q = tp / max((r["yt"] == c).sum(), 1)
        P_.append(100 * p)
        R_.append(100 * q)
        F_.append(100 * 2 * p * q / max(p + q, 1e-12))
    cls_rows[name] = {"precision": pm(P_), "recall": pm(R_), "F1": pm(F_),
                      "test images (seed 0)": int((R[0]["yt"] == c).sum())}
accs = [100 * (r["Pt"]["raw"].argmax(1) == r["yt"]).mean() for r in R]
cls_rows["accuracy"] = {"precision": pm(accs), "recall": "", "F1": "", "test images (seed 0)": len(R[0]["yt"])}
save(pd.DataFrame(cls_rows).T, "table_classifier")
summary["accuracy"] = {"mean": float(np.mean(accs)), "sd": float(np.std(accs, ddof=1)),
                       "min": float(np.min(accs)), "max": float(np.max(accs))}
summary["plastic_f1"] = cls_rows["plastic"]["F1"]
cm0 = np.zeros((3, 3), int)
for t, p in zip(R[0]["yt"], R[0]["Pt"]["raw"].argmax(1)):
    cm0[t, p] += 1
summary["confusion_seed0"] = cm0.tolist()

cal = {}
for m, lab in (("raw", "raw softmax"), ("cal", "temperature-scaled")):
    cal[lab] = {"T": pm([r["T"] for r in R], "{:.2f}") if m == "cal" else "1 (fixed)",
                "NLL": pm([nll(r["Pt"][m], r["yt"]) for r in R], "{:.3f}"),
                "ECE (%)": pm([100 * ece(r["Pt"][m], r["yt"])[0] for r in R]),
                "Brier": pm([brier(r["Pt"][m], r["yt"]) for r in R], "{:.3f}")}
save(pd.DataFrame(cal).T, "table_calibration")
summary["calibration"] = cal
summary["T_values"] = [round(r["T"], 2) for r in R]
ece_raw = [100 * ece(r["Pt"]["raw"], r["yt"])[0] for r in R]
ece_cal = [100 * ece(r["Pt"]["cal"], r["yt"])[0] for r in R]
summary["ece_improved_seeds"] = int(sum(c < a for a, c in zip(ece_raw, ece_cal)))
nll_raw = [nll(r["Pt"]["raw"], r["yt"]) for r in R]
nll_cal = [nll(r["Pt"]["cal"], r["yt"]) for r in R]
summary["nll_improved_seeds"] = int(sum(c < a for a, c in zip(nll_raw, nll_cal)))

# ---------------------------------------------------------------- threshold table (cal MSP)
TAB_T = [0.0, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
rows = []
for t in TAB_T:
    ms = []
    for r in R:
        P = r["Pt"]["cal"]
        pred = P.argmax(1)
        ms.append(metrics(route(pred, scores(P, "msp"), t), r["yt"], pred))
    df = pd.DataFrame(ms)
    rows.append({"tau": "ungated" if t == 0 else t,
                 **{c: pm(df[c]) for c in ["admitted_%", "paper_purity_%", "plastic_purity_%",
                                           "contamination_%", "intercepted", "lost_to_gate", "cost_k9"]}})
save(pd.DataFrame(rows), "table_threshold_cal", index=False)


# ---------------------------------------------------------------- methods compared at selected thresholds
def run_method(r, method, k, member_P=None):
    """Return (routed_test, pred_test, tau) for one seed with threshold chosen on validation."""
    if method == "ungated":
        P = r["Pt"]["cal"]
        pred = P.argmax(1)
        return pred, pred, 0.0
    if method == "ensemble":
        Pv, Pt = member_P
        pv, pt = Pv.argmax(1), Pt.argmax(1)
        tau = choose_tau(pv, scores(Pv, "msp"), r["yv"], k)
        return route(pt, scores(Pt, "msp"), tau), pt, tau
    if method == "conformal":
        Pv, Pt = r["Pv"]["cal"], r["Pt"]["cal"]
        # Validation is split into disjoint halves: alpha is tuned on one half (qhat fitted on it,
        # cost scored on it), and the final qhat comes from the other half only, so the reported
        # coverage guarantee holds for the calibration half. Same alpha resolution as the tau grid.
        n = len(r["yv"])
        idx = np.random.default_rng(r["seed"]).permutation(n)
        tune, calib = idx[: n // 2], idx[n // 2:]
        alphas = np.round(np.arange(0.005, 0.505, 0.005), 3)
        cs = [cost(conformal_route(Pv[tune], conformal_qhat(Pv[tune], r["yv"][tune], al)), r["yv"][tune], k)
              for al in alphas]
        alpha = plateau_mid(cs, alphas)
        q = conformal_qhat(Pv[calib], r["yv"][calib], alpha)
        cover = float((Pt[np.arange(len(r["yt"])), r["yt"]] >= 1 - q).mean())
        r.setdefault("conformal_coverage", []).append((k, alpha, cover))
        return conformal_route(Pt, q), Pt.argmax(1), alpha
    if method in ("theory_raw", "theory_cal"):  # no tuning: tau fixed at k/(k + 1)
        P = r["Pt"]["raw" if method == "theory_raw" else "cal"]
        pt = P.argmax(1)
        return route(pt, scores(P, "msp"), k / (k + 1)), pt, k / (k + 1)
    conf, kind = {"msp_raw": ("raw", "msp"), "msp_cal": ("cal", "msp"), "margin": ("cal", "margin"),
                  "entropy": ("cal", "entropy"), "per_stream": ("cal", "msp"),
                  "risk_2pct": ("cal", "msp")}[method]
    Pv, Pt = r["Pv"][conf], r["Pt"][conf]
    pv, pt = Pv.argmax(1), Pt.argmax(1)
    sv, st = scores(Pv, kind), scores(Pt, kind)
    if method == "per_stream":
        tau = choose_tau_per_stream(pv, sv, r["yv"], k)
    elif method == "risk_2pct":
        tau, feasible = choose_tau_risk(pv, sv, r["yv"], target=0.02)
        r.setdefault("risk_feasible", []).append(feasible)
    else:
        tau = choose_tau(pv, sv, r["yv"], k)
    return route(pt, st, tau), pt, tau


def ensemble_P(s):
    Pv, Pt = [], []
    for m in range(5):
        d = runs[(s, m)]
        T = fit_temperature(d["logp_val"], d["y_val"])
        Pv.append(softmax_T(d["logp_val"], T))
        Pt.append(softmax_T(d["logp_test"], T))
    return np.mean(Pv, 0), np.mean(Pt, 0)


ENS = {s: ensemble_P(s) for s in ENS_SEEDS}
METHODS = ["ungated", "theory_raw", "theory_cal", "msp_raw", "msp_cal", "margin", "entropy", "per_stream", "conformal", "risk_2pct"]
LABEL = {"theory_raw": "τ = k/(k + 1), raw softmax (no tuning)",
         "theory_cal": "τ = k/(k + 1), temperature-scaled (no tuning)", "ungated": "Ungated", "msp_raw": "Max softmax, raw", "msp_cal": "Max softmax, temperature-scaled",
         "margin": "Margin (top-1 − top-2)", "entropy": "1 − normalised entropy",
         "per_stream": "Per-stream thresholds", "conformal": "Conformal prediction sets (LAC)",
         "risk_2pct": "Risk-controlled τ (contamination ≤ 2%)", "ensemble": "Deep ensemble (5 models)",
         "msp_cal_ens_seeds": "Max softmax, temperature-scaled (same 5 splits)"}

per_seed = []  # long table: seed, method, k, metrics
for k in KS:
    for r in R:
        for meth in METHODS:
            routed, pred, tau = run_method(r, meth, k)
            per_seed.append({"seed": r["seed"], "method": meth, "k": k,
                             "tau": json.dumps({CLASSES[c]: v for c, v in tau.items()}) if isinstance(tau, dict) else tau,
                             **metrics(routed, r["yt"], pred, k)})
        if r["seed"] in ENS:
            routed, pred, tau = run_method(r, "ensemble", k, ENS[r["seed"]])
            Pt_ens = ENS[r["seed"]][1]
            # the ensemble has its own ungated baseline (its own argmax)
            per_seed.append({"seed": r["seed"], "method": "ensemble", "k": k, "tau": tau,
                             **metrics(routed, r["yt"], pred, k),
                             "ungated_cost": cost(pred, r["yt"], k)})
L = pd.DataFrame(per_seed)
L.to_csv(OUT / "per_seed_methods.csv", index=False)


def ratio(a, b):
    if b == 0:
        raise RuntimeError("ungated cost is zero; relative cost undefined")
    return a / b


L["cost"] = [row[f"cost_k{int(row.k)}"] for _, row in L.iterrows()]
base = {(int(r.seed), int(r.k)): r.cost for _, r in L[L.method == "ungated"].iterrows()}
L["rel_cost"] = [ratio(row.cost, row.ungated_cost if row.method == "ensemble" else base[(int(row.seed), int(row.k))])
                 for _, row in L.iterrows()]
L.to_csv(OUT / "per_seed_methods.csv", index=False)

comp = []
for k in KS:
    for meth in METHODS + ["ensemble"]:
        sub = L[(L.k == k) & (L.method == meth)]
        if sub.empty:
            continue
        comp.append({"k": k, "method": LABEL[meth], "seeds": len(sub),
                     "τ (α for conformal)": pm(sub.tau.astype(float), "{:.2f}") if meth not in ("per_stream", "ungated") else "",
                     "admitted (%)": pm(sub["admitted_%"]), "contamination (%)": pm(sub["contamination_%"], "{:.2f}"),
                     "intercepted": pm(sub["intercepted"]), "lost to gate": pm(sub["lost_to_gate"]),
                     "cost / ungated": pm(sub.rel_cost, "{:.2f}"),
                     "seeds with lower cost": int((sub.rel_cost < 1).sum())})
    # same-split comparison for the ensemble
    sub = L[(L.k == k) & (L.method == "msp_cal") & (L.seed.isin(ENS_SEEDS))]
    if len(sub):
        comp.append({"k": k, "method": LABEL["msp_cal_ens_seeds"], "seeds": len(sub),
                     "τ (α for conformal)": pm(sub.tau.astype(float), "{:.2f}"), "admitted (%)": pm(sub["admitted_%"]),
                     "contamination (%)": pm(sub["contamination_%"], "{:.2f}"),
                     "intercepted": pm(sub["intercepted"]), "lost to gate": pm(sub["lost_to_gate"]),
                     "cost / ungated": pm(sub.rel_cost, "{:.2f}"),
                     "seeds with lower cost": int((sub.rel_cost < 1).sum())})
C = pd.DataFrame(comp)
save(C, "table_methods", index=False)
NB_TEST, NB_TRAIN = len(R[0]["yt"]), len(pd.read_csv(PRED / f"split_seed{SEEDS[0]}_train.csv"))
# paired ensemble vs single model (member 0) on the same splits
ens_pair = {}
for k in KS:
    e = L[(L.k == k) & (L.method == "ensemble")].set_index("seed")
    s1 = L[(L.k == k) & (L.method == "msp_cal")].set_index("seed").loc[e.index]
    d_cont = (e["contamination_%"] - s1["contamination_%"]).values
    d_cost = (e["cost"] - s1["cost"]).values
    ens_pair[k] = {"seeds": len(e), "contamination_diff_pp_mean": float(np.mean(d_cont)) if len(e) else None,
                   "contamination_diff_pp_sd": float(np.std(d_cont, ddof=1)) if len(e) > 1 else None,
                   "cost_diff_mean": float(np.mean(d_cost)) if len(e) else None,
                   "cost_diff_sd": float(np.std(d_cost, ddof=1)) if len(e) > 1 else None}
summary["ensemble_vs_single"] = ens_pair
summary["risk_feasible_fraction"] = float(np.mean([f for r in R for f in r.get("risk_feasible", [])])) \
    if any("risk_feasible" in r for r in R) else None
rk = L[L.method == "risk_2pct"]
summary["risk_test_exceeds_target"] = {int(k): int((rk[rk.k == k]["contamination_%"] > 2).sum()) for k in KS}
summary["conformal_coverage"] = [(r["seed"], *c) for r in R for c in r.get("conformal_coverage", [])]
summary["methods"] = C.to_dict(orient="records")
summary["per_stream_taus"] = {k: L[(L.k == k) & (L.method == "per_stream")].tau.tolist() for k in KS}

# ---------------------------------------------------------------- policy rates and corrected tests (cal MSP)
# Per-stream rates pooled over seeds, used as inputs by simulation/market_model.py and district_sim.py:
# share of all processed items admitted to each stream, and the contamination of that stream.
stream_rates = {}
for name, k in (("ungated", None), ("gated_k3", 3), ("gated_k9", 9)):
    tot = {"items": 0, "paper_adm": 0, "paper_cont": 0, "plastic_adm": 0, "plastic_cont": 0}
    for r in R:
        P = r["Pt"]["cal"]
        pred = P.argmax(1)
        if k is None:
            routed = pred
        else:
            tau = choose_tau(r["Pv"]["cal"].argmax(1), scores(r["Pv"]["cal"], "msp"), r["yv"], k)
            routed = route(pred, scores(P, "msp"), tau)
        tot["items"] += len(pred)
        for c, s in ((PAPER, "paper"), (PLASTIC, "plastic")):
            tot[f"{s}_adm"] += int((routed == c).sum())
            tot[f"{s}_cont"] += int(((routed == c) & (r["yt"] != c)).sum())
    adm_all = tot["paper_adm"] + tot["plastic_adm"]
    stream_rates[name] = {
        "admitted": adm_all / tot["items"],
        "contamination": (tot["paper_cont"] + tot["plastic_cont"]) / adm_all,
        **{f"{s}_share": tot[f"{s}_adm"] / tot["items"] for s in ("paper", "plastic")},
        **{f"{s}_contamination": tot[f"{s}_cont"] / max(tot[f"{s}_adm"], 1) for s in ("paper", "plastic")},
        "pooled_items": tot["items"]}
(OUT / "policy_rates.json").write_text(json.dumps(stream_rates, indent=2))
summary["policy_rates"] = stream_rates

# Repeated random subsampling: the 10 splits re-use the same images, so test sets overlap and seeds
# are not independent. Intervals and p-values use the corrected resampled t-test of Nadeau and
# Bengio (2003): variance factor (1/J + n_test/n_train). Pooled counts are sums over overlapping
# test sets and are descriptive only.
from scipy.stats import t as tdist  # noqa: E402


def nb_test(diffs, n_test, n_train):
    d = np.asarray(diffs, float)
    J = len(d)
    if J < 2:
        return {"mean": float(d.mean()), "ci95": [np.nan, np.nan], "p_two_sided": np.nan}
    var = np.var(d, ddof=1) * (1 / J + n_test / n_train)
    se = np.sqrt(var)
    tt = d.mean() / se if se > 0 else np.inf
    q = tdist.ppf(0.975, J - 1)
    return {"mean": float(d.mean()), "ci95": [float(d.mean() - q * se), float(d.mean() + q * se)],
            "p_two_sided": float(2 * tdist.sf(abs(tt), J - 1))}


N_TEST = len(R[0]["yt"])
N_TRAIN = len(pd.read_csv(PRED / f"split_seed{SEEDS[0]}_train.csv"))
stats_k = {}
for k in KS:
    d_cont, d_rel, tot = [], [], np.zeros(6)
    for r in R:
        P = r["Pt"]["cal"]
        pred = P.argmax(1)
        tau = choose_tau(r["Pv"]["cal"].argmax(1), scores(r["Pv"]["cal"], "msp"), r["yv"], k)
        routed = route(pred, scores(P, "msp"), tau)
        ua, uc, ul = outcome(pred, r["yt"])
        ga, gc, gl = outcome(routed, r["yt"])
        u_rate = 100 * uc.sum() / ua.sum()
        g_rate = 100 * gc.sum() / ga.sum() if ga.sum() else np.nan
        d_cont.append(u_rate - g_rate)
        d_rel.append(ratio(k * gc.sum() + gl.sum(), k * uc.sum() + ul.sum()) - 1)
        tot += [ua.sum(), uc.sum(), ul.sum(), ga.sum(), gc.sum(), gl.sum()]
    stats_k[k] = {"contamination_reduction_pp": nb_test(d_cont, N_TEST, N_TRAIN),
                  "relative_cost_change": nb_test(d_rel, N_TEST, N_TRAIN),
                  "seeds_lower_cost": int(sum(x < 0 for x in d_rel)),
                  "seeds_higher_cost": int(sum(x > 0 for x in d_rel)),
                  "pooled_descriptive": {"ungated_admitted": int(tot[0]), "ungated_contaminants": int(tot[1]),
                                         "gated_admitted": int(tot[3]), "gated_contaminants": int(tot[4])}}
(OUT / "table_stats.json").write_text(json.dumps(stats_k, indent=2))
summary["stats"] = stats_k
summary["n_test"], summary["n_train"] = N_TEST, N_TRAIN

# ---------------------------------------------------------------- continuous cost-ratio sweep
k_grid = np.round(np.exp(np.linspace(np.log(0.25), np.log(50), 40)), 3)
sweep = []
for k in k_grid:
    for r in R:
        for conf in ("raw", "cal"):
            Pv, Pt = r["Pv"][conf], r["Pt"][conf]
            tau = choose_tau(Pv.argmax(1), scores(Pv, "msp"), r["yv"], k)
            pred = Pt.argmax(1)
            routed = route(pred, scores(Pt, "msp"), tau)
            sweep.append({"k": k, "seed": r["seed"], "conf": conf, "tau": tau,
                          "rel_cost": ratio(cost(routed, r["yt"], k), cost(pred, r["yt"], k))})
S = pd.DataFrame(sweep)
S.to_csv(OUT / "sweep_k.csv", index=False)
g = S[S.conf == "cal"].groupby("k")
summary["sweep"] = {"k_sig_benefit_first": float(min(
    (kk for kk, sub in g if nb_test(sub.rel_cost.values - 1, N_TEST, N_TRAIN)["ci95"][1] < 0), default=np.nan)),
    "k_sig_harm_last": float(max(
    (kk for kk, sub in g if nb_test(sub.rel_cost.values - 1, N_TEST, N_TRAIN)["ci95"][0] > 0), default=np.nan)),
    "tau_vs_theory": [{"k": float(kk), "tau_mean": float(sub.tau.mean()), "theory": float(kk / (kk + 1)),
                       "rel_cost_mean": float(sub.rel_cost.mean())} for kk, sub in g]}

fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0))
for conf, col, lab in (("raw", "#C8553D", "raw softmax"), ("cal", "#3B6EA5", "temperature-scaled")):
    sub = S[S.conf == conf].groupby("k")
    m = sub.rel_cost.mean()
    lo = sub.rel_cost.quantile(0.1)
    hi = sub.rel_cost.quantile(0.9)
    axes[0].plot(m.index, m.values, color=col, label=lab)
    axes[0].fill_between(m.index, lo.values, hi.values, color=col, alpha=0.15)
    t = sub.tau.mean()
    axes[1].plot(t.index, t.values, color=col, label=lab)
axes[0].axhline(1, color="grey", lw=0.8, ls="--")
axes[0].set_xscale("log"); axes[0].set_xlabel("Cost ratio k"); axes[0].set_ylabel("Test cost / ungated cost")
axes[0].legend(fontsize=8)
kk = np.array(k_grid)
axes[1].plot(kk, kk / (kk + 1), color="black", ls=":", label="k/(k + 1)")
axes[1].set_xscale("log"); axes[1].set_xlabel("Cost ratio k"); axes[1].set_ylabel("τ chosen on validation")
axes[1].legend(fontsize=8)
fig.tight_layout(); fig.savefig(OUT / "fig_k_sweep.png"); plt.close(fig)

# ---------------------------------------------------------------- shift test on RealWaste
shift_rows, shift_seed = [], []
for r in R:
    n = len(y_rw)
    perm = np.random.default_rng(1000 + r["seed"]).permutation(n)
    n_cal = int(args.rw_calib_frac * n)
    cal_i, ev_i = perm[:n_cal], perm[n_cal:]
    for subset, mask in (("matched classes", rw_matched), ("all classes", np.ones(n, bool))):
        ev = ev_i[mask[ev_i]]
        cal_sub = cal_i[mask[cal_i]]
        T_rw = fit_temperature(r["logp_rw"][cal_sub], y_rw[cal_sub])
        y = y_rw[ev]
        # the recalibrated setting uses labelled target images: an oracle for the "calibrate on the
        # line's own camera" step of Section VI, not a deployable method without such labels
        for setting in ("TrashNet-calibrated", "recalibrated on RealWaste (oracle)"):
            if setting == "TrashNet-calibrated":
                P = r["Prw"]["cal"][ev]
                taus = {k: choose_tau(r["Pv"]["cal"].argmax(1), scores(r["Pv"]["cal"], "msp"), r["yv"], k) for k in KS}
            else:
                P = softmax_T(r["logp_rw"][ev], T_rw)
                Pc = softmax_T(r["logp_rw"][cal_sub], T_rw)
                taus = {k: choose_tau(Pc.argmax(1), scores(Pc, "msp"), y_rw[cal_sub], k) for k in KS}
            pred = P.argmax(1)
            row = {"seed": r["seed"], "subset": subset, "setting": setting,
                   "T": r["T"] if setting == "TrashNet-calibrated" else T_rw,
                   "accuracy_%": 100 * (pred == y).mean(), "ECE_%": 100 * ece(P, y)[0], "NLL": nll(P, y),
                   "ungated_contamination_%": metrics(pred, y)["contamination_%"],
                   "ungated_admitted_%": metrics(pred, y)["admitted_%"]}
            for k in KS:
                routed = route(pred, scores(P, "msp"), taus[k])
                mm = metrics(routed, y, pred, k)
                row.update({f"k{k}_tau": taus[k], f"k{k}_contamination_%": mm["contamination_%"],
                            f"k{k}_admitted_%": mm["admitted_%"],
                            f"k{k}_rel_cost": ratio(cost(routed, y, k), cost(pred, y, k))})
            shift_seed.append(row)
SH = pd.DataFrame(shift_seed)
SH.to_csv(OUT / "per_seed_shift.csv", index=False)
agg = []
for (subset, setting), sub in SH.groupby(["subset", "setting"], sort=False):
    agg.append({"subset": subset, "setting": setting, "T": pm(sub["T"], "{:.2f}"),
                "accuracy (%)": pm(sub["accuracy_%"]), "ECE (%)": pm(sub["ECE_%"]),
                "ungated contamination (%)": pm(sub["ungated_contamination_%"]),
                **{f"k={k}: τ": pm(sub[f"k{k}_tau"], "{:.2f}") for k in (3, 9)},
                **{f"k={k}: contamination (%)": pm(sub[f"k{k}_contamination_%"]) for k in (3, 9)},
                **{f"k={k}: admitted (%)": pm(sub[f"k{k}_admitted_%"]) for k in (3, 9)},
                **{f"k={k}: cost / ungated": pm(sub[f"k{k}_rel_cost"], "{:.2f}") for k in (3, 9)}})
save(pd.DataFrame(agg), "table_shift", index=False)
summary["shift"] = agg
summary["realwaste_eval_items"] = {"matched": int(rw_matched.sum()), "all": int(len(y_rw)),
                                   "calib_frac": args.rw_calib_frac}

# shift reliability figure (seed 0, matched classes)
r0 = R[0]
fig, ax = plt.subplots(figsize=(3.6, 3.4))
ax.plot([0, 1], [0, 1], "--", color="grey", lw=1, label="perfect calibration")
for P, y, col, lab in ((r0["Pt"]["cal"], r0["yt"], "#3B6EA5", "TrashNet test"),
                       (r0["Prw"]["cal"][rw_matched], y_rw[rw_matched], "#C8553D", "RealWaste")):
    e, rows_ = ece(P, y)
    cx, acc, _ = zip(*rows_)
    ax.plot(cx, acc, "o-", ms=4, color=col, label=f"{lab} (ECE {100 * e:.1f}%)")
ax.set_xlim(0.3, 1); ax.set_ylim(0, 1.02); ax.set_xlabel("Mean confidence"); ax.set_ylabel("Accuracy")
ax.legend(fontsize=8); fig.tight_layout(); fig.savefig(OUT / "fig_shift_reliability.png"); plt.close(fig)

# ---------------------------------------------------------------- figures carried over from v1 (seed 0)
P0, y0 = r0["Pt"], r0["yt"]
fig, ax = plt.subplots(figsize=(3.6, 3.2))
ax.imshow(cm0, cmap="Blues")
for i in range(3):
    for j in range(3):
        ax.text(j, i, cm0[i, j], ha="center", va="center", color="white" if cm0[i, j] > cm0.max() / 2 else "black")
ax.set_xticks(range(3), CLASSES); ax.set_yticks(range(3), CLASSES)
ax.set_xlabel("Predicted class"); ax.set_ylabel("True class")
fig.tight_layout(); fig.savefig(OUT / "fig_confusion_matrix.png"); plt.close(fig)

fig, axes = plt.subplots(1, 2, figsize=(7, 2.8), sharey=True)
sep = {}
for ax, m, title in zip(axes, ("raw", "cal"), ("Raw softmax", f"Temperature-scaled (T = {r0['T']:.2f})")):
    P = P0[m]; ok = P.argmax(1) == y0; c = P.max(1); b = np.linspace(1 / 3, 1, 21)
    ax.hist(c[ok], bins=b, alpha=.65, label=f"correct (n = {ok.sum()})", color="#3B6EA5")
    ax.hist(c[~ok], bins=b, alpha=.75, label=f"incorrect (n = {(~ok).sum()})", color="#C8553D")
    ax.set_yscale("log"); ax.set_xlabel("Confidence p"); ax.set_title(title); ax.legend(fontsize=8)
    sep[m] = {"correct": int(ok.sum()), "incorrect": int((~ok).sum()),
              "errors_above_0.96": int(((~ok) & (c > 0.96)).sum())}
axes[0].set_ylabel("Items (log scale)")
fig.tight_layout(); fig.savefig(OUT / "fig_confidence_separation.png"); plt.close(fig)
summary["separation_seed0"] = sep
# confident errors across all seeds
summary["confident_errors_all_seeds"] = {
    m: {"errors": int(sum((r["Pt"][m].argmax(1) != r["yt"]).sum() for r in R)),
        "above_0.96": int(sum(((r["Pt"][m].argmax(1) != r["yt"]) & (r["Pt"][m].max(1) > 0.96)).sum() for r in R))}
    for m in ("raw", "cal")}

fig, ax = plt.subplots(figsize=(3.6, 3.4))
ax.plot([0, 1], [0, 1], "--", color="grey", lw=1, label="perfect calibration")
for m, col, lab in (("raw", "#C8553D", "raw"), ("cal", "#3B6EA5", "temperature-scaled")):
    e, rows_ = ece(P0[m], y0)
    cx, acc, _ = zip(*rows_)
    ax.plot(cx, acc, "o-", color=col, ms=4, label=f"{lab} (ECE {100 * e:.1f}%)")
ax.set_xlabel("Mean confidence"); ax.set_ylabel("Accuracy"); ax.set_xlim(.3, 1); ax.set_ylim(0, 1.02)
ax.legend(fontsize=8); fig.tight_layout(); fig.savefig(OUT / "fig_reliability.png"); plt.close(fig)

fig, ax = plt.subplots(figsize=(4.2, 3.2))
grid = np.round(np.arange(0, 0.996, 0.005), 3)
for m, col, lab in (("raw", "#C8553D", "raw softmax"), ("cal", "#3B6EA5", "temperature-scaled")):
    for i, r in enumerate(R):
        P = r["Pt"][m]; pred = P.argmax(1); sc = scores(P, "msp")
        pts = [metrics(route(pred, sc, t), r["yt"]) for t in grid]
        ax.plot([p["admitted_%"] for p in pts], [p["contamination_%"] for p in pts], color=col, lw=0.8,
                alpha=0.5, label=lab if i == 0 else None)
ax.set_xlabel("Items admitted to recycling (%)"); ax.set_ylabel("Recycling contamination (%)")
ax.legend(fontsize=8); fig.tight_layout(); fig.savefig(OUT / "fig_contamination_coverage.png"); plt.close(fig)

summary["methods_nb"] = {f"{m}_k{k}": nb_test(L[(L.k == k) & (L.method == m)].rel_cost.values - 1, N_TEST, N_TRAIN)
                         for k in KS for m in METHODS + ["ensemble"] if m != "ungated"
                         and len(L[(L.k == k) & (L.method == m)]) > 1}
summary["cpu_latency_ms_seed0"] = 1000 * float(runs[(0, 0)]["cpu_latency_s"]) if (0, 0) in runs else None
(OUT / "summary.json").write_text(json.dumps(summary, indent=2, default=float))
print(json.dumps({k: summary[k] for k in ("accuracy", "T_values", "stats")}, indent=1, default=float)[:3000])
