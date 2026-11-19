"""
Exp-ED: the adversarial-prior study of exp_prior_ess.py, part (i), at the ESS the
main pipeline deploys. exp_prior_ess.py ran it at alpha0 = 8 and 50 only, so the
deployed alpha0 = 2 had no reversed-prior test. Added after the main results.
Core files and exp_prior_ess.py are not modified (its functions are imported);
results go to results/expED_*.

Same design as that part: prior bias lambda in {0, 0.25, 0.5, 0.75, 1}, N in
{10, 50, 200, 400}, 20 seeds, one fixed test set (seed 999) and one fixed
calibration split (seed 777), methods MAP and cMAP-full, prior-free MLE as the
reference, with the prior of seed s drawn from default_rng(1000 + s) for each
lambda. The proposed model (ordinal-ICI threat node) is run on the whole grid
here, since at alpha0 = 2 it is the deployed estimator.

Run:  python exp_prior_deployed_ess.py [--workers 8]
      python exp_prior_deployed_ess.py --summary-only
Outputs: results/expED_raw.csv, results/expED_summary.md
"""
from __future__ import annotations
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import sys, time, argparse
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import pandas as pd
from scipy import stats as st

RES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
P = "expED"
ESS = 2.0
LAMS = (0.0, 0.25, 0.5, 0.75, 1.0)
NS = (10, 50, 200, 400)
SEEDS = 20
TEST_N = 2500
METHODS = ("MAP", "cMAP", "Proposed")
METRICS = ("acc", "macroAUC", "brier", "ece")


def one_seed(seed):
    import model_spec as ms
    import dataio
    import metrics as mt
    import exp_prior_ess as ep
    gt = ms.ground_truth_bn()
    test = dataio.generate_dataset(gt, TEST_N, seed=999)
    calib = dataio.generate_dataset(gt, 300, seed=777)
    est = dataio.estimate_confusion(calib)
    alpha, _ = dataio.fit_reliability(calib)
    rows = []
    for N in NS:
        data = dataio.generate_dataset(gt, N, seed=seed)
        met, _ = mt.evaluate(ep.ground("MLE", data, None, 0), test, sensor_cpts=est, alpha=alpha)
        rows.append({"seed": seed, "N": N, "lam": np.nan, "method": "MLE", **{k: met[k] for k in METRICS}})
        for lam in LAMS:
            prior = ep.biased_prior_cpts(np.random.default_rng(1000 + seed), lam)
            for m in METHODS:
                met, _ = mt.evaluate(ep.ground(m, data, prior, ESS, seed=seed), test, sensor_cpts=est, alpha=alpha)
                rows.append({"seed": seed, "N": N, "lam": lam, "method": m, **{k: met[k] for k in METRICS}})
    return rows


def summarize():
    df = pd.read_csv(os.path.join(RES, f"{P}_raw.csv"), float_precision="round_trip")
    L = ["# Exp-ED: adversarial prior at the deployed ESS (alpha0 = 2)", "",
         f"- {df.seed.nunique()} seeds; test seed 999, calibration seed 777, as exp_prior_ess.py part (i); added after the main results", "",
         "## means", "| lambda | N | method | acc | macro-AUC | Brier | ECE |", "|---|---|---|---|---|---|---|"]
    for N in NS:
        q = df[(df.method == "MLE") & (df.N == N)]
        L.append(f"| - | {N} | MLE | {q.acc.mean():.3f} | {q.macroAUC.mean():.3f} | {q.brier.mean():.3f} | {q.ece.mean():.3f} |")
    for lam in LAMS:
        for N in NS:
            for m in METHODS:
                q = df[(df.lam == lam) & (df.N == N) & (df.method == m)]
                assert len(q) == SEEDS
                L.append(f"| {lam:g} | {N} | {m} | {q.acc.mean():.3f} | {q.macroAUC.mean():.3f} | {q.brier.mean():.3f} | {q.ece.mean():.3f} |")
    L += ["", "## fully reversed prior (lambda = 1) minus MLE at the same N (paired t over the seeds)", "| N | method | d acc (p) | d macro-AUC (p) | d Brier (p) | d ECE (p) |", "|---|---|---|---|---|---|"]
    for N in NS:
        b = df[(df.method == "MLE") & (df.N == N)].sort_values("seed")
        for m in METHODS:
            a = df[(df.lam == 1.0) & (df.N == N) & (df.method == m)].sort_values("seed")
            cells = []
            for k in METRICS:
                d = a[k].values - b[k].values
                cells.append(f"{d.mean():+.3f} ({st.ttest_1samp(d, 0.0).pvalue:.2g})")
            L.append(f"| {N} | {m} | " + " | ".join(cells) + " |")
    txt = "\n".join(L) + "\n"
    with open(os.path.join(RES, f"{P}_summary.md"), "w", encoding="utf-8", newline="\n") as f:
        f.write(txt)
    sys.stdout.reconfigure(encoding="utf-8")
    print(txt)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--summary-only", action="store_true")
    a = ap.parse_args()
    if not a.summary_only:
        t0 = time.time()
        rows = []
        with ProcessPoolExecutor(max_workers=a.workers) as ex:
            for i, r in enumerate(ex.map(one_seed, range(SEEDS))):
                rows.extend(r)
                print(f"[expED] seed {i + 1}/{SEEDS} ({time.time() - t0:.0f}s)", flush=True)
        pd.DataFrame(rows).to_csv(os.path.join(RES, f"{P}_raw.csv"), index=False)
    summarize()
