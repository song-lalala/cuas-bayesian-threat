"""
Exp-A (3rd revision): is the headline sample-efficiency claim an artifact of
the chosen ESS? Recompute the Table-4 comparison over the SAME 20
replications at alpha_0 in {1, 2, 5} -- the flat plateau of Fig. 5 -- and
report the Proposed(N=10) vs MLE(N=50) contrast per metric, plus the
Proposed-vs-MLE crossings that define the "fold-reduction" claim.

Also records the class-prior (constant) predictor and the CPT-oracle as
results files, so both baselines live in results/ rather than only in prose.

Run:  python exp_ess_sensitivity.py [--reps 20]
Outputs: results/expA_ess_sensitivity.csv, results/expA_baselines.csv,
         results/expA_summary.md
"""
from __future__ import annotations
import os, sys, time, argparse
import numpy as np
import pandas as pd
from scipy import stats as st

import model_spec as ms
import dataio
import metrics as mt
import pipeline as pl

RES = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(RES, exist_ok=True)
TEST_N = 2500
ESS_GRID = (1.0, 2.0, 5.0)
N_GRID = (10, 50, 100, 200, 400)
METRICS = ("acc", "macroAUC", "ece", "brier")


def run(reps=20):
    gt = ms.ground_truth_bn()
    # P(T) implied by the generative specification (no test data involved)
    prior_marginal = ms.assemble_bn({k: gt.cpts[k] for k in ms.INTERNAL}) \
        .query(["T"]).table
    rows, base_rows = [], []
    t0 = time.time()
    for r in range(reps):
        test = dataio.generate_dataset(gt, TEST_N, seed=9000 + r)
        calib = dataio.generate_dataset(gt, 300, seed=5000 + r)
        est = dataio.estimate_confusion(calib)
        alpha, _ = dataio.fit_reliability(calib)
        y = np.array([s["T"] for s in test])

        # --- baselines: class-prior constant predictor and the CPT-oracle ---
        # The constant predictor emits the threat marginal implied by the
        # GENERATIVE specification -- a quantity known before any test data.
        # (Using the test set's own label frequencies would hand it a free
        # look at the answers and drive its ECE to zero.)
        pri = prior_marginal
        P = np.tile(pri, (len(y), 1))
        base_rows.append({"rep": r, "model": "class-prior constant",
                          "acc": float((P.argmax(1) == y).mean()),
                          "macroAUC": 0.5, "ece": mt.ece(P, y),
                          "brier": mt.brier(P, y),
                          "far@pd0.9": mt.far_at_pd(P, y, 0.90)})
        met, _ = mt.evaluate({k: gt.cpts[k] for k in ms.INTERNAL}, test)
        base_rows.append({"rep": r, "model": "CPT-oracle", **met})

        # --- ESS sensitivity of the scarcity comparison --------------------
        for N in N_GRID:
            data = dataio.generate_dataset(gt, N, seed=r)
            cp, _ = pl.ground_internal("B1-MLE", data, r, gt=gt)
            met, _ = mt.evaluate(cp, test, sensor_cpts=est, alpha=alpha)
            rows.append({"rep": r, "N": N, "method": "B1-MLE",
                         "ess": np.nan, **met})
            for a0 in ESS_GRID:
                cp, _ = pl.ground_internal("Proposed", data, r, ess=a0, gt=gt)
                met, _ = mt.evaluate(cp, test, sensor_cpts=est, alpha=alpha)
                rows.append({"rep": r, "N": N, "method": "Proposed",
                             "ess": a0, **met})
        print(f"[expA] rep {r+1}/{reps} ({time.time()-t0:.0f}s)", flush=True)
        pd.DataFrame(rows).to_csv(
            os.path.join(RES, "expA_ess_sensitivity.csv"), index=False)
        pd.DataFrame(base_rows).to_csv(
            os.path.join(RES, "expA_baselines.csv"), index=False)
    summarize()
    print(f"[expA] ALL DONE in {time.time()-t0:.0f}s", flush=True)


def summarize():
    df = pd.read_csv(os.path.join(RES, "expA_ess_sensitivity.csv"))
    bl = pd.read_csv(os.path.join(RES, "expA_baselines.csv"))
    lines = ["# Exp-A: ESS sensitivity of the sample-efficiency claim, "
             "and the two reference baselines", ""]

    lines.append("## reference baselines (means over replications)")
    for model in bl.model.unique():
        d = bl[bl.model == model]
        parts = [f"{k} {d[k].mean():.3f}" for k in
                 ("acc", "macroAUC", "ece", "brier", "far@pd0.9") if k in d]
        lines.append(f"- {model}: " + ", ".join(parts))

    lines.append("\n## per-(N, alpha0) means")
    for N in N_GRID:
        m = df[(df.N == N) & (df.method == "B1-MLE")]
        lines.append(f"- N={N} MLE: " +
                     ", ".join(f"{k} {m[k].mean():.3f}" for k in METRICS))
        for a0 in ESS_GRID:
            p = df[(df.N == N) & (df.method == "Proposed") & (df.ess == a0)]
            lines.append(f"    Proposed a0={a0:g}: " +
                         ", ".join(f"{k} {p[k].mean():.3f}" for k in METRICS))

    lines.append("\n## Proposed(N=10) - MLE(N=50), paired by replication")
    for a0 in ESS_GRID:
        p = df[(df.N == 10) & (df.method == "Proposed") &
               (df.ess == a0)].set_index("rep")
        m = df[(df.N == 50) & (df.method == "B1-MLE")].set_index("rep")
        ix = p.index.intersection(m.index)
        out = []
        for k in METRICS:
            d = (p.loc[ix, k] - m.loc[ix, k]).values
            pv = st.ttest_1samp(d, 0).pvalue
            out.append(f"{k} {d.mean():+.3f} (p={pv:.2f})")
        lines.append(f"- a0={a0:g}: " + ", ".join(out))

    lines.append("\n## Proposed(N=50) vs MLE at larger N, paired "
                 "(a0=2, the reported setting)")
    p = df[(df.N == 50) & (df.method == "Proposed") &
           (df.ess == 2.0)].set_index("rep")
    for Nm in (100, 200):
        m = df[(df.N == Nm) & (df.method == "B1-MLE")].set_index("rep")
        ix = p.index.intersection(m.index)
        out = []
        for k in METRICS:
            d = (p.loc[ix, k] - m.loc[ix, k]).values
            pv = st.ttest_1samp(d, 0).pvalue
            out.append(f"{k} {d.mean():+.3f} (p={pv:.2f})")
        lines.append(f"- vs MLE N={Nm}: " + ", ".join(out))

    txt = "\n".join(lines)
    with open(os.path.join(RES, "expA_summary.md"), "w",
              encoding="utf-8") as f:
        f.write(txt + "\n")
    print(txt.encode(sys.stdout.encoding or "utf-8", errors="replace")
             .decode(sys.stdout.encoding or "utf-8", errors="replace"))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=20)
    ap.add_argument("--summary-only", action="store_true")
    a = ap.parse_args()
    if a.summary_only:
        summarize()
    else:
        run(a.reps)
