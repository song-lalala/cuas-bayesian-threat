"""
Exp-N (new-reviewer round): does the headline conclusion depend on how noisy
the evaluation world is?

The generative world's threat node uses an ordinal-response sharpness
TAU_T = 0.40, which leaves substantial aleatoric overlap between adjacent
threat levels -- so much that the CPT-oracle itself reaches only ~0.59
accuracy and ~0.46 FAR at Pd = 0.9. That level was a design choice and the
manuscript owed the reader both a justification and a sensitivity check.

This script rebuilds the world at TAU_T in {0.25, 0.40, 0.60} (sharper,
as reported, and blurrier) and re-runs the scarcity comparison -- Proposed
vs unsmoothed MLE vs the CPT-oracle -- over the same 20 replications, so the
reader can see whether the sample-efficiency conclusion is an artifact of
one noise setting.

Run:  python exp_noise_level.py [--reps 20]
Outputs: results/expN_noise_raw.csv, results/expN_summary.md
"""
from __future__ import annotations
import os, sys, time, argparse
import numpy as np
import pandas as pd
from scipy import stats as st

import model_spec as ms
import dataio
import grounding as g
import metrics as mt
import pipeline as pl

RES = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(RES, exist_ok=True)
TEST_N = 2500
TAUS = (0.25, 0.40, 0.60)
N_GRID = (10, 50, 200, 400)


def run(reps=20):
    rows = []
    t0 = time.time()
    tau_orig = ms.TAU_T
    try:
        for tau in TAUS:
            ms.TAU_T = tau                      # rebuild the world's T node
            gt = ms.ground_truth_bn()
            for r in range(reps):
                test = dataio.generate_dataset(gt, TEST_N, seed=9000 + r)
                calib = dataio.generate_dataset(gt, 300, seed=5000 + r)
                est = dataio.estimate_confusion(calib)
                alpha, _ = dataio.fit_reliability(calib)
                y = np.array([s["T"] for s in test])
                # oracle for this world
                m, _ = mt.evaluate({k: gt.cpts[k] for k in ms.INTERNAL}, test,
                                   sensor_cpts=None, alpha=None)
                rows.append({"tau": tau, "rep": r, "N": -1,
                             "method": "CPT-oracle",
                             "majority": float(np.bincount(y, minlength=4).max()
                                                / len(y)), **m})
                for N in N_GRID:
                    data = dataio.generate_dataset(gt, N, seed=r)
                    for name in ("Proposed", "B1-MLE"):
                        cpts, _ = pl.ground_internal(name, data, r, gt=gt)
                        m, _ = mt.evaluate(cpts, test, sensor_cpts=est,
                                           alpha=alpha)
                        rows.append({"tau": tau, "rep": r, "N": N,
                                     "method": name, **m})
            print(f"[expN] tau={tau} done ({time.time()-t0:.0f}s)", flush=True)
            pd.DataFrame(rows).to_csv(
                os.path.join(RES, "expN_noise_raw.csv"), index=False)
    finally:
        ms.TAU_T = tau_orig
    summarize()
    print(f"[expN] ALL DONE in {time.time()-t0:.0f}s", flush=True)


def summarize():
    df = pd.read_csv(os.path.join(RES, "expN_noise_raw.csv"))
    lines = ["# Exp-N: sensitivity of the headline result to the world's "
             "aleatoric noise level (TAU_T)", ""]
    for tau in sorted(df.tau.unique()):
        d = df[df.tau == tau]
        orc = d[d.method == "CPT-oracle"]
        lines.append(f"## TAU_T = {tau:g}  (oracle acc "
                     f"{orc['acc'].mean():.3f}, AUC {orc['macroAUC'].mean():.3f}, "
                     f"ECE {orc['ece'].mean():.3f}, FAR "
                     f"{orc['far@pd0.9'].mean():.3f}; majority "
                     f"{orc['majority'].mean():.3f})")
        for N in N_GRID:
            p = d[(d.N == N) & (d.method == "Proposed")].set_index("rep")
            m = d[(d.N == N) & (d.method == "B1-MLE")].set_index("rep")
            ix = p.index.intersection(m.index)
            out = []
            for k in ("acc", "macroAUC", "ece", "brier"):
                delta = (p.loc[ix, k] - m.loc[ix, k]).dropna().values
                out.append(f"d{k} {delta.mean():+.3f}"
                           f"{'*' if st.ttest_1samp(delta, 0).pvalue < 0.05 else ' '}")
            lines.append(f"- N={N}: Proposed {p['acc'].mean():.3f}/"
                         f"{p['macroAUC'].mean():.3f}/{p['ece'].mean():.3f} vs "
                         f"MLE {m['acc'].mean():.3f}/{m['macroAUC'].mean():.3f}/"
                         f"{m['ece'].mean():.3f}  |  " + ", ".join(out))
        # five-fold crossing check: Proposed(10) vs MLE(50)
        p10 = d[(d.N == 10) & (d.method == "Proposed")].set_index("rep")
        m50 = d[(d.N == 50) & (d.method == "B1-MLE")].set_index("rep")
        ix = p10.index.intersection(m50.index)
        out = []
        for k in ("acc", "macroAUC", "ece", "brier"):
            delta = (p10.loc[ix, k] - m50.loc[ix, k]).dropna().values
            out.append(f"d{k} {delta.mean():+.3f} "
                       f"(p={st.ttest_1samp(delta, 0).pvalue:.2f})")
        lines.append("- five-fold check, Proposed(N=10) - MLE(N=50): "
                     + ", ".join(out))
        lines.append("")
    txt = "\n".join(lines)
    with open(os.path.join(RES, "expN_summary.md"), "w",
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
