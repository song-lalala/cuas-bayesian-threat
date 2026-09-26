"""
Exp-K: is the headline calibration number an artifact of the bin count?

The manuscript reports top-label ECE with 15 equal-width bins. That choice
is conventional but arbitrary, and with four classes and 2,500 test
scenarios a reviewer is entitled to ask whether the number moves when the
bin count moves. This sweeps it over the deployed pipeline (Proposed CPTs,
estimated confusion matrices, fitted reliability) on the same 20
replications as everything else, and reports the CPT-oracle alongside so
the reader can see the finite-sample floor at each setting.

Run:  python exp_ece_bins.py [--reps 20]
Outputs: results/expK_ece_bins.csv, results/expK_summary.md
"""
from __future__ import annotations
import argparse
import os
import time

import numpy as np
import pandas as pd
from scipy import stats as st

import model_spec as ms
import dataio
import metrics as mt
import pipeline as pl

RES = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(RES, exist_ok=True)
BINS = (5, 10, 15, 20, 30, 50)
TRAIN_N = 200
TEST_N = 2500


def ci(x):
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    h = x.std(ddof=1) / np.sqrt(len(x)) * st.t.ppf(0.975, len(x) - 1)
    return x.mean(), h


def run(reps=20):
    gt = ms.ground_truth_bn()
    oracle_cpts = {k: gt.cpts[k] for k in ms.INTERNAL}
    rows = []
    t0 = time.time()
    for r in range(reps):
        test = dataio.generate_dataset(gt, TEST_N, seed=9000 + r)
        calib = dataio.generate_dataset(gt, 300, seed=5000 + r)
        est = dataio.estimate_confusion(calib)
        alpha, _ = dataio.fit_reliability(calib)
        data = dataio.generate_dataset(gt, TRAIN_N, seed=r)

        cpts, _ = pl.ground_internal("Proposed", data, r, gt=gt)
        P, y = mt.posteriors(cpts, test, ms.SENSORS, sensor_cpts=est,
                             alpha=alpha)
        Po, yo = mt.posteriors(oracle_cpts, test, ms.SENSORS)

        for b in BINS:
            rows.append({"rep": r, "bins": b, "model": "Proposed",
                         "ece": mt.ece(P, y, b),
                         "cwece": mt.classwise_ece(P, y, b)})
            rows.append({"rep": r, "bins": b, "model": "CPT-oracle",
                         "ece": mt.ece(Po, yo, b),
                         "cwece": mt.classwise_ece(Po, yo, b)})
        print(f"[expK] rep {r + 1}/{reps} ({time.time() - t0:.0f}s)",
              flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RES, "expK_ece_bins.csv"), index=False)

    with open(os.path.join(RES, "expK_summary.md"), "w",
              encoding="utf-8") as f:
        f.write("# Exp-K: top-label ECE vs. bin count "
                f"(N={TRAIN_N}, {reps} replications)\n\n")
        f.write("| bins | Proposed ECE | CPT-oracle ECE | "
                "Proposed classwise |\n|---|---|---|---|\n")
        for b in BINS:
            d = df[df.bins == b]
            m, h = ci(d[d.model == "Proposed"]["ece"])
            mo, ho = ci(d[d.model == "CPT-oracle"]["ece"])
            mc, hc = ci(d[d.model == "Proposed"]["cwece"])
            f.write(f"| {b} | {m:.3f} ± {h:.3f} | {mo:.3f} ± {ho:.3f} "
                    f"| {mc:.3f} ± {hc:.3f} |\n")
        prop = [ci(df[(df.bins == b) & (df.model == "Proposed")]["ece"])[0]
                for b in BINS]
        f.write(f"\n- Proposed ECE range over bins in {list(BINS)}: "
                f"{min(prop):.3f}-{max(prop):.3f} "
                f"(spread {max(prop) - min(prop):.3f})\n")
    print(open(os.path.join(RES, "expK_summary.md"),
               encoding="utf-8").read())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=20)
    a = ap.parse_args()
    run(a.reps)
