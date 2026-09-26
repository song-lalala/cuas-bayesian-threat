"""
Table-5 (calibration-set size) cells re-evaluated over the same 20 independent
replications as exp_significance.py, so all headline tables carry means +/-
95% CIs. Writes results/expS_calibsize_raw.csv and prints the summary.
"""
from __future__ import annotations
import os, sys, time
import numpy as np
import pandas as pd

import model_spec as ms
import dataio
import grounding as g
import metrics as mt
from exp_significance import ground_internal, t_ci

RES = os.path.join(os.path.dirname(__file__), "results")
TEST_N = 2500
GRID = (0, 30, 100, 300, 1000, -1)   # 0=abstract, -1=oracle true matrices


def run(reps=20):
    gt = ms.ground_truth_bn()
    rows = []
    t0 = time.time()
    for r in range(reps):
        test = dataio.generate_dataset(gt, TEST_N, seed=9000 + r)
        data = dataio.generate_dataset(gt, 200, seed=r)
        cpts = ground_internal("Proposed-cMAP", data, r)
        for cn in GRID:
            if cn == 0:
                met, _ = mt.evaluate(cpts, test, sensor_mode="abstract",
                                     reliability=False, use_soft_eoir=True)
                src = "abstract"
            elif cn == -1:
                met, _ = mt.evaluate(cpts, test, sensor_mode="calibrated",
                                     reliability=False, use_soft_eoir=True,
                                     sensor_cpts=None)
                src = "oracle-true-CM"
            else:
                est = dataio.estimate_confusion(
                    dataio.generate_dataset(gt, cn, seed=5000 + r))
                met, _ = mt.evaluate(cpts, test, sensor_mode="calibrated",
                                     reliability=False, use_soft_eoir=True,
                                     sensor_cpts=est)
                src = "estimated"
            rows.append({"rep": r, "calib_n": cn, "source": src, **met})
        print(f"[calibsize-CI] rep {r+1}/{reps} ({time.time()-t0:.0f}s)",
              flush=True)
        pd.DataFrame(rows).to_csv(os.path.join(RES, "expS_calibsize_raw.csv"),
                                  index=False)
    df = pd.DataFrame(rows)
    for cn in GRID:
        d = df[df.calib_n == cn]
        out = []
        for k in ["acc", "macroF1", "macroAUC", "brier", "ece"]:
            m, lo, hi = t_ci(d[k].values)
            out.append(f"{k} {m:.3f}+-{(hi-lo)/2:.3f}")
        print(f"calib_n={cn} ({d['source'].iloc[0]}): " + ", ".join(out),
              flush=True)


if __name__ == "__main__":
    run(int(sys.argv[1]) if len(sys.argv) > 1 else 20)
