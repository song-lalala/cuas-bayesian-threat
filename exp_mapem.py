"""
Exp-M: MAP-EM grounding with LATENT H, N, C (C2-iii executed).

The scarcity study observes the latent layer (simulation labels). This
experiment removes that advantage: training rows carry only the observed
inputs (v, rdot, d, z), the four sensor reports, and the threat label T;
H, N, C (and cls, emit) are latent, and the internal CPTs are learned by
MAP-EM seeded from the elicited prior, with the constrained M-step on the
final iteration and the calibration-split sensor estimates in the E-step
(never the true sensor CPTs).

Compared per replication (same test draw): prior-only (B2), MAP-EM (latent),
and the fully-observed Proposed upper reference.

Run:  python exp_mapem.py [--reps 20] [--smoke]
Outputs: results/expM_raw.csv, results/expM_summary.md
"""
from __future__ import annotations
import os, sys, time, argparse
import numpy as np
import pandas as pd

import model_spec as ms
import dataio
import grounding as g
import metrics as mt
import pipeline as pl

RES = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(RES, exist_ok=True)
TEST_N = 2500
EM_ITERS = 12
KEEP = ["v", "rdot", "d", "z", "Rr", "Re", "Ra", "Rf", "T"]


def run(reps=20, Ns=(50, 200), ess=pl.ESS_FIXED):
    gt = ms.ground_truth_bn()
    rows = []
    t0 = time.time()
    for r in range(reps):
        test = dataio.generate_dataset(gt, TEST_N, seed=9000 + r)
        calib = dataio.generate_dataset(gt, 300, seed=5000 + r)
        est = dataio.estimate_confusion(calib)
        alpha, _ = dataio.fit_reliability(calib)
        rng = np.random.default_rng(1000 + r)
        prior = ms.expert_prior_cpts(rng)
        met, _ = mt.evaluate(prior, test, sensor_cpts=est, alpha=alpha)
        rows.append({"rep": r, "N": 0, "method": "B2-expert", **met})
        for N in Ns:
            full = dataio.generate_dataset(gt, N, seed=r)
            partial = [{k: row[k] for k in KEEP} for row in full]
            cpts_em = g.em_internal(partial, prior, ess, n_iter=EM_ITERS,
                                    seed=r, constrained=True,
                                    sensor_cpts=est)
            met, _ = mt.evaluate(cpts_em, test, sensor_cpts=est, alpha=alpha)
            rows.append({"rep": r, "N": N, "method": "MAP-EM (latent)", **met})
            cpts_fo, _ = pl.ground_internal("Proposed", full, r, ess=ess,
                                            gt=gt)
            met, _ = mt.evaluate(cpts_fo, test, sensor_cpts=est, alpha=alpha)
            rows.append({"rep": r, "N": N, "method": "fully-observed", **met})
        print(f"[expM] rep {r+1}/{reps} ({time.time()-t0:.0f}s)", flush=True)
        pd.DataFrame(rows).to_csv(os.path.join(RES, "expM_raw.csv"),
                                  index=False)
    summarize()
    print(f"[expM] ALL DONE in {time.time()-t0:.0f}s", flush=True)


def summarize():
    df = pd.read_csv(os.path.join(RES, "expM_raw.csv"))
    lines = ["# Exp-M: MAP-EM with latent H,N,C (C2-iii executed)", ""]
    b2 = df[df.method == "B2-expert"]
    lines.append(f"- prior-only (B2): acc {b2['acc'].mean():.3f}, "
                 f"AUC {b2['macroAUC'].mean():.3f}, ECE {b2['ece'].mean():.3f}")
    for N in sorted(df[df.N > 0].N.unique()):
        for m in ["MAP-EM (latent)", "fully-observed"]:
            d = df[(df.N == N) & (df.method == m)]
            sem = d["acc"].std(ddof=1) / max(np.sqrt(len(d)), 1)
            lines.append(f"- N={N} {m}: acc {d['acc'].mean():.3f}"
                         f"(+-{2.093*sem:.3f}), AUC {d['macroAUC'].mean():.3f}, "
                         f"ECE {d['ece'].mean():.3f}, "
                         f"cwECE {d['cwece'].mean():.3f}")
    txt = "\n".join(lines)
    with open(os.path.join(RES, "expM_summary.md"), "w", encoding="utf-8") as f:
        f.write(txt + "\n")
    print(txt.encode(sys.stdout.encoding or "utf-8", errors="replace")
             .decode(sys.stdout.encoding or "utf-8", errors="replace"))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=20)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--summary-only", action="store_true")
    a = ap.parse_args()
    if a.summary_only:
        summarize()
    elif a.smoke:
        run(reps=2, Ns=(50,))
    else:
        run(reps=a.reps)
