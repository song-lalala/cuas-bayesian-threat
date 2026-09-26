"""
Exp-L: computational cost of the deployed model.

Three costs are paid at different times and are reported separately:
  (a) offline grounding      -- once per (re)training, supervised or MAP-EM,
  (b) deployment assembly    -- once per range bin when the model is loaded,
  (c) per-track inference     -- every sensor update, and the only one that
                                competes with a real-time budget.

The per-update cost is timed end to end: extracting the evidence from the
track record plus the exact junction-free variable-elimination query for
P(T | evidence) on the deployed network.

Run:  python exp_latency.py [--reps 5] [--test 2500]
Outputs: results/expL_latency.csv, results/expL_summary.md
"""
from __future__ import annotations
import os, sys, time, platform, argparse
import numpy as np
import pandas as pd

import model_spec as ms
import dataio
import grounding as g
import metrics as mt
import pipeline as pl

RES = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(RES, exist_ok=True)
TRAIN_N = 200
EM_ITERS = 12
KEEP = ["v", "rdot", "d", "z", "Rr", "Re", "Ra", "Rf", "T"]


def run(reps=5, test_n=2500):
    gt = ms.ground_truth_bn()
    rows, per_query = [], []
    for r in range(reps):
        test = dataio.generate_dataset(gt, test_n, seed=9000 + r)
        calib = dataio.generate_dataset(gt, 300, seed=5000 + r)
        est = dataio.estimate_confusion(calib)
        alpha, _ = dataio.fit_reliability(calib)
        data = dataio.generate_dataset(gt, TRAIN_N, seed=r)

        # (a) offline grounding -----------------------------------------
        t = time.perf_counter()
        cpts, _ = pl.ground_internal("Proposed", data, r, gt=gt)
        t_ground = time.perf_counter() - t

        rng = np.random.default_rng(1000 + r)
        prior = ms.expert_prior_cpts(rng)
        partial = [{k: row[k] for k in KEEP} for row in data]
        t = time.perf_counter()
        g.em_internal(partial, prior, pl.ESS_FIXED, n_iter=EM_ITERS,
                      seed=r, constrained=True, sensor_cpts=est)
        t_em = time.perf_counter() - t

        # (b) deployment assembly, one network per range bin -------------
        t = time.perf_counter()
        bycache = {dv: dataio.build_inference_bn(
            cpts, ms.SENSORS, dv, "calibrated", sensor_cpts=est, alpha=alpha)
            for dv in range(ms.CARD["d"])}
        t_assemble = time.perf_counter() - t

        # (c) per-track inference ----------------------------------------
        q = np.empty(len(test))
        for i, scn in enumerate(test):
            t = time.perf_counter()
            bn = bycache[scn["d"]]
            ev = dataio.evidence_from_scenario(scn, ms.SENSORS)
            bn.query(["T"], ev)
            q[i] = time.perf_counter() - t
        per_query.append(q)
        rows.append({"rep": r, "n_queries": len(test),
                     "ground_s": t_ground, "mapem_s": t_em,
                     "assemble_s": t_assemble,
                     "q_mean_ms": q.mean() * 1e3,
                     "q_median_ms": float(np.median(q)) * 1e3,
                     "q_p95_ms": float(np.percentile(q, 95)) * 1e3,
                     "q_max_ms": q.max() * 1e3,
                     "throughput_hz": 1.0 / q.mean()})
        print(f"  rep {r}: ground {t_ground*1e3:.0f} ms, MAP-EM "
              f"{t_em*1e3:.0f} ms, assemble {t_assemble*1e3:.1f} ms, "
              f"query {q.mean()*1e3:.3f} ms (p95 {np.percentile(q,95)*1e3:.3f})")

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RES, "expL_latency.csv"), index=False)
    allq = np.concatenate(per_query)
    with open(os.path.join(RES, "expL_summary.md"), "w", encoding="utf-8") as f:
        f.write("# Exp-L: computational cost\n\n")
        f.write(f"- machine: {platform.processor() or platform.machine()}, "
                f"Python {platform.python_version()}, "
                f"NumPy {np.__version__}; single core, no GPU\n")
        f.write(f"- replications: {len(df)}; queries per replication: "
                f"{int(df.n_queries.iloc[0])}\n\n")
        f.write(f"- offline grounding (supervised, N={TRAIN_N}): "
                f"{df.ground_s.mean()*1e3:.0f} ms\n")
        f.write(f"- offline grounding (MAP-EM, {EM_ITERS} iterations, latent "
                f"layer): {df.mapem_s.mean()*1e3:.0f} ms "
                f"({df.mapem_s.mean()/df.ground_s.mean():.0f}x supervised)\n")
        f.write(f"- deployment assembly ({ms.CARD['d']} range bins): "
                f"{df.assemble_s.mean()*1e3:.1f} ms\n")
        f.write(f"- per-track inference: mean {allq.mean()*1e3:.3f} ms, "
                f"median {np.median(allq)*1e3:.3f} ms, "
                f"p95 {np.percentile(allq,95)*1e3:.3f} ms, "
                f"max {allq.max()*1e3:.3f} ms\n")
        f.write(f"- sustained single-core throughput: "
                f"{1.0/allq.mean():.0f} track updates/s\n")
    print(f"\nsaved results/expL_latency.csv and expL_summary.md "
          f"(mean query {allq.mean()*1e3:.3f} ms, "
          f"{1.0/allq.mean():.0f} updates/s)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--test", type=int, default=2500)
    a = ap.parse_args()
    t0 = time.time()
    run(a.reps, a.test)
    print(f"total {time.time()-t0:.1f}s")
