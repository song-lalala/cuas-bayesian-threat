"""
Exp-TS: the worked track of Secs. VII-A and VII-I under other training seeds, and
how many table cells actually move its posterior. Core files and exp_sensitivity.py
are not modified (its functions are imported); results go to results/expTS_*.

Seeds: training seed s = 0..19 (N = 200, prior rng(1000 + s), the path of
pipeline.ground_internal), calibration seed 777 as in exp_sensitivity.py, the
evidence e* of that script. For each s: the drone-class posterior, P(T >= medium)
and P(T = high) under the learned CPTs. Reference: the same queries in the true
world with its range-dependent sensor tables. Seed 0 is the model of Sec. VII-A
and of the tornado.

RF report (added after the main results): the same queries, and P(emit = silent), with
and without the RF report in e*, for the true world, the calibrated model of every seed
and the abstract sensor model on the tables of seed 0.

Cells: for seed 0, the swing of P(T = high | e*) over theta +- 0.1 of every learned
table cell, computed as in exp_sensitivity.main. A cell "moves the posterior" if
|swing| exceeds 1e-12 (numerical zero), and more strongly if it exceeds 1e-3. The
count of cells whose three-point linear-fractional system solves (128 of 183 in
exp_sensitivity) counts solvable systems, not cells that move the posterior.

Run:  python exp_worked_track_seeds.py
Outputs: results/expTS_seeds.csv, results/expTS_cells.csv, results/expTS_rf_report.csv,
results/expTS_summary.md
"""
from __future__ import annotations
import os, sys
from itertools import product as ip
import numpy as np
import pandas as pd

import model_spec as ms
import dataio
import pipeline as pl
import exp_sensitivity as es

RES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
P = "expTS"
SEEDS = range(20)


def queries(bn, ev):
    pt = bn.query(["T"], ev).table
    return float(bn.query(["cls"], ev).table[2]), float(pt[2:].sum()), float(pt[3])


def rf_row(model, seed, bn, ev):
    """The same evidence with the RF report and without it: P(drone), P(silent), P(T>=medium), P(T=high)."""
    out = []
    for name, e in (("with", ev), ("without", {k: v for k, v in ev.items() if k != "Rf"})):
        c, m, h = queries(bn, e)
        out.append({"model": model, "seed": seed, "rf_report": name, "p_drone": c, "p_silent": float(bn.query(["emit"], e).table[1]), "p_medium_up": m, "p_high": h})
    return out


def main():
    gt = ms.ground_truth_bn()
    ev = es.EVIDENCE
    calib = dataio.generate_dataset(gt, 300, seed=777)
    est = dataio.estimate_confusion(calib)
    alpha, _ = dataio.fit_reliability(calib)
    true_q = queries(gt, ev)
    assert ms.STATES["emit"][1] == "silent" and ms.STATES["Rf"][ev["Rf"]] == "none"
    rows = []
    rf_rows = rf_row("true world", -1, gt, ev)
    cpts0 = None
    for s in SEEDS:
        data = dataio.generate_dataset(gt, 200, seed=s)
        cpts, _ = pl.ground_internal("Proposed", data, s, gt=gt)
        if s == 0:
            cpts0 = cpts
        bn = dataio.build_inference_bn(cpts, ms.SENSORS, ev["d"], sensor_cpts=est, alpha=alpha)
        c, m, h = queries(bn, ev)
        rows.append({"seed": s, "p_drone": c, "p_medium_up": m, "p_high": h})
        rf_rows += rf_row("calibrated", s, bn, ev)
        print(f"[expTS] seed {s}: {c:.3f} {m:.3f} {h:.3f}", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RES, f"{P}_seeds.csv"), index=False)
    rf_rows += rf_row("abstract", 0, dataio.build_inference_bn(cpts0, ms.SENSORS, ev["d"], sensor_mode="abstract"), ev)
    rc = pd.DataFrame(rf_rows)
    rc.to_csv(os.path.join(RES, f"{P}_rf_report.csv"), index=False)
    cal = rc[rc.model == "calibrated"]
    rise = cal[cal.rf_report == "with"].sort_values("seed").p_high.values - cal[cal.rf_report == "without"].sort_values("seed").p_high.values
    none_given_silent = float(est["Rf"].table[2, 1, ev["Rf"]])
    assert abs(df.p_high.iloc[0] - es.posterior_T_high(cpts0, est, alpha)) < 1e-12

    cells = []
    for node in ms.INTERNAL:
        tab = cpts0[node].table
        for j in ip(*[range(x) for x in tab.shape[:-1]]):
            for k in range(tab.shape[-1]):
                th = float(tab[j + (k,)])
                lo, hi = max(es.EPS, th - es.DELTA), min(1 - es.EPS, th + es.DELTA)
                f_lo = es.posterior_T_high(es.perturbed(cpts0, node, j, k, lo), est, alpha)
                f_hi = es.posterior_T_high(es.perturbed(cpts0, node, j, k, hi), est, alpha)
                cells.append({"node": node, "parent_cfg": str(j), "state": k, "theta": th, "swing": f_hi - f_lo})
    cf = pd.DataFrame(cells)
    cf.to_csv(os.path.join(RES, f"{P}_cells.csv"), index=False)
    n12 = cf[cf.swing.abs() > 1e-12]
    n3 = cf[cf.swing.abs() > 1e-3]
    L = ["# Exp-TS: worked track under other training seeds; cells that move its posterior", "",
         f"- true world: drone {true_q[0]:.3f}, P(T>=medium) {true_q[1]:.3f}, P(T=high) {true_q[2]:.3f}",
         f"- seed 0: drone {df.p_drone[0]:.3f}, P(T>=medium) {df.p_medium_up[0]:.3f}, P(T=high) {df.p_high[0]:.3f}",
         f"- seeds 0-19, P(T=high): mean {df.p_high.mean():.3f}, range {df.p_high.min():.3f} to {df.p_high.max():.3f}, below 0.5: {(df.p_high < 0.5).sum()}, above the true world's: {(df.p_high > true_q[2]).sum()}",
         f"- seeds 0-19, P(T>=medium): range {df.p_medium_up.min():.3f} to {df.p_medium_up.max():.3f}",
         f"- seeds 0-19, drone posterior: range {df.p_drone.min():.3f} to {df.p_drone.max():.3f}",
         f"- cells: {len(cf)}; |swing| > 1e-12: {len(n12)} ({', '.join(f'{k} {(n12.node == k).sum()}' for k in ms.INTERNAL)}); |swing| > 1e-3: {len(n3)} ({', '.join(f'{k} {(n3.node == k).sum()}' for k in ms.INTERNAL)})",
         f"- cells per table: {', '.join(f'{k} {(cf.node == k).sum()}' for k in ms.INTERNAL)}",
         "", "## the RF report on this evidence (none), with and without it", "| model | seed | RF report | P(drone) | P(silent) | P(T>=medium) | P(T=high) |", "|---|---|---|---|---|---|---|"]
    for _, q in rc[(rc.model != "calibrated") | (rc.seed == 0)].iterrows():
        L.append(f"| {q.model} | {q.seed} | {q.rf_report} | {q.p_drone:.3f} | {q.p_silent:.3f} | {q.p_medium_up:.3f} | {q.p_high:.3f} |")
    L += ["", f"- seeds 0-19, rise of P(T=high) from the RF report: {rise.min():+.3f} to {rise.max():+.3f}; below zero: {(rise < 0).sum()}",
          f"- P(Rf = none | drone, silent) in the estimated matrix (calibration seed 777): {none_given_silent:.4f}"]
    txt = "\n".join(L) + "\n"
    with open(os.path.join(RES, f"{P}_summary.md"), "w", encoding="utf-8", newline="\n") as f:
        f.write(txt)
    sys.stdout.reconfigure(encoding="utf-8")
    print(txt)


if __name__ == "__main__":
    main()
