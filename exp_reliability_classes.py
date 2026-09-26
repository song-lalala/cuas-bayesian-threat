"""
Exp-R: binned reliability decomposition for ALL FOUR threat classes.

Sec. VII-D reports the decomposition for the high class (the one that drives
an engagement decision). A reader is entitled to ask whether the residual
bias found there is specific to that class or a property of the estimator
everywhere, so this script repeats the analysis for negligible / low / medium /
high on the same 20 replications, with Clopper-Pearson intervals and the
CPT-oracle's own decomposition alongside.

Run:  python exp_reliability_classes.py [--reps 20] [--N 200]
Outputs: results/expR_reliability_classes.csv, results/expR_summary.md
"""
from __future__ import annotations
import os, time, argparse
import numpy as np
from scipy import stats as st

import model_spec as ms
import dataio
import metrics as mt
import pipeline as pl

RES = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(RES, exist_ok=True)
BINS = [0.0, 0.10, 0.25, 0.50, 0.75, 1.0]
# Names come from the model spec, never hardcoded: the threat levels are
# negligible/low/medium/high and a stale copy here would mislabel the table.
CLASSES = ms.STATES["T"]
MIN_N = 30


def cp_interval(k, n):
    lo = st.beta.ppf(0.025, k, n - k + 1) if k > 0 else 0.0
    hi = st.beta.ppf(0.975, k + 1, n - k) if k < n else 1.0
    return float(lo), float(hi)


def decompose(p, hit):
    """Return per-bin (label, predicted, observed, ci_lo, ci_hi, n)."""
    out = []
    for lo, hi_ in zip(BINS[:-1], BINS[1:]):
        m = (p > lo) & (p <= hi_)
        if m.sum() < MIN_N:
            continue
        k, n = int(hit[m].sum()), int(m.sum())
        cl, ch = cp_interval(k, n)
        out.append((f"({lo:.2f},{hi_:.2f}]", float(p[m].mean()),
                    float(hit[m].mean()), cl, ch, n))
    return out


def run(reps=20, N=200):
    gt = ms.ground_truth_bn()
    mod_p, mod_y, ora_p, ora_y = [], [], [], []
    for r in range(reps):
        test = dataio.generate_dataset(gt, 2500, seed=9000 + r)
        calib = dataio.generate_dataset(gt, 300, seed=5000 + r)
        est = dataio.estimate_confusion(calib)
        alpha, _ = dataio.fit_reliability(calib)
        data = dataio.generate_dataset(gt, N, seed=r)
        cpts, _ = pl.ground_internal("Proposed", data, r, gt=gt)
        P, y = mt.posteriors(cpts, test, ms.SENSORS, sensor_cpts=est,
                             alpha=alpha)
        mod_p.append(P); mod_y.append(y)
        Po, yo = mt.posteriors({k: gt.cpts[k] for k in ms.INTERNAL}, test,
                               ms.SENSORS)
        ora_p.append(Po); ora_y.append(yo)
    P = np.concatenate(mod_p); y = np.concatenate(mod_y)
    Po = np.concatenate(ora_p); yo = np.concatenate(ora_y)

    rows, summary = [], []
    for k, name in enumerate(CLASSES):
        mod = decompose(P[:, k], (y == k).astype(float))
        ora = dict((b[0], b) for b in decompose(Po[:, k],
                                                (yo == k).astype(float)))
        gap_w, gap_n, worst = 0.0, 0, (0.0, "", 0.0, 0.0)
        for lab, pm, om, cl, ch, n in mod:
            o = ora.get(lab)
            rows.append({"class": name, "bin": lab, "predicted": pm,
                         "observed": om, "ci_lo": cl, "ci_hi": ch, "n": n,
                         "outside_ci": int(not (cl <= pm <= ch)),
                         "z": (om - pm) / np.sqrt(max(om * (1 - om), 1e-12) / n),
                         "oracle_predicted": o[1] if o else float("nan"),
                         "oracle_observed": o[2] if o else float("nan"),
                         "oracle_n": o[5] if o else 0})
            gap_w += n * abs(om - pm); gap_n += n
            if abs(om - pm) > abs(worst[0]):
                worst = (om - pm, lab, pm, om)
        summary.append((name, gap_w / max(gap_n, 1), worst,
                        sum(1 for b in mod if not (b[3] <= b[1] <= b[4])),
                        len(mod)))

    import pandas as pd
    pd.DataFrame(rows).to_csv(
        os.path.join(RES, "expR_reliability_classes.csv"), index=False)
    with open(os.path.join(RES, "expR_summary.md"), "w", encoding="utf-8") as f:
        f.write(f"# Exp-R: per-class reliability decomposition "
                f"(N={N}, {reps} replications pooled)\n\n")
        f.write("| class | weighted mean |pred-obs| | worst bin | predicted "
                "-> observed | bins outside CI |\n|---|---|---|---|---|\n")
        for name, gap, (d, lab, pm, om), out, tot in summary:
            f.write(f"| {name} | {gap:.3f} | {lab} | {pm:.3f} -> {om:.3f} "
                    f"({d:+.3f}) | {out}/{tot} |\n")
            print(f"  {name:8s}: weighted gap {gap:.3f}; worst bin {lab} "
                  f"{pm:.3f}->{om:.3f} ({d:+.3f}); {out}/{tot} bins outside CI")
    print("\nsaved results/expR_reliability_classes.csv and expR_summary.md")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=20)
    ap.add_argument("--N", type=int, default=200)
    a = ap.parse_args()
    t0 = time.time()
    run(a.reps, a.N)
    print(f"total {time.time()-t0:.1f}s")
