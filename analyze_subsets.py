"""
Post-analysis for the revision (runs on expS raw files; separate from the
experiment scripts so it never races the freshness gate):
  1) reliability on-vs-off paired tests on the hard subsets (N4)
  2) far-range full-subset rows for Table 5 (N4)
  3) FAR@Pd paired test calibrated-vs-abstract (N6)
  4) high-class reliability table: predicted vs observed P(T=high) bins (N6)
"""
from __future__ import annotations
import os
import numpy as np
import pandas as pd
from scipy import stats as st

import model_spec as ms
import dataio
import metrics as mt
import pipeline as pl

RES = os.path.join(os.path.dirname(__file__), "results")
fus = pd.read_csv(os.path.join(RES, "expS_fusion_raw.csv"))


def t_ci(x):
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    h = x.std(ddof=1) / np.sqrt(len(x)) * st.t.ppf(0.975, len(x) - 1)
    return x.mean(), h, len(x)


print("== (1) reliability on-off paired deltas (calib+reliab - calibrated) ==")
for sub in ["silent-drone", "far-range", "far-silent"]:
    on = fus[fus.config == f"_subset_{sub}_calib+reliab"].set_index("rep")
    off = fus[fus.config == f"_subset_{sub}_calibrated"].set_index("rep")
    common = on.index.intersection(off.index)
    for met in ["ece", "brier", "acc"]:
        d = (on.loc[common, met] - off.loc[common, met]).values
        m, h, n = t_ci(d)
        p = st.ttest_1samp(d, 0).pvalue
        print(f"  {sub:12s} {met:5s}: {m:+.4f} [{m-h:+.4f},{m+h:+.4f}] "
              f"p={p:.3f} (n={n})")

print("\n== (2) far-range full-subset rows (mean +- CIhalf) ==")
for cfg in ["abstract", "calibrated", "calib+reliab"]:
    d = fus[fus.config == f"_subset_far-range_{cfg}"]
    parts = []
    for met in ["acc", "macroAUC", "brier", "ece"]:
        m, h, _ = t_ci(d[met].values)
        parts.append(f"{met} {m:.3f}+-{h:.3f}")
    print(f"  {cfg:12s}: " + ", ".join(parts) +
          f"  n_sub~{d['n_sub'].mean():.0f}")

print("\n== (3) FAR@Pd=0.9 paired (calibrated - abstract) ==")
a = fus[fus.config == "all (abstract model)"].set_index("rep")["far@pd0.9"]
c = fus[fus.config == "all (calibrated)"].set_index("rep")["far@pd0.9"]
d = (c - a).dropna().values
m, h, n = t_ci(d)
print(f"  dFAR = {m:+.4f} [{m-h:+.4f},{m+h:+.4f}] "
      f"p={st.ttest_1samp(d,0).pvalue:.3f} (n={n})")
print("\n== (4) high-class reliability (P(T=high) bins; 20 draws, N=200) ==")
gt = ms.ground_truth_bn()
allp, ally, orp, ory = [], [], [], []
for r in range(20):
    test = dataio.generate_dataset(gt, 2500, seed=9000 + r)
    calib = dataio.generate_dataset(gt, 300, seed=5000 + r)
    est = dataio.estimate_confusion(calib)
    alpha, _ = dataio.fit_reliability(calib)
    data = dataio.generate_dataset(gt, 200, seed=r)
    cpts, _ = pl.ground_internal("Proposed", data, r, gt=gt)
    Pm, y = mt.posteriors(cpts, test, ms.SENSORS, sensor_cpts=est, alpha=alpha)
    allp.append(Pm[:, 3]); ally.append((y == 3).astype(float))
    Po, yo = mt.posteriors({k: gt.cpts[k] for k in ms.INTERNAL}, test,
                           ms.SENSORS)
    orp.append(Po[:, 3]); ory.append((yo == 3).astype(float))
pv = np.concatenate(allp); hy = np.concatenate(ally)
po = np.concatenate(orp); hyo = np.concatenate(ory)
bins = [0.0, 0.1, 0.25, 0.5, 0.75, 1.0]
rows = []
for lo, hi_ in zip(bins[:-1], bins[1:]):
    m = (pv > lo) & (pv <= hi_)
    mo = (po > lo) & (po <= hi_)
    if m.sum() < 30:
        continue
    k, n = int(hy[m].sum()), int(m.sum())
    lo_ci = st.beta.ppf(0.025, k, n - k + 1) if k > 0 else 0.0
    hi_ci = st.beta.ppf(0.975, k + 1, n - k) if k < n else 1.0
    rows.append((f"({lo:.2f},{hi_:.2f}]", pv[m].mean(), hy[m].mean(),
                 float(lo_ci), float(hi_ci), n,
                 po[mo].mean() if mo.sum() >= 30 else float("nan"),
                 hyo[mo].mean() if mo.sum() >= 30 else float("nan"),
                 int(mo.sum())))
for lab, pm, om, l, h, n, opm, oom, on in rows:
    orc = (f"oracle {opm:.3f}->{oom:.3f} (n={on})"
           if opm == opm else "oracle n/a")
    print(f"  bin {lab}: predicted {pm:.3f} -> observed {om:.3f} "
          f"[{l:.3f},{h:.3f}] (n={n}); {orc}")
with open(os.path.join(RES, "expS_highclass_reliability.csv"), "w") as f:
    f.write("bin,predicted,observed,ci_lo,ci_hi,n,oracle_predicted,"
            "oracle_observed,oracle_n\n")
    for lab, pm, om, l, h, n, opm, oom, on in rows:
        f.write(f"{lab},{pm:.4f},{om:.4f},{l:.4f},{h:.4f},{n},"
                f"{opm:.4f},{oom:.4f},{on}\n")
print("saved results/expS_highclass_reliability.csv "
      "(20 draws, binomial CIs, oracle row)")
