"""
Exp-BS: replication-level intervals for the binned reliability of P(T = high)
(Table 9 and the mixed-world decomposition of Sec. VII-H). Predictions within one
replication share its training set, calibration split and CPTs, so binomial
intervals on the pooled scenarios are too narrow for a statement about the
estimator. Here the 20 replications are resampled with replacement and the
deviation predicted - observed is recomputed on the pooled bins of each resample.
Core files are not modified; results go to results/expBS_*.

Synthetic world: the runs of exp_reliability_classes.py (N = 200, test 9000+r,
calibration 5000+r, training r), repeated to keep the per-replication bins of the
pipeline and of the CPT oracle; the pooled bins are checked against
results/expR_reliability_classes.csv. Mixed world: results/expI_highthreat_bins.csv,
splits P0, P1, P1c, P2, P3, variants calib-hard and CPT-oracle.

Bins (0,.10], (.10,.25], (.25,.50], (.50,.75], (.75,1]. Percentile intervals (2.5,
97.5) from 10,000 resamples, numpy default_rng(20261106); the same resampled
replications serve every bin and variant. An interval "excludes 0" if both ends
have one sign. The binomial count is the Clopper-Pearson interval on the pooled
bin, as in Table 9.

Pipeline against CPT oracle: counting bins whose own interval excludes 0 does not
compare the two, so results/expBS_paired.csv gives, on the same resampled
replications, the signed difference (pipeline deviation - oracle deviation) and
the difference in size (|pipeline deviation| - |oracle deviation|), with percentile
intervals, per bin. A resample that leaves a bin empty in either is left out of
that bin. No further random numbers are drawn, so the intervals above do not move.

Run:  python exp_replication_intervals.py
      python exp_replication_intervals.py --summary-only
Outputs: results/expBS_synthetic_bins.csv, results/expBS_intervals.csv,
         results/expBS_paired.csv, results/expBS_summary.md
"""
from __future__ import annotations
import os, sys, argparse
import numpy as np
import pandas as pd
from scipy import stats as st

import model_spec as ms
import dataio
import metrics as mt
import pipeline as pl

RES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
P = "expBS"
BINS = [0.0, 0.10, 0.25, 0.50, 0.75, 1.0]
B = 10000
SEED = 20261106
PROTOS = ("P0", "P1", "P1c", "P2", "P3")


def bins_of(p, hit, r, variant):
    out = []
    for lo, hi in zip(BINS[:-1], BINS[1:]):
        m = (p > lo) & (p <= hi)
        out.append({"rep": r, "variant": variant, "bin_lo": lo, "bin_hi": hi, "n": int(m.sum()),
                    "sum_pred": float(p[m].sum()), "hits": int(hit[m].sum())})
    return out


def run_synthetic(reps=20, N=200):
    gt = ms.ground_truth_bn()
    rows = []
    for r in range(reps):
        test = dataio.generate_dataset(gt, 2500, seed=9000 + r)
        calib = dataio.generate_dataset(gt, 300, seed=5000 + r)
        est = dataio.estimate_confusion(calib)
        alpha, _ = dataio.fit_reliability(calib)
        data = dataio.generate_dataset(gt, N, seed=r)
        cpts, _ = pl.ground_internal("Proposed", data, r, gt=gt)
        Pm, y = mt.posteriors(cpts, test, ms.SENSORS, sensor_cpts=est, alpha=alpha)
        rows += bins_of(Pm[:, 3], (y == 3), r, "pipeline")
        Po, yo = mt.posteriors({k: gt.cpts[k] for k in ms.INTERNAL}, test, ms.SENSORS)
        rows += bins_of(Po[:, 3], (yo == 3), r, "CPT-oracle")
        print(f"[expBS] rep {r + 1}/{reps}", flush=True)
    pd.DataFrame(rows).to_csv(os.path.join(RES, f"{P}_synthetic_bins.csv"), index=False)


def cp(k, n):
    lo = st.beta.ppf(0.025, k, n - k + 1) if k > 0 else 0.0
    hi = st.beta.ppf(0.975, k + 1, n - k) if k < n else 1.0
    return float(lo), float(hi)


def intervals(df, idx):
    """idx: B x reps array of resampled replication indices."""
    out = []
    nb = len(BINS) - 1
    tab = {}
    for k in ("n", "sum_pred", "hits"):
        tab[k] = df.pivot(index="rep", columns="bin_lo", values=k).sort_index(axis=1).values
    assert tab["n"].shape == (20, nb)
    for b in range(nb):
        n_, p_, h_ = (tab[k][:, b] for k in ("n", "sum_pred", "hits"))
        pooled = (p_.sum() - h_.sum()) / n_.sum()
        ns = n_[idx].sum(axis=1)
        ok = ns > 0
        dev = (p_[idx].sum(axis=1)[ok] - h_[idx].sum(axis=1)[ok]) / ns[ok]
        lo, hi = np.percentile(dev, [2.5, 97.5])
        cl, ch = cp(int(h_.sum()), int(n_.sum()))
        pred = p_.sum() / n_.sum()
        out.append({"bin_lo": BINS[b], "bin_hi": BINS[b + 1], "n": int(n_.sum()), "predicted": float(pred),
                    "observed": float(h_.sum() / n_.sum()), "dev": float(pooled), "boot_lo": float(lo), "boot_hi": float(hi),
                    "boot_excludes_0": bool(lo > 0 or hi < 0), "binomial_outside": bool(not (cl <= pred <= ch)),
                    "resamples_used": int(ok.sum())})
    return out


def deviations(df, idx):
    """Pooled deviation per bin, and the deviation of every resample (bins x resamples, NaN if the bin is empty)."""
    tab = {k: df.pivot(index="rep", columns="bin_lo", values=k).sort_index(axis=1).values for k in ("n", "sum_pred", "hits")}
    pooled, res = [], []
    for b in range(len(BINS) - 1):
        n_, p_, h_ = (tab[k][:, b] for k in ("n", "sum_pred", "hits"))
        pooled.append((p_.sum() - h_.sum()) / n_.sum())
        ns = n_[idx].sum(axis=1).astype(float)
        ns[ns == 0] = np.nan
        res.append((p_[idx].sum(axis=1) - h_[idx].sum(axis=1)) / ns)
    return np.array(pooled), np.array(res)


def paired_intervals(a, o, idx):
    """Pipeline (a) minus oracle (o) on the same resamples: signed difference, difference in size."""
    pa, ra = deviations(a, idx)
    po, ro = deviations(o, idx)
    out = []
    for b in range(len(BINS) - 1):
        ok = ~(np.isnan(ra[b]) | np.isnan(ro[b]))
        d, s = ra[b][ok] - ro[b][ok], np.abs(ra[b][ok]) - np.abs(ro[b][ok])
        lo, hi = np.percentile(d, [2.5, 97.5])
        slo, shi = np.percentile(s, [2.5, 97.5])
        out.append({"bin_lo": BINS[b], "bin_hi": BINS[b + 1], "dev_pipeline": float(pa[b]), "dev_oracle": float(po[b]),
                    "diff": float(pa[b] - po[b]), "diff_lo": float(lo), "diff_hi": float(hi), "diff_excludes_0": bool(lo > 0 or hi < 0),
                    "size_diff": float(abs(pa[b]) - abs(po[b])), "size_lo": float(slo), "size_hi": float(shi), "size_excludes_0": bool(slo > 0 or shi < 0),
                    "resamples_used": int(ok.sum())})
    return out


def summarize():
    rng = np.random.default_rng(SEED)
    idx = rng.integers(0, 20, size=(B, 20))
    syn = pd.read_csv(os.path.join(RES, f"{P}_synthetic_bins.csv"), float_precision="round_trip")
    mix = pd.read_csv(os.path.join(RES, "expI_highthreat_bins.csv"), float_precision="round_trip")
    ref = pd.read_csv(os.path.join(RES, "expR_reliability_classes.csv"))
    ref = ref[ref["class"] == "high"]
    rows = []
    for v in ("pipeline", "CPT-oracle"):
        for o in intervals(syn[syn.variant == v], idx):
            rows.append({"world": "synthetic", "proto": "-", "variant": v, **o})
    for proto in PROTOS:
        for v in ("calib-hard", "CPT-oracle"):
            for o in intervals(mix[(mix.proto == proto) & (mix.variant == v)], idx):
                rows.append({"world": "mixed", "proto": proto, "variant": v, **o})
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RES, f"{P}_intervals.csv"), index=False)
    rows2 = [{"world": "synthetic", "proto": "-", **o} for o in paired_intervals(syn[syn.variant == "pipeline"], syn[syn.variant == "CPT-oracle"], idx)]
    for proto in PROTOS:
        q = mix[mix.proto == proto]
        rows2 += [{"world": "mixed", "proto": proto, **o} for o in paired_intervals(q[q.variant == "calib-hard"], q[q.variant == "CPT-oracle"], idx)]
    pr = pd.DataFrame(rows2)
    pr.to_csv(os.path.join(RES, f"{P}_paired.csv"), index=False)
    # the pooled synthetic bins are those of Table 9
    q = df[(df.world == "synthetic") & (df.variant == "pipeline")].reset_index(drop=True)
    assert len(ref) == 5 and np.allclose(q.predicted.values, ref.predicted.values, atol=1e-9) and np.allclose(q.observed.values, ref.observed.values, atol=1e-9) and (q.n.values == ref.n.values).all()
    L = ["# Exp-BS: replication-level intervals for the binned reliability of P(T = high)", "",
         f"- {B} resamples of the 20 replications, default_rng({SEED}); deviation = predicted - observed",
         f"- pooled synthetic bins equal expR_reliability_classes.csv (high class): 5 bins", "",
         "## synthetic world", "| variant | bin | n | deviation | 95% replication interval | excludes 0 | outside binomial |", "|---|---|---|---|---|---|---|"]
    for _, r in df[df.world == "synthetic"].iterrows():
        L.append(f"| {r.variant} | ({r.bin_lo:.2f},{r.bin_hi:.2f}] | {r.n} | {r.dev:+.4f} | [{r.boot_lo:+.4f}, {r.boot_hi:+.4f}] | {r.boot_excludes_0} | {r.binomial_outside} |")
    L += ["", "## mixed world, bins excluding 0 (of 25 over five splits)"]
    for v in ("calib-hard", "CPT-oracle"):
        q = df[(df.world == "mixed") & (df.variant == v)]
        L.append(f"- {v}: replication intervals {int(q.boot_excludes_0.sum())}/{len(q)}; binomial intervals {int(q.binomial_outside.sum())}/{len(q)}")
    L += ["", "| split | variant | bin | deviation | 95% replication interval | excludes 0 | outside binomial |", "|---|---|---|---|---|---|---|"]
    for _, r in df[df.world == "mixed"].iterrows():
        L.append(f"| {r.proto} | {r.variant} | ({r.bin_lo:.2f},{r.bin_hi:.2f}] | {r.dev:+.4f} | [{r.boot_lo:+.4f}, {r.boot_hi:+.4f}] | {r.boot_excludes_0} | {r.binomial_outside} |")
    L += ["", "## pipeline minus CPT oracle on the same resampled replications",
          "signed = deviation of the pipeline - deviation of the oracle; size = |deviation of the pipeline| - |deviation of the oracle|", ""]
    qm = pr[pr.world == "mixed"]
    L.append(f"- mixed world: signed difference excludes 0 in {int(qm.diff_excludes_0.sum())}/{len(qm)} bins, difference in size in {int(qm.size_excludes_0.sum())}/{len(qm)}")
    for lo in BINS[:-1]:
        qb = qm[qm.bin_lo == lo]
        L.append(f"- mixed world, bin from {lo:.2f}: signed {int(qb.diff_excludes_0.sum())}/{len(qb)}, size {int(qb.size_excludes_0.sum())}/{len(qb)}")
    L += ["", "| split | bin | pipeline | oracle | signed difference | 95% interval | excludes 0 | difference in size | 95% interval | excludes 0 |", "|---|---|---|---|---|---|---|---|---|---|"]
    for _, r in pr.iterrows():
        L.append(f"| {r.proto} | ({r.bin_lo:.2f},{r.bin_hi:.2f}] | {r.dev_pipeline:+.4f} | {r.dev_oracle:+.4f} | {r['diff']:+.4f} | [{r.diff_lo:+.4f}, {r.diff_hi:+.4f}] | {r.diff_excludes_0} | {r.size_diff:+.4f} | [{r.size_lo:+.4f}, {r.size_hi:+.4f}] | {r.size_excludes_0} |")
    txt = "\n".join(L) + "\n"
    with open(os.path.join(RES, f"{P}_summary.md"), "w", encoding="utf-8", newline="\n") as f:
        f.write(txt)
    sys.stdout.reconfigure(encoding="utf-8")
    print(txt)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary-only", action="store_true")
    a = ap.parse_args()
    if not a.summary_only:
        run_synthetic()
    summarize()
