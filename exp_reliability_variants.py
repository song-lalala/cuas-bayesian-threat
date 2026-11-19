"""
Double discounting of sensor uncertainty.

The deployed sensor layer (V0) mixes a range-MARGINAL confusion matrix,
which already averages the far-range degradation, with a uniform report using
alpha = clip(0.98 p/max p, .05, .98), a ratio of raw accuracies that does not
subtract the chance rate. This script compares V0 with the chance-corrected
alternative of reliability_cc.py and with the reference models, in the
standard world and in a structural-degradation world (world_options.py).
Core files are not modified; results go to results/expD_*.

Variants (names are fixed and never mixed):
  V0                        range-marginal C + alpha (dataio.fit_reliability)
  V1                        reference matrix C_ref + chance-corrected kappa,
                            cap 1 (reliability_cc.calibrate_v1)
  V1-cap0.98                the same C_ref, deployed weight min(kappa, 0.98):
                            isolates V1's cap from V0's 0.98
  V2                        range-conditional counts (beta = 0.5), no discount
  V3                        range-marginal C, no discount (= 'all (calibrated)')
  sensor-oracle-marginal    true range-marginal tables of the world
                            (Table 7 'sensor-oracle'), grounded internal CPTs
  sensor-oracle-conditional true range-conditional tables of the world,
                            grounded internal CPTs
  CPT-oracle                true internal CPTs + true range-marginal tables
                            (the manuscript's CPT-oracle)
  Bayes-oracle              true internal CPTs + true range-conditional tables
                            (the exact posterior of the world)

Protocol (identical to the fusion block of exp_significance.py): S = 20
replications; rep r draws the test set (2500, seed 9000+r), the training set
(N = 200, seed r; internal CPTs grounded with the Proposed pipeline) and the
calibration split (seed 5000+r; n = 300 for the main comparison, the sweep
uses n in {30, 100, 300, 1000} from the same seed, so smaller splits are
prefixes of larger ones). Both worlds use the same seeds; sampling consumes
one uniform per node in a fixed topological order, so the two worlds share
every root, latent and threat draw and differ only in the sensor reports.

Subsets: all; far / near / imminent; radio-silent drones (cls = drone,
emit = silent) and their intersection with each range.
Metrics: accuracy, macro-F1, macro-AUC (NaN when a threat level is absent
from the subset, as in Table 6), Brier, top-label ECE (15 bins), classwise
ECE; FAR@Pd=0.9 on the full set.

Adoption rule (fixed in the plan before any result; reproduced verbatim):
  "In the standard world, adopt V1 as the deployed model if, against V0, its
   mean Brier score and ECE are not significantly worse (paired test over
   the 20 replications) and it reduces the calibration error in the near and
   imminent bins. If V1 is significantly worse in Brier, keep V0 and report
   the analysis and interpretation only. Report the structural-world results
   together with V2 regardless of the decision."
Operationalisation (written into this file before the run):
  - paired test = two-sided one-sample t-test of the per-replication deltas
    (V1 - V0) over the 20 replications, alpha = 0.05; the Wilcoxon signed-
    rank p is reported beside it but does not decide;
  - "significantly worse" in a metric = mean delta > 0 and t-test p < 0.05,
    on the full test set (subset 'all');
  - "reduces the calibration error in the near and imminent bins" = the
    mean top-label ECE of V1 is below that of V0 in the 'near' subset AND in
    the 'imminent' subset (significance reported, not required);
  - adopt V1 iff Brier not significantly worse AND ECE not significantly
    worse AND the near/imminent condition holds; otherwise V0 stays.

Run:  python exp_reliability_variants.py                 (full: S = 20)
      python exp_reliability_variants.py --repro-only 3  (standard world,
                         first 3 reps, checks bit-reproduction, writes nothing)
      python exp_reliability_variants.py --summary-only
"""
from __future__ import annotations
import os, sys, time, argparse
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

import model_spec as ms
import dataio
import metrics as mt
import pipeline as pl
import reliability_cc as rcc
import world_options as wo
import exp_significance as es        # paired-test helpers and macro_auc only


def read_csv(path):
    """Exact parse. pandas' default float converter is not correctly rounded
    (it differs from float() by one ulp in about a third of the published
    values), which would break the bit-level reproduction check."""
    return pd.read_csv(path, float_precision="round_trip")

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
P = "expD"
TEST_N, FUSION_N, CALIB_MAIN = 2500, 200, 300
CALIB_NS = (30, 100, 300, 1000)
MAIN_VARIANTS = ["V0", "V1", "V1-cap0.98", "V2", "V3",
                 "sensor-oracle-marginal", "sensor-oracle-conditional",
                 "CPT-oracle", "Bayes-oracle"]
SWEEP_VARIANTS = ["V0", "V1", "V2", "V3"]
SUBSETS = ["all", "far", "near", "imminent", "silent-drone", "far-silent",
           "near-silent", "imminent-silent"]
METRICS = ["acc", "macroF1", "macroAUC", "brier", "ece", "cwece"]
LABELS = [0, 1, 2, 3]
REFERENCE_TABLE = {   # independently computed population values, for the A1 diff only
    "Re": {"V0": (0.594, 0.767, 0.804), "truth": (0.55, 0.92, 1.00), "V3": 0.821},
    "Ra": {"V0": (0.618, 0.737, 0.775), "truth": (0.50, 0.88, 1.00), "V3": 0.790},
    "Rr": {"V0": (0.870, 0.915, 0.930), "truth": (0.88, 0.97, 1.00), "V3": 0.949},
}


# ------------------------------------------------------------ evaluation --
def subset_masks(test):
    d = np.array([s["d"] for s in test])
    silent = np.array([s["cls"] == 2 and s["emit"] == 1 for s in test])
    return {"all": np.ones(len(test), bool), "far": d == 0, "near": d == 1,
            "imminent": d == 2, "silent-drone": silent,
            "far-silent": silent & (d == 0), "near-silent": silent & (d == 1),
            "imminent-silent": silent & (d == 2)}


def sub_metrics(Pm, ym):
    pred = Pm.argmax(1)
    return {"acc": float((pred == ym).mean()),
            "macroF1": float(f1_score(ym, pred, labels=LABELS,
                                      average="macro", zero_division=0)),
            "macroAUC": es.macro_auc(Pm, ym),
            "brier": mt.brier(Pm, ym), "ece": mt.ece(Pm, ym),
            "cwece": mt.classwise_ece(Pm, ym)}


def posteriors(variant, cpts, cpts_true, test, cal, tabs):
    S = ms.SENSORS
    if variant == "V0":
        return mt.posteriors(cpts, test, S, "calibrated",
                             sensor_cpts=cal["est"], alpha=cal["alpha"])
    if variant == "V1":
        return mt.posteriors(cpts, test, S, "calibrated",
                             sensor_cpts=cal["v1"]["c_ref"],
                             alpha=cal["v1"]["kappa"])
    if variant == "V1-cap0.98":
        kap = {s: np.minimum(cal["v1"]["kappa"][s], 0.98) for s in S}
        return mt.posteriors(cpts, test, S, "calibrated",
                             sensor_cpts=cal["v1"]["c_ref"], alpha=kap)
    if variant == "V2":
        return mt.posteriors(cpts, test, S, "conditional",
                             sensor_cpts=cal["estc"], alpha=None)
    if variant == "V3":
        return mt.posteriors(cpts, test, S, "calibrated",
                             sensor_cpts=cal["est"], alpha=None)
    if variant == "sensor-oracle-marginal":
        return mt.posteriors(cpts, test, S, "calibrated",
                             sensor_cpts=tabs["marg"], alpha=None)
    if variant == "sensor-oracle-conditional":
        return mt.posteriors(cpts, test, S, "conditional",
                             sensor_cpts=tabs["cond"], alpha=None)
    if variant == "CPT-oracle":
        return mt.posteriors(cpts_true, test, S, "calibrated",
                             sensor_cpts=tabs["marg"], alpha=None)
    if variant == "Bayes-oracle":
        return mt.posteriors(cpts_true, test, S, "conditional",
                             sensor_cpts=tabs["cond"], alpha=None)
    raise ValueError(variant)


def calibrate(bn, n, r):
    calib = dataio.generate_dataset(bn, n, seed=5000 + r)
    est = dataio.estimate_confusion(calib)
    alpha, _ = dataio.fit_reliability(calib)
    v1 = rcc.calibrate_v1(calib)
    for s in ms.SENSORS:                  # V1 starts from V0's own inputs
        assert v1["c_marg"][s].table.tobytes() == est[s].table.tobytes()
        assert v1["alpha_v0"][s].tobytes() == alpha[s].tobytes()
    return {"calib": calib, "est": est, "alpha": alpha, "v1": v1,
            "estc": dataio.estimate_confusion_conditional(calib)}


def run_world_rep(world, r, variants_main=MAIN_VARIANTS, calib_ns=CALIB_NS):
    bn = wo.world_bn(world)
    test = dataio.generate_dataset(bn, TEST_N, seed=9000 + r)
    data = dataio.generate_dataset(bn, FUSION_N, seed=r)
    cpts, _ = pl.ground_internal("Proposed", data, r, gt=bn)
    cpts_true = {k: bn.cpts[k] for k in ms.INTERNAL}
    tabs = {"marg": wo.true_marginal_tables(bn),
            "cond": wo.true_conditional_tables(bn)}
    masks = subset_masks(test)
    rows, crow = [], []
    for n in calib_ns:
        cal = calibrate(bn, n, r)
        for s in ms.SENSORS:
            st = cal["v1"]["stats"][s]
            crow.append({"world": world, "rep": r, "calib_n": n, "sensor": s,
                         **{f"p_{k}": cal["v1"]["p_hat"][s][k] for k in range(3)},
                         **{f"alpha_{k}": cal["alpha"][s][k] for k in range(3)},
                         **{f"kappa_{k}": cal["v1"]["kappa"][s][k] for k in range(3)},
                         "chance_exact": (rcc.chance_accuracy(cal["calib"], s)
                                          if n == CALIB_MAIN else np.nan),
                         **st})
        variants = variants_main if n == CALIB_MAIN else SWEEP_VARIANTS
        for v in variants:
            Pv, y = posteriors(v, cpts, cpts_true, test, cal, tabs)
            for sub in SUBSETS:
                m = masks[sub]
                row = {"world": world, "rep": r, "calib_n": n, "variant": v,
                       "subset": sub, "n_sub": int(m.sum()),
                       **sub_metrics(Pv[m], y[m])}
                if sub == "all":
                    row["far@pd0.9"] = mt.far_at_pd(Pv, y, 0.90)
                rows.append(row)
    return rows, crow, cpts


# ------------------------------------------------- bit-reproduction check --
def reproduction_mismatches(runs, reps):
    """Standard-world rows that must equal the published per-replication
    rows exactly (same code path, same seeds)."""
    fus = read_csv(os.path.join(RES, "expS_fusion_raw.csv"))
    cs = read_csv(os.path.join(RES, "expS_calibsize_raw.csv"))
    ab = read_csv(os.path.join(RES, "expA_baselines.csv"))
    std = runs[runs.world == "standard"]
    out, n_cmp = [], 0

    def cmp(label, mine, ref, cols):
        nonlocal n_cmp
        for c in cols:
            a = np.asarray(mine[c], float)
            b = np.asarray(ref[c], float)
            n_cmp += len(a)
            if len(a) != len(b) or not np.array_equal(a, b, equal_nan=True):
                d = np.nanmax(np.abs(a - b)) if len(a) == len(b) else np.nan
                out.append(f"{label} {c}: max|diff| = {d}")

    full = METRICS + ["far@pd0.9"]
    for v, cfg in (("V0", "all (calib+reliab)"), ("V3", "all (calibrated)")):
        mine = std[(std.calib_n == CALIB_MAIN) & (std.variant == v)
                   & (std.subset == "all")].sort_values("rep")
        ref = fus[(fus.config == cfg) & (fus.rep < reps)].sort_values("rep")
        cmp(f"{v} vs '{cfg}'", mine, ref, full)
        for sub, rsub in (("silent-drone", "silent-drone"), ("far", "far-range"),
                          ("far-silent", "far-silent")):
            tag = "calib+reliab" if v == "V0" else "calibrated"
            mine = std[(std.calib_n == CALIB_MAIN) & (std.variant == v)
                       & (std.subset == sub)].sort_values("rep")
            ref = fus[(fus.config == f"_subset_{rsub}_{tag}")
                      & (fus.rep < reps)].sort_values("rep")
            cmp(f"{v} {sub}", mine, ref,
                ["n_sub", "acc", "macroAUC", "brier", "ece", "cwece"])
    mine = std[(std.calib_n == CALIB_MAIN) & (std.variant == "sensor-oracle-marginal")
               & (std.subset == "all")].sort_values("rep")
    ref = cs[(cs.calib_n == -1) & (cs.rep < reps)].sort_values("rep")
    cmp("sensor-oracle-marginal vs calibsize oracle", mine, ref, full)
    for n in CALIB_NS:
        mine = std[(std.calib_n == n) & (std.variant == "V3")
                   & (std.subset == "all")].sort_values("rep")
        ref = cs[(cs.calib_n == n) & (cs.rep < reps)].sort_values("rep")
        cmp(f"V3 n={n} vs calibsize estimated", mine, ref, full)
    mine = std[(std.calib_n == CALIB_MAIN) & (std.variant == "CPT-oracle")
               & (std.subset == "all")].sort_values("rep")
    ref = ab[(ab.model == "CPT-oracle") & (ab.rep < reps)].sort_values("rep")
    cmp("CPT-oracle vs expA", mine, ref, full)
    return out, n_cmp


# --------------------------------------------------- population (A1) -----
def _pop_accuracy(bn, s):
    """P(report correct | d) at population level (exact sums over the
    world's tables, correctness by dataio._is_correct)."""
    f = bn.cpts[s]
    pc = bn.cpts["cls"].table
    pe = bn.cpts["emit"].table
    K = ms.CARD[s]
    p = np.zeros(ms.CARD["d"])
    for d in range(ms.CARD["d"]):
        tot = 0.0
        for c in range(ms.CARD["cls"]):
            for e in range(ms.CARD["emit"]):
                row = f.table[c, d] if "d" in f.vars else f.table[c, e]
                ok = sum(row[r] for r in range(K)
                         if dataio._is_correct({"cls": c, "emit": e, s: r}, s))
                tot += pc[c] * pe[e] * ok
        p[d] = tot
    return p


def _share(L, base):
    """Clean-matrix share s in L = s base + (1 - s) U (least squares over
    every entry) and the residual norm."""
    K = base.shape[-1]
    x = (base - 1.0 / K).ravel()
    yv = (L - 1.0 / K).ravel()
    s = float(x @ yv / (x @ x))
    return s, float(np.linalg.norm(yv - s * x))


def _kl(T, L, pc):
    """sum_c P(c) KL(T(.|c) || L(.|c)) in nats. Rounding can leave
    KL(T || T) at -1e-17; that is reported as 0. A zero in L where T > 0
    gives +inf (a hard zero in the deployed likelihood), reported as such."""
    with np.errstate(divide="ignore"):
        v = float(sum(pc[c] * np.sum(T[c] * np.log(T[c] / L[c]))
                      for c in range(T.shape[0])))
    return 0.0 if -1e-12 < v < 0 else v


def population_tables(big_n=100_000, big_seed=777_001):
    rows = []
    for world in wo.WORLDS:
        bn = wo.world_bn(world)
        pd_true = bn.cpts["d"].table
        pc = bn.cpts["cls"].table
        tm = wo.true_marginal_tables(bn)
        p = {s: _pop_accuracy(bn, s) for s in ms.SENSORS}
        alpha = {s: np.clip(p[s] / max(p[s].max(), 1e-6) * 0.98, 0.05, 0.98)
                 for s in ms.SENSORS}            # = dataio.fit_reliability rule
        kappa = {s: rcc.kappa_from_p(p[s], ms.CARD[s]) for s in ms.SENSORS}
        cref, st = rcc.reference_confusion(tm, kappa, pd_true, None)
        for s in wo.DEGRADED:
            T = bn.cpts[s].table                       # (cls, d, R)
            base = ms.ground_truth_bn().cpts[s].table[:, 2, :]
            for d in range(3):
                Ld = {"V0": dataio._discount_cpt(tm[s], alpha[s][d]).table,
                      "V1": dataio._discount_cpt(cref[s], kappa[s][d]).table,
                      "V1-cap0.98": dataio._discount_cpt(
                          cref[s], min(kappa[s][d], 0.98)).table,
                      "V3": tm[s].table, "truth": T[:, d, :]}
                for v, L in Ld.items():
                    sh, res = _share(L, base)
                    rows.append({"world": world, "source": "population",
                                 "sensor": s, "d": ms.STATES["d"][d],
                                 "variant": v, "share": sh,
                                 "share_resid": res,
                                 "kl_truth_nats": _kl(T[:, d, :], L, pc),
                                 "p_correct": p[s][d], "alpha": alpha[s][d],
                                 "kappa": kappa[s][d],
                                 "kbar": st[s]["kbar"]})
        # large-sample check with the actual estimation functions
        big = dataio.generate_dataset(bn, big_n, seed=big_seed)
        v1 = rcc.calibrate_v1(big)
        for s in wo.DEGRADED:
            T = bn.cpts[s].table
            base = ms.ground_truth_bn().cpts[s].table[:, 2, :]
            for d in range(3):
                Ld = {"V0": dataio._discount_cpt(v1["c_marg"][s],
                                                 v1["alpha_v0"][s][d]).table,
                      "V1": dataio._discount_cpt(v1["c_ref"][s],
                                                 v1["kappa"][s][d]).table,
                      "V3": v1["c_marg"][s].table}
                for v, L in Ld.items():
                    sh, res = _share(L, base)
                    rows.append({"world": world, "source": f"estimated n={big_n}",
                                 "sensor": s, "d": ms.STATES["d"][d],
                                 "variant": v, "share": sh, "share_resid": res,
                                 "kl_truth_nats": _kl(T[:, d, :], L, pc),
                                 "p_correct": v1["p_hat"][s][d],
                                 "alpha": v1["alpha_v0"][s][d],
                                 "kappa": v1["kappa"][s][d],
                                 "kbar": v1["stats"][s]["kbar"]})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ run ----
def run(reps):
    t0 = time.time()
    os.makedirs(RES, exist_ok=True)
    differ, drows = wo.identity_report()
    assert differ == ["Rr", "Re", "Ra"], differ
    print(f"[expD] structural world differs from standard only in {differ} "
          f"(rows {drows}); every other CPT byte-identical", flush=True)
    pop = population_tables()
    pop.to_csv(os.path.join(RES, f"{P}_population.csv"), index=False)
    print(f"[expD] population tables done ({time.time()-t0:.0f}s)", flush=True)
    rows, crows = [], []
    for r in range(reps):
        cp = {}
        for world in wo.WORLDS:
            rw, cw, cpts = run_world_rep(world, r)
            rows += rw
            crows += cw
            cp[world] = cpts
        for k in ms.INTERNAL:      # internal CPTs never see a sensor report
            assert cp["standard"][k].table.tobytes() == \
                cp["structural"][k].table.tobytes(), (r, k)
        pd.DataFrame(rows).to_csv(os.path.join(RES, f"{P}_runs_raw.csv"),
                                  index=False)
        pd.DataFrame(crows).to_csv(os.path.join(RES, f"{P}_calib_raw.csv"),
                                   index=False)
        print(f"[expD] rep {r+1}/{reps} done ({time.time()-t0:.0f}s)",
              flush=True)
    runs = read_csv(os.path.join(RES, f"{P}_runs_raw.csv"))
    mism, n_cmp = reproduction_mismatches(runs, reps)
    print(f"[expD] bit-reproduction: {n_cmp} values compared, "
          f"{len(mism)} mismatching columns", flush=True)
    assert n_cmp > 0
    for m in mism:
        print("   MISMATCH", m, flush=True)
    if mism:
        sys.exit(1)
    summarize()
    print(f"[expD] ALL DONE in {time.time()-t0:.0f}s", flush=True)


def repro_only(k):
    t0 = time.time()
    rows = []
    for r in range(k):
        rw, _, _ = run_world_rep("standard", r)
        rows += rw
    runs = pd.DataFrame(rows)
    mism, n_cmp = reproduction_mismatches(runs, k)
    print(f"[expD repro-only] reps 0..{k-1}: {n_cmp} values compared, "
          f"{len(mism)} mismatching columns ({time.time()-t0:.0f}s)")
    assert n_cmp > 0
    for m in mism:
        print("   MISMATCH", m)
    return 1 if mism else 0


def run_true_internal(reps):
    """Diagnostic added after the main run (it does not enter the adoption
    rule): the same sensor layers on top of the TRUE internal CPTs, to test
    whether V0's calibration edge compensates for the learned internal
    CPTs' overconfidence rather than reflecting a better sensor model.
    Writes results/expD_trueCPT_raw.csv."""
    t0 = time.time()
    rows = []
    for r in range(reps):
        for world in wo.WORLDS:
            bn = wo.world_bn(world)
            test = dataio.generate_dataset(bn, TEST_N, seed=9000 + r)
            data = dataio.generate_dataset(bn, FUSION_N, seed=r)
            cpts_l, _ = pl.ground_internal("Proposed", data, r, gt=bn)
            cpts_true = {k: bn.cpts[k] for k in ms.INTERNAL}
            tabs = {"marg": wo.true_marginal_tables(bn),
                    "cond": wo.true_conditional_tables(bn)}
            masks = subset_masks(test)
            cal = calibrate(bn, CALIB_MAIN, r)
            for tag, cp in (("learned-internal", cpts_l),
                            ("true-internal", cpts_true)):
                for v in ("V0", "V1", "V3", "sensor-oracle-conditional"):
                    Pv, y = posteriors(v, cp, cpts_true, test, cal, tabs)
                    for sub in SUBSETS:
                        m = masks[sub]
                        conf = Pv[m].max(1)
                        hit = (Pv[m].argmax(1) == y[m]).astype(float)
                        rows.append({"world": world, "rep": r,
                                     "calib_n": CALIB_MAIN,
                                     "variant": f"{v}|{tag}",
                                     "subset": sub, "n_sub": int(m.sum()),
                                     **sub_metrics(Pv[m], y[m]),
                                     "mean_conf": float(conf.mean()),
                                     "conf_gap": float(conf.mean() - hit.mean())})
        pd.DataFrame(rows).to_csv(os.path.join(RES, f"{P}_trueCPT_raw.csv"),
                                  index=False)
        print(f"[expD trueCPT] rep {r+1}/{reps} ({time.time()-t0:.0f}s)",
              flush=True)


def paired_between(df, world, a, b, subsets, metrics, n=CALIB_MAIN):
    """Paired a - b deltas over replications for any two variants."""
    g = df[(df.world == world) & (df.calib_n == n)]
    out = []
    for sub in subsets:
        for k in metrics:
            xa = g[(g.variant == a) & (g.subset == sub)].set_index("rep")[k]
            xb = g[(g.variant == b) & (g.subset == sub)].set_index("rep")[k]
            common = xa.index.intersection(xb.index)
            dl = (xa.loc[common] - xb.loc[common]).values
            dl = dl[~np.isnan(dl)]
            t = es.across_rep_tests(dl)
            out.append((sub, k, float(np.nanmean(xa)), float(np.nanmean(xb)),
                        t))
    return out


# -------------------------------------------------------------- summary ----
def _pm(x):
    m, lo, hi = es.t_ci(np.asarray(x, float))
    return m, (hi - lo) / 2


def paired_table(runs):
    out = []
    for (world, n), g in runs.groupby(["world", "calib_n"]):
        base = g[g.variant == "V0"].set_index(["subset", "rep"])
        for v in g.variant.unique():
            if v == "V0":
                continue
            gv = g[g.variant == v].set_index(["subset", "rep"])
            for sub in SUBSETS:
                for k in METRICS:
                    a = gv.loc[sub][k]
                    b = base.loc[sub][k]
                    common = a.index.intersection(b.index)
                    dlt = (a.loc[common] - b.loc[common]).values
                    dlt = dlt[~np.isnan(dlt)]
                    if len(dlt) < 3:
                        continue
                    t = es.across_rep_tests(dlt)
                    out.append({"world": world, "calib_n": n, "variant": v,
                                "subset": sub, "metric": k,
                                "mean_variant": float(np.nanmean(a.values)),
                                "mean_V0": float(np.nanmean(b.values)),
                                "delta": t["mean"], "lo": t["lo"],
                                "hi": t["hi"], "t_p": t["t_p"],
                                "wilcoxon_p": t["wilcoxon_p"],
                                "n_reps": len(dlt)})
    return pd.DataFrame(out)


def decomposition(runs, world, v, ref="V3", metric="brier"):
    """Exact range decomposition of a silent-subset delta: the silent-drone
    mean is the n-weighted mean of its far/near/imminent parts."""
    g = runs[(runs.world == world) & (runs.calib_n == CALIB_MAIN)]
    res = {"total": [], "far": [], "near": [], "imminent": []}
    for r in sorted(g.rep.unique()):
        gr = g[g.rep == r]

        def val(var, sub):
            x = gr[(gr.variant == var) & (gr.subset == sub)]
            return float(x[metric].iloc[0]), int(x["n_sub"].iloc[0])
        tv, nt = val(v, "silent-drone")
        tr, _ = val(ref, "silent-drone")
        res["total"].append(tv - tr)
        acc = 0.0
        for dname in ("far", "near", "imminent"):
            av, nd = val(v, f"{dname}-silent")
            ar, _ = val(ref, f"{dname}-silent")
            c = nd / nt * (av - ar)
            res[dname].append(c)
            acc += c
        assert abs(acc - (tv - tr)) < 1e-12, (r, acc, tv - tr)
    return {k: es.across_rep_tests(np.array(x)) for k, x in res.items()}


def decide(pt):
    q = pt[(pt.world == "standard") & (pt.calib_n == CALIB_MAIN)
           & (pt.variant == "V1")].set_index(["subset", "metric"])
    b, e = q.loc[("all", "brier")], q.loc[("all", "ece")]
    worse_b = b.delta > 0 and b.t_p < 0.05
    worse_e = e.delta > 0 and e.t_p < 0.05
    near, imm = q.loc[("near", "ece")], q.loc[("imminent", "ece")]
    reduce_ni = near.delta < 0 and imm.delta < 0
    adopt = (not worse_b) and (not worse_e) and reduce_ni
    lines = [
        f"- Brier (all): dV1-V0 = {b.delta:+.4f} [{b.lo:+.4f},{b.hi:+.4f}], "
        f"t p = {b.t_p:.2e}, W p = {b.wilcoxon_p:.2e} -> "
        f"{'significantly WORSE' if worse_b else 'not significantly worse'}",
        f"- ECE (all): dV1-V0 = {e.delta:+.4f} [{e.lo:+.4f},{e.hi:+.4f}], "
        f"t p = {e.t_p:.2e}, W p = {e.wilcoxon_p:.2e} -> "
        f"{'significantly WORSE' if worse_e else 'not significantly worse'}",
        f"- ECE near: {near.mean_V0:.4f} -> {near.mean_variant:.4f} "
        f"(d {near.delta:+.4f}, t p = {near.t_p:.2e}); ECE imminent: "
        f"{imm.mean_V0:.4f} -> {imm.mean_variant:.4f} (d {imm.delta:+.4f}, "
        f"t p = {imm.t_p:.2e}) -> "
        f"{'reduced in both' if reduce_ni else 'NOT reduced in both'}",
    ]
    if adopt:
        verdict = "ADOPT V1 as the deployed model"
    elif worse_b:
        verdict = "KEEP V0 (V1 significantly worse in Brier); analysis only"
    else:
        verdict = "KEEP V0 (adoption conditions not all met); analysis only"
    return adopt, verdict, lines


def summarize():
    runs = read_csv(os.path.join(RES, f"{P}_runs_raw.csv"))
    cal = read_csv(os.path.join(RES, f"{P}_calib_raw.csv"))
    pop = read_csv(os.path.join(RES, f"{P}_population.csv"))
    reps = runs.rep.nunique()
    pt = paired_table(runs)
    pt.to_csv(os.path.join(RES, f"{P}_paired_tests.csv"), index=False)
    L = [f"# Exp-D: double discounting of sensor uncertainty (S={reps}, "
         f"test n={TEST_N}, N={FUSION_N}, calibration n={CALIB_MAIN})", ""]

    L.append("## A1 population level: clean-matrix share of the deployed "
             "likelihood (standard world)")
    p0 = pop[(pop.world == "standard") & (pop.source == "population")]
    for s in wo.DEGRADED:
        parts = []
        for v in ("V0", "V1", "V1-cap0.98", "V3", "truth"):
            x = p0[(p0.sensor == s) & (p0.variant == v)].set_index("d")
            parts.append(f"{v} " + "/".join(f"{x.loc[d, 'share']:.3f}"
                                            for d in ms.STATES["d"]))
        rv = REFERENCE_TABLE[s]
        parts.append("reference V0 " + "/".join(f"{a:.3f}" for a in rv["V0"])
                     + f", V3 {rv['V3']:.3f}")
        L.append(f"- {s}: " + "; ".join(parts))
    L.append("- max residual of the share fit: "
             f"{p0.share_resid.max():.1e}")
    L.append("\n## A1b population level: KL(truth || deployed) in nats, "
             "far/near/imminent")
    for world in wo.WORLDS:
        pw = pop[(pop.world == world) & (pop.source == "population")]
        for s in wo.DEGRADED:
            parts = []
            for v in ("V0", "V1", "V1-cap0.98", "V3"):
                x = pw[(pw.sensor == s) & (pw.variant == v)].set_index("d")
                parts.append(f"{v} " + "/".join(
                    f"{x.loc[d, 'kl_truth_nats']:.4f}" for d in ms.STATES["d"]))
            L.append(f"- {world} {s}: " + "; ".join(parts))
    big = pop[pop.source.str.startswith("estimated")]
    L.append("\n## A1c large-sample check (actual estimators, "
             f"{big.source.iloc[0]}): share far/near/imminent, standard world")
    for s in wo.DEGRADED:
        parts = []
        for v in ("V0", "V1", "V3"):
            x = big[(big.world == "standard") & (big.sensor == s)
                    & (big.variant == v)].set_index("d")
            parts.append(f"{v} " + "/".join(f"{x.loc[d, 'share']:.3f}"
                                            for d in ms.STATES["d"]))
        L.append(f"- {s}: " + "; ".join(parts))

    for world in wo.WORLDS:
        g = runs[(runs.world == world) & (runs.calib_n == CALIB_MAIN)]
        L.append(f"\n## A5 {world} world, n={CALIB_MAIN}: full test set "
                 "(mean +- 95% CI half-width)")
        for v in MAIN_VARIANTS:
            d = g[(g.variant == v) & (g.subset == "all")]
            vals = []
            for k in METRICS + ["far@pd0.9"]:
                m, h = _pm(d[k])
                vals.append(f"{k} {m:.4f}+-{h:.4f}")
            L.append(f"- {v}: " + ", ".join(vals))
        for k in ("brier", "ece", "acc"):
            L.append(f"\n### {world}: {k} by subset (mean over replications)")
            hdr = "| variant | " + " | ".join(SUBSETS) + " |"
            L.append(hdr)
            L.append("|" + "---|" * (len(SUBSETS) + 1))
            for v in MAIN_VARIANTS:
                cells = []
                for sub in SUBSETS:
                    x = g[(g.variant == v) & (g.subset == sub)][k]
                    cells.append(f"{np.nanmean(x):.4f}")
                L.append(f"| {v} | " + " | ".join(cells) + " |")
        L.append(f"\n### {world}: paired deltas vs V0 (t p / Wilcoxon p)")
        q = pt[(pt.world == world) & (pt.calib_n == CALIB_MAIN)]
        for v in MAIN_VARIANTS[1:]:
            for sub in ("all", "far", "near", "imminent", "silent-drone",
                        "far-silent"):
                cells = []
                for k in ("brier", "ece", "acc", "macroAUC"):
                    x = q[(q.variant == v) & (q.subset == sub) & (q.metric == k)]
                    if len(x):
                        x = x.iloc[0]
                        cells.append(f"d{k} {x.delta:+.4f} "
                                     f"(p {x.t_p:.1e}/{x.wilcoxon_p:.1e})")
                L.append(f"- {v} | {sub}: " + "; ".join(cells))

    L.append("\n## A5 calibration-size sweep (full test set; ECE near / "
             "imminent)")
    for world in wo.WORLDS:
        for n in CALIB_NS:
            g = runs[(runs.world == world) & (runs.calib_n == n)]
            for v in SWEEP_VARIANTS:
                d = g[(g.variant == v) & (g.subset == "all")]
                dn = g[(g.variant == v) & (g.subset == "near")]
                di = g[(g.variant == v) & (g.subset == "imminent")]
                a, ah = _pm(d["macroAUC"]); b, bh = _pm(d["brier"])
                e, eh = _pm(d["ece"])
                L.append(f"- {world} n={n} {v}: AUC {a:.4f}+-{ah:.4f}, "
                         f"Brier {b:.4f}+-{bh:.4f}, ECE {e:.4f}+-{eh:.4f}; "
                         f"ECE near {dn['ece'].mean():.4f}, imminent "
                         f"{di['ece'].mean():.4f}")
            q = pt[(pt.world == world) & (pt.calib_n == n)
                   & (pt.subset == "all")]
            for v in ("V1", "V2", "V3"):
                x = q[(q.variant == v) & (q.metric == "brier")]
                y = q[(q.variant == v) & (q.metric == "ece")]
                if len(x):
                    L.append(f"    {v}-V0: dBrier {x.delta.iloc[0]:+.4f} "
                             f"(p {x.t_p.iloc[0]:.1e}), dECE "
                             f"{y.delta.iloc[0]:+.4f} (p {y.t_p.iloc[0]:.1e})")

    L.append("\n## reliability weights (n=300, mean over replications) and "
             "projection activity")
    for world in wo.WORLDS:
        c3 = cal[(cal.world == world) & (cal.calib_n == CALIB_MAIN)]
        for s in ms.SENSORS:
            x = c3[c3.sensor == s]
            al = "/".join(f"{x[f'alpha_{k}'].mean():.3f}" for k in range(3))
            ka = "/".join(f"{x[f'kappa_{k}'].mean():.3f}" for k in range(3))
            kmin = "/".join(f"{x[f'kappa_{k}'].min():.3f}" for k in range(3))
            L.append(f"- {world} {s}: alpha_V0 {al}; kappa {ka} (min {kmin}); "
                     f"kbar {x.kbar.mean():.3f}")
        for n in CALIB_NS:
            cn = cal[(cal.world == world) & (cal.calib_n == n)]
            L.append(f"- {world} n={n}: projected rows "
                     f"{int(cn.rows_projected.sum())}/{int(cn.rows.sum())} "
                     f"({cn.rows_projected.sum() / cn.rows.sum():.1%}); "
                     "by sensor " + ", ".join(
                         f"{s} {int(cn[cn.sensor == s].rows_projected.sum())}"
                         f"/{int(cn[cn.sensor == s].rows.sum())}"
                         for s in ms.SENSORS))
    rf = cal[(cal.calib_n == CALIB_MAIN) & (cal.sensor == "Rf")]
    L.append(f"- RF exact chance accuracy (mean) {rf.chance_exact.mean():.4f} "
             f"vs 1/K = {1/3:.4f}")

    L.append("\n## A6 range decomposition of the radio-silent-drone delta "
             "(n=300; contribution = n_d/n_silent x subset delta)")
    for world in wo.WORLDS:
        for v in ("V0", "V1"):
            for metric in ("brier", "acc"):
                dc = decomposition(runs, world, v, "V3", metric)
                L.append(f"- {world} {v}-V3 {metric}: total "
                         f"{dc['total']['mean']:+.5f} (p {dc['total']['t_p']:.1e})"
                         + "".join(f"; {dn} {dc[dn]['mean']:+.5f} "
                                   f"(p {dc[dn]['t_p']:.1e})"
                                   for dn in ("far", "near", "imminent")))

    L.append("\n## V1 cap: V1-cap0.98 - V1 (same C_ref; only the deployed "
             "weight is capped at 0.98 like V0)")
    for world in wo.WORLDS:
        for sub, k, ma, mb, t in paired_between(
                runs, world, "V1-cap0.98", "V1",
                ("all", "far", "near", "imminent", "silent-drone"),
                ("brier", "ece", "acc")):
            L.append(f"- {world} {sub} {k}: {mb:.4f} -> {ma:.4f} "
                     f"(d {t['mean']:+.4f}, t p {t['t_p']:.1e})")

    L.append("\n## subset-selection check: exact vs range-marginal sensor "
             "tables (positive d = the exact model scores worse)")
    for world in wo.WORLDS:
        for a, b in (("sensor-oracle-conditional", "sensor-oracle-marginal"),
                     ("Bayes-oracle", "CPT-oracle")):
            for sub, k, ma, mb, t in paired_between(
                    runs, world, a, b,
                    ("all", "far", "silent-drone", "far-silent"), ("brier",)):
                L.append(f"- {world} {a} - {b} | {sub} {k}: {mb:.4f} -> "
                         f"{ma:.4f} (d {t['mean']:+.4f}, t p {t['t_p']:.1e})")

    tpath = os.path.join(RES, f"{P}_trueCPT_raw.csv")
    if os.path.exists(tpath):
        tc = read_csv(tpath)
        # the learned-internal rows must equal the main run's rows exactly
        n_chk = 0
        for v in ("V0", "V1", "V3", "sensor-oracle-conditional"):
            a = tc[tc.variant == f"{v}|learned-internal"].sort_values(
                ["world", "rep", "subset"])
            b = runs[(runs.calib_n == CALIB_MAIN) & (runs.variant == v)] \
                .sort_values(["world", "rep", "subset"])
            for k in METRICS:
                assert np.array_equal(a[k].values, b[k].values,
                                      equal_nan=True), (v, k)
                n_chk += len(a)
        assert n_chk > 0
        both = pd.concat([runs[runs.calib_n == CALIB_MAIN], tc])
        L.append("\n## diagnostic (added after the main run; not part of the "
                 "rule): sensor layers on the TRUE internal CPTs")
        L.append(f"- consistency: {n_chk} learned-internal values equal the "
                 "main run exactly")
        L.append("- signed calibration gap = mean top-label confidence - "
                 "accuracy (positive = over-confident), full test set:")
        for world in wo.WORLDS:
            for tag in ("learned-internal", "true-internal"):
                cells = []
                for v in ("V0", "V1", "V3", "sensor-oracle-conditional"):
                    d = tc[(tc.world == world) & (tc.variant == f"{v}|{tag}")
                           & (tc.subset == "all")]
                    m, h = _pm(d.conf_gap)
                    cells.append(f"{v} {m:+.4f}+-{h:.4f} (conf "
                                 f"{d.mean_conf.mean():.4f})")
                L.append(f"    {world} {tag}: " + "; ".join(cells))
            for tag in ("learned-internal", "true-internal"):
                for sub, k, ma, mb, t in paired_between(
                        tc, world, f"V1|{tag}", f"V0|{tag}",
                        ("all", "near", "imminent"), ("conf_gap",)):
                    L.append(f"    {world} {tag} V1 - V0 conf_gap | {sub}: "
                             f"{mb:+.4f} -> {ma:+.4f} (d {t['mean']:+.4f}, "
                             f"t p {t['t_p']:.1e})")
                for sub, k, ma, mb, t in paired_between(
                        tc, world, f"V0|{tag}",
                        f"sensor-oracle-conditional|{tag}", ("all",),
                        ("conf_gap",)):
                    L.append(f"    {world} {tag} V0 - exact-sensor conf_gap: "
                             f"{mb:+.4f} -> {ma:+.4f} (d {t['mean']:+.4f} "
                             f"[{t['lo']:+.4f},{t['hi']:+.4f}], t p "
                             f"{t['t_p']:.1e}, W p {t['wilcoxon_p']:.1e})")
        for world in wo.WORLDS:
            g = both[both.world == world]
            for v in ("V0|true-internal", "V1|true-internal",
                      "V3|true-internal", "CPT-oracle", "Bayes-oracle"):
                d = g[(g.variant == v) & (g.subset == "all")]
                L.append(f"- {world} {v}: Brier {d.brier.mean():.4f}, ECE "
                         f"{d.ece.mean():.4f}, cwECE {d.cwece.mean():.4f}, "
                         f"acc {d.acc.mean():.4f}; ECE near "
                         f"{g[(g.variant == v) & (g.subset == 'near')].ece.mean():.4f}"
                         f", imminent "
                         f"{g[(g.variant == v) & (g.subset == 'imminent')].ece.mean():.4f}")
            for a, b in (("V1|true-internal", "V0|true-internal"),
                         ("V3|true-internal", "V0|true-internal"),
                         ("V0|true-internal", "Bayes-oracle"),
                         ("V1|true-internal", "Bayes-oracle")):
                for sub, k, ma, mb, t in paired_between(
                        both, world, a, b, ("all", "near", "imminent"),
                        ("brier", "ece")):
                    L.append(f"    {a} - {b} | {sub} {k}: d {t['mean']:+.4f} "
                             f"(t p {t['t_p']:.1e}, W p {t['wilcoxon_p']:.1e})")

    adopt, verdict, lines = decide(pt)
    L.append("\n## A7 decision (rule fixed before the run, standard world, n=300)")
    L += lines
    L.append(f"- VERDICT: {verdict}")
    txt = "\n".join(L)
    with open(os.path.join(RES, f"{P}_summary.md"), "w", encoding="utf-8") as f:
        f.write(txt + "\n")
    print(txt)
    return adopt


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=20)
    ap.add_argument("--repro-only", type=int, default=0)
    ap.add_argument("--summary-only", action="store_true")
    ap.add_argument("--population-only", action="store_true")
    ap.add_argument("--diag-true-internal", action="store_true")
    a = ap.parse_args()
    if a.repro_only:
        sys.exit(repro_only(a.repro_only))
    if a.population_only:
        population_tables().to_csv(os.path.join(RES, f"{P}_population.csv"),
                                   index=False)
    elif a.diag_true_internal:
        run_true_internal(a.reps)
    elif a.summary_only:
        summarize()
    else:
        run(a.reps)
