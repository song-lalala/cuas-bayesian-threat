"""
Exp-J: scaling the shared nuisance parent to simultaneous jamming of all four
modalities.

DESIGN FIXED IN THIS DOCSTRING BEFORE THE FIRST RUN. The planning note expected
"parameters linear, computation linear in |W|, the real bottleneck is calibration
data per jamming state"; that is an expectation, not a result. The rules [P], [C] and
[R] below decide what is written, whichever way they come out. exp_correlated.py and
results/expF_* are only read.

World. exp_correlated's world (world v2 with range-dependent sensors) with the shared
nuisance root W now a parent of ALL FOUR reports (Rr, Re, Ra, Rf).
  |W| = K in {2, 3, 4}. P(W = 0) = 0.65 (benign) as now, the remaining 0.35 split
  equally over the K - 1 degraded states.
  Severity of degraded state w: s_w = w / (K - 1), so K = 2: 1; K = 3: 1/2, 1;
  K = 4: 1/3, 2/3, 1 (equal steps up to s = 1).
  Report table in state w: (1 - gamma_s s_w) base + gamma_s s_w profile_s.
  Rr, Re, Ra: gamma and profile of exp_correlated (imported, not copied): a weak-
  signature drone is misread as bird/clutter/none.
  Rf: gamma = 0.90 (as Re and Ra), profile [consumer, mil, none] = [0.03, 0.02, 0.95]
  for every (cls, emit). Chosen because wide-band jamming raises the noise floor of an
  RF detector, so the report disappears rather than moving to another class: an
  emitting drone then looks radio-silent, the failure that most plausibly correlates
  with the other sensors' misses and that the latent emission state is exposed to.
  The Wobs node of exp_correlated is kept (binary, W != 0, 90%) only so that the draw
  sequence for K = 2 equals exp_correlated's; no model here uses it.
This world is an assumption. No measured jamming data enter this study.

Models, on the same draws, as in exp_correlated.py part (i):
  naive-CI   : deployed model, W unmodeled (marginal matrices, fitted reliability).
  W-observed : W a parent of the four reports, W-conditional matrices, W known at run
               time.
  W-latent   : the same tables, W never observed at run time (marginalized).
W is labeled in the calibration split for both W models, and P(W) is the true prior
(both as in exp_correlated.py; the prior is not estimated and not counted below).

Protocol. Replications r = 0..19: test 2,500 (seed 9000 + r), calibration (seed
5000 + r), training N = 200 (seed r), Proposed latent and threat tables, fitted
reliability on the W-mixed calibration split, exactly as exp_correlated.py. Calibration
n in {300, 1200}; the 300 are the first 300 of the 1200 (asserted).

Measured.
  [P] Parameters: extra free parameters of the sensor layer against naive-CI,
      (K - 1) * sum_s (prod of the card of s's marginal parents) (r_s - 1). Checked
      against the shapes of the estimated tables, and it must give the manuscript's 15
      at three sensors, K = 2. Stop if it does not.
  [C] Computation: per-update time in the way of exp_latency.py (evidence extraction
      plus exact variable elimination, time.perf_counter, one process, one thread,
      5 replications x 2,500 queries; condition order shuffled in each replication),
      with exp_latency's deployed path in the standard world as the same-session
      reference. Largest intermediate factor of the implemented elimination order
      (entries and variables), overall and among factors containing W.
      Reading: consistent with "linear in |W|" if the largest factor containing W grows
      in proportion to K and t(K = 4) / t(K = 2) <= 2 for W-latent; "computation is not
      the constraint" if the mean update stays under 1 ms in every condition. Otherwise
      the measured values are written instead.
  [R] Calibration recovery: ECE, macro-AUC and Brier overall, on the degraded subset
      (W != 0), on the worst state (W = K - 1, s = 1) and on the benign subset; pairwise
      error correlation; calibration samples per state and per parent cell.
      ECE of small subsets is biased upward with the subset size, so ECE on the worst
      state is compared only within one K. Across K the comparison is the gain on the
      degraded subset, G(K, n) = ECE_adv(naive-CI) - ECE_adv(model), and the Brier score.
      Paired t over the 20 replications, Holm within each family beside the raw p.
      Reading: "calibration data is the bottleneck" is written only if, for W-observed,
      G(4, 1200) > G(4, 300) with paired p < 0.05 and computation passes [C]. W-latent is
      read separately. A loss of any W model against naive-CI is listed, not averaged away.
Error correlation. rho3: mean over the pairs of Rr, Re, Ra of the correlation of
  (report != drone) on drones, exp_correlated's definition. rho_rf: mean over the pairs
  with Rf of the correlation with (Rf = none), on emitting drones. rho6: their mean.
Path check. Three sensors, K = 2, s = 1, n = 300 reproduces results/expF_corr_raw.csv
  (severity 1) on rho, acc, macroAUC, ece, brier, cwece, acc_adv, ece_adv, auc_adv for
  the three fusions and 20 seeds within 1e-9. Stop if it does not. The other cells have
  no earlier counterpart.

Outputs (results/): expJ_raw.csv, expJ_calib_counts.csv, expJ_params.csv, expJ_factors.csv,
  expJ_paired.csv, expJ_checks.txt, expJ_latency.csv, expJ_summary.md, expJ_run.log,
  expJ_latency_run.log. Timing varies between runs; the other files do not.

Run:  python exp_jam_scaling.py --selftest
      python exp_jam_scaling.py --stage main --workers 20
      python exp_jam_scaling.py --stage timing      (alone on the machine, one process)
      python exp_jam_scaling.py --summary-only
"""
from __future__ import annotations
import os
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import sys, time, platform, argparse
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
from cuas_bn import Factor, BayesNet, sample_ancestral
import model_spec as ms
import dataio
import metrics as mt
import pipeline as pl
import exp_correlated as ec

RES = os.path.join(HERE, "results")
P = "expJ"
REPS = 20
KS = (2, 3, 4)
NS = (300, 1200)
S3 = ("Rr", "Re", "Ra")
S4 = ("Rr", "Re", "Ra", "Rf")
FUSIONS = ("naive-CI", "W-observed", "W-latent")
MODE = {"naive-CI": "none", "W-observed": "observed", "W-latent": "latent"}
GAMMA_RF = 0.90
RF_PROFILE = np.array([0.03, 0.02, 0.95])
PATH_COLS = ("rho", "acc", "macroAUC", "ece", "brier", "cwece", "acc_adv", "ece_adv", "auc_adv")
TIME_REPS = 5
TIME_TEST = 2500


# ------------------------------------------------------------------- world ---
def prior_W(K):
    return np.array([1 - ec.P_ADVERSE] + [ec.P_ADVERSE / (K - 1)] * (K - 1))


def severities(K):
    return [w / (K - 1) for w in range(1, K)]


def _gamma(s):
    return GAMMA_RF if s == "Rf" else ec.GAMMA[s]


def _profile(s):
    return RF_PROFILE[None, None, :] if s == "Rf" else ec.ADVERSE_PROFILE[s][:, None, :]


def jam_world(K, w_sensors=S4):
    gt = ms.ground_truth_bn()
    card = dict(ms.CARD); card["W"] = K; card["Wobs"] = 2
    parents = {k: list(v) for k, v in ms.PARENTS.items()}
    parents["W"] = []
    parents["Wobs"] = ["W"]
    cpts = dict(gt.cpts)
    cpts["W"] = Factor(("W",), prior_W(K))
    a = ec.W_OBS_ACC
    cpts["Wobs"] = Factor(("W", "Wobs"), np.array([[a, 1 - a]] + [[1 - a, a]] * (K - 1)))
    for s in w_sensors:
        base = gt.cpts[s].table
        tabs = [base]
        for sev in severities(K):
            gam = _gamma(s) * sev
            tabs.append((1 - gam) * base + gam * _profile(s))
        parents[s] = list(ms.PARENTS[s]) + ["W"]
        cpts[s] = Factor(tuple(ms.PARENTS[s]) + ("W", s), np.stack(tabs, axis=-2))
    return BayesNet(card, parents, cpts)


# ------------------------------------------------------------------ models ---
def estimate_confusion_W(calib, K, w_sensors, smoothing=0.5):
    out = {}
    for s in ms.SENSORS:
        pa = dataio.MARGINAL_PARENTS[s] + (["W"] if s in w_sensors else [])
        shape = tuple((K if p == "W" else ms.CARD[p]) for p in pa) + (ms.CARD[s],)
        n = np.zeros(shape)
        for row in calib:
            n[tuple(row[p] for p in pa) + (row[s],)] += 1
        n = n + smoothing
        out[s] = Factor(tuple(pa) + (s,), n / n.sum(axis=-1, keepdims=True))
    return out


def build_bn(internal_cpts, sensor_cpts, w_mode, K, alpha=None, d_value=0):
    gt = ms.ground_truth_bn()
    cpts = dict(gt.cpts)
    parents = {k: list(v) for k, v in ms.PARENTS.items()}
    for k in ms.INTERNAL:
        cpts[k] = internal_cpts[k]
    for s in ms.SENSORS:
        cpts.pop(s, None)
        parents.pop(s, None)
    card = dict(ms.CARD)
    if w_mode != "none":
        card["W"] = K
        parents["W"] = []
        cpts["W"] = Factor(("W",), prior_W(K))
    for s in ms.SENSORS:
        c = sensor_cpts[s]
        if alpha is not None:
            c = dataio._discount_cpt(c, float(alpha[s][d_value]))
        cpts[s] = c
        parents[s] = list(c.vars[:-1])
    return BayesNet({k: card[k] for k in cpts}, {k: parents[k] for k in cpts}, cpts)


def evidence(scn, w_mode):
    ev = {k: scn[k] for k in ms.OBSERVED}
    for s in ms.SENSORS:
        ev[s] = scn[s]
    if w_mode == "observed":
        ev["W"] = scn["W"]
    return ev


def posteriors(internal_cpts, sensor_cpts, test, w_mode, K, alpha):
    Pm = np.zeros((len(test), ms.CARD["T"]))
    y = np.zeros(len(test), dtype=int)
    cache = {}
    for i, scn in enumerate(test):
        if scn["d"] not in cache:
            cache[scn["d"]] = build_bn(internal_cpts, sensor_cpts, w_mode, K, alpha, scn["d"])
        Pm[i] = cache[scn["d"]].query(["T"], evidence(scn, w_mode)).table
        y[i] = scn["T"]
    return Pm, y


def _corr(a, b):
    return float(np.corrcoef(a, b)[0, 1]) if a.std() > 0 and b.std() > 0 else np.nan


def error_correlations(test):
    sub = [x for x in test if x["cls"] == 2]
    e = {s: np.array([x[s] != 0 for x in sub], float) for s in S3}
    c3 = [_corr(e[S3[i]], e[S3[j]]) for i in range(3) for j in range(i + 1, 3)]
    rho3 = float(np.mean([v for v in c3 if not np.isnan(v)]))
    em = [x for x in sub if x["emit"] == 0]
    rf = np.array([x["Rf"] == 2 for x in em], float)
    crf = [_corr(np.array([x[s] != 0 for x in em], float), rf) for s in S3]
    crf = [v for v in crf if not np.isnan(v)]
    rho_rf = float(np.mean(crf)) if crf else np.nan
    return rho3, rho_rf, float(np.mean(crf + [v for v in c3 if not np.isnan(v)])) if crf else rho3


def block_metrics(Pm, y, w, K):
    adv, worst, ben = w >= 1, w == K - 1, w == 0
    m = {"acc": float((Pm.argmax(1) == y).mean()), "macroAUC": ec.macro_auc(Pm, y), "ece": mt.ece(Pm, y),
         "brier": mt.brier(Pm, y), "cwece": mt.classwise_ece(Pm, y)}
    for tag, q in (("adv", adv), ("worst", worst), ("ben", ben)):
        m[f"acc_{tag}"] = float((Pm.argmax(1) == y)[q].mean())
        m[f"ece_{tag}"] = mt.ece(Pm[q], y[q])
        m[f"auc_{tag}"] = ec.macro_auc(Pm[q], y[q])
        m[f"brier_{tag}"] = mt.brier(Pm[q], y[q])
        m[f"n_{tag}"] = int(q.sum())
    return m


def calib_counts(calib, S, K, n, seed, w_sensors):
    rows = []
    empty, cells = 0, 0
    for s in w_sensors:
        pa = dataio.MARGINAL_PARENTS[s] + ["W"]
        shape = tuple((K if p == "W" else ms.CARD[p]) for p in pa)
        c = np.zeros(shape)
        for row in calib:
            c[tuple(row[p] for p in pa)] += 1
        empty += int((c == 0).sum())
        cells += c.size
    for w in range(K):
        z = [x for x in calib if x["W"] == w]
        rows.append({"sensors": S, "K": K, "n": n, "seed": seed, "W": w, "n_state": len(z),
                     "n_state_drone": sum(x["cls"] == 2 for x in z),
                     "n_state_emitting_drone": sum(x["cls"] == 2 and x["emit"] == 0 for x in z),
                     "empty_parent_cells": empty, "parent_cells": cells})
    return rows


def run_cell(task):
    """One replication: every n at this (sensors, K), so that the calibration splits nest."""
    S, K, seed, ns = task
    w_sensors = S4 if S == 4 else S3
    t0 = time.time()
    world = jam_world(K, w_sensors)
    test = sample_ancestral(world, np.random.default_rng(9000 + seed), ec.TEST_N)
    big = sample_ancestral(world, np.random.default_rng(5000 + seed), max(ns))
    train = sample_ancestral(world, np.random.default_rng(seed), ec.GROUND_N)
    if len(ns) > 1:
        small = sample_ancestral(world, np.random.default_rng(5000 + seed), min(ns))
        assert small == big[:min(ns)], "the calibration splits do not nest"
    cpts, _ = pl.ground_internal("Proposed", train, seed, gt=ms.ground_truth_bn())
    rho3, rho_rf, rho6 = error_correlations(test)
    w = np.array([x["W"] for x in test])
    rows, counts = [], []
    for n in ns:
        calib = big[:n]
        est_n = dataio.estimate_confusion(calib)
        alpha, _ = dataio.fit_reliability(calib)
        est_w = estimate_confusion_W(calib, K, w_sensors)
        counts += calib_counts(calib, S, K, n, seed, w_sensors)
        for fus in FUSIONS:
            Pm, y = posteriors(cpts, est_n if fus == "naive-CI" else est_w, test, MODE[fus], K, alpha)
            rows.append({"sensors": S, "K": K, "n": n, "seed": seed, "fusion": fus, "rho": rho3, "rho_rf": rho_rf,
                         "rho6": rho6, **block_metrics(Pm, y, w, K)})
    return rows, counts, time.time() - t0


# ---------------------------------------------------------------- [P], [C] ---
def param_rows():
    rows = []
    for S, w_sensors in ((3, S3), (4, S4)):
        one = sum(int(np.prod([ms.CARD[p] for p in dataio.MARGINAL_PARENTS[s]])) * (ms.CARD[s] - 1) for s in w_sensors)
        for K in KS:
            calib = sample_ancestral(jam_world(K, w_sensors), np.random.default_rng(0), 5)
            est = estimate_confusion_W(calib, K, w_sensors)
            base = dataio.estimate_confusion(calib)
            free = lambda tabs: sum(int(np.prod(t.table.shape[:-1])) * (t.table.shape[-1] - 1) for t in tabs.values())
            rows.append({"sensors": S, "K": K, "per_state_params": one, "formula_extra": (K - 1) * one,
                         "tables_extra": free(est) - free(base), "layer_total": free(est)})
    return pd.DataFrame(rows)


def elimination_trace(bn, targets, ev):
    """Scopes of the factors built by BayesNet.query's elimination order."""
    scopes = [tuple(x for x in bn.cpts[v].vars if x not in ev) for v in bn.order]
    scopes = [x for x in scopes if x]
    keep = set(targets)
    size = lambda sc: int(np.prod([bn.card[v] for v in sc]))
    best, best_vars, best_w, best_w_vars = 0, (), 0, ()
    for v in [x for x in bn.order if x not in keep and x not in ev]:
        inv = [x for x in scopes if v in x]
        if not inv:
            continue
        rest = [x for x in scopes if v not in x]
        u = tuple(dict.fromkeys(y for x in inv for y in x))
        if size(u) > best:
            best, best_vars = size(u), u
        if "W" in u and size(u) > best_w:
            best_w, best_w_vars = size(u), u
        scopes = rest + [tuple(y for y in u if y != v)]
    return best, best_vars, best_w, best_w_vars


def factor_rows():
    rows = []
    for S, w_sensors, Ks in ((3, S3, (2,)), (4, S4, KS)):
        for K in Ks:
            world = jam_world(K, w_sensors)
            calib = sample_ancestral(world, np.random.default_rng(0), 50)
            cpts = {k: ms.ground_truth_bn().cpts[k] for k in ms.INTERNAL}
            scn = calib[0]
            for fus in FUSIONS:
                est = dataio.estimate_confusion(calib) if fus == "naive-CI" else estimate_confusion_W(calib, K, w_sensors)
                bn = build_bn(cpts, est, MODE[fus], K, None, scn["d"])
                b, bv, bw, bwv = elimination_trace(bn, ["T"], evidence(scn, MODE[fus]))
                rows.append({"sensors": S, "K": K, "fusion": fus, "largest": b, "largest_vars": " ".join(bv),
                             "largest_with_W": bw, "largest_with_W_vars": " ".join(bwv)})
    return pd.DataFrame(rows)


def stage_timing(res):
    conds = [(4, K, f) for K in KS for f in FUSIONS] + [(3, 2, f) for f in FUSIONS] + [("deployed", 0, "deployed")]
    rows, pooled = [], {c: [] for c in conds}
    print(f"machine {platform.processor() or platform.machine()}, Python {platform.python_version()}, NumPy {np.__version__}, "
          f"threads {os.environ.get('OMP_NUM_THREADS')}", flush=True)
    for r in range(TIME_REPS):
        prep = {}
        for c in conds:
            S, K, fus = c
            if S == "deployed":
                if "dep" not in prep:
                    gt = ms.ground_truth_bn()
                    test = dataio.generate_dataset(gt, TIME_TEST, seed=9000 + r)
                    calib = dataio.generate_dataset(gt, 300, seed=5000 + r)
                    est, (alpha, _) = dataio.estimate_confusion(calib), dataio.fit_reliability(calib)
                    cp, _ = pl.ground_internal("Proposed", dataio.generate_dataset(gt, 200, seed=r), r, gt=gt)
                    prep["dep"] = (test, {dv: dataio.build_inference_bn(cp, ms.SENSORS, dv, "calibrated", sensor_cpts=est, alpha=alpha)
                                          for dv in range(ms.CARD["d"])})
                continue
            if (S, K) not in prep:
                ws = S4 if S == 4 else S3
                world = jam_world(K, ws)
                test = sample_ancestral(world, np.random.default_rng(9000 + r), TIME_TEST)
                calib = sample_ancestral(world, np.random.default_rng(5000 + r), 300)
                train = sample_ancestral(world, np.random.default_rng(r), ec.GROUND_N)
                alpha, _ = dataio.fit_reliability(calib)
                cp, _ = pl.ground_internal("Proposed", train, r, gt=ms.ground_truth_bn())
                prep[(S, K)] = (test, cp, dataio.estimate_confusion(calib), estimate_confusion_W(calib, K, ws), alpha)
        order = np.random.default_rng(71000 + r).permutation(len(conds))
        for i in order:
            c = conds[int(i)]
            S, K, fus = c
            q = np.empty(TIME_TEST)
            cpu0 = time.process_time()
            if S == "deployed":
                test, cache = prep["dep"]
                for j, scn in enumerate(test):
                    t = time.perf_counter()
                    bn = cache[scn["d"]]
                    ev = dataio.evidence_from_scenario(scn, ms.SENSORS)
                    bn.query(["T"], ev)
                    q[j] = time.perf_counter() - t
            else:
                test, cp, est_n, est_w, alpha = prep[(S, K)]
                mode = MODE[fus]
                est = est_n if fus == "naive-CI" else est_w
                cache = {dv: build_bn(cp, est, mode, K, alpha, dv) for dv in range(ms.CARD["d"])}
                for j, scn in enumerate(test):
                    t = time.perf_counter()
                    bn = cache[scn["d"]]
                    ev = evidence(scn, mode)
                    bn.query(["T"], ev)
                    q[j] = time.perf_counter() - t
            rows.append({"rep": r, "sensors": S, "K": K, "fusion": fus, "n_queries": TIME_TEST, "q_mean_ms": q.mean() * 1e3,
                         "q_median_ms": float(np.median(q)) * 1e3, "q_p95_ms": float(np.percentile(q, 95)) * 1e3,
                         "q_max_ms": q.max() * 1e3, "cpu_s": time.process_time() - cpu0, "position": int(np.where(order == i)[0][0])})
        print(f"[expJ timing] rep {r + 1}/{TIME_REPS} done", flush=True)
    pd.DataFrame(rows).to_csv(os.path.join(res, f"{P}_latency.csv"), index=False)


# ----------------------------------------------------------------- summary ---
def _holm(ps):
    ps = np.asarray(ps, float)
    order = np.argsort(ps)
    out = np.empty(len(ps))
    run = 0.0
    for rank, i in enumerate(order):
        run = max(run, min(1.0, (len(ps) - rank) * ps[i]))
        out[i] = run
    return out


def paired_table(d):
    from scipy import stats as st
    rows = []
    MET = ("ece", "macroAUC", "brier", "ece_adv", "auc_adv", "brier_adv", "ece_worst", "auc_worst", "brier_worst", "ece_ben", "brier_ben")
    get = lambda K, n, f, m: d[(d.sensors == 4) & (d.K == K) & (d.n == n) & (d.fusion == f)].sort_values("seed")[m].values
    for K in KS:
        for n in NS:
            for f in ("W-observed", "W-latent"):
                for m in MET:
                    x = get(K, n, f, m) - get(K, n, "naive-CI", m)
                    x = x[~np.isnan(x)]
                    rows.append({"family": "model_vs_naive", "K": K, "n": n, "a": f, "b": "naive-CI", "metric": m, "delta": float(x.mean()),
                                 "p": float(st.ttest_1samp(x, 0).pvalue)})
        for f in FUSIONS:
            for m in MET:
                x = get(K, 1200, f, m) - get(K, 300, f, m)
                x = x[~np.isnan(x)]
                rows.append({"family": "n1200_vs_n300", "K": K, "n": 1200, "a": f, "b": f, "metric": m, "delta": float(x.mean()),
                             "p": float(st.ttest_1samp(x, 0).pvalue)})
        for f in ("W-observed", "W-latent"):
            ga, gb = get(K, 1200, "naive-CI", "ece_adv") - get(K, 1200, f, "ece_adv"), get(K, 300, "naive-CI", "ece_adv") - get(K, 300, f, "ece_adv")
            x = ga - gb
            rows.append({"family": "gain_n1200_vs_n300", "K": K, "n": 1200, "a": f, "b": "naive-CI", "metric": "G_ece_adv", "delta": float(x.mean()),
                         "p": float(st.ttest_1samp(x, 0).pvalue)})
    t = pd.DataFrame(rows)
    t["p_holm"] = np.nan
    for fam, idx in t.groupby("family").groups.items():
        t.loc[idx, "p_holm"] = _holm(t.loc[idx, "p"].values)
    return t


def summarize(res):
    d = pd.read_csv(os.path.join(res, f"{P}_raw.csv"), float_precision="round_trip")
    pa = pd.read_csv(os.path.join(res, f"{P}_params.csv"))
    fa = pd.read_csv(os.path.join(res, f"{P}_factors.csv"))
    cc = pd.read_csv(os.path.join(res, f"{P}_calib_counts.csv"))
    t = paired_table(d)
    t.to_csv(os.path.join(res, f"{P}_paired.csv"), index=False)
    L = [f"# Exp-J: the shared nuisance parent over all four modalities", ""]
    chk = os.path.join(res, f"{P}_checks.txt")
    if os.path.exists(chk):
        L += ["## Checks", "```", open(chk, encoding="utf-8").read().strip(), "```", ""]
    L += ["## [P] Parameters", "| sensors | K | per state | extra (formula) | extra (tables) | sensor layer total |", "|---|---|---|---|---|---|"]
    L += [f"| {x.sensors} | {x.K} | {x.per_state_params} | {x.formula_extra} | {x.tables_extra} | {x.layer_total} |" for x in pa.itertuples()]
    L += ["", "## Induced error correlation", "| sensors | K | rho3 | rho_rf | rho6 |", "|---|---|---|---|---|"]
    for (S, K), z in d[d.fusion == "naive-CI"].groupby(["sensors", "K"]):
        z = z[z.n == 300]
        L.append(f"| {S} | {K} | {z.rho.mean():.3f} | {z.rho_rf.mean():.3f} | {z.rho6.mean():.3f} |")
    L += ["", "## Calibration samples (mean over replications)", "| K | n | W | per state | drones | emitting drones | empty parent cells of all |", "|---|---|---|---|---|---|---|"]
    for (K, n, w), z in cc[cc.sensors == 4].groupby(["K", "n", "W"]):
        L.append(f"| {K} | {n} | {w} | {z.n_state.mean():.1f} | {z.n_state_drone.mean():.1f} | {z.n_state_emitting_drone.mean():.1f} | {z.empty_parent_cells.mean():.1f} of {z.parent_cells.iloc[0]} |")
    L += ["", "## [R] Means", "| sensors | K | n | fusion | ECE | AUC | Brier | ECE adv | AUC adv | Brier adv | ECE worst | AUC worst | Brier worst | ECE benign | Brier benign |", "|" + "---|" * 15]
    for (S, K, n, f), z in d.groupby(["sensors", "K", "n", "fusion"], sort=False):
        L.append(f"| {S} | {K} | {n} | {f} | " + " | ".join(f"{z[c].mean():.4f}" for c in ("ece", "macroAUC", "brier", "ece_adv", "auc_adv", "brier_adv", "ece_worst", "auc_worst", "brier_worst", "ece_ben", "brier_ben")) + " |")
    show = ("ece", "macroAUC", "brier", "ece_adv", "auc_adv", "brier_adv", "ece_worst", "brier_worst", "ece_ben", "brier_ben")
    L += ["", "## [R] Model minus naive-CI (delta, raw p)", "| K | n | model | " + " | ".join(show) + " |", "|---|---|---|" + "---|" * len(show)]
    z0 = t[t.family == "model_vs_naive"]
    for (K, n, a), z in z0.groupby(["K", "n", "a"], sort=False):
        z = z.set_index("metric")
        L.append(f"| {K} | {n} | {a} | " + " | ".join(f"{z.loc[m].delta:+.4f} ({z.loc[m].p:.2g})" for m in show) + " |")
    L += ["", "## [R] n = 1200 minus n = 300 (delta, raw p)", "| K | model | " + " | ".join(show) + " |", "|---|---|" + "---|" * len(show)]
    for (K, a), z in t[t.family == "n1200_vs_n300"].groupby(["K", "a"], sort=False):
        z = z.set_index("metric")
        L.append(f"| {K} | {a} | " + " | ".join(f"{z.loc[m].delta:+.4f} ({z.loc[m].p:.2g})" for m in show) + " |")
    L += ["", "## [R] Gain on the degraded subset, G(K, n) = ECE_adv(naive-CI) - ECE_adv(model)", "| K | model | G(K, 300) | G(K, 1200) | difference (raw p) |", "|---|---|---|---|---|"]
    get = lambda K, n, f, m: d[(d.sensors == 4) & (d.K == K) & (d.n == n) & (d.fusion == f)].sort_values("seed")[m].values
    for x in t[t.family == "gain_n1200_vs_n300"].itertuples():
        g0 = (get(x.K, 300, "naive-CI", "ece_adv") - get(x.K, 300, x.a, "ece_adv")).mean()
        g1 = (get(x.K, 1200, "naive-CI", "ece_adv") - get(x.K, 1200, x.a, "ece_adv")).mean()
        L.append(f"| {x.K} | {x.a} | {g0:+.4f} | {g1:+.4f} | {x.delta:+.4f} ({x.p:.2g}) |")
    g4 = t[(t.family == "gain_n1200_vs_n300") & (t.K == 4) & (t.a == "W-observed")].iloc[0]
    rule_r = bool(g4.delta > 0 and g4.p < 0.05)
    L += ["", "## [C] Largest intermediate factor", "| sensors | K | fusion | largest (entries, variables) | largest with W |", "|---|---|---|---|---|"]
    L += [f"| {x.sensors} | {x.K} | {x.fusion} | {x.largest} ({x.largest_vars}) | {x.largest_with_W if x.largest_with_W else '-'} ({x.largest_with_W_vars if x.largest_with_W else ''}) |" for x in fa.itertuples()]
    lat = os.path.join(res, f"{P}_latency.csv")
    rule_c = None
    if os.path.exists(lat):
        la = pd.read_csv(lat)
        L += ["", "## [C] Per-update time, ms (5 replications x 2,500 queries)", "| sensors | K | fusion | mean | median | p95 | max | per-replication means |", "|---|---|---|---|---|---|---|---|"]
        for (S, K, f), z in la.groupby(["sensors", "K", "fusion"], sort=False):
            L.append(f"| {S} | {K} | {f} | {z.q_mean_ms.mean():.3f} | {z.q_median_ms.mean():.3f} | {z.q_p95_ms.mean():.3f} | {z.q_max_ms.max():.3f} | {z.q_mean_ms.min():.3f}-{z.q_mean_ms.max():.3f} |")
        tl = lambda K: la[(la.sensors.astype(str) == "4") & (la.K == K) & (la.fusion == "W-latent")].q_mean_ms.mean()
        f4 = fa[(fa.sensors == 4) & (fa.fusion == "W-latent")].set_index("K")
        prop = bool(f4.loc[4].largest_with_W * 2 == f4.loc[2].largest_with_W * 4)
        under = bool(la.groupby(["sensors", "K", "fusion"]).q_mean_ms.mean().max() < 1.0)
        rule_c = (prop, tl(4) / tl(2), under)
        L += ["", f"[C] largest factor with W proportional to K: {prop}; t(4) / t(2) for W-latent: {tl(4) / tl(2):.2f} (rule <= 2: {tl(4) / tl(2) <= 2}); every mean update under 1 ms: {under}"]
    L += ["", f"[R] G(4, 1200) > G(4, 300) for W-observed with p < 0.05: {rule_r} (difference {g4.delta:+.4f}, p {g4.p:.2g}, Holm {g4.p_holm:.2g})"]
    if rule_c is not None:
        L += [f"Reading of the data-bottleneck statement: {'supported' if rule_r and rule_c[0] and rule_c[1] <= 2 and rule_c[2] else 'not supported by the fixed rule; write the measured values'}"]
    txt = "\n".join(L)
    open(os.path.join(res, f"{P}_summary.md"), "w", encoding="utf-8").write(txt + "\n")
    print(txt)


# ---------------------------------------------------------------- selftest ---
def selftest():
    a, b = jam_world(2, S3), ec.world_with_W(1.0)
    assert list(a.card) == list(b.card) and a.parents == b.parents
    n = 0
    for v in b.cpts:
        assert a.cpts[v].vars == b.cpts[v].vars and a.cpts[v].table.tobytes() == b.cpts[v].table.tobytes(), v
        n += 1
    calib = sample_ancestral(b, np.random.default_rng(5000), 300)
    assert calib == sample_ancestral(a, np.random.default_rng(5000), 300)
    e1, e2 = estimate_confusion_W(calib, 2, S3), ec.estimate_confusion_W(calib)
    cp = {k: ms.ground_truth_bn().cpts[k] for k in ms.INTERNAL}
    alpha, _ = dataio.fit_reliability(calib)
    for s in ms.SENSORS:
        assert e1[s].vars == e2[s].vars and e1[s].table.tobytes() == e2[s].table.tobytes(), s
        n += 1
    for mode in ("none", "observed", "latent"):
        x, y = build_bn(cp, e1, mode, 2, alpha, 1), ec.build_bn_W(cp, e2, mode, alpha=alpha, d_value=1)
        assert list(x.card) == list(y.card) and x.order == y.order and x.parents == y.parents
        for v in y.cpts:
            assert x.cpts[v].vars == y.cpts[v].vars and x.cpts[v].table.tobytes() == y.cpts[v].table.tobytes(), (mode, v)
            n += 1
    for K in KS:
        w = jam_world(K, S4)
        assert abs(w.cpts["W"].table.sum() - 1) < 1e-12 and w.cpts["W"].table[0] == 1 - ec.P_ADVERSE
        assert len(severities(K)) == K - 1 and severities(K)[-1] == 1.0
        for s in S4:
            t = w.cpts[s].table
            assert w.cpts[s].vars[-2:] == ("W", s) and np.allclose(t.sum(-1), 1.0) and (t >= 0).all(), (K, s)
            assert t[(Ellipsis, 0, slice(None))].tobytes() == ms.ground_truth_bn().cpts[s].table.tobytes(), (K, s)
            n += 1
    pr = param_rows()
    assert (pr.formula_extra == pr.tables_extra).all()
    assert int(pr[(pr.sensors == 3) & (pr.K == 2)].formula_extra.iloc[0]) == 15 == ec.extra_params()
    print(pr.to_string(index=False))
    print(factor_rows().to_string(index=False))
    print(f"self-test: {n + len(pr)} checks passed")
    return n + len(pr)


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--stage", default="", help="main or timing")
    ap.add_argument("--workers", type=int, default=20)
    ap.add_argument("--reps", type=int, default=REPS)
    ap.add_argument("--res", default=RES)
    ap.add_argument("--summary-only", action="store_true")
    a = ap.parse_args()
    os.makedirs(a.res, exist_ok=True)
    if a.selftest:
        assert selftest() > 0
        return
    if a.summary_only:
        summarize(a.res)
        return
    if a.stage == "timing":
        stage_timing(a.res)
        return
    assert a.stage == "main"
    from multiprocessing import Pool
    n_self = selftest()
    t0 = time.time()
    tasks = [(3, 2, r, (300,)) for r in range(a.reps)] + [(4, K, r, NS) for K in KS for r in range(a.reps)]
    rows, counts = [], []
    with Pool(a.workers) as pool:
        for i, (ro, co, secs) in enumerate(pool.imap(run_cell, tasks), 1):
            rows += ro
            counts += co
            if i % 10 == 0 or i == len(tasks):
                print(f"[expJ] {i}/{len(tasks)} cells ({secs:.0f}s; {time.time() - t0:.0f}s total)", flush=True)
    d = pd.DataFrame(rows)
    d.to_csv(os.path.join(a.res, f"{P}_raw.csv"), index=False)
    pd.DataFrame(counts).to_csv(os.path.join(a.res, f"{P}_calib_counts.csv"), index=False)
    param_rows().to_csv(os.path.join(a.res, f"{P}_params.csv"), index=False)
    factor_rows().to_csv(os.path.join(a.res, f"{P}_factors.csv"), index=False)
    # path check against exp_correlated.py part (i)
    f = pd.read_csv(os.path.join(RES, "expF_corr_raw.csv"), float_precision="round_trip")
    f = f[f.severity == 1.0].set_index(["seed", "fusion"])
    z = d[d.sensors == 3].set_index(["seed", "fusion"])
    dev = [abs(float(z.loc[k, c]) - float(f.loc[k, c])) for k in z.index for c in PATH_COLS]
    assert len(dev) == len(z) * len(PATH_COLS) and len(dev) > 0
    again = pd.DataFrame(run_cell((4, 4, 0, NS))[0])
    same = again.equals(d[(d.sensors == 4) & (d.K == 4) & (d.seed == 0)].reset_index(drop=True))
    txt = (f"self-test checks passed: {n_self}\n"
           f"path check against expF_corr_raw.csv (three sensors, K = 2, s = 1): max |difference| {max(dev):.2e} over {len(dev)} values\n"
           f"K = 4, seed 0 rerun in the parent process equals the pool's rows: {same}\n"
           f"cells: {len(tasks)}; threads per worker: OMP {os.environ.get('OMP_NUM_THREADS')}; workers {a.workers}; elapsed {time.time() - t0:.0f}s")
    open(os.path.join(a.res, f"{P}_checks.txt"), "w", encoding="utf-8").write(txt + "\n")
    print(txt, flush=True)
    assert max(dev) <= 1e-9 and same, "path or determinism check failed"
    print("main stage done; run --stage timing alone, then --summary-only", flush=True)


if __name__ == "__main__":
    main()
