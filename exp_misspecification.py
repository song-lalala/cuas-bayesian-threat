"""
Robustness to structural misspecification, non-monotone dependence and
distribution shift. Core files are not
modified; results go to results/expW_*. Worlds: world_misspec.py (S1, S3, S4)
and world_nonmonotone.py (S2, 'H-v-U'); every builder returns the standard
world unchanged by default.

Scenarios (20 replications each; seeds as in the main study: training data
seed r, calibration 5000+r with n = 300, test 9000+r with 2,500 scenarios,
expert prior rng(1000+r), ESS 2; sensor layer = deployed V0):
  S0  standard world (reference; also checks bit-reproduction of
      expB_baselines_raw.csv)
  S1  non-additive threat response, lam = 0.5 / 1.0 (ICI misspecified,
      monotonicity still true)
  S2  H-v-U (H non-monotone in v; the constraints are wrong for H)
  S3  edges d -> N and emit -> H absent from the model, w = 0.5 / 1.5
  S4  training and calibration in the standard world, test set shifted:
      more drones, more radio-silent, sensor degradation x1.5, C weight x2,
      tau_T 0.60. The deployed model does not know about the shift (its
      root marginals stay the standard ones).
  In S1-S3 training, calibration and test data all come from the scenario
  world. The expert prior is always generated from the standard world's
  weights: in a misspecified world the expert believes the wrong structure.

Methods: Proposed (cMAP H, N, C + ordinal ICI T), cMAP-full (full-table T),
MAP (no constraints), B1-MLE, B5-logistic (exp_baselines.logreg_posteriors:
L2 multinomial logistic regression on one-hot observed evidence, no
structural assumption), and in S2 also Proposed-Hfree (H unconstrained, the
rest as Proposed). Oracles: Bayes oracle = the scenario world's own posterior
(its structure, its true conditional sensor tables); CPT oracle = the
model's structure with the best tables it can hold (the world's true CPTs, or
in S3 the population conditionals P(X | model parents)) and the world's true
range-marginal sensor tables; family-best = S1 the ICI projection of the
true T, S2 the monotone (KL) projection of the true H. Bayes - CPT oracle is
the structural loss, CPT oracle - method the estimation loss.
Metrics: accuracy, macro-AUC, Brier, ECE (15 bins), and for the high-threat
class ece_high (15-bin calibration error of P(T = high)) and bias_high
(mean P(T = high) - frequency of T = high).

RULES FIXED BEFORE THE RUN (in this docstring; not registered externally)

[J] Judgment per scenario and N (primary metric: Brier on that scenario's
    test set).
    best alternative = the non-oracle method other than Proposed with the
      lowest mean Brier among {cMAP-full, MAP, B1-MLE, B5-logistic} (plus
      Proposed-Hfree in S2).
    "breaks"   if the best alternative beats Proposed in Brier: mean of the
               per-replication differences (Proposed - alternative) > 0 and
               two-sided paired t-test p < 0.05 over the 20 replications;
    otherwise "holds"    if Proposed's excess Brier over the scenario's CPT
                         oracle is at most twice its excess in the standard
                         world at the same N,
              "degrades" if that excess is more than twice.
    (excess = mean over replications of Brier(Proposed) - Brier(CPT oracle)
    on the same test set.) ECE, macro-AUC, accuracy and the high-threat
    lines are reported for every cell but do not enter the verdict.

[D] Monotonicity diagnostic (S2). For each node X in {H, N, C, T (full
    table)}: LR_X = 2 [l(MAP) - l(cMAP)], where l(theta) = sum_jk N_jk log
    theta_jk on the N training scenarios and MAP / cMAP are the deployed
    estimators (same prior, ESS 2). Null distribution by parametric
    bootstrap: B = 199 count tables drawn from the cMAP fit with the
    observed column totals; both estimators refitted on each;
    p_X = (1 + #{LR* >= LR}) / (B + 1). Flag X if p_X < 0.05 / 4 = 0.0125
    (Bonferroni over the four nodes); flag the network if any node is
    flagged. Reported at N = 50, 200, 1000: false-alarm rate in the standard
    world (any node, and per node) and detection rate in H-v-U (node H, and
    any node).

Run:  python exp_misspecification.py                (all, 20 replications)
      python exp_misspecification.py --summary-only
"""
from __future__ import annotations
import os, sys, time, argparse
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

import model_spec as ms
import dataio
import grounding as g
import metrics as mt
import pipeline as pl
import world_options as wo
import world_misspec as wm
import world_nonmonotone as wn
import exp_baselines as eb
import exp_estimator_consistency as ec
import exp_significance as es
from cuas_bn import Factor, BayesNet

HERE = os.path.dirname(os.path.abspath(__file__))
RES_IN = os.path.join(HERE, "results")    # existing results (read only)
RES = RES_IN                              # output folder (--res for test runs)
P = "expW"
REPS = 20
NS = (50, 200)
DIAG_NS = (50, 200, 1000)
B_BOOT = 199
DIAG_ALPHA = 0.05 / 4
TEST_N, CALIB_N = 2500, 300
ESS = pl.ESS_FIXED
LABELS = [0, 1, 2, 3]
SCEN = {"S0": ["standard"], "S1": list(wm.WORLDS_S1), "S2": ["H-v-U"],
        "S3": list(wm.WORLDS_S3), "S4": list(wm.WORLDS_S4)}
BN_METHODS = ["Proposed", "cMAP-full", "MAP", "B1-MLE"]
ALTS = ["cMAP-full", "MAP", "B1-MLE", "B5-logistic"]
ROOTS = ["cls", "emit", "v", "rdot", "d", "z"]


def read_csv(path):
    return pd.read_csv(path, float_precision="round_trip")


def world(name):
    return wn.world_bn("H-v-U") if name == "H-v-U" else wm.world_bn(name)


# ------------------------------------------------------------ metrics ------
def metrics_all(P, y):
    pred = P.argmax(1)
    try:
        from sklearn.metrics import roc_auc_score
        auc = float(roc_auc_score(np.eye(4)[y], P, multi_class="ovr",
                                  average="macro", labels=LABELS))
    except Exception:
        auc = np.nan
    ph = P[:, 3]
    hit = (y == 3).astype(float)
    bins = np.linspace(0, 1, mt.N_BINS + 1)
    eh = 0.0
    for b in range(mt.N_BINS):
        m = (ph > bins[b]) & (ph <= bins[b + 1])
        if m.sum():
            eh += m.sum() / len(y) * abs(hit[m].mean() - ph[m].mean())
    return {"acc": float((pred == y).mean()),
            "macroF1": float(f1_score(y, pred, labels=LABELS, average="macro",
                                      zero_division=0)),
            "macroAUC": auc, "brier": mt.brier(P, y), "ece": mt.ece(P, y),
            "cwece": mt.classwise_ece(P, y),
            "far@pd0.9": mt.far_at_pd(P, y, 0.90),
            "ece_high": float(eh), "bias_high": float(ph.mean() - hit.mean())}


# ------------------------------------------------------------ oracles ------
def model_bn(wbn, internal, sensor_tabs):
    """The model's structure with the world's roots, given internal tables
    and sensor tables (no discount)."""
    cpts = {k: wbn.cpts[k] for k in ROOTS}
    cpts.update(internal)
    cpts.update(sensor_tabs)
    parents = {k: list(cpts[k].vars[:-1]) for k in cpts}
    return BayesNet(ms.CARD, parents, cpts)


def bn_posteriors(bn, test):
    obs = ms.OBSERVED + ms.SENSORS
    P = np.zeros((len(test), ms.CARD["T"]))
    for i, s in enumerate(test):
        P[i] = bn.query(["T"], {k: s[k] for k in obs}).table
    return P


def population_tables(wbn):
    """P_world(X | model parents) for X in H, N, C, T: the best tables the
    model's structure can hold (its infinite-data limit without constraints)."""
    out = {}
    for x in ms.INTERNAL:
        pa = ms.PARENTS[x]
        if list(wbn.parents[x]) == list(pa):
            out[x] = wbn.cpts[x]
            continue
        j = wbn.query(list(pa) + [x]).table
        out[x] = Factor(tuple(pa) + (x,), j / j.sum(axis=-1, keepdims=True))
    return out


def oracle_tables(wname, wbn):
    tabs = {"CPT-oracle": population_tables(wbn)}
    if wname.startswith("nonadd-"):
        w = ec.parent_probs(wbn, "T")
        ici = ec.ici_from_counts(1e4 * w[..., None] * wbn.cpts["T"].table,
                                 restarts=8, seed=0)
        tabs["family-best"] = {**tabs["CPT-oracle"],
                               "T": Factor(("H", "N", "C", "T"), ici)}
    if wname == "H-v-U":
        w = ec.parent_probs(wbn, "H")
        tabs["family-best"] = {**tabs["CPT-oracle"],
                               "H": ec.kl_projection("H", wbn.cpts["H"].table, w)}
    return tabs


# ------------------------------------------------------ methods per rep ----
def fit_methods(data, r, with_hfree=False):
    fits = {m: pl.ground_internal(m, data, r)[0] for m in BN_METHODS}
    if with_hfree:
        prior = ms.expert_prior_cpts(np.random.default_rng(1000 + r))
        hf = dict(fits["Proposed"])
        hf["H"] = g.map_cpt("H", data, prior["H"], ESS)
        fits["Proposed-Hfree"] = hf
    return fits


def eval_methods(fits, data, test, est, alpha):
    """Threat posteriors of every fitted BN (deployed sensor layer, the
    metrics.evaluate path) and of the logistic baseline."""
    out = {}
    for m, cp in fits.items():
        out[m] = mt.posteriors(cp, test, ms.SENSORS, "calibrated",
                               sensor_cpts=est, alpha=alpha)[0]
    Pl, _ = eb.logreg_posteriors(data, test)
    if Pl is not None:
        out["B5-logistic"] = Pl
    return out


def run_main(reps=REPS):
    rows = []
    t0 = time.time()
    worlds = {w: world(w) for ws in SCEN.values() for w in ws}
    otabs = {w: oracle_tables(w, b) for w, b in worlds.items()}
    for r in range(reps):
        std = worlds["standard"]
        std_fits = {}
        for scen, wlist in SCEN.items():
            for wname in wlist:
                wbn = worlds[wname]
                test = dataio.generate_dataset(wbn, TEST_N, seed=9000 + r)
                y = np.array([s["T"] for s in test])
                if scen == "S4":
                    src_calib = std_calib
                else:
                    src_calib = dataio.generate_dataset(wbn, CALIB_N, seed=5000 + r)
                    est = dataio.estimate_confusion(src_calib)
                    alpha, _ = dataio.fit_reliability(src_calib)
                    if scen == "S0":
                        std_calib, std_est, std_alpha = src_calib, est, alpha
                if scen == "S4":
                    est, alpha = std_est, std_alpha
                big = None if scen == "S4" else \
                    dataio.generate_dataset(wbn, max(NS), seed=r)
                for N in NS:
                    if scen == "S4":
                        fits, data = std_fits[N]
                    else:
                        data = big[:N]
                        fits = fit_methods(data, r, with_hfree=(scen == "S2"))
                        if scen == "S0":
                            std_fits[N] = (fits, data)
                    Ps = eval_methods(fits, data, test, est, alpha)
                    for m, Pm in Ps.items():
                        rows.append({"scenario": scen, "world": wname, "rep": r,
                                     "N": N, "method": m, **metrics_all(Pm, y)})
                tabs_m = wo.true_marginal_tables(wbn)
                Po = {"Bayes-oracle": bn_posteriors(wbn, test)}
                for oname, tabs in otabs[wname].items():
                    Po[oname] = bn_posteriors(model_bn(wbn, tabs, tabs_m), test)
                for m, Pm in Po.items():
                    rows.append({"scenario": scen, "world": wname, "rep": r,
                                 "N": -1, "method": m, **metrics_all(Pm, y)})
        pd.DataFrame(rows).to_csv(os.path.join(RES, f"{P}_raw.csv"), index=False)
        print(f"[expW main] rep {r+1}/{reps} ({time.time()-t0:.0f}s)", flush=True)


# ----------------------------------------------------------- diagnostic ----
class CmapSolver:
    """grounding.cmap_cpt's convex program with the pseudo-counts as a
    cvxpy Parameter (compiled once, re-solved per bootstrap replicate)."""

    def __init__(self, node):
        import cvxpy as cp
        pa = ms.PARENTS[node]
        K = ms.CARD[node]
        J = int(np.prod([ms.CARD[p] for p in pa]))
        self.shape = tuple(ms.CARD[p] for p in pa) + (K,)
        self.c = cp.Parameter((J, K), nonneg=True)
        self.th = cp.Variable((J, K), nonneg=True)
        cons = [cp.sum(self.th, axis=1) == 1]
        pairs, _ = g._covering_pairs(node)
        pj = [p[0] for p in pairs]
        pjp = [p[1] for p in pairs]
        for m in range(1, K):
            cons.append(cp.sum(self.th[pjp, m:], axis=1)
                        >= cp.sum(self.th[pj, m:], axis=1))
        self.prob = cp.Problem(cp.Maximize(cp.sum(cp.multiply(
            self.c, cp.log(self.th)))), cons)
        self.J, self.K = J, K
        self.stats = {"CLARABEL": 0, "ECOS": 0, "SCS": 0, "fallback": 0}

    def solve(self, c):
        """Same solver cascade as grounding.cmap_cpt: CLARABEL, then ECOS,
        then SCS; if none is optimal, the closed-form MAP (tallied)."""
        self.c.value = c.reshape(self.J, self.K)
        X = None
        for solver in ("CLARABEL", "ECOS", "SCS"):
            try:
                self.prob.solve(solver=solver)
            except Exception:
                continue
            if self.prob.status in ("optimal", "optimal_inaccurate") \
                    and self.th.value is not None:
                self.stats[solver] += 1
                X = np.clip(self.th.value, 1e-9, None)
                break
        if X is None:
            self.stats["fallback"] += 1
            X = self.c.value.copy()
        X = X / X.sum(axis=1, keepdims=True)
        return X.reshape(self.shape)


def _loglik(n, theta):
    return float(np.sum(n * np.log(theta)))


def lr_test(node, n, prior_tab, solver, rng):
    """Parametric-bootstrap likelihood-ratio test of monotonicity for one
    node (rule [D])."""
    a = ESS * prior_tab
    cm = solver.solve(n + a)
    mp = (n + a) / (n + a).sum(axis=-1, keepdims=True)
    lr = 2 * (_loglik(n, mp) - _loglik(n, cm))
    K = n.shape[-1]
    tot = n.reshape(-1, K).sum(1)
    cmf = cm.reshape(-1, K)
    ge = 0
    for _ in range(B_BOOT):
        nb = np.vstack([rng.multinomial(int(t), cmf[j]) if t > 0
                        else np.zeros(K) for j, t in enumerate(tot)]).reshape(n.shape)
        cmb = solver.solve(nb + a)
        mpb = (nb + a) / (nb + a).sum(axis=-1, keepdims=True)
        lrb = 2 * (_loglik(nb, mpb) - _loglik(nb, cmb))
        ge += lrb >= lr - 1e-12
    return lr, (1 + ge) / (B_BOOT + 1)


def run_diagnostic(reps=REPS):
    rows = []
    t0 = time.time()
    solvers = {x: CmapSolver(x) for x in ms.INTERNAL}
    for r in range(reps):
        prior = ms.expert_prior_cpts(np.random.default_rng(1000 + r))
        for wi, wname in enumerate(("standard", "H-v-U")):
            big = dataio.generate_dataset(world(wname), max(DIAG_NS), seed=r)
            for ni, N in enumerate(DIAG_NS):
                for xi, x in enumerate(ms.INTERNAL):
                    rng = np.random.default_rng(100000 + 1000 * r + 100 * wi
                                                + 10 * ni + xi)
                    n = g.counts(x, big[:N])
                    before = dict(solvers[x].stats)
                    lr, p = lr_test(x, n, prior[x].table, solvers[x], rng)
                    used = {k: solvers[x].stats[k] - before[k] for k in before}
                    rows.append({"world": wname, "rep": r, "N": N, "node": x,
                                 "lr": lr, "p": p, "flag": p < DIAG_ALPHA,
                                 "n_ecos": used["ECOS"], "n_scs": used["SCS"],
                                 "n_fallback": used["fallback"]})
        pd.DataFrame(rows).to_csv(os.path.join(RES, f"{P}_diag_raw.csv"),
                                  index=False)
        print(f"[expW diag] rep {r+1}/{reps} ({time.time()-t0:.0f}s)", flush=True)


# ---------------------------------------------------- bit reproduction ----
def reproduction_check():
    w = read_csv(os.path.join(RES, f"{P}_raw.csv"))
    b = read_csv(os.path.join(RES_IN, "expB_baselines_raw.csv"))
    names = {"Proposed": "Proposed", "B1-MLE": "B1-MLE",
             "MAP": "expert MAP (full table)",
             "cMAP-full": "expert cMAP (full table)",
             "B5-logistic": "logistic regression"}
    reps = sorted(w.rep.unique())
    n_cmp, bad = 0, []
    for mine, theirs in names.items():
        for N in NS:
            a = w[(w.scenario == "S0") & (w.method == mine) & (w.N == N)] \
                .sort_values("rep")
            c = b[(b.method == theirs) & (b.N == N) & (b.rep.isin(reps))] \
                .sort_values("rep")
            for k in ("acc", "macroF1", "macroAUC", "brier", "ece", "cwece",
                      "far@pd0.9"):
                n_cmp += len(a)
                if not np.array_equal(a[k].values, c[k].values, equal_nan=True):
                    bad.append(f"{mine} N={N} {k}")
    return n_cmp, bad


# ------------------------------------------------------------ summary ------
def _pm(x):
    m, lo, hi = es.t_ci(np.asarray(x, float))
    return m, (hi - lo) / 2


def judge(w):
    """Rule [J] for every scenario world and N."""
    out = []
    for (scen, wname), g_ in w[w.N > 0].groupby(["scenario", "world"]):
        for N in NS:
            d = g_[g_.N == N]
            alts = [m for m in ALTS + (["Proposed-Hfree"] if scen == "S2" else [])
                    if m in set(d.method)]
            best = min(alts, key=lambda m: d[d.method == m].brier.mean())
            pr = d[d.method == "Proposed"].set_index("rep").brier
            al = d[d.method == best].set_index("rep").brier
            t = es.across_rep_tests((pr - al).values)
            orc = w[(w.world == wname) & (w.method == "CPT-oracle")].set_index("rep").brier
            std_p = w[(w.world == "standard") & (w.N == N) & (w.method == "Proposed")].set_index("rep").brier
            std_o = w[(w.world == "standard") & (w.method == "CPT-oracle")].set_index("rep").brier
            exc = float((pr - orc).mean())
            exc0 = float((std_p - std_o).mean())
            breaks = t["mean"] > 0 and t["t_p"] < 0.05
            verdict = "breaks" if breaks else ("holds" if exc <= 2 * exc0 else "degrades")
            out.append({"scenario": scen, "world": wname, "N": N,
                        "best_alt": best, "d_brier": t["mean"], "p": t["t_p"],
                        "excess": exc, "excess_std": exc0,
                        "ratio": exc / exc0 if exc0 > 0 else np.nan,
                        "verdict": verdict})
    return pd.DataFrame(out)


def summarize():
    w = read_csv(os.path.join(RES, f"{P}_raw.csv"))
    L = [f"# Exp-W: misspecification, non-monotone dependence and shift "
         f"({w.rep.nunique()} replications)", ""]
    n_cmp, bad = reproduction_check()
    L.append(f"- bit-reproduction of expB_baselines_raw.csv (standard world, "
             f"N = 50, 200, five methods): {n_cmp} values, {len(bad)} mismatching")
    for s in bad:
        L.append(f"  - MISMATCH {s}")
    L.append("\n## worlds: KL(true T || ICI family) and nodes changed")
    for wname in SCEN["S1"] + ["standard"]:
        wbn = world(wname)
        wt = ec.parent_probs(wbn, "T")
        ici = ec.ici_from_counts(1e4 * wt[..., None] * wbn.cpts["T"].table,
                                 restarts=8, seed=0)
        L.append(f"- {wname}: KL(T || ICI) {ec.wkl(wbn.cpts['T'].table, ici, wt):.4e}")
    jd = judge(w)
    jd.to_csv(os.path.join(RES, f"{P}_judgment.csv"), index=False)
    L.append("\n## judgment (rule [J], primary metric Brier)")
    L.append("| scenario | world | N | best alternative | d Brier (Prop - alt) | p | "
             "excess over CPT oracle | std excess | ratio | verdict |")
    L.append("|---|---|---|---|---|---|---|---|---|---|")
    for _, x in jd.iterrows():
        L.append(f"| {x.scenario} | {x.world} | {x.N} | {x.best_alt} | "
                 f"{x.d_brier:+.4f} | {x.p:.1e} | {x.excess:.4f} | "
                 f"{x.excess_std:.4f} | {x.ratio:.2f} | {x.verdict} |")
    L.append("\n## loss decomposition (Brier): structural = CPT oracle - Bayes "
             "oracle, estimation = Proposed - CPT oracle")
    L.append("| scenario | world | Bayes oracle | CPT oracle | family-best | "
             "structural | estimation N=50 | estimation N=200 |")
    L.append("|---|---|---|---|---|---|---|---|")
    for (scen, wname), g_ in w.groupby(["scenario", "world"], sort=False):
        bo = g_[g_.method == "Bayes-oracle"].set_index("rep").brier
        co = g_[g_.method == "CPT-oracle"].set_index("rep").brier
        fb = g_[g_.method == "family-best"].brier
        est_ = [float((g_[(g_.method == "Proposed") & (g_.N == N)]
                       .set_index("rep").brier - co).mean()) for N in NS]
        fbs = f"{fb.mean():.4f}" if len(fb) else "-"
        L.append(f"| {scen} | {wname} | {bo.mean():.4f} | {co.mean():.4f} | "
                 f"{fbs} | {(co - bo).mean():+.4f} | {est_[0]:+.4f} | "
                 f"{est_[1]:+.4f} |")
    L.append("\n## detail: mean (95% CI half-width) per scenario world, method, N")
    for (scen, wname), g_ in w.groupby(["scenario", "world"], sort=False):
        L.append(f"\n### {scen} {wname}")
        L.append("| method | N | Brier | ECE | AUC | acc | ece_high | bias_high |")
        L.append("|---|---|---|---|---|---|---|---|")
        for (m, N), d in g_.groupby(["method", "N"], sort=False):
            cells = []
            for k in ("brier", "ece", "macroAUC", "acc"):
                mm, h = _pm(d[k])
                cells.append(f"{mm:.4f} ({h:.4f})")
            L.append(f"| {m} | {'oracle' if N < 0 else N} | " + " | ".join(cells)
                     + f" | {d.ece_high.mean():.4f} | {d.bias_high.mean():+.4f} |")
        for N in NS:
            pr = g_[(g_.method == "Proposed") & (g_.N == N)].set_index("rep")
            for m in [x for x in g_.method.unique() if x not in ("Proposed",) and
                      not x.endswith("oracle") and x != "family-best"]:
                al = g_[(g_.method == m) & (g_.N == N)].set_index("rep")
                if al.empty:
                    continue
                tb = es.across_rep_tests((pr.brier - al.brier).values)
                te = es.across_rep_tests((pr.ece - al.ece).values)
                L.append(f"- N={N} Proposed - {m}: dBrier {tb['mean']:+.4f} "
                         f"(p {tb['t_p']:.1e}), dECE {te['mean']:+.4f} (p {te['t_p']:.1e})")
        if scen != "S0":
            for N in NS:
                for m in ("Proposed", "B5-logistic"):
                    a = g_[(g_.method == m) & (g_.N == N)].set_index("rep").brier
                    s0 = w[(w.world == "standard") & (w.method == m) & (w.N == N)].set_index("rep").brier
                    t = es.across_rep_tests((a - s0).values)
                    L.append(f"- N={N} {m}: Brier change vs standard world "
                             f"{t['mean']:+.4f} (p {t['t_p']:.1e})")
    dg = os.path.join(RES, f"{P}_diag_raw.csv")
    if os.path.exists(dg):
        dd = read_csv(dg)
        L.append("\n## monotonicity diagnostic (rule [D]; flag if p < 0.0125)")
        L.append(f"- tests: {len(dd)}; cMAP solves per test {B_BOOT + 1}; solved "
                 f"by ECOS {int(dd.n_ecos.sum())}, by SCS {int(dd.n_scs.sum())}, "
                 f"closed-form MAP fallback {int(dd.n_fallback.sum())} "
                 f"(all others by CLARABEL)")
        L.append("| N | std any-node false alarm | std per node H/N/C/T | "
                 "H-v-U detection (H) | H-v-U any node | H-v-U per node H/N/C/T |")
        L.append("|---|---|---|---|---|---|")
        for N in DIAG_NS:
            s = dd[(dd.world == "standard") & (dd.N == N)]
            u = dd[(dd.world == "H-v-U") & (dd.N == N)]
            fa = s.groupby("rep").flag.any().mean()
            de = u.groupby("rep").flag.any().mean()
            per = lambda q: "/".join(f"{q[q.node == x].flag.mean():.2f}" for x in ms.INTERNAL)
            L.append(f"| {N} | {fa:.2f} | {per(s)} | "
                     f"{u[u.node == 'H'].flag.mean():.2f} | {de:.2f} | {per(u)} |")
    txt = "\n".join(L)
    with open(os.path.join(RES, f"{P}_summary.md"), "w", encoding="utf-8") as f:
        f.write(txt + "\n")
    print(txt)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=REPS)
    ap.add_argument("--summary-only", action="store_true")
    ap.add_argument("--only", default="", help="main,diag")
    ap.add_argument("--res", default="", help="output folder (test runs)")
    a = ap.parse_args()
    if a.res:
        RES = a.res
    os.makedirs(RES, exist_ok=True)
    if a.summary_only:
        summarize()
    else:
        parts = a.only.split(",") if a.only else ["main", "diag"]
        if "main" in parts:
            run_main(a.reps)
        if "diag" in parts:
            run_diagnostic(a.reps)
        summarize()
