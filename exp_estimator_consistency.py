"""
Statistical interpretation and consistency of the cMAP and MAP-EM estimators.
Core files are not modified; results go
to results/expC_*.

Facts checked numerically (the algebra is in the manuscript's appendix):
 cMAP maximizes sum_jk (N_jk + a_jk) log q_jk over the monotone set M,
 a_jk = ess * phat_jk. This is (i) the constrained MAP under the prior
 Dir(a + 1), (ii) the Bayes decision minimizing the column-weighted posterior
 expected KL loss sum_j (N_j + a_j) E_post KL(theta_j || q_j) under the
 conjugate prior Dir(a), and (iii) the column-weighted KL (M-)projection of
 the Dir(a) posterior mean onto M. Strictly concave -> unique; consistent for
 a monotone truth, and converging to the KL projection of the truth onto M
 (pseudo-true value) otherwise; within O(1/N) of the constrained MLE.

Sections
 2.1 complete data, standard world (truth monotone): N in NS, 20 reps; nodes
     H, N, C, T (full table); estimators cMAP (grounding.cmap_cpt), MAP
     (posterior mean, grounding.map_cpt) and cMLE (the same convex program
     with a vanishing pseudo-count 1e-9/K per cell, so empty cells and empty
     columns are defined: an empty column comes out uniform where the
     constraints allow); metrics: column-weighted KL to the truth (weights =
     true parent-configuration probabilities) and max abs error; plus
     max|cMAP - cMLE|.
 2.2 non-monotone world (world_nonmonotone 'H-v-U'), node H: distances to the
     truth and to the pseudo-true value, the KL projection of the truth onto
     M computed with the same convex program (grounding._cmap_from_counts on
     pseudo-counts proportional to P(j) theta_jk).
 2.3 MAP-EM with latent H, N, C (and cls, emit), protocol of exp_mapem.py
     (ess 2, expert prior, calibration-split sensor estimates in the E-step,
     test 2500). The E-step is vectorized but computes the same posterior
     expected counts as grounding.em_internal and the same M-steps
     (`--check-em` compares the two). N in {200, 1000, 5000}, 50 iterations.
     Records the log posterior log p(e | theta) + sum a log theta per
     iteration (constants dropped); stopping at 12 vs 50; constraint at the
     final M-step only (implemented) vs at every M-step; CPT error by N;
     label check (identity vs best permutation of each latent node's states).
 2.4 ICI threat node, standard world: column-weighted KL of the ordinal ICI
     fit (grounding.ordinal_ici_cpt) vs the full-table cMAP as N grows, and
     the infinite-data ICI fit (pseudo-true value within the family).

Run:  python exp_estimator_consistency.py            (all sections)
      python exp_estimator_consistency.py --check-em
      python exp_estimator_consistency.py --summary-only
"""
from __future__ import annotations
import os, sys, time, argparse
from itertools import permutations
import numpy as np
import pandas as pd

import model_spec as ms
import dataio
import grounding as g
import metrics as mt
import pipeline as pl
import world_nonmonotone as wn
import exp_significance as es        # t_ci / across_rep_tests helpers only
from cuas_bn import Factor

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
P = "expC"
NS = (50, 200, 1000, 5000, 20000)
EM_NS = (200, 1000, 5000)
EM_MAX = 50
EM_CHECK = (12, 50)
REPS = 20
ESS = pl.ESS_FIXED
NODES = ("H", "N", "C", "T")
KEEP = ["v", "rdot", "d", "z", "Rr", "Re", "Ra", "Rf", "T"]
MLE_EPS = 1e-9


def read_csv(path):
    return pd.read_csv(path, float_precision="round_trip")


# ------------------------------------------------------------ helpers ------
def parent_probs(bn, node):
    """True parent-configuration probabilities P(j), shaped like the
    parent axes of node's CPT."""
    pa = ms.PARENTS[node]
    if node == "T":
        return bn.query(["H", "N", "C"]).table
    w = np.ones(())
    for p in pa:
        w = np.multiply.outer(w, bn.cpts[p].table)
    return w


def wkl(true_tab, est_tab, w):
    K = true_tab.shape[-1]
    t = true_tab.reshape(-1, K)
    e = est_tab.reshape(-1, K)
    return float(np.sum(w.reshape(-1) * np.sum(t * np.log(t / e), axis=1)))


def cmle(node, data):
    n = g.counts(node, data)
    pm = np.full(n.shape, 1.0 / n.shape[-1])
    return g._cmap_from_counts(node, n, pm, MLE_EPS)


def kl_projection(node, true_tab, w, scale=1e4):
    """argmax_{q in M} sum_j w_j sum_k theta*_jk log q_jk (same program)."""
    c = scale * w[..., None] * true_tab
    return g._cmap_from_counts(node, c, true_tab, 0.0)


# ------------------------------------------------------- 2.1 / 2.2 --------
def run_consistency(reps=REPS):
    rows = []
    t0 = time.time()
    worlds = {"standard": wn.world_bn("standard"), "H-v-U": wn.world_bn("H-v-U")}
    proj = {}
    for wname, bn in worlds.items():
        w = parent_probs(bn, "H")
        proj[wname] = kl_projection("H", bn.cpts["H"].table, w)
    pt_rows = []
    for wname, bn in worlds.items():
        w = parent_probs(bn, "H")
        pt_rows.append({"world": wname, "node": "H",
                        "kl_truth_to_projection": wkl(bn.cpts["H"].table,
                                                      proj[wname].table, w)})
    pd.DataFrame(pt_rows).to_csv(os.path.join(RES, f"{P}_pseudotrue.csv"),
                                 index=False)
    for r in range(reps):
        prior = ms.expert_prior_cpts(np.random.default_rng(1000 + r))
        for wname, bn in worlds.items():
            big = dataio.generate_dataset(bn, max(NS), seed=r)
            nodes = NODES if wname == "standard" else ("H",)
            for N in NS:
                data = big[:N]            # same seed: prefixes are the N-draws
                for node in nodes:
                    tru = bn.cpts[node].table
                    w = parent_probs(bn, node)
                    est = {"cMAP": g.cmap_cpt(node, data, prior[node], ESS).table,
                           "MAP": g.map_cpt(node, data, prior[node], ESS).table,
                           "cMLE": cmle(node, data).table}
                    for name, tab in est.items():
                        row = {"world": wname, "rep": r, "N": N, "node": node,
                               "estimator": name, "wkl": wkl(tru, tab, w),
                               "maxabs": float(np.abs(tab - tru).max())}
                        if node == "H":
                            row["wkl_to_pseudotrue"] = wkl(proj[wname].table,
                                                           tab, w)
                        if name == "cMAP":
                            row["maxabs_cmap_cmle"] = float(
                                np.abs(tab - est["cMLE"]).max())
                        rows.append(row)
        pd.DataFrame(rows).to_csv(os.path.join(RES, f"{P}_consistency_raw.csv"),
                                  index=False)
        print(f"[expC 2.1/2.2] rep {r+1}/{reps} ({time.time()-t0:.0f}s)",
              flush=True)


# ------------------------------------------------------------- 2.3 --------
class FastEM:
    """Vectorized MAP-EM with the E-step and M-steps of grounding.em_internal
    (latent cls, emit, H, N, C; observed v, rdot, d, z, sensors, T)."""

    def __init__(self, partial, prior, ess, sensor_cpts):
        gt = ms.ground_truth_bn()
        a = {k: np.array([row[k] for row in partial]) for k in KEEP}
        self.a = a
        s = sensor_cpts
        S = (s["Rr"].table[:, a["Rr"]].T[:, :, None]
             * s["Re"].table[:, a["Re"]].T[:, :, None]
             * s["Ra"].table[:, a["Ra"]].T[:, :, None]
             * np.moveaxis(s["Rf"].table[:, :, a["Rf"]], 2, 0))      # (i,c,e)
        pce = np.multiply.outer(gt.cpts["cls"].table, gt.cpts["emit"].table)
        self.A = S * pce[None]
        self.prior = prior
        self.ess = ess
        self.alpha = {k: ess * prior[k].table for k in ms.INTERNAL}

    def estep(self, cur):
        a = self.a
        PH = cur["H"].table[a["v"], a["rdot"], a["d"]]                  # (i,h)
        PC = cur["C"].table[a["d"], a["z"]]                             # (i,k)
        PN = cur["N"].table                                             # (c,e,n)
        PT = np.moveaxis(cur["T"].table, 3, 0)[a["T"]]                  # (i,h,n,k)
        X = self.A[:, :, :, None, None, None] * PN[None, :, :, None, :, None]
        Y = (PH[:, None, None, :, None, None] * PC[:, None, None, None, None, :]
             * PT[:, None, None, :, :, :])
        J = X * Y                                                       # (i,c,e,h,n,k)
        p = J.sum(axis=(1, 2, 3, 4, 5))
        post = J / p[:, None, None, None, None, None]
        acc = {k: np.zeros(cur[k].table.shape) for k in ms.INTERNAL}
        np.add.at(acc["H"], (a["v"], a["rdot"], a["d"]), post.sum(axis=(1, 2, 4, 5)))
        acc["N"] += post.sum(axis=(0, 3, 5))
        np.add.at(acc["C"], (a["d"], a["z"]), post.sum(axis=(1, 2, 3, 4)))
        mT = post.sum(axis=(1, 2))
        for t in range(ms.CARD["T"]):
            acc["T"][..., t] += mT[a["T"] == t].sum(axis=0)
        return acc, float(np.log(p).sum())

    def log_prior(self, cur):
        return float(sum(np.sum(self.alpha[k] * np.log(cur[k].table))
                         for k in ms.INTERNAL))

    def mstep(self, acc, constrained):
        new = {}
        for k in ms.INTERNAL:
            pm = self.prior[k].table
            if constrained:
                new[k] = g._cmap_from_counts(k, acc[k], pm, self.ess)
            else:
                num = acc[k] + self.ess * pm
                new[k] = Factor(self.prior[k].vars,
                                num / num.sum(axis=-1, keepdims=True))
        return new

    def run(self, n_iter, every=False, checkpoints=()):
        """Returns (estimates at checkpoints, log-posterior trace). For
        every=False the iterate sequence is unconstrained and the reported
        estimate at checkpoint t is the constrained M-step from that
        iteration's expected counts, i.e. exactly em_internal(n_iter=t,
        constrained=True)."""
        cur = {k: Factor(self.prior[k].vars, self.prior[k].table.copy())
               for k in ms.INTERNAL}
        trace, out = [], {}
        for it in range(1, n_iter + 1):
            acc, ll = self.estep(cur)
            trace.append({"iter": it - 1, "loglik": ll,
                          "logpost": ll + self.log_prior(cur)})
            if every:
                cur = self.mstep(acc, constrained=True)
                if it in checkpoints:
                    out[it] = cur
            else:
                if it in checkpoints:
                    out[it] = self.mstep(acc, constrained=True)
                cur = self.mstep(acc, constrained=False)
        acc, ll = self.estep(cur)
        trace.append({"iter": n_iter, "loglik": ll,
                      "logpost": ll + self.log_prior(cur)})
        for t, est in out.items():
            _, llc = self.estep(est)
            trace.append({"iter": f"{t}c", "loglik": llc,
                          "logpost": llc + self.log_prior(est)})
        return out, trace


def check_em():
    """FastEM must reproduce grounding.em_internal. The unconstrained
    iterations agree to machine precision. The final constrained M-step is
    the same function (grounding._cmap_from_counts), but the convex solver's
    own accuracy is about 1e-4 in theta (a 1e-12 relative change of the
    counts moves its solution by up to ~8e-5), so there the two agree only
    to that tolerance; the solver noise is reported, not hidden."""
    gt = ms.ground_truth_bn()
    r, N = 0, 200
    calib = dataio.generate_dataset(gt, 300, seed=5000 + r)
    est = dataio.estimate_confusion(calib)
    prior = ms.expert_prior_cpts(np.random.default_rng(1000 + r))
    full = dataio.generate_dataset(gt, N, seed=r)
    partial = [{k: row[k] for k in KEEP} for row in full]
    fem = FastEM(partial, prior, ESS, est)
    ref_u = g.em_internal(partial, prior, ESS, n_iter=12, seed=r,
                          constrained=False, sensor_cpts=est)
    cur = {k: Factor(prior[k].vars, prior[k].table.copy()) for k in ms.INTERNAL}
    for _ in range(12):
        acc, _ = fem.estep(cur)
        cur = fem.mstep(acc, constrained=False)
    du = {k: float(np.abs(cur[k].table - ref_u[k].table).max())
          for k in ms.INTERNAL}
    ref = g.em_internal(partial, prior, ESS, n_iter=12, seed=r,
                        constrained=True, sensor_cpts=est)
    out, _ = fem.run(12, every=False, checkpoints=(12,))
    dc = {k: float(np.abs(out[12][k].table - ref[k].table).max())
          for k in ms.INTERNAL}
    # solver sensitivity on identical-up-to-1e-12 counts
    acc12, _ = fem.estep(ref_u)
    sens = {}
    for k in ms.INTERNAL:
        pm = prior[k].table
        a1 = g._cmap_from_counts(k, acc12[k], pm, ESS).table
        a2 = g._cmap_from_counts(k, acc12[k] * (1 + 1e-12), pm, ESS).table
        sens[k] = float(np.abs(a1 - a2).max())
    print(f"[check-em] unconstrained 12 iterations max|diff|: {du}")
    print(f"[check-em] with the final constrained step max|diff|: {dc}")
    print(f"[check-em] solver sensitivity to a 1e-12 relative count change: {sens}")
    assert max(du.values()) < 1e-12, du
    assert max(dc.values()) < 1e-3, dc
    return du, dc, sens


def label_ok(node, est_tab, true_tab, w):
    """True if the identity labeling of node's states fits the truth at least
    as well as every other permutation of them."""
    K = true_tab.shape[-1]
    base = wkl(true_tab, est_tab, w)
    for perm in permutations(range(K)):
        if perm == tuple(range(K)):
            continue
        if wkl(true_tab, est_tab[..., list(perm)], w) < base:
            return False
    return True


def run_mapem(reps=REPS):
    gt = ms.ground_truth_bn()
    wts = {k: parent_probs(gt, k) for k in NODES}
    rows, trows = [], []
    t0 = time.time()
    for r in range(reps):
        test = dataio.generate_dataset(gt, 2500, seed=9000 + r)
        calib = dataio.generate_dataset(gt, 300, seed=5000 + r)
        est = dataio.estimate_confusion(calib)
        alpha, _ = dataio.fit_reliability(calib)
        prior = ms.expert_prior_cpts(np.random.default_rng(1000 + r))
        big = dataio.generate_dataset(gt, max(EM_NS), seed=r)
        for N in EM_NS:
            partial = [{k: row[k] for k in KEEP} for row in big[:N]]
            fem = FastEM(partial, prior, ESS, est)
            for variant, every in (("final-only", False), ("every-iteration", True)):
                outs, trace = fem.run(EM_MAX, every=every, checkpoints=EM_CHECK)
                for tr in trace:
                    trows.append({"rep": r, "N": N, "variant": variant, **tr})
                for t, cp in outs.items():
                    met, _ = mt.evaluate(cp, test, sensor_cpts=est, alpha=alpha)
                    row = {"rep": r, "N": N, "variant": variant, "iters": t,
                           **{k: met[k] for k in ("acc", "macroAUC", "ece",
                                                  "brier")}}
                    for k in NODES:
                        row[f"wkl_{k}"] = wkl(gt.cpts[k].table, cp[k].table,
                                              wts[k])
                    for k in ("H", "N", "C"):
                        row[f"label_ok_{k}"] = label_ok(k, cp[k].table,
                                                        gt.cpts[k].table, wts[k])
                    rows.append(row)
        pd.DataFrame(rows).to_csv(os.path.join(RES, f"{P}_mapem_raw.csv"),
                                  index=False)
        pd.DataFrame(trows).to_csv(os.path.join(RES, f"{P}_mapem_trace.csv"),
                                   index=False)
        print(f"[expC 2.3] rep {r+1}/{reps} ({time.time()-t0:.0f}s)", flush=True)


# ------------------------------------------------------------- 2.4 --------
def ici_from_counts(c, restarts=4, seed=0):
    """ordinal_ici_cpt's fit on supplied pseudo-counts (same parameterization
    via grounding._oi_table, same optimizer settings)."""
    from itertools import product as ip
    from scipy.optimize import minimize
    pa_cards = [ms.CARD[p] for p in ms.PARENTS["T"]]
    K = ms.CARD["T"]
    cfgs = list(ip(*[range(x) for x in pa_cards]))
    cc = c.reshape(len(cfgs), K)
    dim = sum(x - 1 for x in pa_cards) + (K - 1) + 1
    rng = np.random.default_rng(seed)
    best = None
    for _ in range(restarts):
        x0 = rng.normal(0.0, 0.8, dim)
        res = minimize(lambda x: -float(np.sum(cc * np.log(
            g._oi_table(x, pa_cards, K, cfgs)))), x0, method="L-BFGS-B",
            options={"maxiter": 800, "ftol": 1e-11})
        if best is None or res.fun < best.fun:
            best = res
    return g._oi_table(best.x, pa_cards, K, cfgs).reshape(tuple(pa_cards) + (K,))


def run_ici(reps=REPS):
    gt = ms.ground_truth_bn()
    w = parent_probs(gt, "T")
    tru = gt.cpts["T"].table
    floor = ici_from_counts(1e4 * w[..., None] * tru, restarts=8, seed=0)
    pd.DataFrame([{"node": "T", "family": "ordinal ICI",
                   "kl_truth_to_family": wkl(tru, floor, w),
                   "maxabs_truth_to_family": float(np.abs(floor - tru).max())}]) \
        .to_csv(os.path.join(RES, f"{P}_ici_floor.csv"), index=False)
    rows = []
    t0 = time.time()
    for r in range(reps):
        prior = ms.expert_prior_cpts(np.random.default_rng(1000 + r))
        big = dataio.generate_dataset(gt, max(NS), seed=r)
        for N in NS:
            data = big[:N]
            ici = g.ordinal_ici_cpt("T", data, prior["T"], ESS, seed=r).table
            full = g.cmap_cpt("T", data, prior["T"], ESS).table
            rows.append({"rep": r, "N": N, "wkl_ici": wkl(tru, ici, w),
                         "wkl_full_cmap": wkl(tru, full, w),
                         "wkl_ici_to_floor": wkl(floor, ici, w)})
        pd.DataFrame(rows).to_csv(os.path.join(RES, f"{P}_ici_raw.csv"),
                                  index=False)
        print(f"[expC 2.4] rep {r+1}/{reps} ({time.time()-t0:.0f}s)", flush=True)


# ----------------------------------------------------------- summary ------
def _pm(x):
    m, lo, hi = es.t_ci(np.asarray(x, float))
    return m, (hi - lo) / 2


def summarize():
    L = ["# Exp-C: interpretation and consistency of the cMAP and MAP-EM "
         f"estimators (ESS {ESS:g}, {REPS} replications)", ""]
    c = read_csv(os.path.join(RES, f"{P}_consistency_raw.csv"))
    pt = read_csv(os.path.join(RES, f"{P}_pseudotrue.csv"))
    L.append("## 2.1 complete data, standard world: column-weighted KL to the "
             "truth (mean over replications)")
    for node in NODES:
        L.append(f"\n### node {node}")
        L.append("| N | cMAP | MAP | cMLE | maxabs cMAP | max|cMAP-cMLE| |")
        L.append("|---|---|---|---|---|---|")
        for N in NS:
            d = c[(c.world == "standard") & (c.node == node) & (c.N == N)]
            f = lambda e, k: d[d.estimator == e][k].mean()
            L.append(f"| {N} | {f('cMAP','wkl'):.2e} | {f('MAP','wkl'):.2e} | "
                     f"{f('cMLE','wkl'):.2e} | {f('cMAP','maxabs'):.4f} | "
                     f"{f('cMAP','maxabs_cmap_cmle'):.2e} |")
        d = c[(c.world == "standard") & (c.node == node) & (c.estimator == "cMAP")]
        m = d.groupby("N")["maxabs_cmap_cmle"].mean()
        slope = np.polyfit(np.log(m.index.values.astype(float)),
                           np.log(m.values), 1)[0]
        m2 = d.groupby("N")["wkl"].mean()
        slope2 = np.polyfit(np.log(m2.index.values.astype(float)),
                            np.log(m2.values), 1)[0]
        L.append(f"- log-log slope vs N: max|cMAP-cMLE| {slope:+.2f}; "
                 f"cMAP KL {slope2:+.2f}")
    L.append("\n## 2.2 non-monotone world (H-v-U), node H")
    floor = float(pt[pt.world == "H-v-U"].kl_truth_to_projection.iloc[0])
    fstd = float(pt[pt.world == "standard"].kl_truth_to_projection.iloc[0])
    L.append(f"- KL(truth || KL projection onto M): H-v-U {floor:.4e}; "
             f"standard {fstd:.2e} (truth already monotone)")
    L.append("| N | cMAP->truth | cMAP->pseudo-true | MAP->truth | "
             "cMLE->pseudo-true |")
    L.append("|---|---|---|---|---|")
    for N in NS:
        d = c[(c.world == "H-v-U") & (c.node == "H") & (c.N == N)]
        f = lambda e, k: d[d.estimator == e][k].mean()
        L.append(f"| {N} | {f('cMAP','wkl'):.3e} | "
                 f"{f('cMAP','wkl_to_pseudotrue'):.3e} | {f('MAP','wkl'):.3e} | "
                 f"{f('cMLE','wkl_to_pseudotrue'):.3e} |")
    mp = read_csv(os.path.join(RES, f"{P}_mapem_raw.csv"))
    tr = read_csv(os.path.join(RES, f"{P}_mapem_trace.csv"))
    L.append("\n## 2.3 MAP-EM (latent H, N, C)")
    num = tr[tr.iter.astype(str).str.fullmatch(r"\d+")].copy()
    num["iter"] = num["iter"].astype(int)
    dec_rows = []
    for (r, N, v), gg in num.groupby(["rep", "N", "variant"]):
        lp = gg.sort_values("iter")["logpost"].values
        dlt = np.diff(lp)
        tol = 1e-9 * np.abs(lp[1:])
        start = 0 if v == "final-only" else 1      # every-iteration: from the first constrained iterate
        dec_rows.append({"rep": r, "N": N, "variant": v,
                         "n_steps": len(dlt[start:]),
                         "n_decrease": int((dlt[start:] < -tol[start:]).sum()),
                         "max_drop": float(-min(0.0, dlt[start:].min())),
                         "rel_change_after_12": float(abs(lp[-1] - lp[12]) / abs(lp[12])),
                         "rel_step_at_12": float(abs(lp[12] - lp[11]) / abs(lp[11]))})
    dec = pd.DataFrame(dec_rows)
    for v in ("final-only", "every-iteration"):
        for N in EM_NS:
            d = dec[(dec.variant == v) & (dec.N == N)]
            L.append(f"- {v} N={N}: decreases {int(d.n_decrease.sum())}/"
                     f"{int(d.n_steps.sum())} steps (max drop {d.max_drop.max():.1e}); "
                     f"relative logpost change iter 12 -> 50: mean "
                     f"{d.rel_change_after_12.mean():.2e}, max "
                     f"{d.rel_change_after_12.max():.2e}; relative step at 12: "
                     f"{d.rel_step_at_12.mean():.2e}")
    fin = []
    for (r, N, v), gg in tr.groupby(["rep", "N", "variant"]):
        if v != "final-only":
            continue
        s = gg.set_index(gg.iter.astype(str))["logpost"]
        fin.append({"N": N, "drop12": float(s["12"] - s["12c"]),
                    "drop50": float(s["50"] - s["50c"])})
    fin = pd.DataFrame(fin)
    for N in EM_NS:
        d = fin[fin.N == N]
        L.append(f"- final-only N={N}: logpost(unconstrained iterate) - "
                 f"logpost(final constrained step) at 12: mean {d.drop12.mean():+.3e}"
                 f" (max {d.drop12.max():+.3e}); at 50: mean {d.drop50.mean():+.3e}")
    L.append("\n| N | variant | iters | acc | AUC | ECE | KL H | KL N | KL C | "
             "KL T | labels ok |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for N in EM_NS:
        for v in ("final-only", "every-iteration"):
            for t in EM_CHECK:
                d = mp[(mp.N == N) & (mp.variant == v) & (mp.iters == t)]
                lab = int(d[["label_ok_H", "label_ok_N", "label_ok_C"]].values.sum())
                L.append(f"| {N} | {v} | {t} | {d.acc.mean():.4f} | "
                         f"{d.macroAUC.mean():.4f} | {d.ece.mean():.4f} | "
                         f"{d.wkl_H.mean():.4f} | {d.wkl_N.mean():.4f} | "
                         f"{d.wkl_C.mean():.4f} | {d.wkl_T.mean():.4f} | "
                         f"{lab}/{3*len(d)} |")
    L.append("\n### paired differences over replications (t p)")
    for N in EM_NS:
        a = mp[(mp.N == N) & (mp.variant == "final-only")].set_index(["rep", "iters"])
        b = mp[(mp.N == N) & (mp.variant == "every-iteration")].set_index(["rep", "iters"])
        for k in ("acc", "macroAUC", "ece", "wkl_H", "wkl_N", "wkl_C", "wkl_T"):
            x = (a.xs(50, level="iters")[k] - a.xs(12, level="iters")[k]).values
            y = (b.xs(12, level="iters")[k] - a.xs(12, level="iters")[k]).values
            tx, ty = es.across_rep_tests(x), es.across_rep_tests(y)
            L.append(f"- N={N} {k}: final-only 50-12 {tx['mean']:+.5f} "
                     f"(p {tx['t_p']:.1e}); every-iteration - final-only at 12 "
                     f"{ty['mean']:+.5f} (p {ty['t_p']:.1e})")
    ic = read_csv(os.path.join(RES, f"{P}_ici_raw.csv"))
    fl = read_csv(os.path.join(RES, f"{P}_ici_floor.csv"))
    L.append("\n## 2.4 ordinal ICI threat node vs full-table cMAP (standard "
             "world)")
    L.append(f"- infinite-data ICI fit: KL(truth || ICI family) "
             f"{fl.kl_truth_to_family.iloc[0]:.4e}, max abs "
             f"{fl.maxabs_truth_to_family.iloc[0]:.4f}")
    L.append("| N | ICI KL | full cMAP KL | ICI KL to its limit |")
    L.append("|---|---|---|---|")
    for N in NS:
        d = ic[ic.N == N]
        L.append(f"| {N} | {d.wkl_ici.mean():.4e} | {d.wkl_full_cmap.mean():.4e}"
                 f" | {d.wkl_ici_to_floor.mean():.4e} |")
    txt = "\n".join(L)
    with open(os.path.join(RES, f"{P}_summary.md"), "w", encoding="utf-8") as f:
        f.write(txt + "\n")
    print(txt)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=REPS)
    ap.add_argument("--check-em", action="store_true")
    ap.add_argument("--summary-only", action="store_true")
    ap.add_argument("--only", default="", help="comma list of 21,23,24")
    a = ap.parse_args()
    os.makedirs(RES, exist_ok=True)
    if a.check_em:
        check_em()
    elif a.summary_only:
        summarize()
    else:
        parts = a.only.split(",") if a.only else ["21", "23", "24"]
        if "21" in parts:
            run_consistency(a.reps)
        if "23" in parts:
            run_mapem(a.reps)
        if "24" in parts:
            run_ici(a.reps)
        summarize()
