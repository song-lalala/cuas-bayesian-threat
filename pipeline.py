"""
Shared grounding pipeline used by every experiment script.

Proposed method = {H, N, C: constrained posterior-mean (cmap_cpt, cvxpy)} +
{T: ordinal cumulative-logit ICI (ordinal_ici_cpt, C1)} with the ESS fixed a
priori at ESS_FIXED = 2 (a weakly informative Dirichlet whose equivalent
sample size is of the order of the child cardinalities; Fig. 5 confirms post
hoc that this sits on the flat ECE-optimal plateau alpha_0 in [2,5]). NO
labelled validation split is consumed by the main pipeline, so every method
in a comparison sees exactly the same N labelled scenarios (label-budget
fairness). Held-out ESS selection (`select_ess`) is kept only as the option
studied in Sec. VII-E(iii) for settings where extra labels exist.

Methods:
  B0-heuristic   hand-set surrogate tables (see model_spec.heuristic_cpts)
  B1-MLE         unconstrained maximum likelihood (full tables)
  B2-expert      prior means only, no data
  MAP            Dirichlet posterior mean, full tables
  cMAP-full      constrained posterior mean, full tables ( = Proposed -ICI )
  Proposed       cMAP for H,N,C + Noisy-MAX for T
Ablations map onto: -ICI -> cMAP-full; -constraints -> MAP for H,N,C with
Noisy-MAX T; -prior -> B1-MLE; -reliability -> alpha=None at evaluation.
"""
from __future__ import annotations
import numpy as np
import model_spec as ms
import dataio
import grounding as g
import metrics as mt

ESS_GRID = (1, 2, 5, 10, 20, 50)
VAL_N = 60          # used only by the Sec. VII-E(iii) selection study
ESS_FIXED = 2.0     # a-priori ESS of the main pipeline (no extra labels)


def log_loss(P, y, eps=1e-12):
    return float(-np.mean(np.log(P[np.arange(len(y)), y] + eps)))


def select_ess(data, prior, val, est, alpha, grid=ESS_GRID):
    """Held-out log-loss ESS selection with the MAP closed-form proxy."""
    best, best_ll = grid[0], np.inf
    for a0 in grid:
        cpts = {n: g.map_cpt(n, data, prior[n], a0) for n in ms.INTERNAL}
        P, y = mt.posteriors(cpts, val, ms.SENSORS, "calibrated",
                             sensor_cpts=est, alpha=alpha)
        ll = log_loss(P, y)
        if ll < best_ll:
            best, best_ll = a0, ll
    return best


def ground_internal(method, data, seed, ess=None, val=None, est=None,
                    alpha=None, gt=None):
    """Return internal CPTs {H,N,C,T} for a method name. The prior-based
    methods use the a-priori ESS_FIXED when `ess` is None; no validation
    labels are consumed (label-budget fairness vs. the prior-free
    baselines)."""
    rng = np.random.default_rng(1000 + seed)
    prior = ms.expert_prior_cpts(rng)
    if method == "B0-heuristic":
        return ms.heuristic_cpts(), None
    if method == "B2-expert":
        return prior, None
    if method == "B1-MLE":
        return {n: g.mle_cpt(n, data) for n in ms.INTERNAL}, None
    if ess is None:
        ess = ESS_FIXED
    out = {}
    for n in ms.INTERNAL:
        if method == "MAP":
            out[n] = g.map_cpt(n, data, prior[n], ess)
        elif method == "cMAP-full":
            out[n] = g.cmap_cpt(n, data, prior[n], ess)
        elif method == "Proposed":
            out[n] = (g.ordinal_ici_cpt(n, data, prior[n], ess, seed=seed)
                      if n == "T" else g.cmap_cpt(n, data, prior[n], ess))
        elif method == "Proposed-NM":
            out[n] = (g.noisymax_cpt(n, data, prior[n], ess, seed=seed)
                      if n == "T" else g.cmap_cpt(n, data, prior[n], ess))
        elif method == "Proposed-WS":
            out[n] = (g.weightedsum_cpt(n, data, prior[n], ess, seed=seed)
                      if n == "T" else g.cmap_cpt(n, data, prior[n], ess))
        elif method == "Proposed-noconstraints":
            out[n] = (g.ordinal_ici_cpt(n, data, prior[n], ess, seed=seed)
                      if n == "T" else g.map_cpt(n, data, prior[n], ess))
        else:
            raise ValueError(method)
    return out, ess


if __name__ == "__main__":
    import time
    gt = ms.ground_truth_bn()
    test = dataio.generate_dataset(gt, 2000, seed=999)
    calib = dataio.generate_dataset(gt, 300, seed=777)
    est = dataio.estimate_confusion(calib)
    al, _ = dataio.fit_reliability(calib)
    data = dataio.generate_dataset(gt, 200, seed=0)
    t0 = time.time()
    for method in ["B0-heuristic", "B1-MLE", "MAP", "cMAP-full", "Proposed",
                   "Proposed-NM"]:
        cpts, ess = ground_internal(method, data, 0, est=est, alpha=al, gt=gt)
        m, _ = mt.evaluate(cpts, test, sensor_cpts=est, alpha=al)
        m2, _ = mt.evaluate(cpts, test, sensor_mode="abstract")
        print(f"{method:12s} (ess={ess}): calib acc {m['acc']:.3f} AUC "
              f"{m['macroAUC']:.3f} ECE {m['ece']:.3f} cwECE {m['cwece']:.3f} "
              f"| abstract AUC {m2['macroAUC']:.3f} ECE {m2['ece']:.3f}")
    print(f"({time.time()-t0:.0f}s)  cmap solver: {g.cmap_report()}")
