"""
CPT grounding methods (P1):
  mle_cpt      : maximum likelihood (full table) -- B1, overfits under scarcity
  map_cpt      : Dirichlet posterior mean with ESS prior
  cmap_cpt     : constrained posterior mean = map_cpt objective + first-order
                 stochastic-dominance (monotonicity) constraints, solved as a
                 convex program with cvxpy (CLARABEL, ECOS fallback) -- the
                 proposed grounding for tabular nodes. Solver outcomes are
                 tallied in CMAP_STATS and a non-optimal solve falls back to
                 the closed-form MAP (counted, never silent).
  noisymax_cpt : leaky Noisy-MAX (ICI) grounding for the ordinal threat node:
                 the same pseudo-counts c = n + ess*prior are fitted by a
                 Noisy-MAX likelihood with O(sum_p r_p * (K-1)) parameters
                 instead of the full O(prod_p r_p * (K-1)) table (C1).
  em_internal  : MAP-EM for the latent-variable setting (H,N,C unobserved);
                 the M-step applies the constrained update when
                 constrained=True, and the sensor/root CPTs can be supplied
                 (e.g. calibration-split estimates) instead of ground truth.

All operate on `data` = list of full/partial assignment dicts. Prior means are
Factors from model_spec (expert_prior_cpts). Monotonicity uses the fact that
every parent is encoded threat-increasing, so a +1 step in any parent must
stochastically increase the (ordinal) child.

Objective convention: the smoothed estimators maximize
    sum_{j,k} (N_jk + ess * phat_jk) * log theta_jk
i.e. the coefficients are the POSTERIOR-MEAN pseudo-counts N + alpha (always
positive, hence a well-posed concave program even when alpha_jk < 1), not the
MAP-mode coefficients N + alpha - 1 (which can be negative in scarce cells).
The manuscript states the same convention.
"""
from __future__ import annotations
import numpy as np
from itertools import product as iproduct
from scipy.optimize import minimize
from cuas_bn import Factor, BayesNet
import model_spec as ms

# solver outcome tally for the convergence report: {"solved","fallback"}
CMAP_STATS = {"solved": 0, "fallback": 0}


def _configs(node):
    pa = ms.PARENTS[node]
    return pa, list(iproduct(*[range(ms.CARD[p]) for p in pa])) if pa else [()]


def counts(node, data):
    pa, cfgs = _configs(node)
    K = ms.CARD[node]
    shape = tuple(ms.CARD[p] for p in pa) + (K,)
    n = np.zeros(shape)
    for row in data:
        if node not in row:
            continue
        idx = tuple(row[p] for p in pa) + (row[node],)
        n[idx] += 1
    return n  # shape (parents..., K)


def mle_cpt(node, data, eps=1e-6):
    n = counts(node, data) + eps
    t = n / n.sum(axis=-1, keepdims=True)
    return Factor(tuple(ms.PARENTS[node]) + (node,), t)


def map_cpt(node, data, prior_mean: Factor, ess: float):
    """Dirichlet posterior mean: (n + ess*prior)/(N + ess) per parent config."""
    n = counts(node, data)
    pm = prior_mean.table
    num = n + ess * pm
    t = num / num.sum(axis=-1, keepdims=True)
    return Factor(tuple(ms.PARENTS[node]) + (node,), t)


def _covering_pairs(node):
    """Return list of (flat_j, flat_jprime) where jprime is j with exactly one
    parent stepped +1 (jprime is the more-threatening config)."""
    pa = ms.PARENTS[node]
    cards = [ms.CARD[p] for p in pa]
    cfgs = list(iproduct(*[range(c) for c in cards])) if pa else [()]
    index = {c: i for i, c in enumerate(cfgs)}
    pairs = []
    for c in cfgs:
        for pos in range(len(pa)):
            if c[pos] + 1 < cards[pos]:
                cp = list(c); cp[pos] += 1; cp = tuple(cp)
                pairs.append((index[c], index[cp]))
    return pairs, len(cfgs)


def cmap_cpt(node, data, prior_mean: Factor, ess: float, verbose=False):
    """Constrained posterior-mean estimate: maximize sum c_{jk} log theta_{jk}
    with c = n + ess*prior (> 0, concave), subject to per-config simplex and
    first-order stochastic-dominance constraints, as a disciplined convex
    program (cvxpy; CLARABEL then ECOS/SCS). A non-optimal status falls back
    to the closed-form MAP and is tallied in CMAP_STATS["fallback"]."""
    import cvxpy as cp
    pa = ms.PARENTS[node]
    K = ms.CARD[node]
    n = counts(node, data)
    pm = prior_mean.table
    Jshape = n.shape[:-1]
    J = int(np.prod(Jshape)) if pa else 1
    c = (n + ess * pm).reshape(J, K)

    th = cp.Variable((J, K), nonneg=True)
    cons = [cp.sum(th, axis=1) == 1]
    pairs, _ = _covering_pairs(node)
    if pairs:
        pj = [p[0] for p in pairs]
        pjp = [p[1] for p in pairs]
        for m in range(1, K):
            cons.append(cp.sum(th[pjp, m:], axis=1) >= cp.sum(th[pj, m:], axis=1))
    obj = cp.Maximize(cp.sum(cp.multiply(c, cp.log(th))))
    prob = cp.Problem(obj, cons)
    solved = False
    for solver in ("CLARABEL", "ECOS", "SCS"):
        try:
            prob.solve(solver=solver)
        except Exception:
            continue
        if prob.status in ("optimal", "optimal_inaccurate") and th.value is not None:
            solved = True
            break
    if solved:
        CMAP_STATS["solved"] += 1
        X = np.clip(th.value, 1e-9, None)
    else:
        CMAP_STATS["fallback"] += 1
        if verbose:
            print(f"[cmap] fallback to MAP for node {node} (status={prob.status})")
        X = c.copy()
    X = X / X.sum(axis=1, keepdims=True)
    t = X.reshape(tuple(Jshape) + (K,)) if pa else X.reshape(K)
    return Factor(tuple(pa) + (node,), t)


def cmap_report(reset=False):
    """Return (and optionally reset) the cMAP solver tally."""
    out = dict(CMAP_STATS)
    if reset:
        CMAP_STATS["solved"] = 0
        CMAP_STATS["fallback"] = 0
    return out


# ---------------- leaky Noisy-MAX (ICI) grounding for the threat node -------
def _nm_unpack(x, cards, K):
    """Unpack the flat parameter vector into per-parent-state cumulative
    curves F_p(t|s) in (0,1], t=0..K-2 (F at K-1 is 1), plus the leak
    cumulative F_0(t). Monotone nondecreasing in t by construction
    (cumulative sum of softplus increments through a sigmoid)."""
    Fs = []
    o = 0
    for r in cards:
        raw = x[o:o + r * (K - 1)].reshape(r, K - 1)
        o += r * (K - 1)
        inc = np.logaddexp(0.0, raw)              # softplus > 0
        F = 1.0 / (1.0 + np.exp(-(np.cumsum(inc, axis=1) - 3.0)))
        Fs.append(F)
    rawl = x[o:o + (K - 1)]
    incl = np.logaddexp(0.0, rawl)
    Fl = 1.0 / (1.0 + np.exp(-(np.cumsum(incl) - 3.0)))
    return Fs, Fl


def _nm_table(x, pa_cards, K, cfgs):
    Fs, Fl = _nm_unpack(x, pa_cards, K)
    J = len(cfgs)
    T = np.zeros((J, K))
    for j, cfg in enumerate(cfgs):
        F = Fl.copy()
        for p, s in enumerate(cfg):
            F = F * Fs[p][s]
        Ffull = np.concatenate([F, [1.0]])        # cumulative P(T<=t)
        pmf = np.diff(np.concatenate([[0.0], Ffull]))
        T[j] = np.clip(pmf, 1e-9, None)
        T[j] /= T[j].sum()
    return T


def noisymax_cpt(node, data, prior_mean: Factor, ess: float, restarts=3,
                 seed=0):
    """Leaky Noisy-MAX grounding (C1): fit the ICI parameterization to the
    same pseudo-counts c = n + ess*prior by maximum likelihood (L-BFGS,
    multi-start). Parameters: one cumulative activation curve per parent
    state plus a leak curve -- O((sum_p r_p + 1)(K-1)) instead of
    O(prod_p r_p (K-1))."""
    pa = ms.PARENTS[node]
    K = ms.CARD[node]
    pa_cards = [ms.CARD[p] for p in pa]
    cfgs = list(iproduct(*[range(r) for r in pa_cards]))
    n = counts(node, data)
    c = (n + ess * prior_mean.table).reshape(len(cfgs), K)
    dim = sum(pa_cards) * (K - 1) + (K - 1)
    rng = np.random.default_rng(seed)

    def nll(x):
        T = _nm_table(x, pa_cards, K, cfgs)
        return -float(np.sum(c * np.log(T)))

    best = None
    for r in range(restarts):
        x0 = rng.normal(0.0, 0.5, dim)
        res = minimize(nll, x0, method="L-BFGS-B",
                       options={"maxiter": 400, "ftol": 1e-9})
        if best is None or res.fun < best.fun:
            best = res
    T = _nm_table(best.x, pa_cards, K, cfgs)
    t = T.reshape(tuple(pa_cards) + (K,))
    return Factor(tuple(pa) + (node,), t)


def nm_param_count(node):
    pa_cards = [ms.CARD[p] for p in ms.PARENTS[node]]
    K = ms.CARD[node]
    return (sum(pa_cards) + 1) * (K - 1)


# ---------------- weighted-sum (Das) ICI grounding --------------------------
def _ws_unpack(x, pa_cards, K):
    P = len(pa_cards)
    wl = x[:P]
    w = np.exp(wl - wl.max())
    w = w / w.sum()
    anchors = []
    o = P
    for r in pa_cards:
        a = x[o:o + r * K].reshape(r, K)
        o += r * K
        e = np.exp(a - a.max(axis=1, keepdims=True))
        anchors.append(e / e.sum(axis=1, keepdims=True))
    return w, anchors


def _ws_table(x, pa_cards, K, cfgs):
    w, anchors = _ws_unpack(x, pa_cards, K)
    T = np.zeros((len(cfgs), K))
    for j, cfg in enumerate(cfgs):
        row = np.zeros(K)
        for p, s in enumerate(cfg):
            row += w[p] * anchors[p][s]
        T[j] = np.clip(row, 1e-9, None)
        T[j] /= T[j].sum()
    return T


def weightedsum_cpt(node, data, prior_mean: Factor, ess: float, restarts=4,
                    seed=0):
    """Weighted-sum ICI grounding (C1, Das form): P(T=t|z) =
    sum_p w_p g_p(t|z_p), sum_p w_p = 1, fitted to the pseudo-counts
    c = n + ess*prior by maximum likelihood (L-BFGS on logits, multi-start).
    Parameters: mixing weights (P-1) + per-parent-state anchor distributions
    (sum_p r_p * (K-1)) -- linear in the number of parents."""
    pa = ms.PARENTS[node]
    K = ms.CARD[node]
    pa_cards = [ms.CARD[p] for p in pa]
    cfgs = list(iproduct(*[range(r) for r in pa_cards]))
    n = counts(node, data)
    c = (n + ess * prior_mean.table).reshape(len(cfgs), K)
    dim = len(pa_cards) + sum(pa_cards) * K
    rng = np.random.default_rng(seed)

    def nll(x):
        T = _ws_table(x, pa_cards, K, cfgs)
        return -float(np.sum(c * np.log(T)))

    # informed start: anchors from the prior's per-parent conditional means
    pm = prior_mean.table.reshape(len(cfgs), K)
    x0s = [rng.normal(0.0, 0.5, dim) for _ in range(restarts - 1)]
    xa = np.zeros(dim)
    o = len(pa_cards)
    for p, r in enumerate(pa_cards):
        for s in range(r):
            rows = [pm[j] for j, cfg in enumerate(cfgs) if cfg[p] == s]
            mean_row = np.clip(np.mean(rows, axis=0), 1e-6, None)
            xa[o + s * K:o + (s + 1) * K] = np.log(mean_row)
        o += r * K
    x0s.append(xa)
    best = None
    for x0 in x0s:
        res = minimize(nll, x0, method="L-BFGS-B",
                       options={"maxiter": 600, "ftol": 1e-10})
        if best is None or res.fun < best.fun:
            best = res
    T = _ws_table(best.x, pa_cards, K, cfgs)
    return Factor(tuple(pa) + (node,), T.reshape(tuple(pa_cards) + (K,)))


def ws_param_count(node):
    pa_cards = [ms.CARD[p] for p in ms.PARENTS[node]]
    K = ms.CARD[node]
    return (len(pa_cards) - 1) + sum(pa_cards) * (K - 1)


# ------------- ordinal cumulative-logit ICI grounding (adopted C1 form) -----
def _oi_unpack(x, pa_cards, K):
    o = 0
    betas = []
    for r in pa_cards:
        raw = x[o:o + (r - 1)]
        o += (r - 1)
        b = np.concatenate([[0.0], np.cumsum(np.logaddexp(0.0, raw))])
        betas.append(b)          # nondecreasing in the parent state, b[0]=0
    base = x[o]
    o += 1
    if K > 2:
        inc = np.logaddexp(0.0, x[o:o + (K - 2)])
        o += (K - 2)
        theta = base + np.concatenate([[0.0], np.cumsum(inc)])
    else:
        theta = np.array([base])
    scale = np.logaddexp(0.0, x[o]) + 0.3         # response sharpness > 0.3
    return betas, theta, scale


def _oi_table(x, pa_cards, K, cfgs):
    betas, theta, scale = _oi_unpack(x, pa_cards, K)
    T = np.zeros((len(cfgs), K))
    for j, cfg in enumerate(cfgs):
        eta = sum(betas[p][s] for p, s in enumerate(cfg))
        z = scale * (theta - eta)
        F = 1.0 / (1.0 + np.exp(-z))              # P(T<=t), t=0..K-2
        Ffull = np.concatenate([F, [1.0]])
        pmf = np.diff(np.concatenate([[0.0], Ffull]))
        T[j] = np.clip(pmf, 1e-9, None)
        T[j] /= T[j].sum()
    return T


def ordinal_ici_cpt(node, data, prior_mean: Factor, ess: float, restarts=4,
                    seed=0):
    """Ordinal cumulative-logit ICI grounding (the adopted C1 form for the
    ordered threat node): each parent contributes an independent, monotone
    additive effect beta_p(s) (nondecreasing in the threat-increasing state
    order, beta_p(0)=0) to a latent threat score, converted to P(T=k) through
    ordered thresholds theta -- a proportional-odds response. MONOTONE BY
    CONSTRUCTION in every parent. Fitted to the pseudo-counts
    c = n + ess*prior by maximum likelihood (L-BFGS, multi-start).
    Free parameters: sum_p (r_p - 1) effects + (K-1) thresholds + 1 sharpness
    -- linear in the number of parents (8 for the threat node vs 54 full)."""
    pa = ms.PARENTS[node]
    K = ms.CARD[node]
    pa_cards = [ms.CARD[p] for p in pa]
    cfgs = list(iproduct(*[range(r) for r in pa_cards]))
    n = counts(node, data)
    c = (n + ess * prior_mean.table).reshape(len(cfgs), K)
    dim = sum(r - 1 for r in pa_cards) + (K - 1) + 1
    rng = np.random.default_rng(seed)

    def nll(x):
        T = _oi_table(x, pa_cards, K, cfgs)
        return -float(np.sum(c * np.log(T)))

    best = None
    for r in range(restarts):
        x0 = rng.normal(0.0, 0.8, dim)
        res = minimize(nll, x0, method="L-BFGS-B",
                       options={"maxiter": 800, "ftol": 1e-11})
        if best is None or res.fun < best.fun:
            best = res
    T = _oi_table(best.x, pa_cards, K, cfgs)
    return Factor(tuple(pa) + (node,), T.reshape(tuple(pa_cards) + (K,)))


def oi_param_count(node):
    pa_cards = [ms.CARD[p] for p in ms.PARENTS[node]]
    K = ms.CARD[node]
    return sum(r - 1 for r in pa_cards) + (K - 1) + 1


# ---------------- MAP-EM for latent internal nodes ----------------
def em_internal(data_partial, prior_means, ess, n_iter=25, seed=0,
                constrained=False, sensor_cpts=None):
    """Learn internal CPTs (H,N,C,T) when H,N,C are UNOBSERVED. `data_partial`
    rows contain observed inputs, sensors, and T (threat label). Root CPTs come
    from ground truth (they are directly estimable from observed inputs and are
    not under study); sensor CPTs come from `sensor_cpts` (e.g. the
    calibration-split estimates) when given, else ground truth.
    E-step: expected counts via posterior over latents per row.
    M-step: MAP update from expected counts; when constrained=True the final
    iteration's M-step is the constrained program of cmap_cpt applied to the
    expected counts (intermediate iterations use the closed form for speed).
    """
    gt = ms.ground_truth_bn()
    cur = {k: Factor(prior_means[k].vars, prior_means[k].table.copy())
           for k in ms.INTERNAL}
    obs_vars = set(ms.OBSERVED) | set(ms.SENSORS) | {"T"}

    def m_step(acc, final):
        new = {}
        for k in ms.INTERNAL:
            pm = prior_means[k].table
            if final and constrained:
                new[k] = _cmap_from_counts(k, acc[k], pm, ess)
            else:
                num = acc[k] + ess * pm
                new[k] = Factor(cur[k].vars,
                                num / num.sum(axis=-1, keepdims=True))
        return new

    for it in range(n_iter):
        acc = {k: np.zeros(cur[k].table.shape) for k in ms.INTERNAL}
        bn = ms.assemble_bn(cur, sensor_source=sensor_cpts)
        for row in data_partial:
            ev = {v: row[v] for v in obs_vars if v in row}
            pH = bn.query(["H"], ev).table
            for hk in range(ms.CARD["H"]):
                acc["H"][(row["v"], row["rdot"], row["d"], hk)] += pH[hk]
            pcls_emit = bn.query(["cls", "emit"], ev).table
            for ci in range(ms.CARD["cls"]):
                for ei in range(ms.CARD["emit"]):
                    w = pcls_emit[ci, ei]
                    if w <= 1e-12:
                        continue
                    pN = bn.query(["N"], {**ev, "cls": ci, "emit": ei}).table
                    acc["N"][ci, ei, :] += w * pN
            pC = bn.query(["C"], ev).table
            for ck in range(ms.CARD["C"]):
                acc["C"][(row["d"], row["z"], ck)] += pC[ck]
            pHNC = bn.query(["H", "N", "C"], ev).table
            acc["T"][:, :, :, row["T"]] += pHNC
        cur = m_step(acc, final=(it == n_iter - 1))
    return cur


def _cmap_from_counts(node, n, pm, ess):
    """cmap_cpt on externally supplied (expected) counts."""
    import cvxpy as cp
    pa = ms.PARENTS[node]
    K = ms.CARD[node]
    Jshape = n.shape[:-1]
    J = int(np.prod(Jshape)) if pa else 1
    c = (n + ess * pm).reshape(J, K)
    th = cp.Variable((J, K), nonneg=True)
    cons = [cp.sum(th, axis=1) == 1]
    pairs, _ = _covering_pairs(node)
    if pairs:
        pj = [p[0] for p in pairs]
        pjp = [p[1] for p in pairs]
        for m in range(1, K):
            cons.append(cp.sum(th[pjp, m:], axis=1) >= cp.sum(th[pj, m:], axis=1))
    prob = cp.Problem(cp.Maximize(cp.sum(cp.multiply(c, cp.log(th)))), cons)
    solved = False
    for solver in ("CLARABEL", "ECOS", "SCS"):
        try:
            prob.solve(solver=solver)
        except Exception:
            continue
        if prob.status in ("optimal", "optimal_inaccurate") and th.value is not None:
            solved = True
            break
    X = np.clip(th.value, 1e-9, None) if solved else c
    if solved:
        CMAP_STATS["solved"] += 1
    else:
        CMAP_STATS["fallback"] += 1
    X = X / X.sum(axis=1, keepdims=True)
    return Factor(tuple(pa) + (node,), X.reshape(tuple(Jshape) + (K,)))


if __name__ == "__main__":
    import dataio
    from exp_constraints import mono_violations
    gt = ms.ground_truth_bn()
    data = dataio.generate_dataset(gt, 40, seed=5)
    rng = np.random.default_rng(0)
    prior = ms.expert_prior_cpts(rng)
    for node in ["T"]:
        mle = mle_cpt(node, data)
        mp = map_cpt(node, data, prior[node], 8.0)
        cm = cmap_cpt(node, data, prior[node], 8.0)
        nmx = noisymax_cpt(node, data, prior[node], 8.0)
        true = gt.cpts[node].table

        def kl(a):
            a2 = a.table.reshape(-1, ms.CARD[node])
            b = true.reshape(-1, ms.CARD[node])
            return np.mean(np.sum(b * (np.log(b + 1e-9) - np.log(a2 + 1e-9)), axis=1))
        print(f"{node}: KL MLE={kl(mle):.3f} MAP={kl(mp):.3f} "
              f"cMAP={kl(cm):.3f} NoisyMAX={kl(nmx):.3f}")
        print(f"  mono-violation rate: MAP={mono_violations(mp):.3f} "
              f"cMAP={mono_violations(cm):.3f} NM={mono_violations(nmx):.3f}")
        print(f"  NM params={nm_param_count(node)} vs full="
              f"{int(np.prod([ms.CARD[p] for p in ms.PARENTS[node]])*(ms.CARD[node]-1))}")
    print("cmap solver:", cmap_report())
    print("grounding v2 OK")
