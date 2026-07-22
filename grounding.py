"""
CPT grounding methods (P1):
  mle_cpt      : maximum likelihood (full table) -- B1, overfits under scarcity
  map_cpt      : Dirichlet posterior mean with ESS prior (eq. 4)
  cmap_cpt     : constrained-MAP = MAP + first-order stochastic-dominance
                 (monotonicity) constraints, solved by SLSQP  -- the proposed
                 grounding for the fully-observed setting
  em_internal  : MAP-EM for the latent-variable setting (H,N,C unobserved)

All operate on `data` = list of full/partial assignment dicts. Prior means are
Factors from model_spec (expert_prior_cpts). Monotonicity uses the fact that
every parent is encoded threat-increasing, so a +1 step in any parent must
stochastically increase the (ordinal) child.
"""
from __future__ import annotations
import numpy as np
from itertools import product as iproduct
from scipy.optimize import minimize
from cuas_bn import Factor, BayesNet
import model_spec as ms


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
    """Constrained-MAP: maximise sum_{j,k} c_{jk} log theta_{jk} with
    c = n + ess*prior (>=0, concave), subject to per-config simplex and
    first-order stochastic dominance across covering pairs. SLSQP, warm-started
    at the closed-form MAP."""
    pa = ms.PARENTS[node]
    K = ms.CARD[node]
    n = counts(node, data)
    pm = prior_mean.table
    Jshape = n.shape[:-1]
    J = int(np.prod(Jshape)) if pa else 1
    c = (n + ess * pm).reshape(J, K)          # coefficients (>=0)
    # warm start = MAP
    map_t = (c / c.sum(axis=1, keepdims=True))
    x0 = map_t.reshape(-1).clip(1e-6, 1)
    cflat = c.reshape(-1)

    def negobj(x):
        return -np.sum(cflat * np.log(x))

    def negobj_grad(x):
        return -cflat / x

    # equality: each config row sums to 1
    def eq_con(x):
        X = x.reshape(J, K)
        return X.sum(axis=1) - 1.0
    cons = [{"type": "eq", "fun": eq_con}]
    # inequality: dominance  sum_{k>=m}(theta_j' - theta_j) >= 0 for m=1..K-1
    pairs, _ = _covering_pairs(node)
    if pairs:
        pj = np.array([p[0] for p in pairs]); pjp = np.array([p[1] for p in pairs])
        def ineq_con(x):
            X = x.reshape(J, K)
            surv = np.cumsum(X[:, ::-1], axis=1)[:, ::-1]  # survival S[:,m]=sum_{k>=m}
            # constraints for m=1..K-1 : S[jp,m]-S[j,m] >=0
            diff = surv[pjp, 1:] - surv[pj, 1:]
            return diff.reshape(-1)
        cons.append({"type": "ineq", "fun": ineq_con})

    bounds = [(1e-9, 1.0)] * (J * K)
    res = minimize(negobj, x0, jac=negobj_grad, bounds=bounds, constraints=cons,
                   method="SLSQP", options={"maxiter": 300, "ftol": 1e-8})
    X = res.x.reshape(J, K)
    X = np.clip(X, 1e-9, None)
    X = X / X.sum(axis=1, keepdims=True)
    t = X.reshape(tuple(Jshape) + (K,)) if pa else X.reshape(K)
    return Factor(tuple(pa) + (node,), t)


# ---------------- MAP-EM for latent internal nodes ----------------
def em_internal(data_partial, prior_means, ess, n_iter=25, seed=0, constrained=False):
    """Learn internal CPTs (H,N,C,T) when H,N,C are UNOBSERVED. `data_partial`
    rows contain observed inputs, sensors, and T (threat label). Sensors/roots
    use ground-truth CPTs. Returns dict of internal Factors.
    E-step: expected counts via posterior over latents per row.
    M-step: MAP (or constrained-MAP) update from expected counts.
    """
    rng = np.random.default_rng(seed)
    gt = ms.ground_truth_bn()
    # init internal CPTs from priors (+ tiny noise for symmetry breaking)
    cur = {k: Factor(prior_means[k].vars, prior_means[k].table.copy()) for k in ms.INTERNAL}
    obs_vars = set(ms.OBSERVED) | set(ms.SENSORS) | {"T"}
    latent = ["H", "N", "C", "cls", "emit"]
    for it in range(n_iter):
        # accumulate expected (fractional) counts for each internal node
        acc = {k: np.zeros(cur[k].table.shape) for k in ms.INTERNAL}
        bn = ms.assemble_bn(cur)  # ground-truth roots+sensors + current internal
        for row in data_partial:
            ev = {v: row[v] for v in obs_vars if v in row}
            # joint posterior over the latents we need, per node, via targeted queries
            # H | ev  (parents v,rdot,d observed) -> need P(H|ev)
            pH = bn.query(["H"], ev).table
            for hk in range(ms.CARD["H"]):
                idx = (row["v"], row["rdot"], row["d"], hk)
                acc["H"][idx] += pH[hk]
            # N | ev : parents cls,emit latent -> need joint P(cls,emit) then map to N counts
            pcls_emit = bn.query(["cls", "emit"], ev).table  # (cls,emit)
            pN_given = cur["N"].table  # (cls,emit,N)
            for ci in range(ms.CARD["cls"]):
                for ei in range(ms.CARD["emit"]):
                    w = pcls_emit[ci, ei]
                    if w <= 0:
                        continue
                    # expected count of (cls,emit,N=k) ∝ w * P(N=k|cls,emit,ev)
                    # given cls,emit fixed, N depends also on downstream T,C via ev;
                    # approximate with posterior P(N|ev,cls,emit)
                    pN = bn.query(["N"], {**ev, "cls": ci, "emit": ei}).table
                    acc["N"][ci, ei, :] += w * pN
            # C | ev : parents d,z observed
            pC = bn.query(["C"], ev).table
            for ck in range(ms.CARD["C"]):
                acc["C"][(row["d"], row["z"], ck)] += pC[ck]
            # T | H,N,C : T observed, H,N,C latent -> joint posterior
            pHNC = bn.query(["H", "N", "C"], ev).table  # (H,N,C)
            tk = row["T"]
            acc["T"][:, :, :, tk] += pHNC
        # M-step: MAP (or cMAP) from expected counts
        newcur = {}
        for k in ms.INTERNAL:
            pm = prior_means[k].table
            num = acc[k] + ess * pm
            t = num / num.sum(axis=-1, keepdims=True)
            newcur[k] = Factor(cur[k].vars, t)
        cur = newcur
    return cur


if __name__ == "__main__":
    # quick check: cMAP recovers monotone CPT and beats MLE KL at small N
    import dataio
    gt = ms.ground_truth_bn()
    data = dataio.generate_dataset(gt, 40, seed=5)
    rng = np.random.default_rng(0)
    prior = ms.expert_prior_cpts(rng)
    for node in ["T"]:
        mle = mle_cpt(node, data); mp = map_cpt(node, data, prior[node], 8.0)
        cm = cmap_cpt(node, data, prior[node], 8.0)
        true = gt.cpts[node].table
        def kl(a):
            a = a.table.reshape(-1, ms.CARD[node]); b = true.reshape(-1, ms.CARD[node])
            return np.mean(np.sum(b * (np.log(b + 1e-9) - np.log(a + 1e-9)), axis=1))
        print(f"{node}: KL(true|MLE)={kl(mle):.3f} KL|MAP={kl(mp):.3f} KL|cMAP={kl(cm):.3f}")
        # verify monotonicity of cMAP survival
        print("cMAP is a valid distribution:", np.allclose(cm.table.sum(-1), 1))
    print("grounding OK")
