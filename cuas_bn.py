"""
Discrete Bayesian-network core: factor algebra + exact inference (variable
elimination). Small, dependency-light (numpy only), for the C-UAS hierarchical
threat model. All CPTs are numpy arrays; latent variables are summed out.

A Factor holds:
  - vars : tuple of variable names (axis order matches `table`)
  - table: np.ndarray, shape = tuple(card[v] for v in vars)
Cardinalities are looked up from a global VARS dict provided by model_spec.
"""
from __future__ import annotations
import numpy as np
from itertools import product as iproduct


class Factor:
    __slots__ = ("vars", "table")

    def __init__(self, vars_, table):
        self.vars = tuple(vars_)
        self.table = np.asarray(table, dtype=float)
        assert self.table.ndim == len(self.vars), (self.vars, self.table.shape)

    def copy(self):
        return Factor(self.vars, self.table.copy())

    def __repr__(self):
        return f"Factor({self.vars}, shape={self.table.shape})"


def _align(f: Factor, all_vars):
    """Return a view of f.table broadcast to axis order `all_vars`."""
    shape = []
    axes = []
    for v in all_vars:
        if v in f.vars:
            axes.append(f.vars.index(v))
    # move f's axes into the order they appear in all_vars, then add new axes
    perm = [f.vars.index(v) for v in all_vars if v in f.vars]
    t = np.transpose(f.table, perm)
    # insert singleton dims for vars not in f
    full_shape = []
    idx = 0
    present = [v for v in all_vars if v in f.vars]
    for v in all_vars:
        if v in f.vars:
            full_shape.append(t.shape[present.index(v)])
        else:
            full_shape.append(1)
    return t.reshape(full_shape)


def factor_product(f1: Factor, f2: Factor, card) -> Factor:
    all_vars = list(f1.vars) + [v for v in f2.vars if v not in f1.vars]
    a = _align(f1, all_vars)
    b = _align(f2, all_vars)
    table = a * b
    # ensure full shape (broadcast) materialized
    target = tuple(card[v] for v in all_vars)
    table = np.broadcast_to(table, target).copy()
    return Factor(all_vars, table)


def factor_marginalize(f: Factor, var) -> Factor:
    if var not in f.vars:
        return f.copy()
    ax = f.vars.index(var)
    table = f.table.sum(axis=ax)
    newvars = tuple(v for v in f.vars if v != var)
    return Factor(newvars, table)


def factor_reduce(f: Factor, evidence: dict) -> Factor:
    """Fix observed variables to their values (slice)."""
    sl = [slice(None)] * len(f.vars)
    keepvars = []
    for i, v in enumerate(f.vars):
        if v in evidence:
            sl[i] = evidence[v]
        else:
            keepvars.append(v)
    table = f.table[tuple(sl)]
    return Factor(tuple(keepvars), table)


def normalize(f: Factor):
    s = f.table.sum()
    if s <= 0:
        # degenerate; return uniform
        t = np.ones_like(f.table)
        return Factor(f.vars, t / t.sum())
    return Factor(f.vars, f.table / s)


class BayesNet:
    """A discrete BN: variables (name->cardinality), parents, and CPTs.

    cpts[v] is a Factor over (parents(v) + [v]) i.e. P(v | parents), with `v`
    as the LAST axis. Root nodes have a Factor over [v] alone.
    """

    def __init__(self, card: dict, parents: dict, cpts: dict):
        self.card = dict(card)
        self.parents = {k: list(v) for k, v in parents.items()}
        self.cpts = cpts  # name -> Factor
        self.order = list(card.keys())

    # ---- inference ----
    def query(self, targets, evidence=None, virtual=None):
        """Return normalized Factor over `targets` given `evidence` (dict).

        Exact variable elimination. `targets` is a list of variable names.
        `virtual` is an optional dict {var: likelihood_vector} injecting
        virtual/likelihood (soft) evidence L(var) as an extra factor (the
        correct handling of soft classifier scores; not Jeffrey's rule).
        """
        evidence = dict(evidence or {})
        virtual = dict(virtual or {})
        targets = list(targets)
        factors = []
        for v in self.order:
            f = self.cpts[v]
            f = factor_reduce(f, evidence)
            if f.vars:  # may become scalar if all reduced
                factors.append(f)
        # inject virtual-evidence likelihood factors
        for var, lik in virtual.items():
            lik = np.asarray(lik, dtype=float)
            lf = Factor((var,), lik)
            lf = factor_reduce(lf, evidence)  # no-op unless var observed
            if lf.vars:
                factors.append(lf)
        keep = set(targets)
        # eliminate all vars not in targets and not in evidence
        elim = [v for v in self.order if v not in keep and v not in evidence]
        for v in elim:
            involved = [f for f in factors if v in f.vars]
            if not involved:
                continue
            rest = [f for f in factors if v not in f.vars]
            prod = involved[0]
            for f in involved[1:]:
                prod = factor_product(prod, f, self.card)
            prod = factor_marginalize(prod, v)
            factors = rest + [prod]
        # multiply remaining
        if not factors:
            # everything constant -> uniform over targets
            shape = tuple(self.card[t] for t in targets)
            return normalize(Factor(targets, np.ones(shape)))
        prod = factors[0]
        for f in factors[1:]:
            prod = factor_product(prod, f, self.card)
        # marginalize any stray vars not in targets (e.g. leftover)
        for v in list(prod.vars):
            if v not in keep:
                prod = factor_marginalize(prod, v)
        # reorder axes to `targets`
        perm = [prod.vars.index(t) for t in targets]
        prod = Factor(targets, np.transpose(prod.table, perm))
        return normalize(prod)

def sample_ancestral(bn: BayesNet, rng, n=1):
    """Sample n joint assignments via ancestral sampling (topological order).
    Returns list of dicts. Assumes `bn.order` is a valid topological order.
    """
    # topological sort by parents
    order = _toposort(bn.parents, list(bn.card.keys()))
    out = []
    for _ in range(n):
        assign = {}
        for v in order:
            pa = bn.parents[v]
            f = bn.cpts[v]
            # index CPT by parent values -> distribution over v (last axis)
            idx = tuple(assign[p] for p in f.vars[:-1])
            dist = f.table[idx] if idx else f.table
            dist = np.asarray(dist, dtype=float)
            dist = dist / dist.sum()
            assign[v] = int(rng.choice(len(dist), p=dist))
        out.append(assign)
    return out


def _toposort(parents, nodes):
    from collections import deque
    indeg = {n: 0 for n in nodes}
    children = {n: [] for n in nodes}
    for n in nodes:
        for p in parents.get(n, []):
            children[p].append(n)
            indeg[n] += 1
    q = deque([n for n in nodes if indeg[n] == 0])
    order = []
    while q:
        n = q.popleft()
        order.append(n)
        for c in children[n]:
            indeg[c] -= 1
            if indeg[c] == 0:
                q.append(c)
    assert len(order) == len(nodes), "cycle or missing node"
    return order


if __name__ == "__main__":
    # self-test: 2-node chain  A -> B, verify posterior P(A|B=b)
    card = {"A": 2, "B": 2}
    parents = {"A": [], "B": ["A"]}
    PA = Factor(("A",), np.array([0.7, 0.3]))
    PB_A = Factor(("A", "B"), np.array([[0.9, 0.1], [0.2, 0.8]]))  # P(B|A)
    bn = BayesNet(card, parents, {"A": PA, "B": PB_A})
    # P(A|B=1) ∝ P(A)P(B=1|A) = [0.7*0.1, 0.3*0.8] = [0.07,0.24] -> [0.2258,0.7742]
    post = bn.query(["A"], {"B": 1})
    print("P(A|B=1) =", np.round(post.table, 4), "(expect [0.2258 0.7742])")
    # marginal P(B)
    pb = bn.query(["B"])
    print("P(B) =", np.round(pb.table, 4), "(expect [0.69 0.31])")
    # virtual evidence on B with likelihood [0.1,0.8] should match reduce approx
    print("toposort:", _toposort(parents, list(card.keys())))
