"""
Non-monotone world option (estimator study exp_estimator_consistency.py; reused
by the misspecification study exp_misspecification.py). The DEFAULT world is
model_spec.ground_truth_bn() itself, returned unchanged; model_spec is not
modified (the consistency gate's freshness guard would otherwise invalidate
every published result).

World 'H-v-U'. The hostile-maneuver node H responds to the speed v in a U
shape: hovering and fast targets are threatening, slow ones are not. Exact
definition: the H table is built by the same ordinal-response construction as
model_spec._ordinal_cpt for H (parents v, rdot, d; weights v 0.9, rdot 1.0,
d 1.1; bias 0.05; tau = model_spec.TAU_HC; mean level
mu = bias + sum_p w_p s_p(x_p) / sum_p |w_p|, centre clip(mu, 0, 1)(K-1),
P(H = k) proportional to exp(-(k - centre)^2 / (2 tau^2))), except that the
speed score is s_v(hover, slow, fast) = (1, 0, 1) instead of the monotone
s_v = v / (|v| - 1) = (0, 0.5, 1). The rdot and d scores are unchanged
(x / (|x| - 1)). Every other CPT, including T | H, N, C and all sensors, is
byte-identical to the standard world (`identity_report`).

Consequence: the true H table violates the monotonicity constraints of
grounding.cmap_cpt along v (hover -> slow lowers H), so the constrained
estimator converges to the KL projection of the truth onto the monotone set
(the pseudo-true value), not to the truth.
"""
from __future__ import annotations
from itertools import product
import numpy as np
from cuas_bn import Factor, BayesNet
import model_spec as ms

WORLDS = ("standard", "H-v-U")
H_SPEC = {"weights": {"v": 0.9, "rdot": 1.0, "d": 1.1}, "bias": 0.05}


def _scores(node, override=None):
    """Per-parent score arrays; default x / (|x| - 1) as in _ordinal_cpt."""
    out = {}
    for p in ms.PARENTS[node]:
        if override and p in override:
            out[p] = list(override[p])
        else:
            out[p] = [val / max(ms.CARD[p] - 1, 1) for val in range(ms.CARD[p])]
    return out


def ordinal_table(node, weights, bias, tau, scores):
    """_ordinal_cpt's construction with explicit parent scores (same
    arithmetic in the same order, so the standard scores reproduce it
    byte for byte)."""
    pa = ms.PARENTS[node]
    K = ms.CARD[node]
    w = dict(weights)
    table = np.zeros(tuple(ms.CARD[p] for p in pa) + (K,))
    wsum = sum(abs(x) for x in w.values()) or 1.0
    for cfg in product(*[range(ms.CARD[p]) for p in pa]):
        acc = 0.0
        for p, val in zip(pa, cfg):
            acc += w.get(p, 0.0) * scores[p][val]
        mu = bias + acc / wsum
        center = np.clip(mu, 0, 1) * (K - 1)
        ks = np.arange(K)
        logits = -((ks - center) ** 2) / (2 * tau ** 2)
        row = np.exp(logits - logits.max())
        row /= row.sum()
        table[cfg] = row
    return table


def world_bn(world="standard"):
    gt = ms.ground_truth_bn()
    if world == "standard":
        return gt
    if world != "H-v-U":
        raise ValueError(world)
    cpts = dict(gt.cpts)
    tab = ordinal_table("H", H_SPEC["weights"], H_SPEC["bias"], ms.TAU_HC,
                        _scores("H", {"v": (1.0, 0.0, 1.0)}))
    cpts["H"] = Factor(("v", "rdot", "d", "H"), tab)
    return BayesNet(ms.CARD, ms.PARENTS, cpts)


def identity_report(world="H-v-U"):
    a, b = ms.ground_truth_bn(), world_bn(world)
    assert list(a.card) == list(b.card) and a.parents == b.parents
    return [v for v in a.card
            if a.cpts[v].vars != b.cpts[v].vars
            or a.cpts[v].table.tobytes() != b.cpts[v].table.tobytes()]


if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8")
    std = ordinal_table("H", H_SPEC["weights"], H_SPEC["bias"], ms.TAU_HC,
                        _scores("H"))
    ref = ms.ground_truth_bn().cpts["H"].table
    assert std.tobytes() == ref.tobytes(), "standard construction not reproduced"
    print("standard H reproduced byte for byte")
    diff = identity_report()
    print("H-v-U differs from standard only in:", diff)
    assert diff == ["H"], diff
    u = world_bn("H-v-U").cpts["H"].table
    # mean H level by speed at rdot = steady, d = near: U shape expected
    mean = (u[:, 1, 1, :] * np.arange(3)).sum(-1)
    print("E[H | v, rdot=steady, d=near] (hover, slow, fast):", np.round(mean, 3))
    assert mean[0] > mean[1] < mean[2]
    print("world_nonmonotone OK")
