"""
Misspecified and shifted worlds for the robustness study (exp_misspecification.py).
The DEFAULT world is model_spec.ground_truth_bn() itself; model_spec is
not modified (the consistency gate's freshness guard). The non-monotone world
of S2 is world_nonmonotone.py ('H-v-U'); the structural sensor-degradation
world is world_options.py (Table 8) and is not rerun here.

All ordinal tables use the construction of model_spec._ordinal_cpt: parent
scores s_p in [0, 1] (default s_p(x) = x / (|p| - 1)), a mean level mu, centre
clip(mu, 0, 1)(K - 1), and P(X = k) proportional to
exp(-(k - centre)^2 / (2 tau^2)). With the standard settings each builder
reproduces the standard table byte for byte (self-test).

S1  non-additive threat response ('nonadd-<lam>'). T | H, N, C with
    mu = (1 - lam) * A + lam * M, where
      A = (1.1 s_H + 1.0 s_N + 1.0 s_C) / 3.1   (the standard additive level,
                                               bias 0)
      M = max(s_H, s_N, s_C)                    (driven by the most
                                               threatening latent factor)
    tau = TAU_T. lam = 0.5 (weak) or 1.0 (strong). Monotone in every parent
    (both A and M are), so the monotonicity constraints still hold; only the
    additive (cumulative-logit ICI) form is wrong. lam = 0 is the standard T.
S3  edges absent from the model ('edges-<w>'). The world adds d -> N and
    emit -> H:
      N | cls, emit, d    weights cls 1.4, emit 0.9, d w   (bias -0.10)
      H | v, rdot, d, emit weights v 0.9, rdot 1.0, d 1.1, emit w (bias 0.05)
    (tau = TAU_HC; emit's threat-increasing direction is silent). w = 0.5
    (weak) or 1.5 (strong). The model keeps H | v, rdot, d and N | cls, emit.
    The sampling order changes with the parents, so S3 draws are not paired
    with the standard world's.
S4  test-time shifts (training and calibration stay in the standard world):
    'more-drones'    P(cls) = (0.20, 0.20, 0.60)       (standard 0.40/0.25/0.35)
    'more-silent'    P(emit) = (0.20, 0.80)            (standard 0.55/0.45)
    'sensor-x1.5'    range degradation gamma x 1.5, capped at 1:
                     radar .18/.045/0, EO/IR .675/.12/0, acoustic .75/.18/0
    'C-weight-x2'    T weights H 1.1, N 1.0, C 2.0 (concept shift)
    'tauT-0.60'      tau_T = 0.60 instead of 0.40 (concept shift)
emit-drone-only  N | cls, emit with emission acting on drones only: for clutter
    and birds both emit rows take the emitting row (exp_comparators.py).
"""
from __future__ import annotations
from itertools import product
import numpy as np
from cuas_bn import Factor, BayesNet
import model_spec as ms

T_W = {"H": 1.1, "N": 1.0, "C": 1.0}
H_W = {"v": 0.9, "rdot": 1.0, "d": 1.1}
N_W = {"cls": 1.4, "emit": 0.9}
WORLDS_S1 = ("nonadd-0.5", "nonadd-1.0")
WORLDS_S3 = ("edges-0.5", "edges-1.5")
WORLDS_S4 = ("more-drones", "more-silent", "sensor-x1.5", "C-weight-x2",
             "tauT-0.60")


def _score(p, x):
    return x / max(ms.CARD[p] - 1, 1)


def _row(mu, K, tau):
    center = np.clip(mu, 0, 1) * (K - 1)
    ks = np.arange(K)
    logits = -((ks - center) ** 2) / (2 * tau ** 2)
    row = np.exp(logits - logits.max())
    row /= row.sum()
    return row


def ordinal_table(parents, K, weights, bias, tau):
    """_ordinal_cpt's construction for an explicit parent list (same
    arithmetic in the same order)."""
    w = dict(weights)
    wsum = sum(abs(x) for x in w.values()) or 1.0
    table = np.zeros(tuple(ms.CARD[p] for p in parents) + (K,))
    for cfg in product(*[range(ms.CARD[p]) for p in parents]):
        acc = 0.0
        for p, val in zip(parents, cfg):
            acc += w.get(p, 0.0) * _score(p, val)
        table[cfg] = _row(bias + acc / wsum, K, tau)
    return table


def nonadditive_T(lam, tau=None):
    tau = ms.TAU_T if tau is None else tau
    pa = ["H", "N", "C"]
    K = ms.CARD["T"]
    wsum = sum(abs(x) for x in T_W.values())
    table = np.zeros(tuple(ms.CARD[p] for p in pa) + (K,))
    for cfg in product(*[range(ms.CARD[p]) for p in pa]):
        acc = 0.0
        for p, val in zip(pa, cfg):
            acc += T_W[p] * _score(p, val)
        additive = 0.0 + acc / wsum
        mx = max(_score(p, val) for p, val in zip(pa, cfg))
        mu = additive if lam == 0 else (1 - lam) * additive + lam * mx
        table[cfg] = _row(mu, K, tau)
    return table


def world_bn(world="standard"):
    gt = ms.ground_truth_bn()
    if world == "standard":
        return gt
    cpts = dict(gt.cpts)
    parents = {k: list(v) for k, v in ms.PARENTS.items()}
    if world.startswith("nonadd-"):
        lam = float(world.split("-")[1])
        cpts["T"] = Factor(("H", "N", "C", "T"), nonadditive_T(lam))
    elif world.startswith("edges-"):
        w = float(world.split("-")[1])
        parents["N"] = ["cls", "emit", "d"]
        parents["H"] = ["v", "rdot", "d", "emit"]
        cpts["N"] = Factor(("cls", "emit", "d", "N"),
                           ordinal_table(parents["N"], ms.CARD["N"],
                                         {**N_W, "d": w}, -0.10, ms.TAU_HC))
        cpts["H"] = Factor(("v", "rdot", "d", "emit", "H"),
                           ordinal_table(parents["H"], ms.CARD["H"],
                                         {**H_W, "emit": w}, 0.05, ms.TAU_HC))
    elif world == "emit-drone-only":
        tab = gt.cpts["N"].table.copy()
        drone = ms.STATES["cls"].index("drone")
        for c in range(ms.CARD["cls"]):
            if c != drone:
                tab[c, 1] = tab[c, 0]
        cpts["N"] = Factor(gt.cpts["N"].vars, tab)
    elif world == "more-drones":
        cpts["cls"] = Factor(("cls",), np.array([0.20, 0.20, 0.60]))
    elif world == "more-silent":
        cpts["emit"] = Factor(("emit",), np.array([0.20, 0.80]))
    elif world == "sensor-x1.5":
        for s in ("Rr", "Re", "Ra"):
            base = gt.cpts[s].table[:, 2, :]          # gamma = 0 slice
            K = base.shape[-1]
            gam = [min(1.0, 1.5 * g) for g in ms.SENSOR_DEGRADE[s]]
            tab = np.stack([(1 - g) * base + g * (1.0 / K) for g in gam], axis=1)
            cpts[s] = Factor(("cls", "d", s), tab)
    elif world == "C-weight-x2":
        cpts["T"] = Factor(("H", "N", "C", "T"),
                           ordinal_table(["H", "N", "C"], ms.CARD["T"],
                                         {"H": 1.1, "N": 1.0, "C": 2.0}, 0.0,
                                         ms.TAU_T))
    elif world == "tauT-0.60":
        cpts["T"] = Factor(("H", "N", "C", "T"),
                           ordinal_table(["H", "N", "C"], ms.CARD["T"], T_W,
                                         0.0, 0.60))
    else:
        raise ValueError(world)
    return BayesNet(ms.CARD, parents, cpts)


def identity_report(world):
    a, b = ms.ground_truth_bn(), world_bn(world)
    return [v for v in a.card
            if a.parents[v] != b.parents[v] or a.cpts[v].vars != b.cpts[v].vars
            or a.cpts[v].table.tobytes() != b.cpts[v].table.tobytes()]


def self_test():
    gt = ms.ground_truth_bn()
    checks = [
        ("T additive (lam=0)", nonadditive_T(0.0), gt.cpts["T"].table),
        ("T via ordinal_table", ordinal_table(["H", "N", "C"], 4, T_W, 0.0,
                                               ms.TAU_T), gt.cpts["T"].table),
        ("H via ordinal_table", ordinal_table(["v", "rdot", "d"], 3, H_W, 0.05,
                                               ms.TAU_HC), gt.cpts["H"].table),
        ("N via ordinal_table", ordinal_table(["cls", "emit"], 2, N_W, -0.10,
                                               ms.TAU_HC), gt.cpts["N"].table),
    ]
    for name, a, b in checks:
        assert a.tobytes() == b.tobytes(), name
    sx = world_bn("sensor-x1.5")
    std = {s: [(1 - g) * gt.cpts[s].table[:, 2, :]
               + g * (1.0 / gt.cpts[s].table.shape[-1])
               for g in ms.SENSOR_DEGRADE[s]] for s in ("Rr", "Re", "Ra")}
    for s in ("Rr", "Re", "Ra"):          # gamma x1 must give the standard table
        tab = np.stack(std[s], axis=1)
        assert tab.tobytes() == gt.cpts[s].table.tobytes(), s
        assert np.allclose(sx.cpts[s].table.sum(-1), 1.0)
    return len(checks) + 3


if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8")
    n = self_test()
    print(f"self-test: {n} byte-identity checks passed")
    for w in WORLDS_S1 + WORLDS_S3 + WORLDS_S4:
        print(f"{w:12s} differs from standard in: {identity_report(w)}")
    t1 = world_bn("nonadd-1.0").cpts["T"].table
    print("nonadd-1.0 E[T | H, N, C] at (hi, coop, lo) vs (med, coop, med):",
          round(float((t1[2, 0, 0] * np.arange(4)).sum()), 3),
          round(float((t1[1, 0, 1] * np.arange(4)).sum()), 3))
    print("world_misspec OK")
