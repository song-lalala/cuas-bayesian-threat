"""
C-UAS hierarchical threat model specification (structure + ground-truth CPTs,
expert-prior CPTs, heuristic baseline CPTs, and measured sensor confusion
matrices). Ground-truth CPTs are generated from monotone ordinal-response
functions so that threat is provably monotone in the threatening direction of
each parent -- which is exactly the domain knowledge the monotonicity
constraints in the proposed method exploit.

Structure (evidence -> latent -> threat):
  roots:  cls, emit, v, rdot, d, z
  H | v,rdot,d          (hostile maneuver)
  N | cls,emit          (non-cooperative track)
  C | d,z               (context-based threat)
  T | H,N,C             (threat level; monotone / weighted)
  sensors (children of cls; RF also of emit):
  Rr|cls  Re|cls  Ra|cls  Rf|cls,emit
Observed at test time: v,rdot,d,z and sensor reports; latent: cls,emit,H,N,C.
"""
from __future__ import annotations
import numpy as np
from cuas_bn import Factor, BayesNet

# ---- variables & cardinalities ----
STATES = {
    "cls":  ["clutter", "bird", "drone"],       # ordinal: drone most threat-relevant
    "emit": ["emitting", "silent"],
    "v":    ["hover", "slow", "fast"],
    "rdot": ["opening", "steady", "closing"],
    "d":    ["far", "near", "imminent"],
    "z":    ["permitted", "nofly"],
    "H":    ["lo", "med", "hi"],
    "N":    ["coop", "noncoop"],
    "C":    ["lo", "med", "hi"],
    "T":    ["negligible", "low", "medium", "high"],
    "Rr":   ["drone", "bird", "clutter"],
    "Re":   ["drone", "bird", "other"],
    "Ra":   ["drone", "none"],
    "Rf":   ["consumer", "mil", "none"],
}
CARD = {k: len(v) for k, v in STATES.items()}
TAU_HC = 0.28   # sharpness of H,N,C given their parents
TAU_T = 0.40    # sharpness of T given H,N,C (residual aleatoric threat noise)
PARENTS = {
    "cls": [], "emit": [], "v": [], "rdot": [], "d": [], "z": [],
    "H": ["v", "rdot", "d"],
    "N": ["cls", "emit"],
    "C": ["d", "z"],
    "T": ["H", "N", "C"],
    "Rr": ["cls"], "Re": ["cls"], "Ra": ["cls"], "Rf": ["cls", "emit"],
}
INTERNAL = ["H", "N", "C", "T"]           # CPTs grounded/learned in experiments
OBSERVED = ["v", "rdot", "d", "z"]         # tracker/context evidence
SENSORS = ["Rr", "Re", "Ra", "Rf"]
LATENT = ["cls", "emit", "H", "N", "C"]

# ordinal index of each state = its position (already threat-increasing)


def _ordinal_cpt(node, weights, bias, tau, rng=None, noise=0.0):
    """Build P(node | parents) via an ordinal-response model.
    Target mean level mu = bias + sum_p weights[p] * (parent_ordinal / (card_p-1)).
    P(node=k) proportional to exp(-((k-mu*(K-1))^2)/(2 tau^2)). Monotone in
    each parent whose weight>0. `noise` perturbs weights (for expert/heuristic).
    """
    pa = PARENTS[node]
    K = CARD[node]
    w = dict(weights)
    if noise and rng is not None:
        w = {p: weights[p] * (1 + noise * rng.standard_normal()) for p in weights}
        bias = bias + noise * 0.5 * rng.standard_normal()
    shape = tuple(CARD[p] for p in pa) + (K,)
    table = np.zeros(shape)
    ranges = [range(CARD[p]) for p in pa]
    from itertools import product as ip
    wsum = sum(abs(x) for x in w.values()) or 1.0
    for cfg in ip(*ranges) if pa else [()]:
        acc = 0.0
        for p, val in zip(pa, cfg):
            acc += w.get(p, 0.0) * (val / max(CARD[p] - 1, 1))
        mu = bias + acc / wsum          # weighted average in ~[0,1]
        center = np.clip(mu, 0, 1) * (K - 1)
        ks = np.arange(K)
        logits = -((ks - center) ** 2) / (2 * tau ** 2)
        row = np.exp(logits - logits.max())
        row /= row.sum()
        table[cfg] = row
    return Factor(tuple(pa) + (node,), table)


def _root_cpt(node, probs):
    return Factor((node,), np.asarray(probs, float))


# ---------------- GROUND TRUTH ----------------
def ground_truth_bn():
    cpts = {}
    # roots (spread so threat classes are reasonably populated)
    cpts["cls"]  = _root_cpt("cls",  [0.40, 0.25, 0.35])     # clutter,bird,drone
    cpts["emit"] = _root_cpt("emit", [0.55, 0.45])            # emitting,silent
    cpts["v"]    = _root_cpt("v",    [0.34, 0.33, 0.33])
    cpts["rdot"] = _root_cpt("rdot", [0.34, 0.33, 0.33])
    cpts["d"]    = _root_cpt("d",    [0.34, 0.33, 0.33])
    cpts["z"]    = _root_cpt("z",    [0.50, 0.50])
    # H | v,rdot,d  (threat-increasing in all three); TAU_HC sharp
    cpts["H"] = _ordinal_cpt("H", {"v": 0.9, "rdot": 1.0, "d": 1.1}, bias=0.05, tau=TAU_HC)
    # N | cls,emit  (drone & silent -> non-cooperative)
    cpts["N"] = _ordinal_cpt("N", {"cls": 1.4, "emit": 0.9}, bias=-0.10, tau=TAU_HC)
    # C | d,z
    cpts["C"] = _ordinal_cpt("C", {"d": 1.1, "z": 1.0}, bias=0.05, tau=TAU_HC)
    # T | H,N,C  (monotone increasing); TAU_T leaves genuine aleatoric noise
    cpts["T"] = _ordinal_cpt("T", {"H": 1.1, "N": 1.0, "C": 1.0}, bias=0.0, tau=TAU_T)
    # sensors
    cpts["Rr"] = _sensor_radar()
    cpts["Re"] = _sensor_eoir()
    cpts["Ra"] = _sensor_acoustic()
    cpts["Rf"] = _sensor_rf()
    return BayesNet(CARD, PARENTS, cpts)


# ---------------- SENSOR CONFUSION MATRICES (measured/literature-informed) ----
def _sensor_radar():
    # Rr|cls rows over {drone,bird,clutter}; cls order clutter,bird,drone
    tab = np.array([
        [0.15, 0.20, 0.65],   # cls=clutter
        [0.20, 0.70, 0.10],   # cls=bird
        [0.82, 0.13, 0.05],   # cls=drone  (good micro-Doppler ID)
    ])
    return Factor(("cls", "Rr"), tab)


def _sensor_eoir():
    # Re|cls over {drone,bird,other}
    tab = np.array([
        [0.10, 0.15, 0.75],   # clutter->other
        [0.12, 0.80, 0.08],   # bird
        [0.85, 0.10, 0.05],   # drone
    ])
    return Factor(("cls", "Re"), tab)


def _sensor_acoustic():
    # Ra|cls over {drone,none}; short range, decent on drone
    tab = np.array([
        [0.10, 0.90],   # clutter
        [0.18, 0.82],   # bird (some rotor-like)
        [0.78, 0.22],   # drone
    ])
    return Factor(("cls", "Ra"), tab)


def _sensor_rf():
    # Rf|cls,emit over {consumer,mil,none}. KEY: a SILENT drone -> 'none'
    # (RF is nearly blind to radio-silent/autonomous drones).
    tab = np.zeros((CARD["cls"], CARD["emit"], CARD["Rf"]))
    # clutter
    tab[0, 0] = [0.05, 0.03, 0.92]; tab[0, 1] = [0.03, 0.02, 0.95]
    # bird
    tab[1, 0] = [0.06, 0.02, 0.92]; tab[1, 1] = [0.03, 0.02, 0.95]
    # drone emitting -> consumer/mil; drone silent -> none (blind)
    tab[2, 0] = [0.70, 0.22, 0.08]; tab[2, 1] = [0.05, 0.03, 0.92]
    return Factor(("cls", "emit", "Rf"), tab)


# ---------------- EXPERT PRIOR (biased but monotone) ----------------
def expert_prior_cpts(rng, noise=0.18):
    """Return prior-MEAN CPTs for the internal nodes: biased/noisy version of
    truth, still monotone. Used as Dirichlet prior means (C2-i)."""
    cpts = {}
    cpts["H"] = _ordinal_cpt("H", {"v": 0.9, "rdot": 1.0, "d": 1.1}, 0.05, TAU_HC, rng, noise)
    cpts["N"] = _ordinal_cpt("N", {"cls": 1.4, "emit": 0.9}, -0.10, TAU_HC, rng, noise)
    cpts["C"] = _ordinal_cpt("C", {"d": 1.1, "z": 1.0}, 0.05, TAU_HC, rng, noise)
    cpts["T"] = _ordinal_cpt("T", {"H": 1.1, "N": 1.0, "C": 1.0}, 0.0, TAU_T, rng, noise)
    return {k: cpts[k] for k in INTERNAL}


# ---------------- HEURISTIC B0 (cruder hand-set; flat, rounded) ------------
def heuristic_cpts():
    """Coarse expert guess: rounded weights + larger tau (over-smoothed),
    the kind of hand-set table the base framework uses."""
    cpts = {}
    cpts["H"] = _ordinal_cpt("H", {"v": 0.7, "rdot": 0.7, "d": 0.7}, 0.1, 0.95)
    cpts["N"] = _ordinal_cpt("N", {"cls": 1.0, "emit": 0.6}, 0.0, 0.9)
    cpts["C"] = _ordinal_cpt("C", {"d": 0.8, "z": 0.8}, 0.1, 0.95)
    cpts["T"] = _ordinal_cpt("T", {"H": 0.8, "N": 0.8, "C": 0.8}, 0.05, 1.0)
    return {k: cpts[k] for k in INTERNAL}


def assemble_bn(internal_cpts, sensor_source=None):
    """Build a full BayesNet using given internal CPTs (H,N,C,T) and
    ground-truth roots + sensors (or provided sensor CPTs)."""
    gt = ground_truth_bn()
    cpts = dict(gt.cpts)
    for k in INTERNAL:
        cpts[k] = internal_cpts[k]
    if sensor_source is not None:
        for s in SENSORS:
            cpts[s] = sensor_source[s]
    return BayesNet(CARD, PARENTS, cpts)


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    bn = ground_truth_bn()
    # sanity: threat rises with threatening evidence
    from cuas_bn import sample_ancestral
    lo = bn.query(["T"], {"v": 0, "rdot": 0, "d": 0, "z": 0})  # hover/opening/far/permitted
    hi = bn.query(["T"], {"v": 2, "rdot": 2, "d": 2, "z": 1})  # fast/closing/imminent/nofly
    print("P(T| benign)  =", np.round(lo.table, 3))
    print("P(T| threat)  =", np.round(hi.table, 3))
    assert hi.table[-1] > lo.table[-1], "high-threat prob should increase"
    # RF-blindness: a silent drone closing on a no-fly zone
    # observe sensors consistent with a drone but RF='none'
    ev = {"v": 2, "rdot": 2, "d": 2, "z": 1, "Rr": 0, "Re": 0, "Ra": 0, "Rf": 2}
    post = bn.query(["T"], ev)
    pcls = bn.query(["cls"], ev)
    print("silent-drone-like evidence: P(T)=", np.round(post.table, 3),
          " P(cls)=", np.round(pcls.table, 3))
    print("model_spec OK")
