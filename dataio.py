"""
Synthetic scenario generator (a runnable stand-in for an AirSim/Gazebo digital
twin) and the sensor-evidence layer (confusion-matrix likelihoods, context-
dependent reliability discounting, virtual evidence, missing sensors).

The generator samples labelled scenarios from the ground-truth BN; each
scenario has observed tracker/context evidence (v,rdot,d,z), sensor reports
(Rr,Re,Ra,Rf), latent (cls,emit,H,N,C) and the threat label T. Real AirSim /
public-dataset logs can replace `generate_dataset` and the confusion matrices
without touching the rest of the pipeline.
"""
from __future__ import annotations
import numpy as np
from cuas_bn import Factor, BayesNet, sample_ancestral
import model_spec as ms

# ---- reliability alpha_s as a function of observed range d (far,near,imminent)
ALPHA_BY_D = {
    "Rr": [0.90, 0.95, 0.95],   # radar: range-robust
    "Re": [0.40, 0.85, 0.90],   # EO/IR: needs pixels -> unreliable far
    "Ra": [0.25, 0.80, 0.90],   # acoustic: short range
    "Rf": [0.70, 0.70, 0.70],   # RF: range-robust (silent-blindness is in CPT)
}


def generate_dataset(bn: BayesNet, n: int, seed: int):
    rng = np.random.default_rng(seed)
    return sample_ancestral(bn, rng, n)   # list of full assignment dicts


def estimate_confusion(calib_data, smoothing=0.5):
    """Estimate each sensor's confusion-matrix CPT P(R_s | parents) by counting
    over a LABELLED calibration subset of the same world (Dirichlet/Laplace
    smoothing). This is the real procedure -- identical whether the labelled
    data come from the synthetic world, AirSim, or a public dataset -- and it
    replaces using the (unknown in practice) true sensor CPTs. Unseen parent
    configurations fall back to a smoothed uniform (uninformative)."""
    out = {}
    for s in ms.SENSORS:
        pa = ms.PARENTS[s]
        K = ms.CARD[s]
        shape = tuple(ms.CARD[p] for p in pa) + (K,)
        n = np.zeros(shape)
        for row in calib_data:
            idx = tuple(row[p] for p in pa) + (row[s],)
            n[idx] += 1
        n = n + smoothing
        t = n / n.sum(axis=-1, keepdims=True)
        out[s] = Factor(tuple(pa) + (s,), t)
    return out


# ---------- sensor CPT transforms ----------
def _discount_cpt(cpt: Factor, alpha: float) -> Factor:
    """P'(R|.) = alpha*P(R|.) + (1-alpha)*uniform over the report (last) axis."""
    K = cpt.table.shape[-1]
    table = alpha * cpt.table + (1 - alpha) * (1.0 / K)
    # renormalise last axis (already sums to 1, but guard)
    table = table / table.sum(axis=-1, keepdims=True)
    return Factor(cpt.vars, table)


def abstract_sensor_cpts(base_bn: BayesNet):
    """B0's 'abstract sensor model': over-confident identity-like mapping that
    assumes each report equals the true class with prob 0.94 (ignores real
    confusion / silent-drone blindness)."""
    out = {}
    # map report states to cls states where a natural correspondence exists
    corr = {
        "Rr": {0: 2, 1: 1, 2: 0},              # drone<-cls drone(2), bird<-1, clutter<-0
        "Re": {0: 2, 1: 1, 2: 0},              # drone,bird,other
        "Ra": {0: 2},                           # drone<-cls drone; none-> everything else
        "Rf": {0: 2, 1: 2, 2: 0},              # consumer/mil<-drone; none<-clutter
    }
    for s in ms.SENSORS:
        pa = ms.PARENTS[s]
        K = ms.CARD[s]
        shape = tuple(ms.CARD[p] for p in pa) + (K,)
        tab = np.full(shape, (1 - 0.94) / (K - 1))
        # set the 'expected' report per cls to 0.94
        it = np.ndindex(*[ms.CARD[p] for p in pa])
        for cfg in it:
            clsval = cfg[pa.index("cls")]
            # pick report state that this abstract model thinks corresponds
            rep = None
            for r, c in corr[s].items():
                if c == clsval:
                    rep = r; break
            if rep is None:
                rep = K - 1  # default 'none/other'
            row = np.full(K, (1 - 0.94) / (K - 1))
            row[rep] = 0.94
            tab[cfg] = row
        out[s] = Factor(tuple(pa) + (s,), tab)
    return out


def build_inference_bn(internal_cpts, sensors_on, reliability, d_value,
                       sensor_mode="calibrated", sensor_cpts=None):
    """Assemble a BN for inference with the chosen internal CPTs and a sensor
    layer configured per experiment:
      sensor_mode: 'calibrated' (confusion matrices ESTIMATED from a labelled
                   calibration set via `sensor_cpts`; falls back to the true
                   matrices only if `sensor_cpts` is None -- an oracle sanity
                   mode), or 'abstract' (B0 over-confident identity model).
      reliability: if True, discount each sensor CPT by alpha_s(d_value).
      sensors_on : subset of SENSORS to include; others are removed (missing
                   sensor -> auto-marginalised).
    """
    gt = ms.ground_truth_bn()
    cpts = dict(gt.cpts)
    for k in ms.INTERNAL:
        cpts[k] = internal_cpts[k]
    if sensor_mode == "abstract":
        src = abstract_sensor_cpts(gt)
    elif sensor_cpts is not None:
        src = sensor_cpts               # estimated from calibration data
    else:
        src = {s: gt.cpts[s] for s in ms.SENSORS}   # oracle (sanity only)
    # remove all sensor nodes first
    for s in ms.SENSORS:
        cpts.pop(s, None)
    parents = {k: v for k, v in ms.PARENTS.items()}
    card = dict(ms.CARD)
    keep_parents = dict(parents)
    # add back only sensors_on, discounted if requested
    for s in sensors_on:
        c = src[s]
        if reliability:
            c = _discount_cpt(c, ALPHA_BY_D[s][d_value])
        cpts[s] = c
    # drop removed sensors from parents/card map for a clean net
    net_parents = {k: parents[k] for k in cpts}
    net_card = {k: card[k] for k in cpts}
    return BayesNet(net_card, net_parents, cpts)


def evidence_from_scenario(scn, sensors_on, use_soft_eoir=False, re_cpt=None):
    """Return (evidence, virtual) dicts for a scenario.
    Hard sensor reports go in `evidence`. If use_soft_eoir, the EO/IR report is
    injected as virtual/likelihood evidence on `cls` (a soft classifier score
    derived from the ESTIMATED Re confusion column, `re_cpt`), instead of a hard
    label; the ablation (-virtual) sets use_soft_eoir=False.
    """
    ev = {k: scn[k] for k in ms.OBSERVED}
    virt = {}
    for s in sensors_on:
        if s == "Re" and use_soft_eoir:
            reP = (re_cpt.table if re_cpt is not None
                   else ms.ground_truth_bn().cpts["Re"].table)   # (cls, Re)
            r = scn["Re"]
            lik = reP[:, r].copy()             # L(cls) ∝ P(Re=r|cls)
            lik = lik / lik.sum()
            virt["cls"] = lik
        else:
            ev[s] = scn[s]
    return ev, virt


if __name__ == "__main__":
    bn = ms.ground_truth_bn()
    data = generate_dataset(bn, 2000, seed=1)
    T = np.array([d["T"] for d in data])
    print("threat label dist:", np.round(np.bincount(T, minlength=4) / len(T), 3))
    # oracle: ground-truth model predicting its own data with all sensors
    correct = 0
    for scn in data[:500]:
        ibn = build_inference_bn({k: bn.cpts[k] for k in ms.INTERNAL},
                                 ms.SENSORS, reliability=False, d_value=scn["d"])
        ev, virt = evidence_from_scenario(scn, ms.SENSORS)
        post = ibn.query(["T"], ev, virt).table
        correct += (post.argmax() == scn["T"])
    print("oracle (true CPTs) threat accuracy on 500:", correct / 500)
    # missing sensor + reliability sanity
    scn = data[0]
    ibn = build_inference_bn({k: bn.cpts[k] for k in ms.INTERNAL}, ["Rr", "Ra"],
                             reliability=True, d_value=scn["d"])
    ev, virt = evidence_from_scenario(scn, ["Rr", "Ra"])
    print("2-sensor+reliability P(T)=", np.round(ibn.query(["T"], ev, virt).table, 3),
          " true T=", scn["T"])
    print("dataio OK")
