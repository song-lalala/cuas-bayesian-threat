"""
Synthetic scenario generator (world v2: range-dependent sensor degradation)
and the sensor-evidence layer of the deployed inference model.

World vs model: the GENERATIVE world's environment-sensitive sensors (radar,
EO/IR, acoustic) are d-conditional (model_spec.SENSOR_DEGRADE), while the
DEPLOYED inference model uses range-MARGINAL confusion tables estimated from
a labelled calibration split, plus a reliability layer alpha_s(d) fitted by
logistic regression on the same split (correct-vs-incorrect sensor decisions
against the range context). The structural mismatch is deliberate: it is the
situation the reliability discounting exists to absorb, and it makes the
reliability layer testable (cf. reviewer comment on vacuous reliability).

The generator samples labelled scenarios from the ground-truth BN; each
scenario has observed tracker/context evidence (v,rdot,d,z), sensor reports
(Rr,Re,Ra,Rf), latent (cls,emit,H,N,C) and the threat label T.
"""
from __future__ import annotations
import numpy as np
from cuas_bn import Factor, BayesNet, sample_ancestral
import model_spec as ms

# parents of each sensor in the DEPLOYED (range-marginal) inference model
MARGINAL_PARENTS = {"Rr": ["cls"], "Re": ["cls"], "Ra": ["cls"],
                    "Rf": ["cls", "emit"]}

# "natural" correct report per true class (used by the reliability fit and
# the abstract model): index of the report that names the class
CORRECT_REPORT = {
    "Rr": {0: 2, 1: 1, 2: 0},   # clutter->'clutter', bird->'bird', drone->'drone'
    "Re": {0: 2, 1: 1, 2: 0},
    "Ra": {0: 1, 1: 1, 2: 0},   # non-drone->'none', drone->'drone'
    "Rf": None,                  # handled specially (emission-dependent)
}


def generate_dataset(bn: BayesNet, n: int, seed: int):
    rng = np.random.default_rng(seed)
    return sample_ancestral(bn, rng, n)   # list of full assignment dicts


# ---------------- confusion-matrix estimation ----------------
def _count_cpt(calib_data, s, parents, smoothing):
    K = ms.CARD[s]
    shape = tuple(ms.CARD[p] for p in parents) + (K,)
    n = np.zeros(shape)
    for row in calib_data:
        idx = tuple(row[p] for p in parents) + (row[s],)
        n[idx] += 1
    n = n + smoothing
    return Factor(tuple(parents) + (s,), n / n.sum(axis=-1, keepdims=True))


def estimate_confusion(calib_data, smoothing=0.5):
    """DEPLOYED-pipeline estimate: each sensor's range-MARGINAL confusion CPT
    P(R_s | class parents), counted over a LABELLED calibration subset of the
    same world (Dirichlet/Laplace smoothing). Identical procedure whether the
    labelled data come from the synthetic world, AirSim, or a public
    dataset."""
    return {s: _count_cpt(calib_data, s, MARGINAL_PARENTS[s], smoothing)
            for s in ms.SENSORS}


def estimate_confusion_conditional(calib_data, smoothing=0.5):
    """Range-CONDITIONAL estimate (upper-reference model that conditions on
    d, i.e. learns the world's true structure)."""
    return {s: _count_cpt(calib_data, s, ms.PARENTS[s], smoothing)
            for s in ms.SENSORS}


def true_marginal_confusion():
    """Range-marginal of the TRUE sensor CPTs under the true P(d) (sensor
    oracle for the deployed model structure)."""
    gt = ms.ground_truth_bn()
    pd = gt.cpts["d"].table
    out = {}
    for s in ms.SENSORS:
        f = gt.cpts[s]
        if "d" in f.vars:
            ax = f.vars.index("d")
            tab = np.tensordot(f.table, pd, axes=([ax], [0]))
            vars_ = tuple(v for v in f.vars if v != "d")
            out[s] = Factor(vars_, tab)
        else:
            out[s] = Factor(f.vars, f.table.copy())
    return out


# ---------------- reliability layer: logistic fit on the calibration split --
def _is_correct(row, s):
    if s == "Rf":
        # emitting drone -> 'consumer' or 'mil'; silent drone & non-drone -> 'none'
        if row["cls"] == 2 and row["emit"] == 0:
            return row[s] in (0, 1)
        return row[s] == 2
    return row[s] == CORRECT_REPORT[s][row["cls"]]


def fit_reliability(calib_data, floor=0.05, ceil=0.98):
    """Fit alpha_s = sigma(w_s^T phi(d) + b_s) by logistic regression of
    correct-vs-incorrect sensor decisions on the range context (one-hot d),
    exactly the Sec. V-B procedure instantiated on the features the synthetic
    world exposes. Returns ({s: alpha array over d}, {s: (w, b)}).
    alpha is rescaled from P(correct) to a discount weight by mapping each
    sensor's best-range accuracy to `ceil` (a sensor at its own best range is
    treated as fully reliable; degradation below that lowers alpha)."""
    from sklearn.linear_model import LogisticRegression
    alphas, params = {}, {}
    for s in ms.SENSORS:
        X = np.array([[1.0 if row["d"] == k else 0.0 for k in (1, 2)]
                      for row in calib_data])          # far = reference level
        y = np.array([1.0 if _is_correct(row, s) else 0.0
                      for row in calib_data])
        if y.min() == y.max():                          # degenerate split
            p = np.full(3, y.mean())
            w, b = np.zeros(2), float(np.log(max(y.mean(), 1e-6) /
                                             max(1 - y.mean(), 1e-6)))
        else:
            lr = LogisticRegression(C=10.0).fit(X, y)
            Xd = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
            p = lr.predict_proba(Xd)[:, 1]
            w, b = lr.coef_[0].copy(), float(lr.intercept_[0])
        a = p / max(p.max(), 1e-6) * ceil
        alphas[s] = np.clip(a, floor, ceil)
        params[s] = (w, b)
    return alphas, params


# ---------- sensor CPT transforms ----------
def _discount_cpt(cpt: Factor, alpha: float) -> Factor:
    """P'(R|.) = alpha*P(R|.) + (1-alpha)*uniform over the report (last) axis."""
    K = cpt.table.shape[-1]
    table = alpha * cpt.table + (1 - alpha) * (1.0 / K)
    table = table / table.sum(axis=-1, keepdims=True)
    return Factor(cpt.vars, table)


def abstract_sensor_cpts():
    """B0's 'abstract sensor model': over-confident identity-like mapping that
    assumes each report equals the natural report for the true class with
    probability `diag` (default 0.94), ignoring measured confusion and
    silent-drone blindness. Range-marginal parents. `diag` is swept in the
    baseline-sensitivity study."""
    return abstract_sensor_cpts_diag(0.94)


def abstract_sensor_cpts_diag(diag):
    out = {}
    for s in ms.SENSORS:
        pa = MARGINAL_PARENTS[s]
        K = ms.CARD[s]
        shape = tuple(ms.CARD[p] for p in pa) + (K,)
        tab = np.full(shape, (1 - diag) / (K - 1))
        from itertools import product as ip
        for cfg in ip(*[range(ms.CARD[p]) for p in pa]):
            clsval = cfg[pa.index("cls")]
            if s == "Rf":
                emitval = cfg[pa.index("emit")]
                rep = 0 if clsval == 2 else 2       # drone->'consumer', else 'none'
                _ = emitval                          # abstract model ignores emission
            else:
                rep = CORRECT_REPORT[s][clsval]
            row = np.full(K, (1 - diag) / (K - 1))
            row[rep] = diag
            tab[cfg] = row
        out[s] = Factor(tuple(pa) + (s,), tab)
    return out


def build_inference_bn(internal_cpts, sensors_on, d_value,
                       sensor_mode="calibrated", sensor_cpts=None,
                       alpha=None):
    """Assemble the deployed inference BN:
      sensor_mode 'calibrated': range-marginal confusion CPTs from
                  `sensor_cpts` (estimate_confusion output); if None, the
                  true range-marginal tables (sensor-oracle sanity mode).
                  'conditional': range-conditional CPTs (learns d; upper
                  reference; `sensor_cpts` from estimate_confusion_conditional
                  or None for truth). 'abstract': the B0 model.
      alpha     : {s: array over d} reliability weights (fit_reliability);
                  None disables discounting. Discounting uses alpha[s][d_value].
      sensors_on: subset of SENSORS; others are removed (auto-marginalised).
    """
    gt = ms.ground_truth_bn()
    cpts = dict(gt.cpts)
    parents = {k: list(v) for k, v in ms.PARENTS.items()}
    for k in ms.INTERNAL:
        cpts[k] = internal_cpts[k]
    if sensor_mode == "abstract":
        src = abstract_sensor_cpts()
    elif sensor_mode == "conditional":
        src = sensor_cpts if sensor_cpts is not None \
            else {s: gt.cpts[s] for s in ms.SENSORS}
    else:
        src = sensor_cpts if sensor_cpts is not None \
            else true_marginal_confusion()
    for s in ms.SENSORS:
        cpts.pop(s, None)
        parents.pop(s, None)
    for s in sensors_on:
        c = src[s]
        if alpha is not None:
            c = _discount_cpt(c, float(alpha[s][d_value]))
        cpts[s] = c
        parents[s] = list(c.vars[:-1])
    net_parents = {k: parents[k] for k in cpts}
    net_card = {k: ms.CARD[k] for k in cpts}
    return BayesNet(net_card, net_parents, cpts)


def evidence_from_scenario(scn, sensors_on):
    """Hard tracker/context evidence plus the hard sensor reports. (Soft
    classifier scores enter as virtual evidence only where a real classifier
    produces them -- the DroneRF study of exp_rf_e2e.)"""
    ev = {k: scn[k] for k in ms.OBSERVED}
    for s in sensors_on:
        ev[s] = scn[s]
    return ev


if __name__ == "__main__":
    bn = ms.ground_truth_bn()
    data = generate_dataset(bn, 2000, seed=1)
    T = np.array([d["T"] for d in data])
    print("threat label dist:", np.round(np.bincount(T, minlength=4) / len(T), 3))
    est = estimate_confusion(data[:300])
    al, pr = fit_reliability(data[:300])
    for s in ms.SENSORS:
        print(f"alpha_{s}(d) = {np.round(al[s], 2)}")
    # oracle-marginal sanity
    correct = 0
    for scn in data[:400]:
        ibn = build_inference_bn({k: bn.cpts[k] for k in ms.INTERNAL},
                                 ms.SENSORS, d_value=scn["d"],
                                 sensor_cpts=est, alpha=al)
        ev = evidence_from_scenario(scn, ms.SENSORS)
        post = ibn.query(["T"], ev).table
        correct += (post.argmax() == scn["T"])
    print("marginal+reliability threat accuracy on 400:", correct / 400)
    print("dataio v2 OK")
