"""
Evaluation: run a grounded model over a test set and compute discrimination
and calibration metrics. Reliability discounting depends on the observed range
d, so we cache one inference BN per d-value (0,1,2) per configuration.

ECE convention (stated in the manuscript): top-label ECE with 15 equal-width
confidence bins. `classwise_ece` additionally reports the classwise variant
(mean over classes of the per-class probability-vs-frequency gap, 15 bins),
which matches the "P(T=high)=0.7 should be right 70% of the time" motivation.
"""
from __future__ import annotations
import numpy as np
from sklearn.metrics import f1_score, roc_auc_score
import dataio, model_spec as ms

LABELS = [0, 1, 2, 3]
N_BINS = 15


def posteriors(internal_cpts, test, sensors_on, sensor_mode="calibrated",
               sensor_cpts=None, alpha=None):
    """Return (P [n,4], y [n]) threat posteriors and true labels."""
    bycache = {}
    P = np.zeros((len(test), ms.CARD["T"]))
    y = np.zeros(len(test), dtype=int)
    for i, scn in enumerate(test):
        dv = scn["d"]
        if dv not in bycache:
            bycache[dv] = dataio.build_inference_bn(
                internal_cpts, sensors_on, dv, sensor_mode,
                sensor_cpts=sensor_cpts, alpha=alpha)
        bn = bycache[dv]
        ev = dataio.evidence_from_scenario(scn, sensors_on)
        P[i] = bn.query(["T"], ev).table
        y[i] = scn["T"]
    return P, y


def ece(P, y, n_bins=N_BINS):
    """Top-label ECE, equal-width bins."""
    conf = P.max(axis=1)
    pred = P.argmax(axis=1)
    correct = (pred == y).astype(float)
    bins = np.linspace(0, 1, n_bins + 1)
    e = 0.0
    N = len(y)
    for b in range(n_bins):
        m = (conf > bins[b]) & (conf <= bins[b + 1])
        if m.sum() == 0:
            continue
        e += (m.sum() / N) * abs(correct[m].mean() - conf[m].mean())
    return e


def classwise_ece(P, y, n_bins=N_BINS):
    """Classwise ECE: mean over classes of the binned |P(class) - freq| gap."""
    N, K = P.shape
    bins = np.linspace(0, 1, n_bins + 1)
    total = 0.0
    for k in range(K):
        p = P[:, k]
        hit = (y == k).astype(float)
        e = 0.0
        for b in range(n_bins):
            m = (p > bins[b]) & (p <= bins[b + 1])
            if m.sum() == 0:
                continue
            e += (m.sum() / N) * abs(hit[m].mean() - p[m].mean())
        total += e
    return total / K


def reliability_curve(P, y, n_bins=N_BINS):
    """Top-label reliability-diagram data: (bin_conf, bin_acc, bin_frac)."""
    conf = P.max(axis=1)
    pred = P.argmax(axis=1)
    correct = (pred == y).astype(float)
    bins = np.linspace(0, 1, n_bins + 1)
    out = []
    for b in range(n_bins):
        m = (conf > bins[b]) & (conf <= bins[b + 1])
        if m.sum() == 0:
            continue
        out.append((conf[m].mean(), correct[m].mean(), m.mean()))
    return np.array(out)


def brier(P, y):
    oh = np.zeros_like(P)
    oh[np.arange(len(y)), y] = 1.0
    return np.mean(np.sum((P - oh) ** 2, axis=1))


def far_at_pd(P, y, pd_target=0.90, positive_from=2):
    """Treat T>=positive_from ('elevated' = medium+high) as positive; find the
    score threshold on P(elevated) giving recall>=pd_target, report the false-
    alarm rate FP/(FP+TN)."""
    score = P[:, positive_from:].sum(axis=1)
    pos = y >= positive_from
    neg = ~pos
    if pos.sum() == 0 or neg.sum() == 0:
        return np.nan
    thr_candidates = np.unique(score)[::-1]
    for thr in thr_candidates:
        pred_pos = score >= thr
        recall = (pred_pos & pos).sum() / pos.sum()
        if recall >= pd_target:
            return (pred_pos & neg).sum() / neg.sum()
    return np.nan


def evaluate(internal_cpts, test, sensors_on=None, sensor_mode="calibrated",
             sensor_cpts=None, alpha=None):
    if sensors_on is None:
        sensors_on = ms.SENSORS
    P, y = posteriors(internal_cpts, test, sensors_on, sensor_mode,
                      sensor_cpts=sensor_cpts, alpha=alpha)
    pred = P.argmax(axis=1)
    acc = float((pred == y).mean())
    f1 = float(f1_score(y, pred, labels=LABELS, average="macro",
                        zero_division=0))
    try:
        auc = float(roc_auc_score(np.eye(4)[y], P, multi_class="ovr",
                                  average="macro", labels=LABELS))
    except Exception:
        auc = float("nan")
    return {
        "acc": acc, "macroF1": f1, "macroAUC": auc,
        "brier": brier(P, y), "ece": ece(P, y),
        "cwece": classwise_ece(P, y),
        "far@pd0.9": far_at_pd(P, y, 0.90),
    }, (P, y)


if __name__ == "__main__":
    gt = ms.ground_truth_bn()
    test = dataio.generate_dataset(gt, 2000, seed=999)
    m, _ = evaluate({k: gt.cpts[k] for k in ms.INTERNAL}, test,
                    sensor_mode="conditional")
    print("oracle (true conditional):", {k: round(v, 3) for k, v in m.items()})
    m2, _ = evaluate({k: gt.cpts[k] for k in ms.INTERNAL}, test)
    print("oracle (true marginal):   ", {k: round(v, 3) for k, v in m2.items()})
