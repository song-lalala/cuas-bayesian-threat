"""
Evaluation: run a grounded model over a test set and compute discrimination
and calibration metrics. Reliability discounting depends on the observed range
d, so we cache one inference BN per d-value (0,1,2) per configuration.
"""
from __future__ import annotations
import numpy as np
from sklearn.metrics import f1_score, roc_auc_score
import dataio, model_spec as ms

LABELS = [0, 1, 2, 3]


def posteriors(internal_cpts, test, sensors_on, reliability, sensor_mode,
               use_soft_eoir=False, sensor_cpts=None):
    """Return (P [n,4], y [n]) threat posteriors and true labels.
    `sensor_cpts` = confusion matrices estimated from a labelled calibration set
    (used in 'calibrated' mode); the EO/IR soft score uses the estimated Re."""
    bycache = {}
    re_cpt = sensor_cpts["Re"] if sensor_cpts is not None else None
    P = np.zeros((len(test), ms.CARD["T"]))
    y = np.zeros(len(test), dtype=int)
    for i, scn in enumerate(test):
        dv = scn["d"]
        key = dv
        if key not in bycache:
            bycache[key] = dataio.build_inference_bn(
                internal_cpts, sensors_on, reliability, dv, sensor_mode,
                sensor_cpts=sensor_cpts)
        bn = bycache[key]
        ev, virt = dataio.evidence_from_scenario(scn, sensors_on, use_soft_eoir,
                                                 re_cpt=re_cpt)
        P[i] = bn.query(["T"], ev, virt).table
        y[i] = scn["T"]
    return P, y


def ece(P, y, n_bins=15):
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
    # sort candidate thresholds by score; pick lowest threshold achieving recall>=pd
    order = np.argsort(-score)
    thr_candidates = np.unique(score)[::-1]
    best_far = np.nan
    for thr in thr_candidates:
        pred_pos = score >= thr
        recall = (pred_pos & pos).sum() / pos.sum()
        if recall >= pd_target:
            far = (pred_pos & neg).sum() / neg.sum()
            best_far = far
            break
    return best_far


def evaluate(internal_cpts, test, sensors_on=None, reliability=True,
             sensor_mode="calibrated", use_soft_eoir=True, sensor_cpts=None):
    if sensors_on is None:
        sensors_on = ms.SENSORS
    P, y = posteriors(internal_cpts, test, sensors_on, reliability, sensor_mode,
                      use_soft_eoir, sensor_cpts=sensor_cpts)
    pred = P.argmax(axis=1)
    acc = float((pred == y).mean())
    f1 = float(f1_score(y, pred, labels=LABELS, average="macro", zero_division=0))
    try:
        auc = float(roc_auc_score(np.eye(4)[y], P, multi_class="ovr",
                                  average="macro", labels=LABELS))
    except Exception:
        auc = float("nan")
    return {
        "acc": acc, "macroF1": f1, "macroAUC": auc,
        "brier": brier(P, y), "ece": ece(P, y),
        "far@pd0.9": far_at_pd(P, y, 0.90),
    }, (P, y)


if __name__ == "__main__":
    import grounding as g
    gt = ms.ground_truth_bn()
    test = dataio.generate_dataset(gt, 3000, seed=999)
    # oracle metrics (true CPTs)
    m, _ = evaluate({k: gt.cpts[k] for k in ms.INTERNAL}, test)
    print("oracle:", {k: round(v, 3) for k, v in m.items()})
