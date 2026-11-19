"""
Chance-corrected reliability layer: variant V1 of the double-discounting
study (exp_reliability_variants.py). The deployed V0
layer (dataio.fit_reliability + dataio._discount_cpt) is NOT modified; this
module only adds the alternative. It lives outside the six core files on
purpose: the consistency gate's freshness guard fails every quoted result if
a core file changes, and this study must leave the published results intact.

Reliability-node semantics (manuscript Sec. V-B). Rel_s in {reliable,
unreliable};  P(R_s | c, reliable) = C_ref(R_s | c), the best-condition
confusion matrix;  P(R_s | c, unreliable) = 1/K_s, a report that carries no
class information;  P(reliable | d) = kappa_s(d). Marginalising Rel_s gives
the deployed likelihood

    L_s(r | c, d) = kappa_s(d) C_ref(r | c) + (1 - kappa_s(d)) / K_s .     (1)

Accuracy implied by (1). With one correct report per class and the class
independent of the context (both hold in the generative world),

    p_s(d) = kappa_s(d) acc_ref + (1 - kappa_s(d)) / K_s ,                 (2)

so kappa_s(d) / kappa_s(d*) = (p_s(d) - 1/K_s) / (p_s(d*) - 1/K_s). Taking
C_ref to be the matrix at the sensor's best range (kappa(d*) = 1),

    kappa_s(d) = clip((p_s(d) - 1/K_s) / (max_d' p_s(d') - 1/K_s), floor, 1) (3)

with p_s(d) the SAME logistic fit V0 uses (recovered from the parameters
dataio.fit_reliability returns). The range-marginal matrix that the labelled
calibration split estimates is the d-average of (1):

    C_marg = kbar C_ref + (1 - kbar) U ,  kbar = sum_d P_cal(d) kappa_s(d)  (4)

with P_cal the empirical range distribution of the calibration split, hence

    C_ref = Pi( (C_marg - (1 - kbar) U) / kbar ) .                          (5)

Pi is the row-wise Euclidean projection onto the floored simplex
{x : x_k >= eps, sum_k x_k = 1}: substituting y = x - eps turns it into the
projection of (v - eps) onto the simplex of radius 1 - K eps, solved exactly
by the sort-and-threshold algorithm (Held, Wolfe & Crowder 1974; Duchi et al.
2008). eps_row = beta / (n_row + K beta) is the smallest value the Laplace-
smoothed count estimate of that row can take (beta = 0.5, n_row = calibration
rows with that parent configuration), so the projection never creates an
impossible report -- the same guarantee the manuscript gives for the smoothed
matrices. A row is "projected" only if some raw entry falls below eps_row;
otherwise (5) is used as is (raw rows sum to one exactly by construction).
At population level (no counts) eps = 0.

Why V0 double-discounts: V0 mixes the MARGINAL matrix, which already
contains the average degradation (1 - kbar) toward uniform, with weight
alpha = 0.98 p/max p, which is a ratio of raw accuracies (it does not
subtract the chance rate 1/K). The clean-matrix share of the V0 likelihood is
alpha(d) * kbar, below kbar (no discount at all) at the best range. V1's share
is kappa(d), which in a uniform-mixing world equals the truth 1 - gamma(d).

RF. The correct-report rule of dataio._is_correct accepts TWO reports for an
emitting drone, so the exact chance accuracy of a uniform report is
sum_c P(c) |S(c)| / K, not 1/K. Per the study specification kappa uses 1/K_s
for every sensor; `chance_accuracy` gives the exact value so the difference
can be reported. RF has no range degradation in the world, so kappa_RF is ~1
either way.
"""
from __future__ import annotations
import numpy as np
from scipy.special import expit
from cuas_bn import Factor
import model_spec as ms
import dataio

D_FEATURES = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])  # far = reference


def p_hat_from_params(params_s):
    """Logistic P(correct | d) for d = far, near, imminent, from the (w, b)
    that dataio.fit_reliability returns (identical to its predict_proba)."""
    w, b = params_s
    return expit(D_FEATURES @ np.asarray(w, float) + float(b))


def chance_accuracy(calib_data, s):
    """Exact accuracy of a uniform report under dataio._is_correct, averaged
    over the calibration rows. Equals 1/K_s for radar, EO/IR and acoustic."""
    K = ms.CARD[s]
    n_ok = np.zeros(len(calib_data))
    for i, row in enumerate(calib_data):
        n_ok[i] = sum(dataio._is_correct({**row, s: r}, s) for r in range(K))
    return float(n_ok.mean() / K)


def kappa_from_p(p, K, floor=0.05, cap=1.0):
    """Eq. (3): chance-corrected relative reliability over d."""
    p = np.asarray(p, float)
    denom = p.max() - 1.0 / K
    if denom <= 1e-12:                      # sensor no better than chance
        return np.full_like(p, floor)
    return np.clip((p - 1.0 / K) / denom, floor, cap)


def fit_reliability_cc(calib_data, floor=0.05, cap=1.0):
    """kappa_s(d) for every sensor from the V0 logistic fit.
    Returns ({s: kappa}, {s: p_hat}, {s: alpha_V0})."""
    alpha_v0, params = dataio.fit_reliability(calib_data)
    kap, phat = {}, {}
    for s in ms.SENSORS:
        phat[s] = p_hat_from_params(params[s])
        kap[s] = kappa_from_p(phat[s], ms.CARD[s], floor, cap)
    return kap, phat, alpha_v0


def project_floored_simplex(v, eps=0.0):
    """Euclidean projection of v onto {x >= eps, sum x = 1}."""
    v = np.asarray(v, float)
    K = v.size
    r = 1.0 - K * eps
    if r <= 0:
        raise ValueError("eps too large for the row length")
    y = v - eps
    u = np.sort(y)[::-1]
    css = np.cumsum(u)
    j = np.arange(1, K + 1)
    rho = np.nonzero(u - (css - r) / j > 0)[0][-1]
    theta = (css[rho] - r) / (rho + 1.0)
    return np.maximum(y - theta, 0.0) + eps


def _row_counts(calib_data, parents):
    shape = tuple(ms.CARD[p] for p in parents)
    n = np.zeros(shape)
    for row in calib_data:
        n[tuple(row[p] for p in parents)] += 1
    return n


def reference_confusion(c_marg, kappa, pd, row_counts=None, smoothing=0.5):
    """Eqs. (4)-(5). c_marg: {s: Factor} range-marginal tables; kappa:
    {s: array over d}; pd: range distribution used for kbar; row_counts:
    {s: array of n_row} (None = population level, eps = 0).
    Returns ({s: Factor C_ref}, {s: stats})."""
    out, stats = {}, {}
    pd = np.asarray(pd, float)
    for s in ms.SENSORS:
        f = c_marg[s]
        K = f.table.shape[-1]
        kbar = float(pd @ np.asarray(kappa[s], float))
        raw = (f.table - (1.0 - kbar) / K) / kbar
        flat = raw.reshape(-1, K)
        if row_counts is None:
            eps = np.zeros(flat.shape[0])
        else:
            eps = smoothing / (row_counts[s].reshape(-1) + K * smoothing)
        ref = flat.copy()
        active = 0
        for i in range(flat.shape[0]):
            if (flat[i] < eps[i]).any():
                ref[i] = project_floored_simplex(flat[i], eps[i])
                active += 1
        out[s] = Factor(f.vars, ref.reshape(raw.shape))
        stats[s] = {"kbar": kbar, "rows": int(flat.shape[0]),
                    "rows_projected": int(active),
                    "min_raw": float(flat.min()),
                    "min_eps": float(eps.min())}
    return out, stats


def calibrate_v1(calib_data, smoothing=0.5, floor=0.05):
    """Full V1 calibration from one labelled split: range-marginal counts
    (dataio.estimate_confusion, unchanged), kappa from the V0 logistic fit,
    and the recovered reference matrices. The deployed likelihood (1) is then
    dataio.build_inference_bn(..., sensor_cpts=C_ref, alpha=kappa), whose
    _discount_cpt computes exactly kappa*C_ref + (1-kappa)/K."""
    c_marg = dataio.estimate_confusion(calib_data, smoothing)
    kappa, phat, alpha_v0 = fit_reliability_cc(calib_data, floor)
    dvals = np.array([row["d"] for row in calib_data])
    pd_cal = np.bincount(dvals, minlength=ms.CARD["d"]) / len(dvals)
    counts = {s: _row_counts(calib_data, dataio.MARGINAL_PARENTS[s])
              for s in ms.SENSORS}
    c_ref, stats = reference_confusion(c_marg, kappa, pd_cal, counts,
                                       smoothing)
    return {"c_marg": c_marg, "c_ref": c_ref, "kappa": kappa, "p_hat": phat,
            "alpha_v0": alpha_v0, "pd_cal": pd_cal, "stats": stats}


if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8")
    # projection self-test: identity inside, exact on a negative entry
    x = project_floored_simplex([0.7, 0.2, 0.1], 0.01)
    assert np.allclose(x, [0.7, 0.2, 0.1])
    x = project_floored_simplex([1.05, 0.0, -0.05], 0.01)
    assert abs(x.sum() - 1) < 1e-12 and x.min() >= 0.01 - 1e-15, x
    gt = ms.ground_truth_bn()
    calib = dataio.generate_dataset(gt, 300, seed=5000)
    v1 = calibrate_v1(calib)
    n_checked = 0
    for s in ms.SENSORS:
        a = np.clip(v1["p_hat"][s] / v1["p_hat"][s].max() * 0.98, 0.05, 0.98)
        assert np.allclose(a, v1["alpha_v0"][s], atol=1e-12), s
        print(f"{s}: p_hat={np.round(v1['p_hat'][s], 3)} "
              f"alpha_V0={np.round(v1['alpha_v0'][s], 3)} "
              f"kappa={np.round(v1['kappa'][s], 3)} stats={v1['stats'][s]}")
        n_checked += 1
    assert n_checked == 4
    print("reliability_cc OK (sensors checked:", n_checked, ")")
