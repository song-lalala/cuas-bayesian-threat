"""
Targeted study of the monotonicity constraints (C2-ii): when the elicited
prior is WEAK/noisy, the plain MAP estimate can violate threat-monotonicity;
constrained estimation (cmap_cpt) removes the violations. We report, for the
learned T-CPT: the fraction of covering-pair / threshold monotonicity
VIOLATIONS, plus full-model accuracy and ECE. Sensor confusion matrices are
ESTIMATED from a calibration split of the same world (deployed pipeline), and
reliability discounting uses the fitted alpha (never oracle tables).
"""
from __future__ import annotations
import numpy as np, pandas as pd, os
import model_spec as ms, dataio, grounding as g, metrics as mt

RES = os.path.join(os.path.dirname(__file__), "results")
ESS = 8.0


def mono_violations(cpt, node="T", tol=1e-6):
    """Fraction of (covering-pair, threshold) monotonicity constraints violated
    by a CPT (survival of more-threatening config should dominate)."""
    pairs, _ = g._covering_pairs(node)
    K = ms.CARD[node]
    X = cpt.table.reshape(-1, K)
    surv = np.cumsum(X[:, ::-1], axis=1)[:, ::-1]
    viol = tot = 0
    for j, jp in pairs:
        for m in range(1, K):
            tot += 1
            if surv[jp, m] < surv[j, m] - tol:
                viol += 1
    return viol / max(tot, 1)


def run(noises=(0.15, 0.5), Ns=(20, 50), seeds=8):
    gt = ms.ground_truth_bn()
    test = dataio.generate_dataset(gt, 2500, seed=999)
    calib = dataio.generate_dataset(gt, 300, seed=777)
    est = dataio.estimate_confusion(calib)
    alpha, _ = dataio.fit_reliability(calib)
    rows = []
    for noise in noises:
        for N in Ns:
            for seed in range(seeds):
                data = dataio.generate_dataset(gt, N, seed=seed)
                rng = np.random.default_rng(1000 + seed)
                prior = ms.expert_prior_cpts(rng, noise=noise)
                for method in ["MAP", "cMAP"]:
                    cpts = {}
                    for node in ms.INTERNAL:
                        if method == "MAP":
                            cpts[node] = g.map_cpt(node, data, prior[node], ESS)
                        else:
                            cpts[node] = g.cmap_cpt(node, data, prior[node], ESS)
                    met, _ = mt.evaluate(cpts, test, sensor_cpts=est,
                                         alpha=alpha)
                    rows.append({"prior_noise": noise, "N": N, "seed": seed,
                                 "method": method,
                                 "T_mono_violation": mono_violations(cpts["T"]),
                                 "acc": met["acc"], "ece": met["ece"],
                                 "macroAUC": met["macroAUC"]})
    df = pd.DataFrame(rows)
    summ = (df.groupby(["prior_noise", "N", "method"])
              .mean(numeric_only=True).drop(columns=["seed"]).reset_index())
    df.to_csv(os.path.join(RES, "exp_constraints_raw.csv"), index=False)
    summ.to_csv(os.path.join(RES, "exp_constraints.csv"), index=False)
    print(summ.to_string(index=False))
    print("cmap solver:", g.cmap_report())
    return summ


if __name__ == "__main__":
    run()
