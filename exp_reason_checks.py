"""
Exp-RS: controls behind sentences of the paper that give a reason. Added after the
main results. Core files are not modified; results go to results/expRS_*.

(a) The rule of [5] on RF silence. From results/expB6_subsets.csv: radio-silent
    drones and silent reports from non-drones, for the published and rescaled rule,
    the CPT oracle and Proposed. Control, on the test sets of exp_base_rule.py
    (scenarios 9000 + r): the published rule with the silence input off, that is the
    report read as Detected, so that RF Status scores 0 and rule 1 of [5] cannot
    fire; RF Protocol stays missing, as under Silent.
(b) How concentrated the softmax is. From the CNN outputs of exp_rf_e2e.py (P0, when
    results/expG_cnn.npz is present) and exp_rf_independent.py (the other splits):
    mean top probability, share above 0.9, median of max(background, 1 - background).
    Control: the soft evidence of exp_rf_e2e.posteriors_rf4 with the mass on the three
    drone models replaced by its equal split.
(c) The weight fitted on a binary context of DroneDetect, against no discount: from
    results/expI_dd_reliability.csv, paired over the five seeds.
(d) The additivity test on the threat node by training size, from
    results/expB6w_flags_raw.csv and results/expW_add_diag_raw.csv. The standard world
    was tested from N = 50 only.
(e) The fitted reliability weights over the calibration splits of the main study
    (scenarios 5000 + r, n = 300, 20 replications).
(f) The RF channel's upper bound, with the seeds, pools and draws of exp_rf_e2e.run:
    no channel, hard report, soft evidence, and an error-free split of emitting drones
    from everything else (the most any RF channel can give through this mapping),
    first split, 20 replications. The emission of non-drone targets stays hidden.
(g) The split max(background, 1 - background) on what the scenarios draw, not on the
    test pool: each drone model's share is P(drone) P(active) / 3, the rest is the
    background pool (exp_rf_e2e.map_class). All 20 runs. Also the size of the test
    background of every run: spectrograms, segments at 20 spectrograms each, errors.
(h) The reliability weights scaled (seeds of exp_correlated.part_rel): the sweep of
    results/expF_rel_raw.csv against the fitted weights on all metrics, and two more
    factors, 0.25 and 0. Factor 0 has no floor, so it leaves the reports out; the sweep's
    floor of 0.02 does not bind at 0.25.
(i) The root marginals of cls and emit: the specification against counts on the
    calibration split and on the training rows (each with +0.5), for Proposed at
    N = 10, 50, 200 and MLE at N = 50, on all metrics.
(j) Adverse subset of the naive model in exp_correlated.part_corr, at every severity:
    mean top probability, accuracy, and accuracy with the sensor reports hidden.
(k) Calibrated fusion against two controls (N = 200, the seeds of exp_significance.py),
    on all scenarios and on radio-silent drones: no RF channel (sensors_on without Rf),
    with and without the fitted discount, and a hand-set table that knows emission:
    the abstract diagonal 0.94 on radar, EO/IR and acoustic, with an RF row in which an
    emitting drone reports consumer or mil with 0.47 each and none with 0.06, and
    anything else reports none with 0.94, consumer and mil with 0.03 each. The abstract
    and calibrated rows must equal those of results/expS_fusion_raw.csv.
(l) A network with a uniform prior mean against the rule of [5]: B3 of exp_baselines.py
    (full tables; it keeps the graph and the latent labels) on T = high at N = 10 to 400,
    minus B6-pub. The
    rule read as P(T >= medium) instead, with Proposed on the same target. The contrast
    behind the realistic-expert sentence with the same full-table structure, from
    results/expB_baselines_raw.csv. Rows must equal results/expB_baselines_raw.csv and
    results/expB6_raw.csv where those files have them.
(m) The worked track with only the training data or only the prior redrawn (seeds 0
    to 19, N = 200, calibration seed 777 as in exp_worked_track_seeds.py; the other part
    stays at seed 0, and the restarts of the ICI fit follow the prior's seed). Redrawing
    both must equal results/expTS_seeds.csv.
(n) False alarms of the additivity test at N = 10 and 20 in the standard world (B = 199,
    the seeds of exp_base_rule_worlds.py).
(o) The reversed-prior grid at alpha0 = 2, from results/expED_raw.csv: Proposed and MLE,
    lambda by N, and the contrasts with MLE. No new run.
(p) Contrasts read from the result files for single sentences: 30 labeled scenarios
    against the abstract model, Proposed against B0 and MLE per N, the fitted discount.
(q) The hand-set emission-aware table of (k) swept over its diagonal d (the seeds of (k),
    N = 200, no discount): abstract_sensor_cpts_diag(d) on radar, EO/IR and acoustic, with an
    RF row in which an emitting drone reports consumer or mil with d/2 each and none with
    1 - d, and anything else reports none with d, consumer and mil with (1 - d)/2 each. At
    each d also the abstract model and no RF channel with that diagonal. Two mixed tables:
    the three estimated with the RF row at 0.94, the three at 0.94 with the RF row estimated.
    Subsets: all scenarios, radio-silent drones, all drones, non-drones, scenarios with an RF
    report of none, and those that are also non-drones. The estimated matrices' correct-report
    probability per sensor (mean over the classes; rows with the variant "estimated,
    correct-report probability", value in the acc column). Estimated matrices and the sensor
    oracle by calibration size, from results/expS_calibsize_raw.csv, against d = 0.75. At
    d = 0.94 the hand-set and abstract rows must equal results/expRS_fusion.csv, and the
    calibrated row must equal n = 300 of results/expS_calibsize_raw.csv.

Run:  python exp_reason_checks.py
Outputs: results/expRS_b6_silent.csv, results/expRS_b6_control.csv,
         results/expRS_softmax.csv, results/expRS_transfer.csv,
         results/expRS_flags.csv, results/expRS_weights.csv,
         results/expRS_rf_bound.csv, results/expRS_split.csv, results/expRS_rel_scale.csv,
         results/expRS_roots.csv, results/expRS_adverse.csv, results/expRS_fusion.csv,
         results/expRS_rule_b3.csv, results/expRS_track_split.csv,
         results/expRS_flags_small.csv, results/expRS_lambda.csv, results/expRS_handset.csv,
         results/expRS_summary.md
"""
from __future__ import annotations
import os, sys
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import pandas as pd
from scipy import stats as st

import model_spec as ms
import dataio
import pipeline as pl
import exp_base_rule as br
import exp_rf_e2e as eg
import exp_correlated as ec
import metrics as mt
from cuas_bn import BayesNet, Factor

RES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
P = "expRS"
REPS = 20
SUBSETS = ("silent-drone", "silent-report-nondrone")


def rule_rows():
    sb = pd.read_csv(os.path.join(RES, "expB6_subsets.csv"), float_precision="round_trip")
    rows = []
    for s in SUBSETS:
        for m, N in (("B6-pub", 0), ("B6-recal", 0), ("CPT-oracle", 0), ("Proposed", 10), ("Proposed", 200)):
            q = sb[(sb.subset == s) & (sb.method == m) & (sb.N == N)]
            assert q.rep.nunique() == REPS, (s, m, N)
            rows.append({"subset": s, "method": m, "N": N, "n": q.n.mean(), "pos": q.pos.mean(), "auc": q.auc.mean(), "brier": q.brier_bin.mean(), "mean_p": q.mean_p.mean()})
    pd.DataFrame(rows).to_csv(os.path.join(RES, f"{P}_b6_silent.csv"), index=False)
    return rows


def rule_control():
    gt = ms.ground_truth_bn()
    rows = []
    for r in range(REPS):
        test = dataio.generate_dataset(gt, 2500, seed=9000 + r)
        y = np.array([x["T"] == 3 for x in test], dtype=float)
        pub, off = np.empty(len(test)), np.empty(len(test))
        for i, x in enumerate(test):
            s = br.map_states(x["d"], x["rdot"], x["v"], x["z"], x["Rf"], x["Rr"])
            pub[i] = br.rule5(s)[0]
            s = dict(s, RF_Status="Detected")
            off[i] = br.rule5(s)[0]
        masks = {"silent-drone": np.array([x["cls"] == 2 and x["emit"] == 1 for x in test]),
                 "silent-report-nondrone": np.array([x["Rf"] == 2 and x["cls"] != 2 for x in test]),
                 "RF report not none": np.array([x["Rf"] != 2 for x in test])}
        assert np.array_equal(pub, br.b6_pub(test))
        for k, m in masks.items():
            rows.append({"rep": r, "subset": k, "n": int(m.sum()), "observed_high": float(y[m].mean()), "mean_published": float(pub[m].mean()), "mean_silence_off": float(off[m].mean()),
                         "brier_published": float(((pub[m] - y[m]) ** 2).mean()), "brier_silence_off": float(((off[m] - y[m]) ** 2).mean())})
    pd.DataFrame(rows).to_csv(os.path.join(RES, f"{P}_b6_control.csv"), index=False)
    return pd.DataFrame(rows)


def conc(s):
    return {"mean_top": float(s.max(axis=1).mean()), "share_top_above_0.9": float((s.max(axis=1) > 0.9).mean()), "median_bg_split": float(np.median(np.maximum(s[:, 0], 1 - s[:, 0]))), "n": len(s)}


def softmax_rows():
    rows = []
    p0 = os.path.join(RES, "expG_cnn.npz")
    if os.path.exists(p0):
        z = np.load(p0, allow_pickle=False)
        assert str(z["class_names"][0]) == "background"
        rows.append({"proto": "P0", "run": "-", **conc(z["test_softmax"])})
    z = np.load(os.path.join(RES, "expI_cnn_dronerf.npz"), allow_pickle=False)
    for k in sorted(x for x in z.keys() if x.endswith("__test_softmax")):
        pre = k.split("__")[0]
        rows.append({"proto": pre.split("_")[0], "run": pre.split("_")[1], **conc(z[k])})
    pd.DataFrame(rows).to_csv(os.path.join(RES, f"{P}_softmax.csv"), index=False)
    return pd.DataFrame(rows)


def soft_control(n=300):
    """Algebra of the generator mapping, checked numerically: moving softmax mass among the drone models leaves P(T) alone."""
    z = np.load(os.path.join(RES, "expI_cnn_dronerf.npz"), allow_pickle=False)
    pool, y = z["P1_42__test_softmax"], z["P1_42__te_y"]
    gt = ms.ground_truth_bn()
    test = dataio.generate_dataset(gt, n, seed=9000)
    calib = dataio.generate_dataset(gt, 300, seed=5000)
    est = dataio.estimate_confusion(calib)
    alpha, _ = dataio.fit_reliability(calib)
    cpts, _ = pl.ground_internal("Proposed", dataio.generate_dataset(gt, 200, seed=0), 0, gt=gt)
    rng = np.random.default_rng(20261107)
    soft = np.empty((n, 4))
    for i, x in enumerate(test):
        emitting = x["cls"] == 2 and x["emit"] == 0
        cl = rng.integers(1, 4) if emitting else 0
        soft[i] = pool[rng.choice(np.flatnonzero(y == cl))]
    lumped = soft.copy()
    lumped[:, 1:] = (1 - soft[:, [0]]) / 3
    prior = np.full(4, 0.25)
    a, _ = eg.posteriors_rf4(cpts, test, soft, None, est, "calib-soft", train_prior=prior, alpha=alpha)
    b, _ = eg.posteriors_rf4(cpts, test, lumped, None, est, "calib-soft", train_prior=prior, alpha=alpha)
    return float(np.abs(a - b).max()), float(np.abs(soft - lumped).max())


def transfer_rows():
    d = pd.read_csv(os.path.join(RES, "expI_dd_reliability.csv"), float_precision="round_trip")
    rows = []
    q = d[d.fit != "4-condition"]
    for (fit, cond), g in q.groupby(["fit", "cond"], sort=False):
        g = g.sort_values("seed")
        x = g.ll_transfer.values - g.ll_none.values
        rows.append({"fit": fit, "cond": cond, "seeds": len(g), "gain_over_no_discount": float(x.mean()), "p": float(st.ttest_1samp(x, 0.0).pvalue), "ll_transfer": float(g.ll_transfer.mean()), "ll_own": float(g.ll_own.mean()), "ll_none": float(g.ll_none.mean())})
    pd.DataFrame(rows).to_csv(os.path.join(RES, f"{P}_transfer.csv"), index=False)
    return pd.DataFrame(rows)


def flag_rows():
    f = pd.concat([pd.read_csv(os.path.join(RES, "expB6w_flags_raw.csv")), pd.read_csv(os.path.join(RES, "expW_add_diag_raw.csv"))])
    q = f[(f.test == "A") & (f.node == "T") & f.world.isin(("standard", "nonadd-0.5", "nonadd-1.0"))]
    t = q.groupby(["world", "N"]).flag.agg(["sum", "count"]).reset_index().rename(columns={"sum": "flagged", "count": "reps"})
    t.to_csv(os.path.join(RES, f"{P}_flags.csv"), index=False)
    return t


def weight_rows():
    gt = ms.ground_truth_bn()
    A = {s: [] for s in ms.SENSORS}
    for r in range(REPS):
        alpha, _ = dataio.fit_reliability(dataio.generate_dataset(gt, 300, seed=5000 + r))
        for s in ms.SENSORS:
            A[s].append(alpha[s])
    rows = []
    for s in ms.SENSORS:
        a = np.array(A[s])
        best = a.argmax(axis=1)
        for d in range(3):
            rows.append({"sensor": s, "range": ms.STATES["d"][d], "mean": float(a[:, d].mean()), "min": float(a[:, d].min()), "max": float(a[:, d].max()), "highest_here": int((best == d).sum())})
    pd.DataFrame(rows).to_csv(os.path.join(RES, f"{P}_weights.csv"), index=False)
    return pd.DataFrame(rows)


def rf_bound_one(r):
    cc = np.load(eg.CNN_CACHE, allow_pickle=True)
    sm_te, y_te, sm_va, y_va = cc["test_softmax"], cc["test_y"], cc["val_softmax"], cc["val_y"]
    prior = np.full(4, 0.25)
    pools_te = {c: np.where(y_te == c)[0] for c in range(4)}
    pools_va = {c: np.where(y_va == c)[0] for c in range(4)}
    gt = ms.ground_truth_bn()
    rng = np.random.default_rng(41000 + r)
    test = dataio.generate_dataset(gt, eg.TEST_N, seed=9000 + r)
    calib = dataio.generate_dataset(gt, 300, seed=5000 + r)
    est = dataio.estimate_confusion(calib)
    alpha, _ = dataio.fit_reliability(calib)
    cpts, _ = pl.ground_internal("Proposed", dataio.generate_dataset(gt, eg.GROUND_N, seed=r), r, gt=gt)
    soft = np.zeros((len(test), 4))
    hard = np.zeros(len(test), dtype=int)
    for i, x in enumerate(test):
        j = eg.draw_pool_index(pools_te, eg.map_class(x, rng), rng)
        soft[i] = sm_te[j]
        hard[i] = int(sm_te[j].argmax())
    cnt = np.zeros((ms.CARD["cls"], ms.CARD["emit"], 4)) + 0.5
    for x in calib:
        j = eg.draw_pool_index(pools_va, eg.map_class(x, rng), rng)
        cnt[x["cls"], x["emit"], int(sm_va[j].argmax())] += 1
    L_calib = cnt / cnt.sum(axis=-1, keepdims=True)
    eps = 1e-6
    emitting = np.array([x["cls"] == 2 and x["emit"] == 0 for x in test])
    told = np.where(emitting[:, None], np.array([eps, (1 - eps) / 3, (1 - eps) / 3, (1 - eps) / 3]), np.array([1 - eps, eps / 3, eps / 3, eps / 3]))
    rows = []
    for name, mode, s, Lh in (("none", "calib-soft", np.full((len(test), 4), 0.25), None), ("hard", "calib-hard", soft, L_calib), ("soft", "calib-soft", soft, None), ("error-free split", "calib-soft", told, None)):
        P, y = eg.posteriors_rf4(cpts, test, s, hard, est, mode, L_hard=Lh, train_prior=prior, alpha=alpha)
        rows.append({"rep": r, "variant": name, "acc": float((P.argmax(1) == y).mean()), "macroAUC": eg.macro_auc(P, y), "brier": mt.brier(P, y), "ece": mt.ece(P, y)})
    return rows


def split_row(proto, run, s, y, w_bg, w_model):
    split = np.maximum(s[:, 0], 1 - s[:, 0])
    n = np.bincount(y, minlength=4)
    assert (n > 0).all()
    w = np.where(y == 0, w_bg / n[0], w_model / n[np.clip(y, 0, 3)])
    o = np.argsort(split)
    assert abs(w.sum() - 1) < 1e-12
    bg = y == 0
    assert n[0] % 20 == 0
    return {"proto": proto, "run": run, "n": len(y), "background_spectrograms": int(n[0]), "background_segments": int(n[0]) // 20, "background_errors": int((bg & (s.argmax(1) != 0)).sum()),
            "pool_median": float(np.median(split)), "pool_below_0.9": float((split < 0.9).mean()), "background_median": float(np.median(split[bg])), "background_below_0.9": float((split[bg] < 0.9).mean()),
            "scenario_median": float(split[o][np.searchsorted(np.cumsum(w[o]), 0.5)]), "scenario_below_0.9": float(w[split < 0.9].sum()), "scenario_mean_top": float((w * s.max(axis=1)).sum())}


def split_rows():
    gt = ms.ground_truth_bn()
    w_bg = 1 - float(gt.cpts["cls"].table[2]) * float(gt.cpts["emit"].table[0])
    w_model = (1 - w_bg) / 3
    z = np.load(eg.CNN_CACHE, allow_pickle=True)
    rows = [split_row("P0", "-", z["test_softmax"], z["test_y"], w_bg, w_model)]
    z = np.load(os.path.join(RES, "expI_cnn_dronerf.npz"), allow_pickle=False)
    for k in sorted(x for x in z.keys() if x.endswith("__test_softmax")):
        pre = k.split("__")[0]
        rows.append(split_row(pre.split("_")[0], pre.split("_")[1], z[k], z[pre + "__te_y"], w_bg, w_model))
    d = pd.DataFrame(rows)
    d.to_csv(os.path.join(RES, f"{P}_split.csv"), index=False)
    return d, w_bg, w_model


def rel_one(seed):
    gt = ms.ground_truth_bn()
    test = dataio.generate_dataset(gt, ec.TEST_N, seed=9000 + seed)
    calib = dataio.generate_dataset(gt, 300, seed=5000 + seed)
    est = dataio.estimate_confusion(calib)
    alpha, _ = dataio.fit_reliability(calib)
    cpts, _ = pl.ground_internal("Proposed", dataio.generate_dataset(gt, ec.GROUND_N, seed=seed), seed, gt=gt)
    rows = []
    for m, floor in ((1.0, 0.02), (0.25, 0.02), (0.0, 0.0)):
        am = {s: np.clip(np.asarray(a) * m, floor, 0.995) for s, a in alpha.items()}
        met, _ = mt.evaluate(cpts, test, sensor_cpts=est, alpha=am)
        rows.append({"seed": seed, "mult": m, **met})
    met, _ = mt.evaluate(cpts, test, sensors_on=[], sensor_cpts=est, alpha=alpha)
    rows.append({"seed": seed, "mult": "reports hidden", **met})
    return rows


def _freq(rows, key):
    c = np.bincount([x[key] for x in rows], minlength=ms.CARD[key]) + 0.5
    return c / c.sum()


def roots_one(r):
    gt = ms.ground_truth_bn()
    test = dataio.generate_dataset(gt, 2500, seed=9000 + r)
    calib = dataio.generate_dataset(gt, 300, seed=5000 + r)
    est = dataio.estimate_confusion(calib)
    alpha, _ = dataio.fit_reliability(calib)
    built = dataio.build_inference_bn
    rows = []
    for method, N in (("Proposed", 10), ("Proposed", 50), ("Proposed", 200), ("B1-MLE", 50)):
        train = dataio.generate_dataset(gt, N, seed=r)
        assert all("cls" in x and "emit" in x for x in train)
        cpts, _ = pl.ground_internal(method, train, r, gt=gt)
        for name, q in (("specification", None), ("calibration split", (_freq(calib, "cls"), _freq(calib, "emit"))), ("training rows", (_freq(train, "cls"), _freq(train, "emit")))):
            def patched(*a, q=q, **k):
                bn = built(*a, **k)
                if q is None:
                    return bn
                c = dict(bn.cpts)
                c["cls"], c["emit"] = Factor(("cls",), q[0]), Factor(("emit",), q[1])
                return BayesNet(bn.card, bn.parents, c)
            dataio.build_inference_bn = patched
            try:
                met, _ = mt.evaluate(cpts, test, sensor_cpts=est, alpha=alpha)
            finally:
                dataio.build_inference_bn = built
            rows.append({"rep": r, "method": method, "N": N, "roots": name, **met})
    return rows


def adverse_one(a):
    sev, seed = a
    world = ec.world_with_W(sev)
    test = ec.sample_ancestral(world, np.random.default_rng(9000 + seed), ec.TEST_N)
    calib = ec.sample_ancestral(world, np.random.default_rng(5000 + seed), 300)
    train = ec.sample_ancestral(world, np.random.default_rng(seed), ec.GROUND_N)
    est = dataio.estimate_confusion(calib)
    alpha, _ = dataio.fit_reliability(calib)
    cpts, _ = pl.ground_internal("Proposed", train, seed, gt=ms.ground_truth_bn())
    P, y = ec.posteriors_W(cpts, est, test, "none", alpha=alpha)
    adv = np.array([x["W"] == 1 for x in test])
    cache = {}
    H = np.zeros_like(P)
    for i, x in enumerate(test):
        if x["d"] not in cache:
            cache[x["d"]] = ec.build_bn_W(cpts, est, "none", alpha=alpha, d_value=x["d"])
        H[i] = cache[x["d"]].query(["T"], {k: x[k] for k in ms.OBSERVED}).table
    return {"severity": sev, "seed": seed, "n_adverse": int(adv.sum()), "mean_top": float(P.max(axis=1)[adv].mean()), "acc": float((P.argmax(1) == y)[adv].mean()), "ece": float(mt.ece(P[adv], y[adv])),
            "acc_reports_hidden": float((H.argmax(1) == y)[adv].mean())}


def fusion_one(r):
    gt = ms.ground_truth_bn()
    test = dataio.generate_dataset(gt, 2500, seed=9000 + r)
    calib = dataio.generate_dataset(gt, 300, seed=5000 + r)
    est = dataio.estimate_confusion(calib)
    alpha, _ = dataio.fit_reliability(calib)
    cpts, _ = pl.ground_internal("Proposed", dataio.generate_dataset(gt, 200, seed=r), r, gt=gt)
    t = np.zeros((ms.CARD["cls"], ms.CARD["emit"], ms.CARD["Rf"]))
    for c in range(ms.CARD["cls"]):
        for e in range(ms.CARD["emit"]):
            t[c, e] = (0.47, 0.47, 0.06) if (c == 2 and e == 0) else (0.03, 0.03, 0.94)
    hand = {**dataio.abstract_sensor_cpts(), "Rf": Factor(("cls", "emit", "Rf"), t)}
    no_rf = [s for s in ms.SENSORS if s != "Rf"]
    assert len(no_rf) == 3
    silent = np.array([x["cls"] == 2 and x["emit"] == 1 for x in test])
    rows = []
    for name, kw in (("abstract", {"sensor_mode": "abstract"}), ("calibrated", {"sensor_cpts": est, "alpha": None}), ("calib+rel", {"sensor_cpts": est, "alpha": alpha}),
                     ("no RF", {"sensors_on": no_rf, "sensor_cpts": est, "alpha": None}), ("no RF+rel", {"sensors_on": no_rf, "sensor_cpts": est, "alpha": alpha}),
                     ("hand-set emission-aware", {"sensor_cpts": hand, "alpha": None})):
        met, (P, y) = mt.evaluate(cpts, test, **kw)
        rows.append({"rep": r, "variant": name, "subset": "all", "n": len(y), "acc": met["acc"], "macroF1": met["macroF1"], "macroAUC": met["macroAUC"], "brier": met["brier"], "ece": met["ece"]})
        rows.append({"rep": r, "variant": name, "subset": "radio-silent", "n": int(silent.sum()), "acc": float((P[silent].argmax(1) == y[silent]).mean()), "macroF1": np.nan,
                     "macroAUC": eg.macro_auc(P[silent], y[silent]), "brier": mt.brier(P[silent], y[silent]), "ece": mt.ece(P[silent], y[silent])})
    return rows


def rule_one(r):
    import grounding as g
    import exp_baselines as eb
    from sklearn.metrics import roc_auc_score
    gt = ms.ground_truth_bn()
    test = dataio.generate_dataset(gt, 2500, seed=9000 + r)
    calib = dataio.generate_dataset(gt, 300, seed=5000 + r)
    est = dataio.estimate_confusion(calib)
    alpha, _ = dataio.fit_reliability(calib)
    y = np.array([x["T"] for x in test])
    hi, med = (y == 3).astype(float), (y >= 2).astype(float)

    def row(method, N, p_hi, p_med, macro):
        return {"rep": r, "method": method, "N": N, "auc_high": float(roc_auc_score(hi, p_hi)), "brier_high": float(((p_hi - hi) ** 2).mean()),
                "auc_ge_medium": float(roc_auc_score(med, p_med)), "brier_ge_medium": float(((p_med - med) ** 2).mean()), "macroAUC": macro,
                "base_high": float(hi.mean()), "base_ge_medium": float(med.mean())}
    p6 = br.b6_pub(test)
    above = p6 > 0.75
    rows = [{**row("B6-pub", 0, p6, p6, np.nan), "n_above": int(above.sum()), "high_above": int(hi[above].sum()), "ge_medium_above": int(med[above].sum())}]
    unif = eb.uniform_prior_cpts()
    for N in (10, 20, 50, 100, 200, 400):
        data = dataio.generate_dataset(gt, N, seed=r)
        c3 = {n: g.map_cpt(n, data, unif[n], pl.ESS_FIXED) for n in ms.INTERNAL}
        met, (P, _) = mt.evaluate(c3, test, sensor_cpts=est, alpha=alpha)
        rows.append(row("B3", N, P[:, 3], P[:, 2:].sum(axis=1), met["macroAUC"]))
        cp, _ = pl.ground_internal("Proposed", data, r, gt=gt)
        met, (P, _) = mt.evaluate(cp, test, sensor_cpts=est, alpha=alpha)
        rows.append(row("Proposed", N, P[:, 3], P[:, 2:].sum(axis=1), met["macroAUC"]))
    return rows


def track_one(a):
    import exp_sensitivity as es
    kind, s = a
    gt = ms.ground_truth_bn()
    calib = dataio.generate_dataset(gt, 300, seed=777)
    est = dataio.estimate_confusion(calib)
    alpha, _ = dataio.fit_reliability(calib)
    data_seed, prior_seed = {"both": (s, s), "training data": (s, 0), "prior": (0, s)}[kind]
    cp, _ = pl.ground_internal("Proposed", dataio.generate_dataset(gt, 200, seed=data_seed), prior_seed, gt=gt)
    ev = es.EVIDENCE
    bn = dataio.build_inference_bn(cp, ms.SENSORS, ev["d"], sensor_cpts=est, alpha=alpha)
    return {"redrawn": kind, "seed": s, "p_high": float(bn.query(["T"], ev).table[3])}


def flag_small_one(a):
    import exp_additivity as ea
    N, r = a
    ea.NI.update({10: 3, 20: 4, 100: 5, 400: 6})  # as exp_base_rule_worlds.py
    return ea.a_test(("standard", r, N))[0]


HAND_GRID = (0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.94, 0.99)
HAND_SUBSETS = ("all", "radio-silent drones", "drones", "non-drones", "RF none", "RF none, non-drones")


def hand_rf(d):
    """Rounded, so that d = 0.94 gives the literals of fusion_one: the last bits of a float break ties between scenarios, which moves macro-AUC."""
    hi, lo, off = round(d / 2, 10), round((1 - d) / 2, 10), round(1 - d, 10)
    t = np.zeros((ms.CARD["cls"], ms.CARD["emit"], ms.CARD["Rf"]))
    for c in range(ms.CARD["cls"]):
        for e in range(ms.CARD["emit"]):
            t[c, e] = (hi, hi, off) if (c == 2 and e == 0) else (lo, lo, d)
    return Factor(("cls", "emit", "Rf"), t)


def handset_one(r):
    gt = ms.ground_truth_bn()
    test = dataio.generate_dataset(gt, 2500, seed=9000 + r)
    calib = dataio.generate_dataset(gt, 300, seed=5000 + r)
    est = dataio.estimate_confusion(calib)
    cpts, _ = pl.ground_internal("Proposed", dataio.generate_dataset(gt, 200, seed=r), r, gt=gt)
    cls_, emit, rfr = (np.array([x[k] for x in test]) for k in ("cls", "emit", "Rf"))
    assert ms.STATES["Rf"][2] == "none" and ms.STATES["cls"][2] == "drone" and ms.STATES["emit"][1] == "silent"
    masks = dict(zip(HAND_SUBSETS, (np.ones(len(test), bool), (cls_ == 2) & (emit == 1), cls_ == 2, cls_ != 2, rfr == 2, (rfr == 2) & (cls_ != 2))))
    no_rf = [s for s in ms.SENSORS if s != "Rf"]
    assert len(no_rf) == 3
    runs = [("calibrated", np.nan, {"sensor_cpts": est, "alpha": None})]
    for d in HAND_GRID:
        ab = dataio.abstract_sensor_cpts_diag(d)
        runs += [("hand-set", d, {"sensor_cpts": {**ab, "Rf": hand_rf(d)}, "alpha": None}), ("abstract", d, {"sensor_cpts": ab, "alpha": None}), ("no RF", d, {"sensors_on": no_rf, "sensor_cpts": ab, "alpha": None})]
    ab = dataio.abstract_sensor_cpts_diag(0.94)
    runs += [("estimated, RF row hand-set", 0.94, {"sensor_cpts": {**est, "Rf": hand_rf(0.94)}, "alpha": None}), ("hand-set, RF row estimated", 0.94, {"sensor_cpts": {**ab, "Rf": est["Rf"]}, "alpha": None})]
    rows = []
    for name, d, kw in runs:
        met, (Q, y) = mt.evaluate(cpts, test, **kw)
        for s, mk in masks.items():
            if s == "all":
                q = {"acc": met["acc"], "macroF1": met["macroF1"], "macroAUC": met["macroAUC"], "brier": met["brier"], "ece": met["ece"], "far": met["far@pd0.9"]}
            else:
                q = {"acc": float((Q[mk].argmax(1) == y[mk]).mean()), "macroF1": np.nan, "macroAUC": eg.macro_auc(Q[mk], y[mk]), "brier": mt.brier(Q[mk], y[mk]), "ece": mt.ece(Q[mk], y[mk]), "far": np.nan}
            rows.append({"rep": r, "variant": name, "diagonal": d, "subset": s, "n": int(mk.sum()), **q})
    for s in ("Rr", "Re", "Ra"):
        c = dataio.CORRECT_REPORT[s]
        assert est[s].vars == ("cls", s)
        rows.append({"rep": r, "variant": "estimated, correct-report probability", "diagonal": np.nan, "subset": s, "n": len(calib), "acc": float(np.mean([est[s].table[k, c[k]] for k in range(ms.CARD["cls"])])), "macroF1": np.nan, "macroAUC": np.nan, "brier": np.nan, "ece": np.nan, "far": np.nan})
    return rows


def pooled(f, items, workers=8):
    """The parent keeps its own thread settings, so the sections above come out as before."""
    for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[v] = "1"
    with ProcessPoolExecutor(max_workers=workers) as ex:
        return list(ex.map(f, items))


def paired(a, b, k):
    d = a.sort_values(a.columns[0])[k].values - b.sort_values(b.columns[0])[k].values
    assert len(d) == 20
    return float(d.mean()), float(st.ttest_1samp(d, 0.0).pvalue)


def cell(a, b, k):
    """A paired contrast. A subset can lack a threat level in a replication, which leaves its macro-AUC undefined (as in results/expS_fusion_raw.csv); such pairs drop out."""
    d = a.sort_values(a.columns[0])[k].values - b.sort_values(b.columns[0])[k].values
    assert len(d) == REPS
    d = d[~np.isnan(d)]
    assert len(d) >= 10
    return f"{d.mean():+.4f} ({st.ttest_1samp(d, 0.0).pvalue:.2g})"


def section_k(fu):
    fu.to_csv(os.path.join(RES, f"{P}_fusion.csv"), index=False)
    raw = pd.read_csv(os.path.join(RES, "expS_fusion_raw.csv"), float_precision="round_trip")
    sel = lambda v, s: fu[(fu.variant == v) & (fu.subset == s)].sort_values("rep")
    for v, c, cs in (("abstract", "all (abstract model)", "_subset_silent-drone_abstract"), ("calibrated", "all (calibrated)", "_subset_silent-drone_calibrated"), ("calib+rel", "all (calib+reliab)", "_subset_silent-drone_calib+reliab")):
        for s, name in (("all", c), ("radio-silent", cs)):
            ref = raw[raw.config == name].sort_values("rep")
            assert len(ref) == REPS
            for k in ("acc", "macroAUC", "brier", "ece"):
                assert np.allclose(sel(v, s)[k].values, ref[k].values, atol=1e-12, equal_nan=True), (v, s, k)
    MET = ("acc", "macroAUC", "brier", "ece")
    L = ["", "## (k) calibrated fusion against two controls (N = 200, 20 replications; abstract, calibrated and calib+rel are the rows of expS_fusion_raw.csv)", "| subset | variant | accuracy | macro-F1 | macro-AUC | Brier | ECE |", "|---|---|---|---|---|---|---|"]
    for s in ("all", "radio-silent"):
        for v in ("abstract", "no RF", "no RF+rel", "hand-set emission-aware", "calibrated", "calib+rel"):
            q = sel(v, s)
            f1 = f"{q.macroF1.mean():.4f}" if s == "all" else "-"
            L.append(f"| {s} | {v} | {q.acc.mean():.4f} | {f1} | {q.macroAUC.mean():.4f} | {q.brier.mean():.4f} | {q.ece.mean():.4f} |")
    L += ["", "| contrast | subset | accuracy (p) | macro-AUC (p) | Brier (p) | ECE (p) |", "|---|---|---|---|---|---|"]
    for a, b in (("no RF", "abstract"), ("calibrated", "no RF"), ("calib+rel", "no RF+rel"), ("calibrated", "abstract"), ("hand-set emission-aware", "abstract"), ("calibrated", "hand-set emission-aware"), ("calib+rel", "hand-set emission-aware"), ("calib+rel", "calibrated")):
        for s in ("all", "radio-silent"):
            L.append(f"| {a} - {b} | {s} | " + " | ".join(cell(sel(a, s)[["rep", k]], sel(b, s)[["rep", k]], k) for k in MET) + " |")
    return L


def section_l(ru):
    ru.to_csv(os.path.join(RES, f"{P}_rule_b3.csv"), index=False)
    B = pd.read_csv(os.path.join(RES, "expB_baselines_raw.csv"), float_precision="round_trip")
    b6 = pd.read_csv(os.path.join(RES, "expB6_raw.csv"), float_precision="round_trip")
    m = lambda meth, N: ru[(ru.method == meth) & (ru.N == N)].sort_values("rep")
    for N in (10, 50, 100, 200, 400):
        assert np.allclose(m("B3", N).macroAUC.values, B[(B.method == "uninformative MAP") & (B.N == N)].sort_values("rep").macroAUC.values, atol=1e-12), N
        assert np.allclose(m("Proposed", N).macroAUC.values, B[(B.method == "Proposed") & (B.N == N)].sort_values("rep").macroAUC.values, atol=1e-12), N
    for N in (10, 20, 50, 100, 200, 400):
        q = b6[(b6.method == "Proposed") & (b6.N == N)].sort_values("rep")
        assert np.allclose(m("Proposed", N).auc_high.values, q.auc.values, atol=1e-12) and np.allclose(m("Proposed", N).auc_ge_medium.values, q.auc_ge_medium.values, atol=1e-12), N
    q = b6[(b6.method == "B6-pub") & (b6.N == 0)].sort_values("rep")
    r6 = m("B6-pub", 0)
    assert np.allclose(r6.auc_high.values, q.auc.values, atol=1e-12) and np.allclose(r6.brier_high.values, q.brier_bin.values, atol=1e-12) and np.allclose(r6.auc_ge_medium.values, q.auc_ge_medium.values, atol=1e-12)
    L = ["", "## (l) a network with a uniform prior mean against the rule of [5] (20 replications)",
         f"- the rule on T = high: AUC {r6.auc_high.mean():.4f}, Brier {r6.brier_high.mean():.4f}; scores above 0.75 are high-threat {r6.high_above.sum() / r6.n_above.sum():.4f} of the time ({int(r6.n_above.sum())} scenarios)",
         f"- the rule read as P(T >= medium): AUC {r6.auc_ge_medium.mean():.4f}, Brier {r6.brier_ge_medium.mean():.4f}; scores above 0.75 are at T >= medium {r6.ge_medium_above.sum() / r6.n_above.sum():.4f} of the time; constant forecast at the base rate {r6.base_ge_medium.mean():.4f}: Brier {(r6.base_ge_medium * (1 - r6.base_ge_medium)).mean():.4f}",
         "", "| N | B3 AUC, T = high | minus the rule (p) | B3 Brier, T = high | minus the rule (p) | Proposed AUC, T = high | Proposed AUC, T >= medium | Proposed Brier, T >= medium |", "|---|---|---|---|---|---|---|---|"]
    for N in (10, 20, 50, 100, 200, 400):
        a, c = m("B3", N), m("Proposed", N)
        L.append(f"| {N} | {a.auc_high.mean():.4f} | {cell(a[['rep', 'auc_high']], r6[['rep', 'auc_high']], 'auc_high')} | {a.brier_high.mean():.4f} | {cell(a[['rep', 'brier_high']], r6[['rep', 'brier_high']], 'brier_high')} | {c.auc_high.mean():.4f} | {c.auc_ge_medium.mean():.4f} | {c.brier_ge_medium.mean():.4f} |")
    bq = lambda meth, N: B[(B.method == meth) & (B.N == N)].sort_values("rep")
    L += ["", "| N | contrast with B0-as-prior MAP | ECE (p) | macro-AUC (p) |", "|---|---|---|---|"]
    for N in (10, 50):
        for meth in ("Proposed", "expert MAP (full table)"):
            L.append(f"| {N} | {meth} | {cell(bq(meth, N)[['rep', 'ece']], bq('B0-as-prior MAP', N)[['rep', 'ece']], 'ece')} | {cell(bq(meth, N)[['rep', 'macroAUC']], bq('B0-as-prior MAP', N)[['rep', 'macroAUC']], 'macroAUC')} |")
    return L


def section_m(tk):
    tk.to_csv(os.path.join(RES, f"{P}_track_split.csv"), index=False)
    ts = pd.read_csv(os.path.join(RES, "expTS_seeds.csv"), float_precision="round_trip").sort_values("seed")
    assert np.allclose(tk[tk.redrawn == "both"].sort_values("seed").p_high.values, ts.p_high.values, atol=1e-9)
    L = ["", "## (m) P(T = high) of the worked track, with the training data and the prior redrawn apart (seeds 0 to 19, calibration seed 777)", "| redrawn | mean | min | max |", "|---|---|---|---|"]
    for k in ("both", "training data", "prior"):
        q = tk[tk.redrawn == k]
        L.append(f"| {k} | {q.p_high.mean():.4f} | {q.p_high.min():.4f} | {q.p_high.max():.4f} |")
    return L


def section_n(fs):
    fs = fs.sort_values(["N", "rep"])
    fs.to_csv(os.path.join(RES, f"{P}_flags_small.csv"), index=False)
    L = ["", "## (n) false alarms of the additivity test at small N, standard world (B = 199)", "| N | flagged | reps |", "|---|---|---|"]
    for N, q in fs.groupby("N"):
        L.append(f"| {N} | {int(q.flag.sum())} | {len(q)} |")
    return L


def section_o():
    e = pd.read_csv(os.path.join(RES, "expED_raw.csv"), float_precision="round_trip")
    MET = ("acc", "macroAUC", "brier", "ece")
    ml = lambda N: e[(e.method == "MLE") & (e.N == N)].sort_values("seed")
    pr = lambda lam, N: e[(e.method == "Proposed") & (e.lam == lam) & (e.N == N)].sort_values("seed")
    rows = []
    for N in (10, 50, 200, 400):
        rows.append({"method": "MLE", "lam": np.nan, "N": N, **{k: float(ml(N)[k].mean()) for k in MET}})
    for lam in (0.0, 0.25, 0.5, 0.75, 1.0):
        for N in (10, 50, 200, 400):
            rows.append({"method": "Proposed", "lam": lam, "N": N, **{k: float(pr(lam, N)[k].mean()) for k in MET}})
    g = pd.DataFrame(rows)
    g.to_csv(os.path.join(RES, f"{P}_lambda.csv"), index=False)
    L = ["", "## (o) the reversed-prior grid at alpha0 = 2, from expED_raw.csv (20 seeds; test seed 999, calibration seed 777)", "| method | lambda | N | accuracy | macro-AUC | Brier | ECE |", "|---|---|---|---|---|---|---|"]
    for _, q in g.iterrows():
        lam = "-" if q.method == "MLE" else f"{q.lam:g}"
        L.append(f"| {q.method} | {lam} | {q.N} | {q.acc:.4f} | {q.macroAUC:.4f} | {q.brier:.4f} | {q.ece:.4f} |")
    L += ["", "| lambda | contrast | accuracy (p) | macro-AUC (p) | Brier (p) | ECE (p) |", "|---|---|---|---|---|---|"]
    for lam in (0.0, 0.25, 0.5, 0.75, 1.0):
        for N, M in ((10, 50), (50, 50), (10, 10), (200, 200), (400, 400)):
            L.append(f"| {lam:g} | Proposed at N={N} minus MLE at N={M} | " + " | ".join(cell(pr(lam, N)[["seed", k]], ml(M)[["seed", k]], k) for k in MET) + " |")
    return L


def section_p():
    k = pd.read_csv(os.path.join(RES, "expS_calibsize_raw.csv"), float_precision="round_trip")
    a, n30 = k[k.source == "abstract"].sort_values("rep"), k[(k.source == "estimated") & (k.calib_n == 30)].sort_values("rep")
    L = ["", "## (p) contrasts read from the result files", "- 30 labeled scenarios minus the abstract model: " + ", ".join(f"{m} {cell(n30[['rep', m]], a[['rep', m]], m)}" for m in ("acc", "macroAUC", "brier", "ece"))]
    s = pd.read_csv(os.path.join(RES, "expS_scarcity_raw.csv"), float_precision="round_trip")
    q = lambda meth, N: s[(s.method == meth) & (s.N == N)].sort_values("rep")
    L += ["", "| N | Proposed - B0: macro-AUC (p) | ECE (p) | FAR (p) | Proposed - MLE: FAR (p) |", "|---|---|---|---|---|"]
    for N in (10, 20, 50, 100, 200, 400):
        L.append(f"| {N} | " + " | ".join(cell(q("Proposed", N)[["rep", m]], q("B0-heuristic", N)[["rep", m]], m) for m in ("macroAUC", "ece", "far@pd0.9")) + f" | {cell(q('Proposed', N)[['rep', 'far@pd0.9']], q('B1-MLE', N)[['rep', 'far@pd0.9']], 'far@pd0.9')} |")
    f = pd.read_csv(os.path.join(RES, f"{P}_fusion.csv"), float_precision="round_trip")
    sel = lambda v, x: f[(f.variant == v) & (f.subset == x)].sort_values("rep")
    L += ["", "- the fitted discount minus none: " + ", ".join(f"{m} {cell(sel('calib+rel', 'all')[['rep', m]], sel('calibrated', 'all')[['rep', m]], m)}" for m in ("brier", "ece")) + f"; radio-silent Brier {cell(sel('calib+rel', 'radio-silent')[['rep', 'brier']], sel('calibrated', 'radio-silent')[['rep', 'brier']], 'brier')}"]
    return L


def section_q(hs):
    hs.to_csv(os.path.join(RES, f"{P}_handset.csv"), index=False)
    MET = ("acc", "macroAUC", "brier", "ece")
    sel = lambda v, d, s: hs[(hs.variant == v) & (hs.subset == s) & (hs.diagonal.isna() if d is None else hs.diagonal == d)].sort_values("rep")

    def qc(a, b, k):
        x = a[k].values - b[k].values
        assert len(x) == REPS
        x = x[~np.isnan(x)]
        return f"{x.mean():+.4f} ({st.ttest_1samp(x, 0.0).pvalue:.2g})" if len(x) >= 10 else f"n/a ({len(x)} of {REPS})"

    fu = pd.read_csv(os.path.join(RES, f"{P}_fusion.csv"), float_precision="round_trip")
    for v, d, w in (("hand-set", 0.94, "hand-set emission-aware"), ("abstract", 0.94, "abstract"), ("calibrated", None, "calibrated")):
        for s, x in (("all", "all"), ("radio-silent drones", "radio-silent")):
            ref = fu[(fu.variant == w) & (fu.subset == x)].sort_values("rep")
            for k in MET:
                assert np.allclose(sel(v, d, s)[k].values, ref[k].values, atol=1e-9, equal_nan=True), (v, s, k)
    ks = pd.read_csv(os.path.join(RES, "expS_calibsize_raw.csv"), float_precision="round_trip")
    n300 = ks[(ks.source == "estimated") & (ks.calib_n == 300)].sort_values("rep")
    for k in MET:
        assert np.allclose(sel("calibrated", None, "all")[k].values, n300[k].values, atol=1e-9), k
    L = ["", "## (q) the hand-set emission-aware table swept over its diagonal (N = 200, no discount, 20 replications; at 0.94 the rows of expRS_fusion.csv, calibrated the row n = 300 of expS_calibsize_raw.csv)", "| diagonal | accuracy | macro-F1 | macro-AUC | Brier | ECE | FAR |", "|---|---|---|---|---|---|---|"]
    q = sel("calibrated", None, "all")
    L.append(f"| calibrated | {q.acc.mean():.4f} | {q.macroF1.mean():.4f} | {q.macroAUC.mean():.4f} | {q.brier.mean():.4f} | {q.ece.mean():.4f} | {q.far.mean():.4f} |")
    for d in HAND_GRID:
        q = sel("hand-set", d, "all")
        L.append(f"| {d:.2f} | {q.acc.mean():.4f} | {q.macroF1.mean():.4f} | {q.macroAUC.mean():.4f} | {q.brier.mean():.4f} | {q.ece.mean():.4f} | {q.far.mean():.4f} |")
    L += ["", "Calibrated minus hand-set", "| subset | scenarios per replication | diagonal | accuracy (p) | macro-AUC (p) | Brier (p) | ECE (p) |", "|---|---|---|---|---|---|---|"]
    for s in HAND_SUBSETS:
        for d in HAND_GRID:
            a, b = sel("calibrated", None, s), sel("hand-set", d, s)
            L.append(f"| {s} | {a.n.mean():.0f} | {d:.2f} | " + " | ".join(qc(a, b, k) for k in MET) + " |")
    L += ["", "The mixed tables: is the RF row the cause?", "| contrast | subset | accuracy (p) | macro-AUC (p) | Brier (p) | ECE (p) |", "|---|---|---|---|---|---|"]
    for s in ("all", "radio-silent drones"):
        L.append(f"| estimated, RF row hand-set 0.94 - calibrated | {s} | " + " | ".join(qc(sel("estimated, RF row hand-set", 0.94, s), sel("calibrated", None, s), k) for k in MET) + " |")
        L.append(f"| hand-set 0.94, RF row estimated - hand-set 0.94 | {s} | " + " | ".join(qc(sel("hand-set, RF row estimated", 0.94, s), sel("hand-set", 0.94, s), k) for k in MET) + " |")
    L += ["", "The estimated matrices' correct-report probability (mean over the classes, 20 calibration splits, n = 300)", "| sensor | mean | smallest | largest |", "|---|---|---|---|"]
    for s in ("Rr", "Re", "Ra"):
        q = hs[(hs.variant == "estimated, correct-report probability") & (hs.subset == s)]
        assert len(q) == REPS
        L.append(f"| {s} | {q.acc.mean():.4f} | {q.acc.min():.4f} | {q.acc.max():.4f} |")
    L += ["", "By calibration size, minus hand-set at 0.75", "| matrices | accuracy (p) | macro-AUC (p) | Brier (p) | ECE (p) |", "|---|---|---|---|---|"]
    b = sel("hand-set", 0.75, "all")
    for lab, q in [("abstract", ks[ks.source == "abstract"])] + [(f"estimated, n = {n}", ks[(ks.source == "estimated") & (ks.calib_n == n)]) for n in (30, 100, 300, 1000)] + [("sensor oracle", ks[ks.source == "oracle-marginal"])]:
        L.append(f"| {lab} | " + " | ".join(qc(q.sort_values("rep"), b, k) for k in MET) + " |")
    L += ["", "At one level: the emission-aware RF row, no RF channel and the emission-blind RF row, all with the same diagonal", "| contrast | subset | diagonal | accuracy (p) | macro-AUC (p) | Brier (p) | ECE (p) |", "|---|---|---|---|---|---|---|"]
    for u, v in (("hand-set", "abstract"), ("no RF", "abstract"), ("hand-set", "no RF")):
        for s in ("all", "radio-silent drones", "non-drones"):
            for d in HAND_GRID:
                L.append(f"| {u} - {v} | {s} | {d:.2f} | " + " | ".join(qc(sel(u, d, s), sel(v, d, s), k) for k in MET) + " |")
    return L


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    rr = pd.DataFrame(rule_rows())
    rc = rule_control()
    sm = softmax_rows()
    ctl, moved = soft_control()
    tr = transfer_rows()
    fl = flag_rows()
    wr = weight_rows()
    L = ["# Exp-RS: controls behind sentences that give a reason", "", "## (a) the rule of [5] on RF silence", "| subset | method | N | AUC | Brier | mean output |", "|---|---|---|---|---|---|"]
    for _, q in rr.iterrows():
        L.append(f"| {q.subset} | {q.method} | {q.N} | {q.auc:.3f} | {q.brier:.3f} | {q.mean_p:.3f} |")
    L += ["", "Control: the published rule with the silence input off (same test sets, means over 20 replications)", "| subset | scenarios | observed high | mean, published | mean, silence off | Brier, published | Brier, silence off |", "|---|---|---|---|---|---|---|"]
    for s in ("silent-drone", "silent-report-nondrone", "RF report not none"):
        q = rc[rc.subset == s]
        L.append(f"| {s} | {q.n.mean():.1f} | {q.observed_high.mean():.3f} | {q.mean_published.mean():.3f} | {q.mean_silence_off.mean():.3f} | {q.brier_published.mean():.3f} | {q.brier_silence_off.mean():.3f} |")
    L += ["", "## (b) softmax concentration", "| split | runs | mean top | share top > 0.9 | median max(background, drones) | smallest of that median |", "|---|---|---|---|---|---|"]
    for pr in ("P0", "P1", "P1c", "P2", "P3"):
        q = sm[sm.proto == pr]
        if len(q):
            L.append(f"| {pr} | {len(q)} | {q.mean_top.mean():.3f} | {q['share_top_above_0.9'].mean():.3f} | {q.median_bg_split.mean():.4f} | {q.median_bg_split.min():.4f} |")
    L += [f"- runs: mean top from {sm.mean_top.min():.3f} to {sm.mean_top.max():.3f}; median of max(background, drones) at least {sm.median_bg_split.min():.4f}",
          f"- control: moving up to {moved:.3f} of softmax mass among the drone models moves P(T) by at most {ctl:.1e} (300 scenarios)"]
    L += ["", "## (c) binary-context weight of DroneDetect against no discount (five seeds)", "| fitted on | applied to | gain (nats) | p | log-lik., transfer | log-lik., own weight |", "|---|---|---|---|---|---|"]
    for _, q in tr.iterrows():
        L.append(f"| {q.fit} | {q.cond} | {q.gain_over_no_discount:+.3f} | {q.p:.2g} | {q.ll_transfer:.3f} | {q.ll_own:.3f} |")
    L += ["", "## (d) additivity test on the threat node, flagged of 20", "| world | N | flagged |", "|---|---|---|"]
    for _, q in fl.iterrows():
        L.append(f"| {q.world} | {q.N} | {q.flagged}/{q.reps} |")
    L += ["", "## (e) fitted reliability weights, calibration seeds 5000 + r, n = 300", "| sensor | range | mean | min | max | highest at this range (of 20) |", "|---|---|---|---|---|---|"]
    for _, q in wr.iterrows():
        L.append(f"| {q.sensor} | {q['range']} | {q['mean']:.3f} | {q['min']:.3f} | {q['max']:.3f} | {q.highest_here} |")
    # ---- (f) to (j)
    fb = pd.DataFrame([x for r in pooled(rf_bound_one, range(REPS)) for x in r])
    fb.to_csv(os.path.join(RES, f"{P}_rf_bound.csv"), index=False)
    gx = pd.read_csv(os.path.join(RES, "expG_rf_e2e_raw.csv"), float_precision="round_trip")
    rc = pd.read_csv(os.path.join(RES, "expRC_raw.csv"), float_precision="round_trip")
    for name, ref in (("hard", gx[gx.rf_likelihood == "calib-hard"]), ("soft", gx[gx.rf_likelihood == "calib-soft"]), ("none", rc[rc.variant == "no-RF"])):
        assert np.allclose(fb[fb.variant == name].sort_values("rep").macroAUC.values, ref.sort_values("rep").macroAUC.values, atol=1e-9), name
    L += ["", "## (f) the RF channel's upper bound (first split, 20 replications; hard and soft are the rows of expG_rf_e2e_raw.csv, none the row of expRC_raw.csv)", "| variant | accuracy | macro-AUC | Brier | ECE |", "|---|---|---|---|---|"]
    for name in ("none", "hard", "soft", "error-free split"):
        q = fb[fb.variant == name]
        L.append(f"| {name} | {q.acc.mean():.4f} | {q.macroAUC.mean():.4f} | {q.brier.mean():.4f} | {q.ece.mean():.4f} |")
    L += ["", "| contrast | accuracy (p) | macro-AUC (p) | Brier (p) | ECE (p) |", "|---|---|---|---|---|"]
    for a, b in (("hard", "none"), ("soft", "none"), ("error-free split", "none"), ("soft", "hard"), ("error-free split", "hard")):
        cells = [paired(fb[fb.variant == a], fb[fb.variant == b], k) for k in ("acc", "macroAUC", "brier", "ece")]
        L.append(f"| {a} - {b} | " + " | ".join(f"{v[0]:+.4f} ({v[1]:.2g})" for v in cells) + " |")
    sp, w_bg, w_model = split_rows()
    L += ["", f"## (g) the split on what the scenarios draw (background pool {w_bg:.4f}, each drone model {w_model:.4f}) and the test background", "| split | run | background spectrograms | segments | errors | pool median | scenario median | scenario below 0.9 | scenario mean top | background median | background below 0.9 |", "|---|---|---|---|---|---|---|---|---|---|---|"]
    for _, q in sp.iterrows():
        L.append(f"| {q.proto} | {q.run} | {q.background_spectrograms} | {q.background_segments} | {q.background_errors} | {q.pool_median:.4f} | {q.scenario_median:.4f} | {q['scenario_below_0.9']:.3f} | {q.scenario_mean_top:.3f} | {q.background_median:.4f} | {q['background_below_0.9']:.3f} |")
    L += [f"- runs: scenario median from {sp.scenario_median.min():.3f} to {sp.scenario_median.max():.3f}; scenario share below 0.9 from {sp['scenario_below_0.9'].min():.3f} to {sp['scenario_below_0.9'].max():.3f}; scenario mean top from {sp.scenario_mean_top.min():.3f} to {sp.scenario_mean_top.max():.3f}",
          f"- test background: {sp.background_segments.min()} to {sp.background_segments.max()} segments, {sp.background_spectrograms.min()} to {sp.background_spectrograms.max()} spectrograms, {sp.background_errors.min()} to {sp.background_errors.max()} errors"]
    rs = pd.DataFrame([x for r in pooled(rel_one, range(REPS)) for x in r])
    rs["mult"] = rs["mult"].astype(str)
    rs.to_csv(os.path.join(RES, f"{P}_rel_scale.csv"), index=False)
    fr = pd.read_csv(os.path.join(RES, "expF_rel_raw.csv"), float_precision="round_trip")
    fr = fr[(fr.part == "miscal") & (fr.world == "nominal")]
    base = rs[rs["mult"] == "1.0"].sort_values("seed")
    assert np.allclose(base.macroAUC.values, fr[fr.mult == 1.0].sort_values("seed").macroAUC.values, atol=1e-9)
    L += ["", "## (h) the fitted weights scaled, minus the fitted weights (20 seeds)", "| factor | ECE, mean | accuracy (p) | macro-AUC (p) | Brier (p) | ECE (p) |", "|---|---|---|---|---|---|"]
    for m in (1.5, 1.25, 0.75, 0.5):
        q = fr[fr.mult == m]
        L.append(f"| {m:g} | {q.ece.mean():.4f} | " + " | ".join(f"{v[0]:+.4f} ({v[1]:.2g})" for v in [paired(q[["seed", k]], base[["seed", k]], k) for k in ("acc", "macroAUC", "brier", "ece")]) + " |")
    for m in ("0.25", "0.0", "reports hidden"):
        q = rs[rs["mult"] == m]
        L.append(f"| {m} | {q.ece.mean():.4f} | " + " | ".join(f"{v[0]:+.4f} ({v[1]:.2g})" for v in [paired(q[["seed", k]], base[["seed", k]], k) for k in ("acc", "macroAUC", "brier", "ece")]) + " |")
    L.append(f"- fitted weights: ECE {base.ece.mean():.4f}, macro-AUC {base.macroAUC.mean():.4f}, Brier {base.brier.mean():.4f}, accuracy {base.acc.mean():.4f}")
    ro = pd.DataFrame([x for r in pooled(roots_one, range(REPS)) for x in r])
    ro.to_csv(os.path.join(RES, f"{P}_roots.csv"), index=False)
    MET = ("acc", "macroF1", "macroAUC", "brier", "ece", "cwece", "far@pd0.9")
    L += ["", "## (i) the root marginals of cls and emit, minus the specification (20 replications)", "| method | N | roots | " + " | ".join(MET) + " |", "|---|---|---|" + "---|" * len(MET)]
    for (m, n) in (("Proposed", 10), ("Proposed", 50), ("Proposed", 200), ("B1-MLE", 50)):
        g = ro[(ro.method == m) & (ro.N == n)]
        for name in ("calibration split", "training rows"):
            cells = [paired(g[g.roots == name][["rep", k]], g[g.roots == "specification"][["rep", k]], k) for k in MET]
            L.append(f"| {m} | {n} | {name} | " + " | ".join(f"{v[0]:+.4f} ({v[1]:.2g})" for v in cells) + " |")
    c = ro[ro.roots == "calibration split"].merge(ro[ro.roots == "specification"], on=["rep", "method", "N"], suffixes=("", "_s"))
    big = max(abs(c.groupby(["method", "N"])[k].mean() - c.groupby(["method", "N"])[k + "_s"].mean()).max() for k in MET)
    L.append(f"- calibration split: the largest mean change of any metric is {big:.4f}")
    for name in ("specification", "calibration split", "training rows"):
        a, b = ro[(ro.roots == name) & (ro.method == "Proposed") & (ro.N == 10)], ro[(ro.roots == name) & (ro.method == "B1-MLE") & (ro.N == 50)]
        cells = [paired(a[["rep", k]], b[["rep", k]], k) for k in ("acc", "macroAUC", "brier", "ece")]
        L.append(f"- Proposed at N=10 minus MLE at N=50, roots from {name}: " + ", ".join(f"{k} {v[0]:+.4f} (p={v[1]:.2g})" for k, v in zip(("acc", "macroAUC", "brier", "ece"), cells)))
    sevs = sorted(pd.read_csv(os.path.join(RES, "expF_corr_raw.csv")).severity.unique().tolist())
    ad = pd.DataFrame(pooled(adverse_one, [(s, seed) for s in sevs for seed in range(REPS)]))
    ad.to_csv(os.path.join(RES, f"{P}_adverse.csv"), index=False)
    cr = pd.read_csv(os.path.join(RES, "expF_corr_raw.csv"), float_precision="round_trip")
    cr = cr[cr.fusion == "naive-CI"].sort_values(["severity", "seed"])
    assert np.allclose(ad.sort_values(["severity", "seed"]).acc.values, cr.acc_adv.values, atol=1e-12) and np.allclose(ad.sort_values(["severity", "seed"]).ece.values, cr.ece_adv.values, atol=1e-9)
    L += ["", "## (j) adverse subset, naive model (20 seeds; accuracy and ECE are the rows of expF_corr_raw.csv)", "| severity | mean top probability | accuracy | top minus accuracy | ECE | accuracy, reports hidden |", "|---|---|---|---|---|---|"]
    for s in sevs:
        q = ad[ad.severity == s]
        L.append(f"| {s:g} | {q.mean_top.mean():.3f} | {q.acc.mean():.3f} | {(q.mean_top - q.acc).mean():+.3f} | {q.ece.mean():.3f} | {q.acc_reports_hidden.mean():.3f} |")
    L += section_k(pd.DataFrame([x for r in pooled(fusion_one, range(REPS)) for x in r]))
    L += section_l(pd.DataFrame([x for r in pooled(rule_one, range(REPS)) for x in r]))
    L += section_m(pd.DataFrame(pooled(track_one, [(k, s) for k in ("both", "training data", "prior") for s in range(REPS)])))
    L += section_n(pd.DataFrame(pooled(flag_small_one, [(N, r) for N in (10, 20) for r in range(REPS)], workers=16)))
    L += section_o()
    L += section_p()
    L += section_q(pd.DataFrame([x for r in pooled(handset_one, range(REPS)) for x in r]))
    txt = "\n".join(L) + "\n"
    with open(os.path.join(RES, f"{P}_summary.md"), "w", encoding="utf-8", newline="\n") as f:
        f.write(txt)
    print(txt)


if __name__ == "__main__":
    main()
