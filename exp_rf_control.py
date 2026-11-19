"""
Exp-RC: a control for the mixed-world RF study of Sec. VII-H: the same pipeline
with no RF channel at all. Core files, exp_rf_e2e.py and exp_rf_independent.py
are not modified (their functions are imported); results go to results/expRC_*.

Seeds are those of exp_rf_e2e.run and exp_rf_independent.e2e_protocol: test
scenarios 9000+r, calibration 5000+r (n = 300), training r (N = 200), Proposed
CPTs, 20 replications.

Variants computed here, on the same scenario draws:
  no-RF        radar, EO/IR and acoustic reports only, through the deployed path
               (metrics.posteriors with sensors_on = Rr, Re, Ra). The RF report
               is not instantiated. Checked against exp_rf_e2e.posteriors_rf4
               with a constant RF likelihood, which is the same model written
               the other way.
  synthetic-RF all four reports, the RF report drawn from the world's synthetic
               table and read through its calibration-split estimate and
               reliability weight, as in the synthetic study. Checked against
               results/expS_fusion_raw.csv.
Neither depends on the DroneRF split, since no spectrogram enters them. The
variants (i) abstract and (ii) calib-hard of each split (P0, P1, P1c, P2, P3)
are read from results/expI_e2e_raw.csv, replication by replication.

Contrasts: paired t-tests over the 20 replications, metric(A) - metric(B), on
accuracy, macro-AUC, Brier score and top-label ECE. Replication r of a split
uses CNN r mod 5 (P1, P1c, P2) or r mod 4 (P3), and P0 has one CNN, so these
p-values cover scenario draws and, at most, those few CNNs.

Per CNN: for P1, P1c, P2 and P3 the same contrasts with each CNN counted once
(the mean over its replications, then a paired t over 5 or 4 values), for
(ii) - (i), (ii) - no-RF and (iii) - (ii). P0 has one CNN, so it has none.

Run:  python exp_rf_control.py [--reps 20]
      python exp_rf_control.py --summary-only
Outputs: results/expRC_raw.csv, results/expRC_contrasts.csv,
         results/expRC_per_cnn.csv, results/expRC_summary.md
"""
from __future__ import annotations
import os, sys, time, argparse
import numpy as np
import pandas as pd
from scipy import stats as st

import model_spec as ms
import dataio
import metrics as mt
import pipeline as pl
import exp_rf_e2e as eg

RES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
P = "expRC"
SENSORS3 = ["Rr", "Re", "Ra"]
METRICS = ("acc", "macroAUC", "brier", "ece")
PROTOS = ("P0", "P1", "P1c", "P2", "P3")
N_CNN = {"P1": 5, "P1c": 5, "P2": 5, "P3": 4}


def row(rep, name, Pm, y):
    return {"rep": rep, "variant": name, "acc": float((Pm.argmax(1) == y).mean()),
            "macroAUC": eg.macro_auc(Pm, y), "brier": float(mt.brier(Pm, y)),
            "ece": float(mt.ece(Pm, y))}


def run(reps=20):
    gt = ms.ground_truth_bn()
    rows, t0 = [], time.time()
    for r in range(reps):
        test = dataio.generate_dataset(gt, eg.TEST_N, seed=9000 + r)
        calib = dataio.generate_dataset(gt, 300, seed=5000 + r)
        est = dataio.estimate_confusion(calib)
        alpha, _ = dataio.fit_reliability(calib)
        train = dataio.generate_dataset(gt, eg.GROUND_N, seed=r)
        cpts, _ = pl.ground_internal("Proposed", train, r, gt=gt)
        Pn, y = mt.posteriors(cpts, test, SENSORS3, sensor_cpts=est, alpha=alpha)
        rows.append(row(r, "no-RF", Pn, y))
        Ps, ys = mt.posteriors(cpts, test, ms.SENSORS, sensor_cpts=est, alpha=alpha)
        assert np.array_equal(y, ys)
        rows.append(row(r, "synthetic-RF", Ps, y))
        if r == 0:
            flat = np.full((ms.CARD["cls"], ms.CARD["emit"], 4), 0.25)
            Pc, yc = eg.posteriors_rf4(cpts, test[:300], None, np.zeros(300, dtype=int), est,
                                       "abstract", L_hard=flat, alpha=alpha)
            assert np.array_equal(yc, y[:300]) and np.abs(Pc - Pn[:300]).max() < 1e-12
        print(f"[expRC] rep {r + 1}/{reps} ({time.time() - t0:.0f}s)", flush=True)
    pd.DataFrame(rows).to_csv(os.path.join(RES, f"{P}_raw.csv"), index=False)


def paired(a, b):
    d = np.asarray(a, float) - np.asarray(b, float)
    return float(d.mean()), float(st.ttest_1samp(d, 0.0).pvalue)


def summarize():
    raw = pd.read_csv(os.path.join(RES, f"{P}_raw.csv"), float_precision="round_trip").sort_values("rep")
    e2 = pd.read_csv(os.path.join(RES, "expI_e2e_raw.csv"), float_precision="round_trip").sort_values("rep")
    fu = pd.read_csv(os.path.join(RES, "expS_fusion_raw.csv"), float_precision="round_trip")
    L = ["# Exp-RC: the mixed-world RF study against a control with no RF channel", ""]
    no, sy = raw[raw.variant == "no-RF"], raw[raw.variant == "synthetic-RF"]
    reps = len(no)
    assert reps == 20 and len(sy) == 20
    # the synthetic-RF variant is the main study's all-sensor fusion on the same seeds
    mine = fu[fu.config == "all (calib+reliab)"].sort_values("rep")
    assert len(mine) == reps
    same = all(np.allclose(mine[k].values, sy[k].values, rtol=0, atol=1e-12) for k in ("acc", "macroAUC", "ece"))
    L.append(f"- synthetic-RF equals expS_fusion_raw.csv on accuracy, macro-AUC and ECE over {reps} replications: {same}")
    assert same
    L.append("\n## variants (means over 20 replications)")
    L.append("| variant | acc | macro-AUC | Brier | ECE |\n|---|---|---|---|---|")
    for nm, q in (("no-RF", no), ("synthetic-RF", sy)):
        L.append(f"| {nm} | {q.acc.mean():.4f} | {q.macroAUC.mean():.4f} | {q.brier.mean():.4f} | {q.ece.mean():.4f} |")
    for proto in PROTOS:
        for v in ("abstract", "calib-hard"):
            q = e2[(e2.proto == proto) & (e2.rf_likelihood == v)]
            assert len(q) == 20
            L.append(f"| {proto} {v} | {q.acc.mean():.4f} | {q.macroAUC.mean():.4f} | {q.brier.mean():.4f} | {q.ece.mean():.4f} |")
    rows = []
    L.append("\n## contrasts (A - B, paired t over 20 replications)")
    L.append("| split | A | B | d acc (p) | d macro-AUC (p) | d Brier (p) | d ECE (p) |\n|---|---|---|---|---|---|---|")

    def add(proto, an, a, bn, b):
        cells, rec = [], {"proto": proto, "A": an, "B": bn}
        for k in METRICS:
            m, p = paired(a[k].values, b[k].values)
            rec[f"d_{k}"], rec[f"p_{k}"] = m, p
            cells.append(f"{m:+.4f} ({p:.2g})")
        rows.append(rec)
        L.append(f"| {proto} | {an} | {bn} | " + " | ".join(cells) + " |")
    add("-", "synthetic-RF", sy, "no-RF", no)
    for proto in PROTOS:
        ab = e2[(e2.proto == proto) & (e2.rf_likelihood == "abstract")]
        ch = e2[(e2.proto == proto) & (e2.rf_likelihood == "calib-hard")]
        add(proto, "abstract", ab, "no-RF", no)
        add(proto, "calib-hard", ch, "no-RF", no)
        add(proto, "calib-hard", ch, "synthetic-RF", sy)
    pd.DataFrame(rows).to_csv(os.path.join(RES, f"{P}_contrasts.csv"), index=False)
    L.append("\n## the same contrasts with each CNN counted once (mean over its replications, paired t over the CNNs)")
    L.append("| split | CNNs | A | B | d acc (p) | d macro-AUC (p) | d Brier (p) | d ECE (p) |\n|---|---|---|---|---|---|---|---|")
    per = []
    for proto, k in N_CNN.items():
        ab = e2[(e2.proto == proto) & (e2.rf_likelihood == "abstract")]
        ch = e2[(e2.proto == proto) & (e2.rf_likelihood == "calib-hard")]
        cs = e2[(e2.proto == proto) & (e2.rf_likelihood == "calib-soft")]
        for an, a, bn, b in (("calib-hard", ch, "abstract", ab), ("calib-hard", ch, "no-RF", no), ("calib-soft", cs, "calib-hard", ch)):
            assert len(a) == 20 and len(b) == 20
            cells, rec = [], {"proto": proto, "n_cnn": k, "A": an, "B": bn}
            for m in METRICS:
                g = pd.Series(a[m].values - b[m].values).groupby(np.arange(20) % k).mean().values
                assert len(g) == k
                d, pv = float(g.mean()), float(st.ttest_1samp(g, 0.0).pvalue)
                rec[f"d_{m}"], rec[f"p_{m}"] = d, pv
                cells.append(f"{d:+.4f} ({pv:.2g})")
            per.append(rec)
            L.append(f"| {proto} | {k} | {an} | {bn} | " + " | ".join(cells) + " |")
    pd.DataFrame(per).to_csv(os.path.join(RES, f"{P}_per_cnn.csv"), index=False)
    txt = "\n".join(L) + "\n"
    with open(os.path.join(RES, f"{P}_summary.md"), "w", encoding="utf-8", newline="\n") as f:
        f.write(txt)
    sys.stdout.reconfigure(encoding="utf-8")
    print(txt)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=20)
    ap.add_argument("--summary-only", action="store_true")
    a = ap.parse_args()
    if not a.summary_only:
        run(a.reps)
    summarize()
