"""
Exp-B (new-reviewer round): standard baselines and a realistic-expert prior.

The manuscript's original comparison set was entirely author-constructed
(B0 surrogate, unsmoothed MLE, prior-only, abstract sensor model). A reader
cannot then tell whether the sample-efficiency gain comes from the expert
knowledge or merely from smoothing, and the introduction's claim that
discriminative models need more labels than a defense setting affords went
untested. This script adds the baselines that settle both questions, over
the SAME 20 replications and seeds as the main study:

  uninformative MAP   Dirichlet posterior mean with a UNIFORM prior mean at
                      the same ESS as the proposed method (the BDeu-style
                      default of BN parameter learning) -- isolates smoothing
                      from knowledge.
  Laplace             add-one counts (the other standard default).
  B1-MLE              unsmoothed maximum likelihood (already in the paper).
  B0-as-prior MAP     the hand-set surrogate table used as the PRIOR MEAN --
                      the "realistic expert" condition: what a practitioner
                      with the base framework's rules, and nothing better,
                      would actually have.
  logistic regression L2 multinomial logistic regression on one-hot encoded
                      observed evidence (same N, same features the BN sees).

Run:  python exp_baselines.py [--reps 20]
Outputs: results/expB_baselines_raw.csv, results/expB_summary.md
"""
from __future__ import annotations
import os, sys, time, argparse
import numpy as np
import pandas as pd
from scipy import stats as st

import model_spec as ms
import dataio
import grounding as g
import metrics as mt
import pipeline as pl
from cuas_bn import Factor

RES = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(RES, exist_ok=True)
TEST_N = 2500
N_GRID = (10, 50, 80, 100, 200, 400)   # 80 matches the Sec. VII-B ablation
METHODS = ["Proposed", "uninformative MAP", "Laplace", "B1-MLE",
           "B0-as-prior MAP", "logistic regression",
           "expert MAP (full table)", "expert cMAP (full table)"]
# The last two exist so the Sec. VII-B ablation can be split along a single
# path with exactly one thing changing per step:
#   B1-MLE -> uninformative MAP   : Dirichlet smoothing only
#          -> expert MAP          : the elicited prior mean only
#          -> expert cMAP         : the monotonicity constraints only
#          -> Proposed            : the ordinal ICI threat node only
# Without them, "Proposed - uninformative MAP" silently bundles the prior
# mean, the constraints and the ICI parameterization into one number.


def uniform_prior_cpts():
    out = {}
    for n in ms.INTERNAL:
        pa = ms.PARENTS[n]
        shape = tuple(ms.CARD[p] for p in pa) + (ms.CARD[n],)
        out[n] = Factor(tuple(pa) + (n,), np.full(shape, 1.0 / ms.CARD[n]))
    return out


def laplace_cpt(node, data):
    n = g.counts(node, data) + 1.0
    return Factor(tuple(ms.PARENTS[node]) + (node,),
                  n / n.sum(axis=-1, keepdims=True))


def logreg_posteriors(train, test):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import OneHotEncoder
    feats = ms.OBSERVED + ms.SENSORS

    def X(rows):
        return np.array([[r[f] for f in feats] for r in rows])
    enc = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    Xtr = enc.fit_transform(X(train))
    ytr = np.array([r["T"] for r in train])
    yte = np.array([r["T"] for r in test])
    if len(set(ytr)) < 2:
        return None, yte
    lr = LogisticRegression(max_iter=2000, C=1.0).fit(Xtr, ytr)
    P = np.zeros((len(yte), ms.CARD["T"]))
    P[:, lr.classes_] = lr.predict_proba(enc.transform(X(test)))
    return np.clip(P, 1e-9, None), yte


def metrics_of(P, y):
    from sklearn.metrics import roc_auc_score, f1_score
    try:
        auc = float(roc_auc_score(np.eye(4)[y], P, multi_class="ovr",
                                  average="macro", labels=[0, 1, 2, 3]))
    except Exception:
        auc = np.nan
    return {"acc": float((P.argmax(1) == y).mean()),
            "macroF1": float(f1_score(y, P.argmax(1), labels=[0, 1, 2, 3],
                                      average="macro", zero_division=0)),
            "macroAUC": auc, "brier": mt.brier(P, y), "ece": mt.ece(P, y),
            "cwece": mt.classwise_ece(P, y),
            "far@pd0.9": mt.far_at_pd(P, y, 0.90)}


def run(reps=20):
    gt = ms.ground_truth_bn()
    unif = uniform_prior_cpts()
    b0 = ms.heuristic_cpts()
    rows = []
    t0 = time.time()
    for r in range(reps):
        test = dataio.generate_dataset(gt, TEST_N, seed=9000 + r)
        calib = dataio.generate_dataset(gt, 300, seed=5000 + r)
        est = dataio.estimate_confusion(calib)
        alpha, _ = dataio.fit_reliability(calib)
        for N in N_GRID:
            data = dataio.generate_dataset(gt, N, seed=r)
            variants = {
                "Proposed": pl.ground_internal("Proposed", data, r, gt=gt)[0],
                "uninformative MAP": {n: g.map_cpt(n, data, unif[n],
                                                   pl.ESS_FIXED)
                                      for n in ms.INTERNAL},
                "Laplace": {n: laplace_cpt(n, data) for n in ms.INTERNAL},
                "B1-MLE": {n: g.mle_cpt(n, data) for n in ms.INTERNAL},
                "B0-as-prior MAP": {n: g.map_cpt(n, data, b0[n], pl.ESS_FIXED)
                                    for n in ms.INTERNAL},
                "expert MAP (full table)":
                    pl.ground_internal("MAP", data, r, gt=gt)[0],
                "expert cMAP (full table)":
                    pl.ground_internal("cMAP-full", data, r, gt=gt)[0],
            }
            for name, cpts in variants.items():
                m, _ = mt.evaluate(cpts, test, sensor_cpts=est, alpha=alpha)
                rows.append({"rep": r, "N": N, "method": name, **m})
            P, y = logreg_posteriors(data, test)
            if P is not None:
                rows.append({"rep": r, "N": N, "method": "logistic regression",
                             **metrics_of(P, y)})
        print(f"[expB] rep {r+1}/{reps} ({time.time()-t0:.0f}s)", flush=True)
        pd.DataFrame(rows).to_csv(
            os.path.join(RES, "expB_baselines_raw.csv"), index=False)
    summarize()
    print(f"[expB] ALL DONE in {time.time()-t0:.0f}s", flush=True)


def summarize():
    df = pd.read_csv(os.path.join(RES, "expB_baselines_raw.csv"))
    lines = ["# Exp-B: standard baselines and the realistic-expert prior", ""]

    def ci(d, k):
        x = d[k].dropna().values
        h = x.std(ddof=1) / np.sqrt(len(x)) * st.t.ppf(0.975, len(x) - 1)
        return x.mean(), h

    lines.append("## means +- 95% CI half-width")
    for N in N_GRID:
        lines.append(f"### N={N}")
        for m in METHODS:
            d = df[(df.N == N) & (df.method == m)]
            if len(d) == 0:
                continue
            parts = []
            for k in ("acc", "macroAUC", "ece", "brier"):
                mu, h = ci(d, k)
                parts.append(f"{k} {mu:.3f}+-{h:.3f}")
            lines.append(f"- {m:20s} " + ", ".join(parts))

    lines.append("\n## paired: Proposed - baseline")
    for N in N_GRID:
        for m in METHODS[1:]:
            a = df[(df.N == N) & (df.method == "Proposed")].set_index("rep")
            b = df[(df.N == N) & (df.method == m)].set_index("rep")
            ix = a.index.intersection(b.index)
            if len(ix) < 2:
                continue
            out = []
            for k in ("macroAUC", "ece", "brier"):
                d = (a.loc[ix, k] - b.loc[ix, k]).dropna().values
                out.append(f"d{k} {d.mean():+.3f} "
                           f"(p={st.ttest_1samp(d, 0).pvalue:.1e})")
            lines.append(f"- N={N} vs {m}: " + ", ".join(out))

    lines.append("\n## realistic-expert crossing: B0-as-prior vs MLE")
    for N in N_GRID:
        a = df[(df.N == N) & (df.method == "B0-as-prior MAP")].set_index("rep")
        b = df[(df.N == N) & (df.method == "B1-MLE")].set_index("rep")
        ix = a.index.intersection(b.index)
        out = []
        for k in ("acc", "macroAUC", "ece", "brier"):
            d = (a.loc[ix, k] - b.loc[ix, k]).dropna().values
            out.append(f"d{k} {d.mean():+.3f} "
                       f"(p={st.ttest_1samp(d, 0).pvalue:.2f})")
        lines.append(f"- N={N}: " + ", ".join(out))

    txt = "\n".join(lines)
    with open(os.path.join(RES, "expB_summary.md"), "w",
              encoding="utf-8") as f:
        f.write(txt + "\n")
    print(txt.encode(sys.stdout.encoding or "utf-8", errors="replace")
             .decode(sys.stdout.encoding or "utf-8", errors="replace"))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=20)
    ap.add_argument("--summary-only", action="store_true")
    a = ap.parse_args()
    if a.summary_only:
        summarize()
    else:
        run(a.reps)
