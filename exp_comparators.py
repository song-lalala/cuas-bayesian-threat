"""
Exp-CM: the free values of the comparators, the attribution of the gains to
the prior mean or to the structure, and two checks on the world. Added after
the main results; nothing here was part of the main study's plan, and the
values chosen for the comparators below were chosen after the fact. Core
files are not modified; results go to results/expCM_*. Seeds as in the main
study: training data seed r, calibration 5000+r with n = 300, test 9000+r with
2,500 scenarios, expert prior rng(1000+r), 20 replications, deployed sensor
layer (estimated matrices and fitted weights) unless stated.

Stages
  factorial  prior mean (uniform, elicited) x structure (full table, monotone
             constraints, constraints + ordinal ICI threat node) x ESS per
             column (uniform 0.01, 0.05, 0.1, 0.2, 0.5, 1, 2; elicited 0.2,
             0.5, 1, 2, 5), BDeu full tables (total ESS 1, 2, 5, 10, 20, so
             ESS / q per column), MLE, and Proposed through the pipeline,
             at N in {10, 20, 50, 80, 100, 200, 400}.
  b0         the B0 surrogate with its four tau multiplied by f in {1.0, 0.8,
             0.6, 0.5, 0.4, 0.3}, alone and as a prior mean (ESS 2, full
             tables). f = 1.0 is model_spec.heuristic_cpts.
  b5         the logistic baseline at L2 strength C in {0.001, 0.003, 0.01,
             0.03, 0.1, 0.3, 1, 3, 10}, N in {10, 50, 200, 400}.
  scalar     one correct-report probability per sensor, estimated on the
             calibration split and put into the hand-set emission-aware table,
             against the estimated matrices, n in {30, 100, 300, 1000}
             (Proposed at N = 200, no reliability discount).
  errors     the CNN errors of expI_node_dronerf.csv under the scenarios' mix
             of RF evidence, not the test pool's.
  world      P(T = high), P(T >= medium), P(N = top) by class and emission, by
             exact inference, in the standard world and in emit-drone-only.
  sens       the world 'emit-drone-only' of world_misspec.py: emission acts on
             N for drones only. Same convention as exp_misspecification.py
             (N = 50, 200; the expert prior comes from the standard world),
             with Proposed at N = 10 against MLE at N = 50 and the value of
             the RF channel (calibrated - no RF) at N = 200.
  s1         the non-additive worlds S1 and the standard world: Proposed at
             N = 10 against MLE at N = 50, and the constrained full table with
             the elicited or a uniform mean at ESS 2 or 0.2, N in {10, 50, 200}.
  summary    expCM_summary.md, from the files above.

Run:  python exp_comparators.py [--stage STAGE] [--workers 8] [--reps 20]
"""
from __future__ import annotations
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import argparse, re, sys, time, warnings
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import pandas as pd
from scipy import stats as st

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
P = "expCM"
REPS = 20
TEST_N, CALIB_N = 2500, 300
NS = (10, 20, 50, 80, 100, 200, 400)
ESS_UNIFORM = (0.01, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0)
ESS_ELICITED = (0.2, 0.5, 1.0, 2.0, 5.0)
ESS_BDEU = (1.0, 2.0, 5.0, 10.0, 20.0)
F_TAU = (1.0, 0.8, 0.6, 0.5, 0.4, 0.3)
C_L2 = (0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0)
N_B5 = (10, 50, 200, 400)
N_CAL = (30, 100, 300, 1000)
METRICS = ("acc", "macroF1", "macroAUC", "brier", "ece", "far@pd0.9")


def _quiet():
    warnings.filterwarnings("ignore")
    np.seterr(all="ignore")


def _setup(r, wbn=None):
    import model_spec as ms, dataio
    w = ms.ground_truth_bn() if wbn is None else wbn
    test = dataio.generate_dataset(w, TEST_N, seed=9000 + r)
    calib = dataio.generate_dataset(w, CALIB_N, seed=5000 + r)
    est = dataio.estimate_confusion(calib)
    alpha, _ = dataio.fit_reliability(calib)
    return w, test, est, alpha


def _pick(m):
    return {k: m[k] for k in METRICS}


def factorial_one(r):
    _quiet()
    import model_spec as ms, dataio, grounding as g, metrics as mt, pipeline as pl, exp_baselines as eb
    gt, test, est, alpha = _setup(r)
    unif = eb.uniform_prior_cpts()
    prior = ms.expert_prior_cpts(np.random.default_rng(1000 + r))
    q = {n: int(np.prod([ms.CARD[p] for p in ms.PARENTS[n]])) for n in ms.INTERNAL}
    rows = []

    def add(N, mean, structure, a0, cpts):
        m, _ = mt.evaluate(cpts, test, sensor_cpts=est, alpha=alpha)
        rows.append({"rep": r, "N": N, "mean": mean, "structure": structure, "a0": a0, **_pick(m)})

    for N in NS:
        data = dataio.generate_dataset(gt, N, seed=r)
        add(N, "none", "full", 0.0, {n: g.mle_cpt(n, data) for n in ms.INTERNAL})
        add(N, "Proposed", "pipeline", 2.0, pl.ground_internal("Proposed", data, r, gt=gt)[0])
        for mean, pr, grid in (("uniform", unif, ESS_UNIFORM), ("elicited", prior, ESS_ELICITED)):
            for a in grid:
                add(N, mean, "full", a, {n: g.map_cpt(n, data, pr[n], a) for n in ms.INTERNAL})
                hnc = {n: g.cmap_cpt(n, data, pr[n], a) for n in ("H", "N", "C")}
                add(N, mean, "constrained", a, {**hnc, "T": g.cmap_cpt("T", data, pr["T"], a)})
                add(N, mean, "constrained+ICI", a, {**hnc, "T": g.ordinal_ici_cpt("T", data, pr["T"], a, seed=r)})
        for e in ESS_BDEU:
            add(N, "BDeu", "full", e, {n: g.map_cpt(n, data, unif[n], e / q[n]) for n in ms.INTERNAL})
    return rows, dict(g.CMAP_STATS)


def b0_tables(f):
    import model_spec as ms
    return {"H": ms._ordinal_cpt("H", {"v": 0.7, "rdot": 0.7, "d": 0.7}, 0.1, 0.95 * f),
            "N": ms._ordinal_cpt("N", {"cls": 1.0, "emit": 0.6}, 0.0, 0.9 * f),
            "C": ms._ordinal_cpt("C", {"d": 0.8, "z": 0.8}, 0.1, 0.95 * f),
            "T": ms._ordinal_cpt("T", {"H": 0.8, "N": 0.8, "C": 0.8}, 0.05, 1.0 * f)}


def b0_one(r):
    _quiet()
    import model_spec as ms, dataio, grounding as g, metrics as mt
    gt, test, est, alpha = _setup(r)
    h, b1 = ms.heuristic_cpts(), b0_tables(1.0)
    assert all(h[k].table.tobytes() == b1[k].table.tobytes() for k in ms.INTERNAL)
    rows = []
    for f in F_TAU:
        b = b0_tables(f)
        m, _ = mt.evaluate(b, test, sensor_cpts=est, alpha=alpha)
        rows.append({"rep": r, "f": f, "use": "alone", "N": 0, **_pick(m)})
        for N in NS:
            data = dataio.generate_dataset(gt, N, seed=r)
            c = {n: g.map_cpt(n, data, b[n], 2.0) for n in ms.INTERNAL}
            m, _ = mt.evaluate(c, test, sensor_cpts=est, alpha=alpha)
            rows.append({"rep": r, "f": f, "use": "prior", "N": N, **_pick(m)})
    return rows, {}


def b5_one(r):
    _quiet()
    import model_spec as ms, dataio, exp_baselines as eb
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import OneHotEncoder
    gt = ms.ground_truth_bn()
    test = dataio.generate_dataset(gt, TEST_N, seed=9000 + r)
    feats = ms.OBSERVED + ms.SENSORS
    X = lambda rs: np.array([[x[k] for k in feats] for x in rs])
    yte = np.array([x["T"] for x in test])
    rows = []
    for N in N_B5:
        data = dataio.generate_dataset(gt, N, seed=r)
        ytr = np.array([x["T"] for x in data])
        if len(set(ytr)) < 2:
            continue
        enc = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        Xtr, Xte = enc.fit_transform(X(data)), None
        Xte = enc.transform(X(test))
        for C in C_L2:
            lr = LogisticRegression(max_iter=5000, C=C).fit(Xtr, ytr)
            Pm = np.zeros((len(yte), ms.CARD["T"]))
            Pm[:, lr.classes_] = lr.predict_proba(Xte)
            m = eb.metrics_of(np.clip(Pm, 1e-9, None), yte)
            rows.append({"rep": r, "C": C, "N": N, **_pick(m)})
    return rows, {}


def one_number_tables(est):
    """One correct-report probability per sensor in the hand-set emission-aware table; RF's counts every class and emission state alike."""
    import model_spec as ms, dataio, exp_reason_checks as rc
    tabs, nums = {}, {}
    for s in ("Rr", "Re", "Ra"):
        c = dataio.CORRECT_REPORT[s]
        d = float(np.mean([est[s].table[k, c[k]] for k in range(ms.CARD["cls"])]))
        tabs[s], nums[s] = dataio.abstract_sensor_cpts_diag(d)[s], d
    rf = est["Rf"].table
    d = float(np.mean([rf[2, 0, 0] + rf[2, 0, 1]] + [rf[c, e, 2] for c in range(ms.CARD["cls"]) for e in range(ms.CARD["emit"]) if not (c == 2 and e == 0)]))
    tabs["Rf"], nums["Rf"] = rc.hand_rf(d), d
    return tabs, nums


def scalar_one(r):
    _quiet()
    import model_spec as ms, dataio, metrics as mt, pipeline as pl
    gt = ms.ground_truth_bn()
    test = dataio.generate_dataset(gt, TEST_N, seed=9000 + r)
    cp, _ = pl.ground_internal("Proposed", dataio.generate_dataset(gt, 200, seed=r), r, gt=gt)
    rows = []
    for n in N_CAL:
        calib = dataio.generate_dataset(gt, n, seed=5000 + r)
        est = dataio.estimate_confusion(calib)
        tabs, nums = one_number_tables(est)
        for layer, tb in (("matrix", est), ("one number", tabs)):
            m, _ = mt.evaluate(cp, test, sensor_cpts=tb, alpha=None)
            rows.append({"rep": r, "n": n, "layer": layer, **_pick(m), **({f"d_{k}": v for k, v in nums.items()} if layer == "one number" else {})})
    return rows, {}


def run_stage(fn, items, workers, out):
    t0 = time.time()
    rows, stats = [], {}
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for i, (x, s) in enumerate(ex.map(fn, items)):
            rows.extend(x)
            for k, v in s.items():
                stats[k] = stats.get(k, 0) + v
            print(f"[{P}] {out} {i + 1}/{len(items)} ({time.time() - t0:.0f}s)", flush=True)
    pd.DataFrame(rows).to_csv(os.path.join(RES, f"{P}_{out}.csv"), index=False)
    if stats:
        print(f"[{P}] {out} solver {stats}", flush=True)
    return stats


def scenario_mix():
    """P(RF evidence is background) and P(each drone model) under the world: an emitting drone draws a random model, everything else draws background."""
    import model_spec as ms
    gt = ms.ground_truth_bn()
    pc, pe = float(gt.cpts["cls"].table[2]), float(gt.cpts["emit"].table[0])
    m = pc * pe / 3
    return np.array([1 - pc * pe, m, m, m])


def run_errors():
    nd = pd.read_csv(os.path.join(RES, "expI_node_dronerf.csv"), float_precision="round_trip")
    w = scenario_mix()
    rows = []
    for _, r in nd.iterrows():
        M = np.array([float(x) for x in re.findall(r"-?\d+\.?\d*", str(r.confusion))]).reshape(4, 4)
        R = M / M.sum(axis=1, keepdims=True)
        err = float((w * (1 - np.diag(R))).sum())
        dd = float(sum(w[i] * R[i, j] for i in (1, 2, 3) for j in (1, 2, 3) if i != j))
        pool_err = M.sum() - np.trace(M)
        pool_dd = M[1:, 1:].sum() - np.trace(M[1:, 1:])
        rows.append({"proto": r.proto, "key": r.key, "pool_acc": np.trace(M) / M.sum(), "pool_share_models": pool_dd / pool_err, "pool_drone_background": (pool_err - pool_dd) / M.sum(),
                     "scenario_error": err, "scenario_share_models": dd / err, "scenario_drone_background": err - dd})
    pd.DataFrame(rows).to_csv(os.path.join(RES, f"{P}_node_errors.csv"), index=False)
    print(f"[{P}] node_errors {len(rows)} rows, background share of the scenarios {w[0]:.4f}", flush=True)


def run_world():
    import model_spec as ms, world_misspec as wm
    rows = []
    for name in ("standard", "emit-drone-only"):
        b = wm.world_bn(name)
        for c, cn in enumerate(ms.STATES["cls"]):
            for e, en in enumerate(ms.STATES["emit"]):
                q = b.query(["T"], {"cls": c, "emit": e}).table
                n = b.query(["N"], {"cls": c, "emit": e}).table
                rows.append({"world": name, "cls": cn, "emit": en, "p_high": float(q[3]), "p_medium_up": float(q[2:].sum()), "p_N_top": float(n[-1])})
        rows.append({"world": name, "cls": "all", "emit": "all", "p_high": float(b.query(["T"]).table[3]), "p_medium_up": float(b.query(["T"]).table[2:].sum()), "p_N_top": float(b.query(["N"]).table[-1])})
    pd.DataFrame(rows).to_csv(os.path.join(RES, f"{P}_world.csv"), index=False)
    print(f"[{P}] world {len(rows)} rows", flush=True)


def sens_one(a):
    """Exp_misspecification's conventions, and the two contrasts it does not have."""
    _quiet()
    wname, r = a
    import model_spec as ms, dataio, metrics as mt, pipeline as pl, world_misspec as wm, world_options as wo
    import exp_misspecification as em
    wbn = wm.world_bn(wname)
    test = dataio.generate_dataset(wbn, TEST_N, seed=9000 + r)
    y = np.array([s["T"] for s in test])
    calib = dataio.generate_dataset(wbn, CALIB_N, seed=5000 + r)
    est = dataio.estimate_confusion(calib)
    alpha, _ = dataio.fit_reliability(calib)
    big = dataio.generate_dataset(wbn, max(em.NS), seed=r)
    rows = []
    fits200 = None
    for N in em.NS:
        data = big[:N]
        fits = em.fit_methods(data, r)
        if N == 200:
            fits200 = fits
        for m, Pm in em.eval_methods(fits, data, test, est, alpha).items():
            rows.append({"world": wname, "rep": r, "N": N, "method": m, **em.metrics_all(Pm, y)})
    data10 = big[:10]
    Pm = mt.posteriors(pl.ground_internal("Proposed", data10, r)[0], test, ms.SENSORS, "calibrated", sensor_cpts=est, alpha=alpha)[0]
    rows.append({"world": wname, "rep": r, "N": 10, "method": "Proposed", **em.metrics_all(Pm, y)})
    tabs_m = wo.true_marginal_tables(wbn)
    Po = {"Bayes-oracle": em.bn_posteriors(wbn, test), "CPT-oracle": em.bn_posteriors(em.model_bn(wbn, em.population_tables(wbn), tabs_m), test)}
    for m, Pm in Po.items():
        rows.append({"world": wname, "rep": r, "N": -1, "method": m, **em.metrics_all(Pm, y)})
    # the RF channel at N = 200, with the fitted discount as deployed and without
    no_rf = [s for s in ms.SENSORS if s != "Rf"]
    assert len(no_rf) == 3
    cp = fits200["Proposed"]
    cls_, emit = np.array([s["cls"] for s in test]), np.array([s["emit"] for s in test])
    masks = {"all": np.ones(len(test), bool), "radio-silent drones": (cls_ == 2) & (emit == 1), "radio-silent non-drones": (cls_ != 2) & (emit == 1)}
    for name, kw in (("calib+rel", {"sensor_cpts": est, "alpha": alpha}), ("no RF+rel", {"sensors_on": no_rf, "sensor_cpts": est, "alpha": alpha}), ("calibrated", {"sensor_cpts": est, "alpha": None}), ("no RF", {"sensors_on": no_rf, "sensor_cpts": est, "alpha": None})):
        Pv, yy = mt.evaluate(cp, test, **kw)[1]
        for sub, mk in masks.items():
            rows.append({"world": wname, "rep": r, "N": 200, "method": "channel: " + name, "subset": sub, "acc": float((Pv[mk].argmax(1) == yy[mk]).mean()), "brier": mt.brier(Pv[mk], yy[mk]), "ece": mt.ece(Pv[mk], yy[mk]), "macroAUC": em.metrics_all(Pv[mk], yy[mk])["macroAUC"]})
    return rows, {}


def s1_one(a):
    _quiet()
    wname, r = a
    import model_spec as ms, dataio, grounding as g, metrics as mt, pipeline as pl, exp_baselines as eb, world_misspec as wm
    wbn, test, est, alpha = _setup(r, wm.world_bn(wname))
    big = dataio.generate_dataset(wbn, 200, seed=r)
    unif = eb.uniform_prior_cpts()
    prior = ms.expert_prior_cpts(np.random.default_rng(1000 + r))
    rows = []
    for N in (10, 50, 200):
        data = big[:N]
        V = {("MLE", "-", 0.0): {n: g.mle_cpt(n, data) for n in ms.INTERNAL}, ("Proposed", "-", 2.0): pl.ground_internal("Proposed", data, r)[0]}
        for mean, pr in (("elicited", prior), ("uniform", unif)):
            for a0 in (2.0, 0.2):
                V[("constrained full table", mean, a0)] = {n: g.cmap_cpt(n, data, pr[n], a0) for n in ms.INTERNAL}
        for (name, mean, a0), c in V.items():
            m, _ = mt.evaluate(c, test, sensor_cpts=est, alpha=alpha)
            rows.append({"world": wname, "rep": r, "N": N, "method": name, "mean": mean, "a0": a0, **_pick(m)})
    return rows, {}


# ------------------------------------------------------------ summary ------
def _rt(name):
    return pd.read_csv(os.path.join(RES, f"{P}_{name}.csv"), float_precision="round_trip")


def _d(a, b):
    x = np.asarray(a, float) - np.asarray(b, float)
    x = x[~np.isnan(x)]
    return x.mean(), st.ttest_1samp(x, 0).pvalue


def _cell(a, b):
    m, p = _d(a, b)
    return f"{m:+.4f} (p={p:.2g})"


def summarize():
    f = _rt("factorial")
    sel = lambda mean, s, a, N: f[(f["mean"] == mean) & (f.structure == s) & (np.isclose(f.a0, a)) & (f.N == N)].sort_values("rep")
    PROP = lambda N: f[(f["mean"] == "Proposed") & (f.N == N)].sort_values("rep")
    MLE = lambda N: sel("none", "full", 0.0, N)
    K = ("acc", "macroAUC", "brier", "ece", "far@pd0.9")
    L = ["# Exp-CM: the comparators' free values and the attribution of the gains", "",
         "Added after the main results. Values chosen for a comparator were chosen after the fact. p: paired t over 20 replications, unadjusted.", ""]
    L.append("## Reproduction of the main study's rows")
    s = pd.read_csv(os.path.join(RES, "expS_scarcity_raw.csv"), float_precision="round_trip")
    b = pd.read_csv(os.path.join(RES, "expB_baselines_raw.csv"), float_precision="round_trip")
    worst = {}
    for N in NS:
        for name, q in (("Proposed", PROP(N)), ("B1-MLE", MLE(N))):
            r0 = s[(s.method == name) & (s.N == N)].sort_values("rep")
            if len(r0):
                worst[f"{name} scarcity"] = max(worst.get(f"{name} scarcity", 0.0), float(np.abs(q.brier.values - r0.brier.values).max()))
        if N in set(b.N):
            for name, q in (("uninformative MAP", sel("uniform", "full", 2.0, N)), ("expert MAP (full table)", sel("elicited", "full", 2.0, N)), ("expert cMAP (full table)", sel("elicited", "constrained", 2.0, N))):
                r0 = b[(b.method == name) & (b.N == N)].sort_values("rep")
                worst[name] = max(worst.get(name, 0.0), float(np.abs(q.brier.values - r0.brier.values).max()))
    grid_prop = max(float(np.abs(PROP(N).brier.values - sel("elicited", "constrained+ICI", 2.0, N).brier.values).max()) for N in NS)
    L.append(f"- largest |Brier difference| against the main study's raw files: {worst}")
    L.append(f"- Proposed through the pipeline against elicited, constrained+ICI, ESS 2 in the grid: {grid_prop:.2e}")
    L += ["", "## Means by arm (macro-AUC / Brier / ECE)"]
    arms = [("MLE", "none", "full", 0.0), ("uniform full, ESS 2 (B3)", "uniform", "full", 2.0), ("uniform full, ESS 0.5", "uniform", "full", 0.5), ("uniform constrained, ESS 0.2", "uniform", "constrained", 0.2), ("uniform constrained+ICI, ESS 0.2", "uniform", "constrained+ICI", 0.2), ("uniform constrained+ICI, ESS 2", "uniform", "constrained+ICI", 2.0),
            ("elicited full, ESS 2", "elicited", "full", 2.0), ("elicited constrained, ESS 2", "elicited", "constrained", 2.0), ("elicited constrained+ICI, ESS 2 (Proposed)", "elicited", "constrained+ICI", 2.0)]
    for name, mean, stru, a in arms:
        L.append(f"- {name}: " + "; ".join(f"N={N} " + "/".join(f"{sel(mean, stru, a, N)[k].mean():.3f}" for k in ("macroAUC", "brier", "ece")) for N in NS))
    L += ["", "## Proposed - the same structure with a weak uniform prior (ESS 0.2)"]
    for N in NS:
        q = sel("uniform", "constrained+ICI", 0.2, N)
        L.append(f"- N={N}: " + "; ".join(f"{k} {_cell(PROP(N)[k].values, q[k].values)}" for k in K))
    L += ["", "## That structure with ESS 0.2 - MLE at a larger N"]
    for Na, Nb in ((10, 50), (20, 50), (50, 100), (50, 200), (80, 200)):
        q, m = sel("uniform", "constrained+ICI", 0.2, Na), MLE(Nb)
        L.append(f"- N={Na} against N={Nb}: " + "; ".join(f"{k} {_cell(q[k].values, m[k].values)}" for k in K))
    L += ["", "## Walks, each step a paired difference"]
    for N in (10, 50, 80):
        p = lambda A, B: "; ".join(f"{k} {_cell(A[k].values, B[k].values)}" for k in ("macroAUC", "brier", "ece"))
        U2, E2, E2c, U2c, U2i, U02i = sel("uniform", "full", 2.0, N), sel("elicited", "full", 2.0, N), sel("elicited", "constrained", 2.0, N), sel("uniform", "constrained", 2.0, N), sel("uniform", "constrained+ICI", 2.0, N), sel("uniform", "constrained+ICI", 0.2, N)
        L.append(f"- N={N} the order of VII-B: uniform - MLE: {p(U2, MLE(N))} | elicited - uniform: {p(E2, U2)} | constraints: {p(E2c, E2)} | ICI: {p(PROP(N), E2c)}")
        L.append(f"- N={N} structure first, ESS 2: constraints - uniform full: {p(U2c, U2)} | ICI: {p(U2i, U2c)} | elicited: {p(PROP(N), U2i)}")
        L.append(f"- N={N} structure first, ESS 0.2: structure - MLE: {p(U02i, MLE(N))} | elicited at ESS 2: {p(PROP(N), U02i)}")
    L += ["", "## Proposed - full tables with a uniform mean, by ESS, and BDeu (ECE, Brier, macro-AUC)"]
    for mean, grid in (("uniform", ESS_UNIFORM), ("BDeu", ESS_BDEU)):
        for N in NS:
            L.append(f"- {mean} N={N}: " + " | ".join(f"{a:g}: " + ", ".join(_cell(PROP(N)[k].values, sel(mean, 'full', a, N)[k].values) for k in ("ece", "brier", "macroAUC")) for a in grid))
    L += ["", "## B0 with its taus multiplied by f"]
    z = _rt("b0")
    al = lambda v: z[(z.use == "alone") & np.isclose(z.f, v)].sort_values("rep")
    pr = lambda v, N: z[(z.use == "prior") & np.isclose(z.f, v) & (z.N == N)].sort_values("rep")
    for v in F_TAU:
        L.append(f"- f={v}: alone " + ", ".join(f"{k} {al(v)[k].mean():.3f}" for k in K) + " | as prior N=10 " + ", ".join(f"{k} {pr(v, 10)[k].mean():.3f}" for k in K) + " | N=50 " + ", ".join(f"{k} {pr(v, 50)[k].mean():.3f}" for k in K))
    for v in (1.0, 0.5):
        for N in NS:
            L.append(f"- Proposed N={N} - B0 f={v} alone: " + "; ".join(f"{k} {_cell(PROP(N)[k].values, al(v)[k].values)}" for k in K))
    for N in (10, 50):
        L.append(f"- Proposed N={N} - B0 f=0.5 as prior at N={N}: " + "; ".join(f"{k} {_cell(PROP(N)[k].values, pr(0.5, N)[k].values)}" for k in K))
    L.append("- B0 f=0.5 as prior N=10 - MLE N=50: " + "; ".join(f"{k} {_cell(pr(0.5, 10)[k].values, MLE(50)[k].values)}" for k in K))
    b1 = b[b.method == "B0-as-prior MAP"]
    L.append(f"- f=1.0 as prior against expB: largest |Brier difference| {max(float(np.abs(pr(1.0, N).brier.values - b1[b1.N == N].sort_values('rep').brier.values).max()) for N in set(b1.N) & set(NS)):.2e}")
    L += ["", "## B5 by L2 strength C"]
    c5 = _rt("b5")
    lr = b[b.method == "logistic regression"]
    L.append(f"- C=1 against expB: largest |Brier difference| {max(float(np.abs(c5[(np.isclose(c5.C, 1.0)) & (c5.N == N)].sort_values('rep').brier.values - lr[lr.N == N].sort_values('rep').brier.values).max()) for N in N_B5):.2e}")
    for N in N_B5:
        for C in C_L2:
            q = c5[np.isclose(c5.C, C) & (c5.N == N)].sort_values("rep")
            L.append(f"- N={N} C={C:g}: means ECE {q.ece.mean():.3f}, Brier {q.brier.mean():.3f}, macro-AUC {q.macroAUC.mean():.3f}; Proposed - B5: " + "; ".join(f"{k} {_cell(PROP(N)[k].values, q[k].values)}" for k in ("macroAUC", "brier", "ece")))
    L += ["", "## One number per sensor"]
    sc = _rt("scalar")
    g3 = lambda n, l: sc[(sc.n == n) & (sc.layer == l)].sort_values("rep")
    cal = pd.read_csv(os.path.join(RES, "expS_calibsize_raw.csv"), float_precision="round_trip")
    L.append(f"- matrix rows against expS_calibsize_raw: largest |Brier difference| {max(float(np.abs(g3(n, 'matrix').brier.values - cal[cal.calib_n == n].sort_values('rep').brier.values).max()) for n in N_CAL):.2e}")
    for n in N_CAL:
        q = g3(n, "one number")
        L.append(f"- n={n}: mean numbers radar {q.d_Rr.mean():.3f}, EO/IR {q.d_Re.mean():.3f}, acoustic {q.d_Ra.mean():.3f}, RF {q.d_Rf.mean():.3f}; one number - matrix: " + "; ".join(f"{k} {_cell(q[k].values, g3(n, 'matrix')[k].values)}" for k in K))
    for n in (30, 100):
        L.append(f"- one number n={n} - matrix n=300: " + "; ".join(f"{k} {_cell(g3(n, 'one number')[k].values, g3(300, 'matrix')[k].values)}" for k in K))
    L += ["", "## CNN errors under the scenarios' mix"]
    ne = _rt("node_errors")
    a = ne.groupby("proto").mean(numeric_only=True)
    for pr_, q in a.iterrows():
        L.append(f"- {pr_}: pool share of model confusions {q.pool_share_models:.3f}, pool drone/background {q.pool_drone_background:.3f}; scenarios: error {q.scenario_error:.3f}, share of model confusions {q.scenario_share_models:.3f}, drone/background {q.scenario_drone_background:.3f}")
    L.append(f"- worst fold: scenario share of model confusions {ne.scenario_share_models.min():.3f}, largest scenario drone/background {ne.scenario_drone_background.max():.3f}, largest pool drone/background {ne.pool_drone_background.max():.3f}")
    L += ["", "## The world, by class and emission"]
    w = _rt("world")
    for _, q in w.iterrows():
        L.append(f"- {q.world}, {q.cls}, {q.emit}: P(T=high) {q.p_high:.3f}, P(T>=medium) {q.p_medium_up:.3f}, P(N=top) {q.p_N_top:.3f}")
    L += ["", "## Emission acting on drones only (emit-drone-only)"]
    se = _rt("sens")
    q1 = se[se.get("subset").isna()] if "subset" in se else se
    m_ = lambda wn, meth, N: q1[(q1.world == wn) & (q1.method == meth) & (q1.N == N)].sort_values("rep")
    jd = pd.read_csv(os.path.join(RES, "expW_judgment.csv"), float_precision="round_trip")
    std = jd[jd.world == "standard"].set_index("N")
    for N in (50, 200):
        wn = "emit-drone-only"
        means = {x: m_(wn, x, N).brier.mean() for x in ("cMAP-full", "MAP", "B1-MLE", "B5-logistic")}
        alt = min(means, key=means.get)
        dm, dp = _d(m_(wn, "Proposed", N).brier.values, m_(wn, alt, N).brier.values)
        exc = (m_(wn, "Proposed", N).brier.values - m_(wn, "CPT-oracle", -1).brier.values).mean()
        verdict = "breaks" if dm > 0 and dp < 0.05 else ("holds" if exc <= 2 * std.excess.loc[N] else "degrades")
        graph = (m_(wn, "CPT-oracle", -1).brier.values - m_(wn, "Bayes-oracle", -1).brier.values).mean()
        L.append(f"- N={N}: Proposed Brier {m_(wn, 'Proposed', N).brier.mean():.4f}, ECE {m_(wn, 'Proposed', N).ece.mean():.4f}; graph {graph:.4f}; excess {exc:.4f} (standard {std.excess.loc[N]:.4f}); best alternative {alt} {dm:+.4f} (p={dp:.2g}); verdict {verdict}")
    a10, b50 = m_("emit-drone-only", "Proposed", 10), m_("emit-drone-only", "B1-MLE", 50)
    L.append("- Proposed N=10 - MLE N=50: " + "; ".join(f"{k} {_cell(a10[k].values, b50[k].values)}" for k in K))
    if "subset" in se:
        for wn in ("standard", "emit-drone-only"):
            for sub in ("all", "radio-silent drones", "radio-silent non-drones"):
                v = lambda n: se[(se.world == wn) & (se.method == "channel: " + n) & (se.subset == sub)].sort_values("rep")
                L.append(f"- {wn}, RF channel, {sub}: calib+rel - no RF+rel: " + "; ".join(f"{k} {_cell(v('calib+rel')[k].values, v('no RF+rel')[k].values)}" for k in ("acc", "macroAUC", "brier", "ece")) + " | calibrated - no RF: " + "; ".join(f"{k} {_cell(v('calibrated')[k].values, v('no RF')[k].values)}" for k in ("acc", "macroAUC", "brier", "ece")))
    L += ["", "## S1: N=10 against MLE at N=50, and the ESS against the prior's direction"]
    s1 = _rt("s1")
    g8 = lambda wn, meth, mean, a, N: s1[(s1.world == wn) & (s1.method == meth) & (s1["mean"] == mean) & np.isclose(s1.a0, a) & (s1.N == N)].sort_values("rep")
    raw_w = pd.read_csv(os.path.join(RES, "expW_raw.csv"), float_precision="round_trip")
    gap = 0.0
    for wn in ("standard", "nonadd-0.5", "nonadd-1.0"):
        for N in (50, 200):
            for meth, tag, a in (("Proposed", "Proposed", 2.0), ("MLE", "B1-MLE", 0.0)):
                r0 = raw_w[(raw_w.world == wn) & (raw_w.method == tag) & (raw_w.N == N)].sort_values("rep")
                gap = max(gap, float(np.abs(g8(wn, meth, "-", a, N).brier.values - r0.brier.values).max()))
    L.append(f"- Proposed and MLE rows against expW_raw: largest |Brier difference| {gap:.2e}")
    for wn in ("standard", "nonadd-0.5", "nonadd-1.0"):
        A, B = g8(wn, "Proposed", "-", 2.0, 10), g8(wn, "MLE", "-", 0.0, 50)
        L.append(f"- {wn}: Proposed N=10 - MLE N=50: " + "; ".join(f"{k} {_cell(A[k].values, B[k].values)}" for k in K))
        for N in (50, 200):
            for mean in ("elicited", "uniform"):
                for a0 in (2.0, 0.2):
                    L.append(f"  - {wn}, N={N}, constrained full table, {mean}, ESS {a0:g} - MLE: " + "; ".join(f"{k} {_cell(g8(wn, 'constrained full table', mean, a0, N)[k].values, g8(wn, 'MLE', '-', 0.0, N)[k].values)}" for k in ("acc", "brier", "ece")))
    open(os.path.join(RES, f"{P}_summary.md"), "w", encoding="utf-8", newline="\n").write("\n".join(L) + "\n")
    print("\n".join(L[:12]))
    print(f"[{P}] summary written, {len(L)} lines", flush=True)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="all")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--reps", type=int, default=REPS)
    a = ap.parse_args()
    stages = ("factorial", "b0", "b5", "scalar", "errors", "world", "sens", "s1", "summary") if a.stage == "all" else (a.stage,)
    reps = list(range(a.reps))
    for s in stages:
        t0 = time.time()
        if s == "factorial":
            run_stage(factorial_one, reps, a.workers, "factorial")
        elif s == "b0":
            run_stage(b0_one, reps, a.workers, "b0")
        elif s == "b5":
            run_stage(b5_one, reps, a.workers, "b5")
        elif s == "scalar":
            run_stage(scalar_one, reps, a.workers, "scalar")
        elif s == "errors":
            run_errors()
        elif s == "world":
            run_world()
        elif s == "sens":
            run_stage(sens_one, [(w, r) for w in ("standard", "emit-drone-only") for r in reps], a.workers, "sens")
        elif s == "s1":
            run_stage(s1_one, [(w, r) for w in ("standard", "nonadd-0.5", "nonadd-1.0") for r in reps], a.workers, "s1")
        elif s == "summary":
            summarize()
        print(f"[{P}] stage {s} done in {time.time() - t0:.0f}s", flush=True)
