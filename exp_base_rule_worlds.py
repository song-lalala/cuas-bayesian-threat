"""
Exp-B6w: the rule of the base framework [5] and the network in worlds where one
latent factor can drive the threat (S1, Sec. VII-G), and at large N.

NOT FIXED IN ADVANCE. This run was added after the first results of exp_base_rule.py
were seen. It is a check
that can go against the network: in the standard world high threat needs joint
evidence, which favors the network's additive threat node over a Noisy-OR, while
in S1 the threat follows the largest latent score. All results are reported
whatever their sign. exp_base_rule.py is not modified; its functions are imported.

Worlds world_misspec.world_bn("nonadd-0.5"), ("nonadd-1.0"): T | H, N, C with mu =
(1 - lam) A + lam max(s_H, s_N, s_C), A the additive level of the standard world.
Seeds and data as exp_misspecification.py: test 9000 + r, calibration 5000 + r,
training seed r (prefixes of one draw); r = 0..19; N in {10, 20, 50, 100, 200, 400}.
The expert prior is the same as in the standard world and so is wrong about T here.

Methods (target and metrics as in exp_base_rule.py: T = high; AUC, Brier, ECE,
ece_pos, Table 9 bins, alarms at 0.5 and 0.25, AUC of T >= medium).
  Rule side: B6-pub, B6-recal, B6-fit, B6-fit-r, B6-fit-p, as in exp_base_rule.py.
  Network side: Proposed; cMAP-full (full-table threat node); Proposed-adaptive, the
    rule of Sec. VII-G (cMAP-full T if the additivity test [A] flags, ICI otherwise;
    MAP for each of H, N, C that the monotonicity test [D] flags); Proposed-EM;
    B0-as-prior MAP.
  Reference: CPT-oracle (the world's true tables, exp_misspecification's path).
The flags at N = 50, 200 come from results/expW_add_diag_raw.csv. At N = 10, 20, 100,
400 they are computed here by exp_additivity's a_test and d_test (199 bootstrap
replicates) with its seed scheme, the N index extended (10: 3, 20: 4, 100: 5, 400: 6),
and saved in results/expB6w_flags_raw.csv.
Large N (form limit against estimation error, descriptive): N = 2000, 10000; B6-fit,
Proposed, cMAP-full; worlds standard and nonadd-1.0; 20 replications.
World facts: prevalence of T = high; P(T = high) by the number of latent variables
at their top state; share of T = high by that number; the best Noisy-OR on the true
latent variables (weighted least squares).
Reproduction: Proposed and cMAP-full at N = 50, 200 and CPT-oracle against
expW_raw.csv, Proposed-adaptive at N = 50, 200 against expW_add_adaptive_raw.csv,
within 1e-9. The other cells have no earlier counterpart.
Reading: as [C] of exp_base_rule.py: signs and paired t over replications; N and
metrics where the rule side is ahead of or not significantly behind each network
method are listed; Holm over each world's family beside the raw p.

Run:  python exp_base_rule_worlds.py --stage flags --workers 24
      python exp_base_rule_worlds.py --stage main --workers 20
      python exp_base_rule_worlds.py --summary-only
"""
from __future__ import annotations
import os
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import sys, time, argparse
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import exp_base_rule as er
import exp_additivity as ea
import exp_misspecification as em_w

ea.NI.update({10: 3, 20: 4, 100: 5, 400: 6})

RES = os.path.join(HERE, "results")
P = "expB6w"
WORLDS = ("nonadd-0.5", "nonadd-1.0")
N_GRID = er.N_GRID
NEW_N = (10, 20, 100, 400)
BIG_N = (2000, 10000)
BIG_WORLDS = ("standard", "nonadd-1.0")
REPS = 20
LATENT = ("H", "N", "C")
RULE = ("B6-pub", "B6-recal", "B6-fit", "B6-fit-r", "B6-fit-p")
NETS = ("Proposed", "cMAP-full", "Proposed-adaptive", "Proposed-EM", "B0-as-prior MAP")
REPRO_KEYS = ("acc", "macroF1", "macroAUC", "brier", "ece", "cwece", "far@pd0.9", "ece_high", "bias_high")


# ------------------------------------------------------------------ flags ---
def flag_tasks():
    tasks = [("A", w, r, N) for w in WORLDS for N in NEW_N for r in range(REPS)]
    tasks += [("D", w, r, N) for w in WORLDS for N in NEW_N for r in range(REPS)]
    return tasks


def run_flags(res, workers):
    from multiprocessing import Pool
    tasks = flag_tasks()
    t0 = time.time()
    rows = []
    with Pool(workers) as pool:
        for i, out in enumerate(pool.imap_unordered(ea.run_task, tasks), 1):
            rows += out
            if i % 40 == 0 or i == len(tasks):
                print(f"[expB6w flags] {i}/{len(tasks)} tasks ({time.time() - t0:.0f}s)", flush=True)
    df = pd.DataFrame(rows).sort_values(["test", "world", "N", "rep", "node"])
    df.to_csv(os.path.join(res, f"{P}_flags_raw.csv"), index=False)
    print(df[df.test == "A"].groupby(["world", "N"]).flag.sum().to_string())
    print(df[df.test == "D"].groupby(["world", "N", "node"]).flag.sum().to_string())


def load_flags(res):
    old = pd.read_csv(os.path.join(RES, "expW_add_diag_raw.csv"))
    new = pd.read_csv(os.path.join(res, f"{P}_flags_raw.csv"))
    d = pd.concat([old[old.world.isin(WORLDS) & old.N.isin((50, 200))], new])
    A = {(x.world, x.rep, x.N): bool(x.flag) for x in d[d.test == "A"].itertuples()}
    D = {(x.world, x.rep, x.N, x.node): bool(x.flag) for x in d[d.test == "D"].itertuples()}
    return A, D


# ------------------------------------------------------------- replication ---
def _eval(method, N, world, r, p, y, ge2, score2, rows, bins, extra=None):
    from sklearn.metrics import roc_auc_score
    rows.append({"world": world, "rep": r, "N": N, "method": method, **er.binary_metrics(p, y),
                 "auc_ge_medium": float(roc_auc_score(ge2, p if score2 is None else score2)), **(extra or {})})
    bins.extend({"world": world, "rep": r, "N": N, "method": method, **b} for b in er.bin_rows(p, y))


def replicate_world(task):
    """One replication of one S1 world, N grid."""
    world, r, A, D = task
    import model_spec as ms, dataio, metrics as mt, pipeline as pl, grounding as g
    import exp_mapem as em
    import world_options as wo
    t0 = time.time()
    wbn = em_w.world(world)
    test = dataio.generate_dataset(wbn, em_w.TEST_N, seed=9000 + r)
    calib = dataio.generate_dataset(wbn, em_w.CALIB_N, seed=5000 + r)
    est = dataio.estimate_confusion(calib)
    alpha, _ = dataio.fit_reliability(calib)
    y = np.array([x["T"] for x in test])
    ge2 = (y >= 2).astype(int)
    rows, bins = [], []
    # no training
    tabs = em_w.oracle_tables(world, wbn)["CPT-oracle"]
    Po = em_w.bn_posteriors(em_w.model_bn(wbn, tabs, wo.true_marginal_tables(wbn)), test)
    _eval("CPT-oracle", -1, world, r, Po[:, 3], y, ge2, Po[:, 2] + Po[:, 3], rows, bins, em_w.metrics_all(Po, y))
    pub = er.b6_pub(test)
    _eval("B6-pub", 0, world, r, pub, y, ge2, None, rows, bins)
    yc = np.array([float(x["T"] == 3) for x in calib])
    prc, a_, b_ = er.recalibrate(er.b6_pub(calib), yc, pub)
    _eval("B6-recal", 0, world, r, prc, y, ge2, None, rows, bins, {"recal_a": a_, "recal_b": b_})
    b0 = ms.heuristic_cpts()
    prior = ms.expert_prior_cpts(np.random.default_rng(1000 + r))
    big = dataio.generate_dataset(wbn, max(N_GRID), seed=r)
    for N in N_GRID:
        data = big[:N]
        partial = [{k: row[k] for k in em.KEEP} for row in data]
        cp_p = pl.ground_internal("Proposed", data, r)[0]
        cp_f = pl.ground_internal("cMAP-full", data, r)[0]
        fa = A[(world, r, N)]
        fd = tuple(x for x in LATENT if D[(world, r, N, x)])
        cp_a = {x: (g.map_cpt(x, data, prior[x], pl.ESS_FIXED) if x in fd else cp_p[x]) for x in LATENT}
        cp_a["T"] = cp_f["T"] if fa else cp_p["T"]
        nets = (("Proposed", cp_p, {}), ("cMAP-full", cp_f, {}),
                ("Proposed-adaptive", cp_a, {"A_flag": fa, "D_flags": "".join(fd)}),
                ("B0-as-prior MAP", {n: g.map_cpt(n, data, b0[n], pl.ESS_FIXED) for n in ms.INTERNAL}, {}),
                ("Proposed-EM", g.em_internal(partial, prior, pl.ESS_FIXED, n_iter=em.EM_ITERS, seed=r, constrained=True,
                                              sensor_cpts=est), {}))
        for name, cp, ex in nets:
            Pm = mt.posteriors(cp, test, ms.SENSORS, "calibrated", sensor_cpts=est, alpha=alpha)[0]
            _eval(name, N, world, r, Pm[:, 3], y, ge2, Pm[:, 2] + Pm[:, 3], rows, bins, {**em_w.metrics_all(Pm, y), **ex})
        for name, kind, sd, up in (("B6-fit", "full", None, False), ("B6-fit-r", "full", er.PRIOR_SD, False),
                                   ("B6-fit-p", "full", er.PRIOR_SD, True)):
            pf, _, deg = er.fit_predict(data, test, kind, sd, np.random.default_rng(61000 + 1000 * r + N), up)
            _eval(name, N, world, r, pf, y, ge2, None, rows, bins, {"degenerate": deg})
    return world, r, rows, bins, time.time() - t0


def replicate_big(task):
    """Large N: B6-fit, Proposed, cMAP-full."""
    world, r = task
    import model_spec as ms, dataio, metrics as mt, pipeline as pl
    t0 = time.time()
    wbn = em_w.world(world)
    test = dataio.generate_dataset(wbn, em_w.TEST_N, seed=9000 + r)
    calib = dataio.generate_dataset(wbn, em_w.CALIB_N, seed=5000 + r)
    est = dataio.estimate_confusion(calib)
    alpha, _ = dataio.fit_reliability(calib)
    y = np.array([x["T"] for x in test])
    ge2 = (y >= 2).astype(int)
    rows, bins = [], []
    big = dataio.generate_dataset(wbn, max(BIG_N), seed=r)
    for N in BIG_N:
        data = big[:N]
        for name in ("Proposed", "cMAP-full"):
            Pm = mt.posteriors(pl.ground_internal(name, data, r)[0], test, ms.SENSORS, "calibrated",
                               sensor_cpts=est, alpha=alpha)[0]
            _eval(name, N, world, r, Pm[:, 3], y, ge2, Pm[:, 2] + Pm[:, 3], rows, bins)
        pf, _, deg = er.fit_predict(data, test, "full", None, np.random.default_rng(61000 + 1000 * r + N), False)
        _eval("B6-fit", N, world, r, pf, y, ge2, None, rows, bins, {"degenerate": deg})
    return world, r, rows, bins, time.time() - t0


# ------------------------------------------------------------- world facts ---
def world_facts():
    from scipy.optimize import minimize
    rows = []
    for world in ("standard",) + WORLDS:
        wbn = em_w.world(world)
        T = wbn.cpts["T"].table
        assert T.shape == (3, 2, 3, 4)
        ph = T[..., 3]
        mar = wbn.query(["H", "N", "C"], {}).table
        prev = float((mar * ph).sum())
        cnt = np.zeros((3, 2, 3), dtype=int)
        for h in range(3):
            for n in range(2):
                for c in range(3):
                    cnt[h, n, c] = int(h == 2) + int(n == 1) + int(c == 2)
        row = {"world": world, "prevalence": prev}
        for k in range(4):
            m = cnt == k
            row[f"p_high_top{k}_min"] = float(ph[m].min())
            row[f"p_high_top{k}_max"] = float(ph[m].max())
            row[f"p_high_top{k}_mean"] = float((mar * ph)[m].sum() / mar[m].sum())
            row[f"share_high_top{k}"] = float((mar * ph)[m].sum() / prev)

        def f(x):
            sig = 1.0 / (1.0 + np.exp(-x))
            q = np.einsum("i,j,k->ijk", 1 - sig[:3], 1 - sig[3:5], 1 - sig[5:8])
            return float((mar * (1 - q - ph) ** 2).sum())
        rng = np.random.default_rng(0)
        best = min((minimize(f, rng.normal(-2, 1, 8), method="L-BFGS-B") for _ in range(20)), key=lambda r_: r_.fun)
        sig = 1.0 / (1.0 + np.exp(-best.x))
        q = np.einsum("i,j,k->ijk", 1 - sig[:3], 1 - sig[3:5], 1 - sig[5:8])
        row.update(noisy_or_rms=float(np.sqrt(best.fun)), noisy_or_all_top=float(1 - q[2, 1, 2]), noisy_or_max_err=float(np.max(np.abs(1 - q - ph))))
        rows.append(row)
    return pd.DataFrame(rows)


# -------------------------------------------------------------------- checks ---
def check_reproduction(d):
    w = pd.read_csv(os.path.join(RES, "expW_raw.csv"), float_precision="round_trip")
    ad = pd.read_csv(os.path.join(RES, "expW_add_adaptive_raw.csv"), float_precision="round_trip")
    out = []
    for method, ref, other in (("Proposed", w, "Proposed"), ("cMAP-full", w, "cMAP-full"), ("CPT-oracle", w, "CPT-oracle"),
                               ("Proposed-adaptive", ad, "Proposed-adaptive")):
        r_ = ref[(ref.method == other) & ref.world.isin(WORLDS)].set_index(["world", "rep", "N"])
        z = d[d.method == method].to_dict("records")
        dev = []
        for x in z:
            key = (x["world"], x["rep"], x["N"])
            if key in r_.index:
                for k in REPRO_KEYS:
                    a, b = float(r_.loc[key, k]), float(x[k])
                    dev.append(0.0 if (np.isnan(a) and np.isnan(b)) else abs(a - b))
        assert len(dev) > 0, method
        out.append((method, max(dev), len(dev)))
    return out


# ------------------------------------------------------------------- summary ---
def summarize(res):
    from scipy import stats as st
    d = pd.read_csv(os.path.join(res, f"{P}_raw.csv"), float_precision="round_trip")
    bn = pd.read_csv(os.path.join(res, f"{P}_bins.csv"))
    big = pd.read_csv(os.path.join(res, f"{P}_big_raw.csv"), float_precision="round_trip")
    wf = pd.read_csv(os.path.join(res, f"{P}_world_facts.csv"))
    fl = pd.read_csv(os.path.join(res, f"{P}_flags_raw.csv"))
    L = ["# Exp-B6w: the rule of [5] and the network where one latent factor can drive the threat (not fixed in advance)", ""]
    chk = os.path.join(res, f"{P}_checks.txt")
    if os.path.exists(chk):
        L += ["## Checks", "```", open(chk, encoding="utf-8").read().strip(), "```", ""]
    L += ["## World facts (P(T = high) by the number of latent variables at their top state)",
          "| world | prevalence | none | one | two | three | share of T = high with one, two, three | best Noisy-OR: RMS, at all-top |",
          "|---|---|---|---|---|---|---|---|"]
    for x in wf.itertuples():
        cell = lambda k: f"{getattr(x, f'p_high_top{k}_min'):.3f}-{getattr(x, f'p_high_top{k}_max'):.3f} (mean {getattr(x, f'p_high_top{k}_mean'):.3f})"
        L.append(f"| {x.world} | {x.prevalence:.4f} | {cell(0)} | {cell(1)} | {cell(2)} | {cell(3)} | "
                 f"{x.share_high_top1:.3f}, {x.share_high_top2:.3f}, {x.share_high_top3:.3f} | {x.noisy_or_rms:.3f}, {x.noisy_or_all_top:.3f} |")
    KS = (("auc", "AUC"), ("brier_bin", "Brier"), ("ece_bin", "ECE"), ("auc_ge_medium", "AUC T>=medium"))
    get = lambda w, m, N, k: d[(d.world == w) & (d.method == m) & (d.N == N)].sort_values("rep")[k].values
    fams = []
    for w in WORLDS:
        L += ["", f"## {w}", "### Means", "| N | method | " + " | ".join(n for _, n in KS) + " |", "|---|---|" + "---|" * len(KS)]
        for m, N in (("CPT-oracle", -1), ("B6-pub", 0), ("B6-recal", 0)):
            L.append(f"| - | {m} | " + " | ".join(f"{get(w, m, N, k).mean():.4f}" for k, _ in KS) + " |")
        for N in N_GRID:
            for m in NETS + RULE[2:]:
                L.append(f"| {N} | {m} | " + " | ".join(f"{get(w, m, N, k).mean():.4f}" for k, _ in KS) + " |")
        L += ["", "### Flags of the rule of Sec. VII-G", "| N | [A] flagged of 20 | [D] flags (H, N, C) of 20 each |", "|---|---|---|"]
        for N in N_GRID:
            z = d[(d.world == w) & (d.method == "Proposed-adaptive") & (d.N == N)]
            L.append(f"| {N} | {int((z.A_flag == True).sum())} | " + ", ".join(f"{x} {int(z.D_flags.fillna('').astype(str).str.contains(x).sum())}" for x in LATENT) + " |")
        rows = []
        for N in N_GRID:
            for net, rule in [(n, r_) for n in ("Proposed", "cMAP-full", "Proposed-adaptive") for r_ in ("B6-pub", "B6-recal", "B6-fit", "B6-fit-r")] + \
                    [("B0-as-prior MAP", "B6-fit-p"), ("Proposed-EM", "B6-fit"), ("Proposed-EM", "B6-fit-r")]:
                nb = 0 if rule in ("B6-pub", "B6-recal") else N
                for k, nm in KS[:3]:
                    diff = get(w, net, N, k) - get(w, rule, nb, k)
                    pv = float(st.ttest_1samp(diff, 0.0).pvalue) if np.ptp(diff) > 0 else np.nan
                    rows.append({"world": w, "N": N, "net": net, "rule": rule, "metric": nm, "delta": float(diff.mean()), "p": pv})
        t = pd.DataFrame(rows)
        t["p_holm"] = er._holm(t.p.fillna(1.0).values)
        fams.append(t)
        L += ["", "### Network minus rule (paired; positive favors the network on AUC, negative on Brier and ECE; p, Holm over this world)",
              "| N | comparison | dAUC | dBrier | dECE |", "|---|---|---|---|---|"]
        for N in N_GRID:
            for (net, rule), z in t[t.N == N].groupby(["net", "rule"], sort=False):
                z = z.set_index("metric")
                L.append(f"| {N} | {net} - {rule} | " + " | ".join(f"{z.loc[k].delta:+.4f} ({z.loc[k].p:.1e}, {z.loc[k].p_holm:.1e})" for k in ("AUC", "Brier", "ECE")) + " |")
        lst = t[t.metric.isin(["AUC", "Brier"])]
        lst = lst[(lst.p >= 0.05) | lst.p.isna() | ((lst.metric == "AUC") & (lst.delta <= 0)) | ((lst.metric == "Brier") & (lst.delta >= 0))]
        L += ["", "### Where the rule side is ahead of, or not significantly behind, the network side (AUC, Brier; raw p >= 0.05 or the sign reversed)"]
        L += [f"- N {int(x.N)}, {x.net} - {x.rule}, {x.metric}: {x.delta:+.4f}, p {x.p:.2g}" for x in lst.itertuples()] or ["- none"]
        L += ["", "### Table 9 form (pooled)"]
        for m, N in (("Proposed", 200), ("cMAP-full", 200), ("Proposed-adaptive", 200), ("B6-pub", 0), ("B6-recal", 0), ("B6-fit", 200), ("B6-fit-p", 200), ("CPT-oracle", -1)):
            L += ["", f"**{m}{f' (N = {N})' if N > 0 else ''}**", "| bin | n | predicted | observed [95% CP] | outside |", "|---|---|---|---|---|"]
            z = bn[(bn.world == w) & (bn.method == m) & (bn.N == N)]
            for lo in er.TBINS[:-1]:
                q = z[z.bin_lo == lo]
                n, k, sp = int(q.n.sum()), int(q.hits.sum()), float(q.sum_pred.sum())
                if n == 0:
                    continue
                l_, h_ = er._cp(k, n)
                L.append(f"| ({lo:.2f}, {q.bin_hi.iloc[0]:.2f}] | {n} | {sp / n:.3f} | {k / n:.3f} [{l_:.3f}, {h_:.3f}] | {'*' if not (l_ <= sp / n <= h_) else ''} |")
        L += ["", "### Alarms (precision, recall, pooled)", "| method | at 0.5 | at 0.25 |", "|---|---|---|"]
        for m, N in (("Proposed", 200), ("cMAP-full", 200), ("Proposed-adaptive", 200), ("B6-pub", 0), ("B6-recal", 0), ("B6-fit", 200), ("B6-fit-p", 200), ("CPT-oracle", -1)):
            z = d[(d.world == w) & (d.method == m) & (d.N == N)]
            cell = []
            for thr in (0.5, 0.25):
                tp, fp, fn = z[f"tp_{thr}"].sum(), z[f"fp_{thr}"].sum(), z[f"fn_{thr}"].sum()
                cell.append(f"{tp / max(tp + fp, 1):.3f}, {tp / max(tp + fn, 1):.3f}")
            L.append(f"| {m}{f' (N = {N})' if N > 0 else ''} | {cell[0]} | {cell[1]} |")
    pd.concat(fams).to_csv(os.path.join(res, f"{P}_paired.csv"), index=False)
    L += ["", "## Large N (descriptive): form limit against estimation error", "| world | N | method | AUC | Brier | ECE | n |", "|---|---|---|---|---|---|---|"]
    for (w, N, m), z in big.groupby(["world", "N", "method"], sort=False):
        L.append(f"| {w} | {N} | {m} | {z.auc.mean():.4f} | {z.brier_bin.mean():.4f} | {z.ece_bin.mean():.4f} | {len(z)} |")
    txt = "\n".join(L)
    open(os.path.join(res, f"{P}_summary.md"), "w", encoding="utf-8").write(txt + "\n")
    print(txt)


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="", help="flags or main")
    ap.add_argument("--workers", type=int, default=20)
    ap.add_argument("--reps", type=int, default=REPS)
    ap.add_argument("--res", default=RES)
    ap.add_argument("--no-summary", action="store_true")
    ap.add_argument("--summary-only", action="store_true")
    a = ap.parse_args()
    os.makedirs(a.res, exist_ok=True)
    if a.summary_only:
        summarize(a.res)
        return
    if a.stage == "flags":
        run_flags(a.res, a.workers)
        return
    assert a.stage == "main"
    from multiprocessing import Pool
    A, D = load_flags(a.res)
    t0 = time.time()
    tasks = [(w, r, A, D) for w in WORLDS for r in range(a.reps)]
    rows, bins, brows, bbins = [], [], [], []
    keep = {}
    with Pool(a.workers) as pool:
        for w, r, ro, bi, secs in pool.imap_unordered(replicate_world, tasks):
            keep[(w, r)] = (ro, bi)
            print(f"[expB6w] {w} rep {r} ({secs:.0f}s; {time.time() - t0:.0f}s total)", flush=True)
        bigs = {}
        for w, r, ro, bi, secs in pool.imap_unordered(replicate_big, [(w, r) for w in BIG_WORLDS for r in range(a.reps)]):
            bigs[(w, r)] = (ro, bi)
            print(f"[expB6w big] {w} rep {r} ({secs:.0f}s; {time.time() - t0:.0f}s total)", flush=True)
    for k in sorted(keep):
        rows += keep[k][0]; bins += keep[k][1]
    for k in sorted(bigs):
        brows += bigs[k][0]; bbins += bigs[k][1]
    d = pd.DataFrame(rows)
    d.to_csv(os.path.join(a.res, f"{P}_raw.csv"), index=False)
    pd.DataFrame(bins).to_csv(os.path.join(a.res, f"{P}_bins.csv"), index=False)
    pd.DataFrame(brows).to_csv(os.path.join(a.res, f"{P}_big_raw.csv"), index=False)
    pd.DataFrame(bbins).to_csv(os.path.join(a.res, f"{P}_big_bins.csv"), index=False)
    world_facts().to_csv(os.path.join(a.res, f"{P}_world_facts.csv"), index=False)
    checks = check_reproduction(d)
    again = replicate_world(("nonadd-1.0", 0, A, D))
    same = pd.DataFrame(again[2]).equals(pd.DataFrame(keep[("nonadd-1.0", 0)][0]))
    txt = ("".join(f"{m} against the expW files: max |difference| {v:.2e} over {n} values\n" for m, v, n in checks)
           + f"nonadd-1.0 replication 0 rerun in the parent process equals the pool's rows: {same}\n"
           + f"threads per worker: OMP {os.environ.get('OMP_NUM_THREADS')}; workers {a.workers}; elapsed {time.time() - t0:.0f}s")
    open(os.path.join(a.res, f"{P}_checks.txt"), "w", encoding="utf-8").write(txt + "\n")
    print(txt, flush=True)
    assert all(v <= 1e-9 for _, v, _ in checks) and same, "reproduction or determinism check failed"
    if not a.no_summary:
        summarize(a.res)


if __name__ == "__main__":
    main()
