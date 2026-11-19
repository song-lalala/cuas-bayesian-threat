"""
Additivity diagnostic [A] for the threat node and adaptive estimator rules.
Companion of exp_misspecification.py
(rule [D], worlds, solver); core files are not modified; results go to
results/expW_add_*.

RULES FIXED BEFORE THE RUN (in this docstring; not registered externally)

[A] Additivity diagnostic. LR_T = 2 [l(cMAP-full T) - l(ICI T)], where
    l(theta) = sum_jk N_jk log theta_jk on the threat-node counts of the N
    training scenarios (the likelihood of [D]) and cMAP-full T and ICI T are
    the deployed estimators (expert prior, ESS 2; the ICI fit with 4 restarts
    seeded by the replication index, as in pipeline.ground_internal). Null
    distribution by parametric bootstrap: B = 199 count tables drawn from the
    ICI fit with the observed column totals; both estimators refitted on each
    with the same settings; p_T = (1 + #{LR* >= LR}) / (B + 1). Flag the
    threat node if p_T < 0.05. Reported at N = 50, 200, 1000 over the 20
    replications of the main study (same training data: the first N rows of
    the replication's dataset): false-alarm rate in the standard world and
    detection rate in S1 weak and strong. The standard world's threat table
    is itself 1.8e-4 from the ICI family in KL, so [A] may flag it at large
    N; that is reported as it is.

Adaptive rules (N = 50, 200; worlds standard, S1 weak, S1 strong, S2):
    (i)  threat node: cMAP-full T if [A] flags, ICI T otherwise;
    (ii) latent nodes: MAP (no constraint) for each of H, N, C that [D]
         flags (p_X < 0.0125), cMAP otherwise.
    Proposed-adaptive applies both; the [D] flag on the full T table is not
    used ((i) governs the threat node). To apply the rules, [A] is also run
    in H-v-U and [D] in S1 weak and strong at N = 50, 200; [D] in the
    standard world and H-v-U is taken from results/expW_diag_raw.csv (same
    replications and training data).
    Metrics of Proposed-adaptive: taken from results/expW_raw.csv when the
    selected model is one of its methods (no flag: Proposed; [A] only:
    cMAP-full; [D] on H only in H-v-U: Proposed-Hfree), otherwise evaluated
    through the same code path (that path is checked bit-identical against
    expW_raw.csv on overlapping cells). Comparisons: paired t-tests over the
    20 replications of Proposed-adaptive against Proposed, against cMAP-full
    (standard, S1) and against Proposed-Hfree (S2), on Brier and ECE.

Seeds: bootstrap rng(200000 + 1000 r + 100 w + 10 n) for [A] with w the
index in A_WI and n the index of N in (50, 200, 1000); [D] in S1 uses
exp_misspecification's scheme rng(100000 + 1000 r + 100 w + 10 n + x) with
w = 2, 3 (0, 1 are taken by the standard world and H-v-U). The tests are
independent and seeded individually, so they are run in parallel worker
processes; the results do not depend on the scheduling.

Run:  python exp_additivity.py [--workers 20]
      python exp_additivity.py --summary-only
"""
from __future__ import annotations
import os
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import sys, time, argparse
import numpy as np
import pandas as pd

import model_spec as ms
import dataio
import grounding as g
import metrics as mt
import exp_misspecification as em
import exp_estimator_consistency as ec
import exp_significance as es

RES_IN = em.RES_IN
RES = RES_IN
P = "expW_add"
REPS = 20
ESS = em.ESS
B_BOOT = 199
A_ALPHA = 0.05
A_GRID = {"standard": (50, 200, 1000), "nonadd-0.5": (50, 200, 1000),
          "nonadd-1.0": (50, 200, 1000), "H-v-U": (50, 200)}
A_WI = {"standard": 0, "nonadd-0.5": 1, "nonadd-1.0": 2, "H-v-U": 3}
D_GRID = {"nonadd-0.5": (50, 200), "nonadd-1.0": (50, 200)}
D_WI = {"nonadd-0.5": 2, "nonadd-1.0": 3}
NI = {50: 0, 200: 1, 1000: 2}
AD_WORLDS = ("standard", "nonadd-0.5", "nonadd-1.0", "H-v-U")
NS = (50, 200)
LATENT = ("H", "N", "C")
METRICS = ["acc", "macroF1", "macroAUC", "brier", "ece", "cwece", "far@pd0.9",
           "ece_high", "bias_high"]

_SOLVERS = {}


def solver(node):
    if node not in _SOLVERS:
        _SOLVERS[node] = em.CmapSolver(node)
    return _SOLVERS[node]


def train_data(world, r, N):
    """First N rows of the replication's dataset (prefix of the main study's
    and of [D]'s datasets: sample_ancestral draws scenario by scenario)."""
    return dataio.generate_dataset(em.world(world), 1000, seed=r)[:N]


def loglik(n, th):
    return float(np.sum(n * np.log(th)))


def t_fits(n, a, r):
    ici = ec.ici_from_counts(n + a, restarts=4, seed=r)
    cm = solver("T").solve(n + a)
    return ici, cm


def a_test(task):
    world, r, N = task
    data = train_data(world, r, N)
    prior = ms.expert_prior_cpts(np.random.default_rng(1000 + r))
    n = g.counts("T", data)
    a = ESS * prior["T"].table
    before = dict(solver("T").stats)
    ici, cm = t_fits(n, a, r)
    lr = 2 * (loglik(n, cm) - loglik(n, ici))
    dep_ici = g.ordinal_ici_cpt("T", data, prior["T"], ESS, seed=r).table
    dep_cm = g.cmap_cpt("T", data, prior["T"], ESS).table
    rng = np.random.default_rng(200000 + 1000 * r + 100 * A_WI[world] + 10 * NI[N])
    K = n.shape[-1]
    tot = n.reshape(-1, K).sum(1)
    icf = ici.reshape(-1, K)
    ge = 0
    for _ in range(B_BOOT):
        nb = np.vstack([rng.multinomial(int(t), icf[j]) if t > 0 else np.zeros(K)
                        for j, t in enumerate(tot)]).reshape(n.shape)
        icb, cmb = t_fits(nb, a, r)
        ge += 2 * (loglik(nb, cmb) - loglik(nb, icb)) >= lr - 1e-12
    used = {k: solver("T").stats[k] - before[k] for k in before}
    p = (1 + ge) / (B_BOOT + 1)
    return [{"test": "A", "world": world, "rep": r, "N": N, "node": "T",
             "lr": lr, "p": p, "flag": p < A_ALPHA,
             "ici_identical_to_deployed": dep_ici.tobytes() == ici.tobytes(),
             "cmap_maxdiff_to_deployed": float(np.abs(dep_cm - cm).max()),
             "n_ecos": used["ECOS"], "n_scs": used["SCS"],
             "n_fallback": used["fallback"]}]


def d_test(task):
    world, r, N = task
    data = train_data(world, r, N)
    prior = ms.expert_prior_cpts(np.random.default_rng(1000 + r))
    out = []
    for xi, x in enumerate(ms.INTERNAL):
        rng = np.random.default_rng(100000 + 1000 * r + 100 * D_WI[world]
                                    + 10 * NI[N] + xi)
        n = g.counts(x, data)
        before = dict(solver(x).stats)
        lr, p = em.lr_test(x, n, prior[x].table, solver(x), rng)
        used = {k: solver(x).stats[k] - before[k] for k in before}
        out.append({"test": "D", "world": world, "rep": r, "N": N, "node": x,
                    "lr": lr, "p": p, "flag": p < em.DIAG_ALPHA,
                    "n_ecos": used["ECOS"], "n_scs": used["SCS"],
                    "n_fallback": used["fallback"]})
    return out


def run_task(task):
    kind, rest = task[0], task[1:]
    return a_test(rest) if kind == "A" else d_test(rest)


def prefix_check():
    """The first N rows agree with the main study's (200-row) and [D]'s
    (1000-row) datasets; checked on a few worlds and replications."""
    n = 0
    for world in ("standard", "nonadd-1.0", "H-v-U"):
        for r in (0, 7):
            a = dataio.generate_dataset(em.world(world), 200, seed=r)
            b = dataio.generate_dataset(em.world(world), 1000, seed=r)[:200]
            assert a == b, (world, r)
            n += 1
    return n


def run_diagnostics(workers):
    from multiprocessing import Pool
    tasks = [("A", w, r, N) for w, Ns in A_GRID.items() for N in Ns
             for r in range(REPS)]
    tasks += [("D", w, r, N) for w, Ns in D_GRID.items() for N in Ns
              for r in range(REPS)]
    t0 = time.time()
    rows = []
    with Pool(workers) as pool:
        for i, out in enumerate(pool.imap_unordered(run_task, tasks), 1):
            rows += out
            if i % 20 == 0 or i == len(tasks):
                print(f"[expW_add diag] {i}/{len(tasks)} tasks "
                      f"({time.time()-t0:.0f}s)", flush=True)
    df = pd.DataFrame(rows).sort_values(["test", "world", "N", "rep", "node"])
    df.to_csv(os.path.join(RES, f"{P}_diag_raw.csv"), index=False)
    return df


# ------------------------------------------------------------ adaptive -----
def flags(diag_add, diag_d0):
    A = {(x.world, x.rep, x.N): bool(x.flag)
         for x in diag_add[diag_add.test == "A"].itertuples()}
    D = {(x.world, x.rep, x.N, x.node): bool(x.flag)
         for x in diag_add[diag_add.test == "D"].itertuples()}
    D.update({(x.world, x.rep, x.N, x.node): bool(x.flag)
              for x in diag_d0.itertuples()})
    return A, D


_CTX = {}


def context(world, r):
    """Test set and sensor layer of the main study for (world, r)."""
    if (world, r) not in _CTX:
        wbn = em.world(world)
        test = dataio.generate_dataset(wbn, em.TEST_N, seed=9000 + r)
        calib = dataio.generate_dataset(wbn, em.CALIB_N, seed=5000 + r)
        est = dataio.estimate_confusion(calib)
        alpha, _ = dataio.fit_reliability(calib)
        _CTX.clear()
        _CTX[(world, r)] = (test, np.array([s["T"] for s in test]), est, alpha)
    return _CTX[(world, r)]


def fit_eval(world, r, N, t_full, relaxed):
    data = dataio.generate_dataset(em.world(world), 200, seed=r)[:N]
    prior = ms.expert_prior_cpts(np.random.default_rng(1000 + r))
    cp = {x: (g.map_cpt(x, data, prior[x], ESS) if x in relaxed
              else g.cmap_cpt(x, data, prior[x], ESS)) for x in LATENT}
    cp["T"] = (g.cmap_cpt("T", data, prior["T"], ESS) if t_full
               else g.ordinal_ici_cpt("T", data, prior["T"], ESS, seed=r))
    test, y, est, alpha = context(world, r)
    Pm = mt.posteriors(cp, test, ms.SENSORS, "calibrated", sensor_cpts=est,
                       alpha=alpha)[0]
    return em.metrics_all(Pm, y)


def run_adaptive():
    w = em.read_csv(os.path.join(RES_IN, "expW_raw.csv"))
    dadd = em.read_csv(os.path.join(RES, f"{P}_diag_raw.csv"))
    d0 = em.read_csv(os.path.join(RES_IN, "expW_diag_raw.csv"))
    A, D = flags(dadd, d0)
    rows, checks = [], []
    t0 = time.time()
    for world in AD_WORLDS:
        for r in range(REPS):
            for N in NS:
                fa = A[(world, r, N)]
                fd = tuple(x for x in LATENT if D[(world, r, N, x)])
                src = None
                if not fa and not fd:
                    src = "Proposed"
                elif fa and not fd:
                    src = "cMAP-full"
                elif not fa and fd == ("H",) and world == "H-v-U":
                    src = "Proposed-Hfree"
                if src is not None:
                    x = w[(w.world == world) & (w.rep == r) & (w.N == N)
                          & (w.method == src)]
                    assert len(x) == 1
                    m = {k: float(x[k].iloc[0]) for k in METRICS}
                else:
                    m = fit_eval(world, r, N, fa, fd)
                rows.append({"world": world, "rep": r, "N": N,
                             "method": "Proposed-adaptive", "A_flag": fa,
                             "D_flags": "".join(fd), "source": src or "fresh",
                             **m})
            # path check: the fresh path reproduces expW_raw bit for bit
            if r < 2:
                for N in NS:
                    for meth, t_full, rel in (("Proposed", False, ()),
                                              ("cMAP-full", True, ())):
                        m = fit_eval(world, r, N, t_full, rel)
                        x = w[(w.world == world) & (w.rep == r) & (w.N == N)
                              & (w.method == meth)]
                        same = all(np.array_equal(np.float64(m[k]),
                                                  np.float64(x[k].iloc[0]),
                                                  equal_nan=True)
                                   for k in METRICS)
                        checks.append({"world": world, "rep": r, "N": N,
                                       "method": meth, "identical": same})
        print(f"[expW_add adaptive] {world} done ({time.time()-t0:.0f}s)",
              flush=True)
    pd.DataFrame(rows).to_csv(os.path.join(RES, f"{P}_adaptive_raw.csv"),
                              index=False)
    pd.DataFrame(checks).to_csv(os.path.join(RES, f"{P}_pathcheck.csv"),
                                index=False)


# ------------------------------------------------------------- summary -----
def summarize():
    dd = em.read_csv(os.path.join(RES, f"{P}_diag_raw.csv"))
    ad = em.read_csv(os.path.join(RES, f"{P}_adaptive_raw.csv"))
    ad["D_flags"] = ad["D_flags"].fillna("")
    pc = em.read_csv(os.path.join(RES, f"{P}_pathcheck.csv"))
    w = em.read_csv(os.path.join(RES_IN, "expW_raw.csv"))
    L = [f"# Exp-W-add: additivity diagnostic [A] and adaptive rules "
         f"({dd.rep.nunique()} replications)", ""]
    a = dd[dd.test == "A"]
    same = a.ici_identical_to_deployed.astype(str).str.lower().eq("true")
    L.append(f"- [A] tests: {len(a)}; ICI fit identical to the deployed fit in "
             f"{int(same.sum())}; max |cMAP-full - "
             f"deployed| {a.cmap_maxdiff_to_deployed.max():.1e}; solver: ECOS "
             f"{int(dd.n_ecos.sum())}, SCS {int(dd.n_scs.sum())}, MAP fallback "
             f"{int(dd.n_fallback.sum())} (all tests, all others CLARABEL)")
    L.append(f"- fresh-path check: {int(pc.identical.sum())} of {len(pc)} cells "
             f"bit-identical to expW_raw.csv")
    L.append("\n## [A] flag rate (p_T < 0.05)")
    L.append("| world | N | flagged | median p |")
    L.append("|---|---|---|---|")
    for world, Ns in A_GRID.items():
        for N in Ns:
            q = a[(a.world == world) & (a.N == N)]
            L.append(f"| {world} | {N} | {int(q.flag.sum())}/{len(q)} | "
                     f"{q.p.median():.3f} |")
    d = dd[dd.test == "D"]
    L.append("\n## [D] in S1 (p_X < 0.0125)")
    L.append("| world | N | any node | per node H/N/C/T |")
    L.append("|---|---|---|---|")
    for world, Ns in D_GRID.items():
        for N in Ns:
            q = d[(d.world == world) & (d.N == N)]
            per = "/".join(f"{int(q[q.node == x].flag.sum())}" for x in ms.INTERNAL)
            L.append(f"| {world} | {N} | {int(q.groupby('rep').flag.any().sum())}"
                     f"/{q.rep.nunique()} | {per} |")
    L.append("\n## adaptive selection (counts over 20 replications)")
    L.append("| world | N | Proposed | cMAP-full | Proposed-Hfree | fresh "
             "(combination) |")
    L.append("|---|---|---|---|---|---|")
    for world in AD_WORLDS:
        for N in NS:
            q = ad[(ad.world == world) & (ad.N == N)]
            fr = q[q.source == "fresh"]
            combo = ", ".join(f"{('T full' if x.A_flag else 'ICI')}+relax "
                              f"{x.D_flags or '-'}" for x in fr.itertuples())
            L.append(f"| {world} | {N} | {int((q.source == 'Proposed').sum())} | "
                     f"{int((q.source == 'cMAP-full').sum())} | "
                     f"{int((q.source == 'Proposed-Hfree').sum())} | "
                     f"{len(fr)}{(' (' + combo + ')') if len(fr) else ''} |")
    L.append("\n## Brier and ECE: Proposed-adaptive against the fixed estimators")
    L.append("| world | N | adaptive Brier | Proposed | cMAP-full | H-free | "
             "adaptive - Proposed (p) | adaptive - alternative (p) | "
             "adaptive ECE | ECE: adaptive - Proposed (p) |")
    L.append("|---|---|---|---|---|---|---|---|---|---|")
    for world in AD_WORLDS:
        for N in NS:
            q = ad[(ad.world == world) & (ad.N == N)].set_index("rep")
            def m_(meth, k="brier"):
                return w[(w.world == world) & (w.N == N) &
                         (w.method == meth)].set_index("rep")[k]
            pr, cm = m_("Proposed"), m_("cMAP-full")
            hf = m_("Proposed-Hfree") if world == "H-v-U" else None
            alt = hf if world == "H-v-U" else cm
            t1 = es.across_rep_tests((q.brier - pr).values)
            t2 = es.across_rep_tests((q.brier - alt).values)
            te = es.across_rep_tests((q.ece - m_("Proposed", "ece")).values)
            hfs = f"{hf.mean():.4f}" if hf is not None else "-"
            L.append(f"| {world} | {N} | {q.brier.mean():.4f} | {pr.mean():.4f} | "
                     f"{cm.mean():.4f} | {hfs} | {t1['mean']:+.4f} ({t1['t_p']:.1e}) | "
                     f"{t2['mean']:+.4f} ({t2['t_p']:.1e}) | {q.ece.mean():.4f} | "
                     f"{te['mean']:+.4f} ({te['t_p']:.1e}) |")
    txt = "\n".join(L)
    with open(os.path.join(RES, f"{P}_summary.md"), "w", encoding="utf-8") as f:
        f.write(txt + "\n")
    print(txt)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=20)
    ap.add_argument("--summary-only", action="store_true")
    ap.add_argument("--only", default="", help="diag,adaptive")
    ap.add_argument("--res", default="", help="output folder (test runs)")
    ap.add_argument("--time-one", action="store_true",
                    help="time one [A] test and one [D] test, write nothing")
    a = ap.parse_args()
    if a.res:
        RES = a.res
    os.makedirs(RES, exist_ok=True)
    if a.time_one:
        t = time.time(); x = a_test(("standard", 0, 200))
        print(f"[A] one test {time.time()-t:.1f}s", x)
        t = time.time(); y = d_test(("nonadd-1.0", 0, 200))
        print(f"[D] one test (4 nodes) {time.time()-t:.1f}s", [(r['node'], r['p']) for r in y])
        print("prefix checks:", prefix_check())
        sys.exit(0)
    if not a.summary_only:
        parts = a.only.split(",") if a.only else ["diag", "adaptive"]
        if "diag" in parts:
            print("prefix checks passed:", prefix_check(), flush=True)
            run_diagnostics(a.workers)
        if "adaptive" in parts:
            run_adaptive()
    if a.summary_only or "adaptive" in (a.only.split(",") if a.only else ["adaptive"]):
        summarize()
