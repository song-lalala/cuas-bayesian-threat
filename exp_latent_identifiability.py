"""
Exp-EM: what the unlabeled case can identify (Sec. IV-D) and how good the latent
rationale is when it is learned without labels (Sec. VII-B). Core files are not
modified; results go to results/expEM_*.

Part 1, identifiability at the world's true tables.
  With H, N and C unlabeled, the tables enter the data only through
      Q(T | v, rdot, d, z, cls, emit)
        = sum_{h,n,c} P(h | v,rdot,d) P(n | cls,emit) P(c | d,z) P(T | h,n,c).
  Free parameters: H 54, N 6, C 12, T 54, 126 in all. The Jacobian of Q in those
  parameters is written exactly, since Q is linear in each table; its rank comes
  from a singular value decomposition (tolerance 1e-10 of the largest). The rank
  is also taken with T known, and for the distribution of what a training row
  actually holds, P(T, Rr, Re, Ra, Rf | v, rdot, d, z), which is a function of Q
  and so cannot have a larger rank.
  Recoding: for a latent X with K states and an invertible K x K matrix R with
  unit row sums, X' = X R and T' = R^-1 T (along the axis of X) leave Q unchanged.
  Dimension: K(K-1) per variable, 6 + 2 + 6 = 14. Their tangents are shown to
  span the null space of the Jacobian.
  Reach: starting from R = I, SLSQP moves one table entry at a time as far as it
  can, up and down, with all recoded tables non-negative and all three recodings
  free together. This is a local search, so the figures are lower bounds on the
  reach near the true tables. Permutations of the states are recodings too, and
  are far from R = I; the monotonicity constraints and the prior exclude them.

Part 2, quality of the latent posteriors on test scenarios.
  Protocol of exp_mapem.py: test 9000+r, calibration 5000+r, prior rng(1000+r),
  training r, N = 50, 200, 20 replications, MAP-EM with 12 iterations and the
  constrained final M-step. Methods: Proposed (labels), MAP-EM (no labels),
  prior only (B2), and the true internal CPTs. Evidence: v, rdot, d, z and the
  four reports, through the calibration-split sensor estimates and reliability
  weights. For each of H, N, C: accuracy of the top state against the scenario's
  true state, and log loss of P(X | e). Paired t over replications.
  P(H | e) and P(C | e) are their table rows, since T is not observed at test
  time and H and C are functions of observed parents only.

Run:  python exp_latent_identifiability.py [--workers 8]
      python exp_latent_identifiability.py --summary-only
Outputs: results/expEM_rank.csv, results/expEM_reach.csv, results/expEM_recoding.csv,
         results/expEM_latent_raw.csv, results/expEM_summary.md
"""
from __future__ import annotations
import os
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import sys, time, argparse
from itertools import product as iproduct
import numpy as np
import pandas as pd
from scipy import stats as st
from scipy.optimize import minimize

import model_spec as ms
import dataio
import grounding as g
import pipeline as pl

RES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
P = "expEM"
TEST_N = 2500
EM_ITERS = 12
NS = (50, 200)
KEEP = ["v", "rdot", "d", "z", "Rr", "Re", "Ra", "Rf", "T"]
TOL = 1e-10
LAT = ("H", "N", "C")


def q_table(H, N, C, T):
    """Q[v, rdot, d, z, cls, emit, t]."""
    return np.einsum("vrdh,cen,dzk,hnkt->vrdzcet", H, N, C, T, optimize=True)


def directions(tab):
    """Tangent of each free parameter of a table: +1 on an entry, -1 on the last
    entry of its column."""
    out = []
    K = tab.shape[-1]
    for j in iproduct(*[range(s) for s in tab.shape[:-1]]):
        for k in range(K - 1):
            d = np.zeros_like(tab)
            d[j + (k,)] = 1.0
            d[j + (K - 1,)] = -1.0
            out.append(d)
    return out


def jacobian(tabs, names=("H", "N", "C", "T")):
    """Exact Jacobian of Q: Q is linear in each table."""
    cols = []
    for nm in names:
        for d in directions(tabs[nm]):
            t2 = dict(tabs)
            t2[nm] = d
            cols.append(q_table(t2["H"], t2["N"], t2["C"], t2["T"]).ravel())
    return np.array(cols).T


def rank_of(J):
    s = np.linalg.svd(J, compute_uv=False)
    r = int((s > TOL * s[0]).sum())
    return r, s


def obs_mixing(gt):
    """M[(v,rdot,d,z,reports,t), (v,rdot,d,z,cls,emit,t)] applied through the
    sensor tables: P(T, reports | inputs) from Q."""
    pc, pe = gt.cpts["cls"].table, gt.cpts["emit"].table
    Rr, Re, Ra, Rf = (gt.cpts[s].table for s in ("Rr", "Re", "Ra", "Rf"))
    # W[d, cls, emit, rr, re, ra, rf] = P(cls) P(emit) prod_s P(R_s | ...)
    return np.einsum("c,e,cdx,cdy,cdz,cew->dcexyzw", pc, pe, Rr, Re, Ra, Rf, optimize=True)




def _obs_one(Q, W):
    return np.einsum("vrdacet,dcexyzw->vrdaxyzwt", Q, W, optimize=True).ravel()


def recoding_tangents(tabs):
    """Tangents, in the 126 free parameters, of the recodings of H, N and C."""
    vecs = []
    axes = {"H": 0, "N": 1, "C": 2}
    for nm in LAT:
        K = tabs[nm].shape[-1]
        for i in range(K):
            for l in range(K):
                if l == K - 1:
                    continue
                A = np.zeros((K, K))
                A[i, l] += 1.0
                A[i, K - 1] -= 1.0
                dX = tabs[nm] @ A
                dT = -np.moveaxis(np.tensordot(A, tabs["T"], axes=(1, axes[nm])), 0, axes[nm])
                v = []
                for m in ("H", "N", "C", "T"):
                    d = dX if m == nm else (dT if m == "T" else np.zeros_like(tabs[m]))
                    v.append(d[..., :-1].ravel())
                vecs.append(np.concatenate(v))
    return np.array(vecs).T


def recode(tabs, x):
    """Apply R_H, R_N, R_C built from the 14 numbers x (R = I + A, A with zero
    row sums); returns the recoded tables."""
    o, Rs = 0, {}
    for nm in LAT:
        K = tabs[nm].shape[-1]
        A = np.zeros((K, K))
        for i in range(K):
            A[i, :K - 1] = x[o:o + K - 1]
            A[i, K - 1] = -x[o:o + K - 1].sum()
            o += K - 1
        Rs[nm] = np.eye(K) + A
    out = {nm: tabs[nm] @ Rs[nm] for nm in LAT}
    T = tabs["T"]
    for ax, nm in enumerate(LAT):
        T = np.moveaxis(np.tensordot(np.linalg.inv(Rs[nm]), T, axes=(1, ax)), 0, ax)
    out["T"] = T
    return out


def part1():
    gt = ms.ground_truth_bn()
    tabs = {nm: gt.cpts[nm].table for nm in ("H", "N", "C", "T")}
    J = jacobian(tabs)
    r, s = rank_of(J)
    J3 = jacobian(tabs, names=("H", "N", "C"))
    r3, s3 = rank_of(J3)
    assert J.shape[1] == 126 and J3.shape[1] == 72
    W = obs_mixing(gt)
    Jo = np.array([_obs_one(J[:, i].reshape((3, 3, 3, 2, 3, 2, 4)), W) for i in range(J.shape[1])]).T
    ro, so = rank_of(Jo)
    ro3, so3 = rank_of(Jo[:, :72])
    V = recoding_tangents(tabs)
    rv, _ = rank_of(V)
    leak = float(np.abs(J @ V).max())
    rows = [{"what": "Q, 126 free parameters", "params": J.shape[1], "rank": r, "s_before_gap": s[r - 1], "s_after_gap": s[r] if r < len(s) else 0.0},
            {"what": "Q, T known, 72 free parameters", "params": J3.shape[1], "rank": r3, "s_before_gap": s3[r3 - 1], "s_after_gap": s3[r3] if r3 < len(s3) else 0.0},
            {"what": "P(T, reports | inputs), 126 free parameters", "params": Jo.shape[1], "rank": ro, "s_before_gap": so[ro - 1], "s_after_gap": so[ro] if ro < len(so) else 0.0},
            {"what": "P(T, reports | inputs), T known, 72 free parameters", "params": 72, "rank": ro3, "s_before_gap": so3[ro3 - 1], "s_after_gap": so3[ro3] if ro3 < len(so3) else 0.0},
            {"what": "recoding tangents", "params": V.shape[1], "rank": rv, "s_before_gap": float("nan"), "s_after_gap": leak}]
    pd.DataFrame(rows).to_csv(os.path.join(RES, f"{P}_rank.csv"), index=False)

    # reach near the true tables
    x0 = np.zeros(14)

    def con(x):
        t = recode(tabs, x)
        return np.concatenate([t[k].ravel() for k in ("H", "N", "C", "T")])

    entries = [(nm, idx) for nm in ("H", "N", "C", "T") for idx in iproduct(*[range(s) for s in tabs[nm].shape])]
    rows, t0 = [], time.time()
    best = None
    for nm, idx in entries:
        base = tabs[nm][idx]
        for sign in (1.0, -1.0):
            if sign < 0 and base < 1e-3:
                rows.append({"node": nm, "entry": str(idx), "sign": sign, "move": 0.0, "feasible": True})
                continue

            def obj(x):
                return -sign * (recode(tabs, x)[nm][idx] - base)
            res = minimize(obj, x0, method="SLSQP", constraints=[{"type": "ineq", "fun": con}],
                           options={"maxiter": 200, "ftol": 1e-10})
            t = recode(tabs, res.x)
            feas = all(t[k].min() >= -1e-9 for k in t)
            mv = abs(t[nm][idx] - base) if feas else 0.0
            if feas and (best is None or mv > best[0]):
                best = (mv, res.x.copy())
            rows.append({"node": nm, "entry": str(idx), "sign": sign, "move": float(mv), "feasible": bool(feas)})
        if len(rows) % 80 < 2:
            print(f"[expEM] reach {len(rows)}/{2 * len(entries)} ({time.time() - t0:.0f}s)", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RES, f"{P}_reach.csv"), index=False)
    # the recoding that moved one entry furthest leaves Q and the observed distribution unchanged
    t2 = recode(tabs, best[1])
    q0 = q_table(tabs["H"], tabs["N"], tabs["C"], tabs["T"])
    q2 = q_table(t2["H"], t2["N"], t2["C"], t2["T"])
    o0, o2 = _obs_one(q0, W), _obs_one(q2, W)
    ex = {"largest_move_" + k: float(np.abs(t2[k] - tabs[k]).max()) for k in t2}
    ex.update({"largest_change_Q": float(np.abs(q2 - q0).max()), "largest_change_observed": float(np.abs(o2 - o0).max()),
               "smallest_entry_recoded": float(min(t2[k].min() for k in t2))})
    assert ex["largest_change_Q"] < 1e-12 and ex["largest_change_observed"] < 1e-12
    pd.DataFrame([ex]).to_csv(os.path.join(RES, f"{P}_recoding.csv"), index=False)
    return ex


def rep_latent(r):
    gt = ms.ground_truth_bn()
    test = dataio.generate_dataset(gt, TEST_N, seed=9000 + r)
    calib = dataio.generate_dataset(gt, 300, seed=5000 + r)
    est = dataio.estimate_confusion(calib)
    alpha, _ = dataio.fit_reliability(calib)
    prior = ms.expert_prior_cpts(np.random.default_rng(1000 + r))
    models = [("prior only", 0, prior), ("true CPTs", 0, {k: gt.cpts[k] for k in ms.INTERNAL})]
    for N in NS:
        full = dataio.generate_dataset(gt, N, seed=r)
        partial = [{k: row[k] for k in KEEP} for row in full]
        em = g.em_internal(partial, prior, pl.ESS_FIXED, n_iter=EM_ITERS, seed=r,
                           constrained=True, sensor_cpts=est)
        sup, _ = pl.ground_internal("Proposed", full, r, ess=pl.ESS_FIXED, gt=gt)
        models += [("Proposed", N, sup), ("MAP-EM", N, em)]
    rows = []
    for name, N, cpts in models:
        bns = {dv: dataio.build_inference_bn(cpts, ms.SENSORS, dv, "calibrated", sensor_cpts=est, alpha=alpha)
               for dv in range(ms.CARD["d"])}
        hit = {x: 0 for x in LAT}
        ll = {x: 0.0 for x in LAT}
        for scn in test:
            bn = bns[scn["d"]]
            ev = dataio.evidence_from_scenario(scn, ms.SENSORS)
            for x in LAT:
                p = bn.query([x], ev).table
                hit[x] += int(p.argmax() == scn[x])
                ll[x] -= float(np.log(max(p[scn[x]], 1e-12)))
        for x in LAT:
            rows.append({"rep": r, "N": N, "method": name, "node": x, "acc": hit[x] / len(test), "logloss": ll[x] / len(test)})
    return rows


def part2(workers):
    from multiprocessing import Pool
    t0, rows = time.time(), []
    with Pool(workers) as pool:
        for i, rr in enumerate(pool.imap_unordered(rep_latent, range(20)), 1):
            rows += rr
            print(f"[expEM] latent {i}/20 ({time.time() - t0:.0f}s)", flush=True)
    pd.DataFrame(rows).sort_values(["rep", "N", "method", "node"]).to_csv(os.path.join(RES, f"{P}_latent_raw.csv"), index=False)


def summarize():
    rk = pd.read_csv(os.path.join(RES, f"{P}_rank.csv"))
    rc = pd.read_csv(os.path.join(RES, f"{P}_reach.csv"))
    ex = pd.read_csv(os.path.join(RES, f"{P}_recoding.csv")).iloc[0]
    la = pd.read_csv(os.path.join(RES, f"{P}_latent_raw.csv"), float_precision="round_trip")
    L = ["# Exp-EM: identifiability of the unlabeled case and quality of the latent posteriors", "",
         "## Jacobian rank at the true tables", "| what | parameters | rank | singular value at the rank | next singular value |", "|---|---|---|---|---|"]
    for _, q in rk.iterrows():
        L.append(f"| {q.what} | {q.params} | {q['rank']} | {q.s_before_gap:.3g} | {q.s_after_gap:.3g} |")
    L.append("\nLast row: the next column gives the largest entry of J times the recoding tangents.")
    L.append("\n## reach of a recoding near the true tables (SLSQP from R = I, lower bounds; permutations excluded)")
    L.append(f"searches that ended infeasible and count as no move: {int((~rc.feasible).sum())} of {len(rc)}")
    L.append("| table | largest move of any entry |\n|---|---|")
    for nm in ("H", "N", "C", "T"):
        L.append(f"| {nm} | {rc[rc.node == nm].move.max():.3f} |")
    L.append(f"\nentries searched, both directions: {len(rc)}")
    L.append(f"\nthe recoding with the largest move: entries move by up to H {ex.largest_move_H:.3f}, N {ex.largest_move_N:.3f}, C {ex.largest_move_C:.3f}, T {ex.largest_move_T:.3f}; Q changes by at most {ex.largest_change_Q:.1e} and P(T, reports | inputs) by at most {ex.largest_change_observed:.1e}")
    L.append("\n## latent posteriors on test scenarios (means over 20 replications)")
    L.append("| method | N | H acc | N acc | C acc | H log loss | N log loss | C log loss |\n|---|---|---|---|---|---|---|---|")

    def cell(m, n, x, k):
        q = la[(la.method == m) & (la.N == n) & (la.node == x)].sort_values("rep")
        assert len(q) == 20, (m, n, x, len(q))
        return q[k].values
    order = [("prior only", 0), ("true CPTs", 0)] + [(m, n) for n in NS for m in ("Proposed", "MAP-EM")]
    for m, n in order:
        L.append(f"| {m} | {n} | " + " | ".join(f"{cell(m, n, x, 'acc').mean():.3f}" for x in LAT) + " | "
                 + " | ".join(f"{cell(m, n, x, 'logloss').mean():.3f}" for x in LAT) + " |")
    L.append("\n## MAP-EM minus Proposed (paired t over 20 replications)")
    L.append("| N | node | d acc (p) | d log loss (p) |\n|---|---|---|---|")
    for n in NS:
        for x in LAT:
            parts = []
            for k in ("acc", "logloss"):
                d = cell("MAP-EM", n, x, k) - cell("Proposed", n, x, k)
                parts.append(f"{d.mean():+.3f} ({st.ttest_1samp(d, 0.0).pvalue:.2g})")
            L.append(f"| {n} | {x} | " + " | ".join(parts) + " |")
    txt = "\n".join(L) + "\n"
    with open(os.path.join(RES, f"{P}_summary.md"), "w", encoding="utf-8", newline="\n") as f:
        f.write(txt)
    sys.stdout.reconfigure(encoding="utf-8")
    print(txt)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--summary-only", action="store_true")
    a = ap.parse_args()
    if not a.summary_only:
        out = part1()
        print(out, flush=True)
        part2(a.workers)
    summarize()
