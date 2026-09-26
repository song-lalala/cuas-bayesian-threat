"""
Main experimental study (revision v2): every headline table/figure from ONE
protocol -- S=20 independent replications, each redrawing the training sets,
the calibration split (confusion estimates + fitted reliability), and the
2500-scenario test draw. All manuscript numbers in Tables 3-5, Fig. 3, the
calibration-size table, and the abstract-diagonal sweep come from this
script's results/ files.

Per replication:
  scarcity  : methods {B0, B1-MLE, B2-expert, MAP, cMAP-full, Proposed}
              x N in {10,20,50,100,200,400}; ESS selected by held-out
              log-loss per (N, replication); paired Proposed-vs-MLE and
              Proposed-vs-B2 deltas with tests.
  fusion    : single sensors vs abstract vs calibrated vs calibrated+fitted
              reliability (Table 4) + per-class paired DeLong, McNemar,
              scenario-level paired bootstraps.
  ablation  : full / -ICI / -constraints / -prior / -reliability at N=80.
  calibsize : confusion-estimation split size sweep (Table 5).
  absweep   : abstract-model diagonal 0.70..0.99 sensitivity (baseline
              fairness check).
Also logs the majority-class baseline, the test class distribution, the
cMAP solver convergence tally, and a reliability-diagram dump (rep 0).

Run:  python exp_significance.py            (full: S=20, B=1000, ~40-60 min)
      python exp_significance.py --reps 2 --boot 100      (smoke)
      python exp_significance.py --summary-only
"""
from __future__ import annotations
import os, sys, time, argparse
import numpy as np
import pandas as pd
from scipy import stats as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({"font.family": "serif", "mathtext.fontset": "stix",
                     "font.size": 10})

import model_spec as ms
import dataio
import grounding as g
import metrics as mt
import pipeline as pl

RES = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(RES, exist_ok=True)
TEST_N = 2500
FUSION_N = 200
ABLATION_N = 80
N_GRID = (10, 20, 50, 100, 200, 400)
TABLE3_NS = (10, 50, 200, 400)
SC_METHODS = ["B0-heuristic", "B1-MLE", "B2-expert", "MAP", "cMAP-full",
              "Proposed"]
CALIB_GRID = (0, 30, 100, 300, 1000, -1)      # 0=abstract, -1=oracle-marginal
DIAG_GRID = (0.70, 0.80, 0.90, 0.94, 0.99)


# ---------------- paired-test helpers (unchanged core) ----------------
def _midrank(x):
    order = np.argsort(x)
    z = x[order]
    n = len(x)
    t = np.zeros(n)
    i = 0
    while i < n:
        j = i
        while j < n and z[j] == z[i]:
            j += 1
        t[i:j] = 0.5 * (i + j - 1) + 1
        i = j
    out = np.empty(n)
    out[order] = t
    return out


def delong_paired(y_bin, s1, s2):
    pos = y_bin == 1
    m, n = int(pos.sum()), int((~pos).sum())
    if m == 0 or n == 0:
        return np.nan, np.nan, np.nan, np.nan
    aucs, v01, v10 = [], [], []
    for s in (s1, s2):
        x, yv = s[pos], s[~pos]
        tx = _midrank(x)
        ty = _midrank(yv)
        tz = _midrank(np.concatenate([x, yv]))
        auc = (tz[:m].sum() - m * (m + 1) / 2.0) / (m * n)
        aucs.append(auc)
        v01.append((tz[:m] - tx) / n)
        v10.append(1.0 - (tz[m:] - ty) / m)
    v01 = np.vstack(v01); v10 = np.vstack(v10)
    S01 = np.cov(v01); S10 = np.cov(v10)
    var = (S01[0, 0] + S01[1, 1] - 2 * S01[0, 1]) / m \
        + (S10[0, 0] + S10[1, 1] - 2 * S10[0, 1]) / n
    d = aucs[0] - aucs[1]
    if var <= 0:
        return aucs[0], aucs[1], np.nan, np.nan
    z = d / np.sqrt(var)
    return aucs[0], aucs[1], z, 2 * st.norm.sf(abs(z))


def macro_auc(P, y):
    from sklearn.metrics import roc_auc_score
    try:
        return float(roc_auc_score(np.eye(4)[y], P, multi_class="ovr",
                                   average="macro", labels=[0, 1, 2, 3]))
    except Exception:
        return np.nan


def mcnemar_exact(pred1, pred2, y):
    c1 = pred1 == y
    c2 = pred2 == y
    b = int((c1 & ~c2).sum())
    c = int((~c1 & c2).sum())
    if b + c == 0:
        return b, c, 1.0
    return b, c, float(st.binomtest(min(b, c), b + c, 0.5).pvalue)


def paired_bootstrap_delta(P1, P2, y, B, rng, fn):
    n = len(y)
    d_hat = fn(P2, y) - fn(P1, y)
    ds = np.empty(B)
    for b in range(B):
        idx = rng.integers(0, n, n)
        ds[b] = fn(P2[idx], y[idx]) - fn(P1[idx], y[idx])
    lo, hi = np.percentile(ds, [2.5, 97.5])
    return d_hat, lo, hi


def t_ci(x, conf=0.95):
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    m = x.mean()
    if len(x) < 2:
        return m, np.nan, np.nan
    se = x.std(ddof=1) / np.sqrt(len(x))
    h = se * st.t.ppf(0.5 + conf / 2, len(x) - 1)
    return m, m - h, m + h


def across_rep_tests(deltas):
    deltas = np.asarray(deltas, float)
    deltas = deltas[~np.isnan(deltas)]
    m, lo, hi = t_ci(deltas)
    tt = st.ttest_1samp(deltas, 0.0)
    try:
        wp = float(st.wilcoxon(deltas).pvalue)
    except ValueError:
        wp = np.nan
    return dict(mean=m, lo=lo, hi=hi, t_p=float(tt.pvalue), wilcoxon_p=wp)


# ---------------------------------------------------------------- main loop
def run(reps=20, B=1000):
    t0 = time.time()
    g.cmap_report(reset=True)
    sc_rows, fus_rows, abl_rows, cs_rows, dg_rows = [], [], [], [], []
    gt = ms.ground_truth_bn()
    for r in range(reps):
        rng = np.random.default_rng(31000 + r)
        test = dataio.generate_dataset(gt, TEST_N, seed=9000 + r)
        calib = dataio.generate_dataset(gt, 300, seed=5000 + r)
        est = dataio.estimate_confusion(calib)
        alpha, _ = dataio.fit_reliability(calib)
        ycls = np.array([s["T"] for s in test])
        majority = float(np.bincount(ycls, minlength=4).max() / len(ycls))

        # ---- scarcity (Table 3 + Fig 3) --------------------------------
        for N in N_GRID:
            data = dataio.generate_dataset(gt, N, seed=r)
            posts = {}
            for m in SC_METHODS:
                cpts, _ = pl.ground_internal(m, data, r, gt=gt)
                met, (P, yy) = mt.evaluate(cpts, test, sensor_cpts=est,
                                           alpha=alpha)
                sc_rows.append({"rep": r, "N": N, "method": m,
                                "ess": pl.ESS_FIXED, "majority": majority,
                                **met})
                posts[m] = P
            for ref in ("B1-MLE", "B2-expert"):
                dA = macro_auc(posts["Proposed"], yy) - macro_auc(posts[ref], yy)
                dE = mt.ece(posts["Proposed"], yy) - mt.ece(posts[ref], yy)
                bb, cc, pp = mcnemar_exact(posts[ref].argmax(1),
                                           posts["Proposed"].argmax(1), yy)
                sc_rows.append({"rep": r, "N": N,
                                "method": f"_delta_vs_{ref}",
                                "dAUC": dA, "dECE": dE, "mcnemar_b": bb,
                                "mcnemar_c": cc, "mcnemar_p": pp})

        # ---- fusion (Table 4) ------------------------------------------
        data = dataio.generate_dataset(gt, FUSION_N, seed=r)
        cpts, _ = pl.ground_internal("Proposed", data, r, gt=gt)
        fus_cfgs = [
            ("radar only",           ["Rr"],     "calibrated", est, alpha),
            ("RF only",              ["Rf"],     "calibrated", est, alpha),
            ("EO/IR only",           ["Re"],     "calibrated", est, alpha),
            ("acoustic only",        ["Ra"],     "calibrated", est, alpha),
            ("all (abstract model)", ms.SENSORS, "abstract",   None, None),
            ("all (calibrated)",     ms.SENSORS, "calibrated", est, None),
            ("all (calib+reliab)",   ms.SENSORS, "calibrated", est, alpha),
        ]
        Ps = {}
        for name, son, mode, sc, al in fus_cfgs:
            met, (P, y) = mt.evaluate(cpts, test, sensors_on=son,
                                      sensor_mode=mode, sensor_cpts=sc,
                                      alpha=al)
            fus_rows.append({"rep": r, "config": name, **met})
            Ps[name] = P
        Pa, Pc, Pcr = (Ps["all (abstract model)"], Ps["all (calibrated)"],
                       Ps["all (calib+reliab)"])
        dl = {}
        for k in range(4):
            _, _, z, p = delong_paired((y == k).astype(int), Pa[:, k], Pc[:, k])
            dl[f"delong_z_c{k}"] = z
            dl[f"delong_p_c{k}"] = p
        dAUC, dAUC_lo, dAUC_hi = paired_bootstrap_delta(Pa, Pc, y, B, rng, macro_auc)
        dECE, dECE_lo, dECE_hi = paired_bootstrap_delta(Pa, Pc, y, B, rng, mt.ece)
        b_mc, c_mc, p_mc = mcnemar_exact(Pa.argmax(1), Pc.argmax(1), y)
        dECEr, dECEr_lo, dECEr_hi = paired_bootstrap_delta(Pc, Pcr, y, B, rng, mt.ece)
        fus_rows.append({
            "rep": r, "config": "_paired_stats",
            "dAUC": dAUC, "dAUC_lo": dAUC_lo, "dAUC_hi": dAUC_hi,
            "dECE": dECE, "dECE_lo": dECE_lo, "dECE_hi": dECE_hi,
            "dECE_reliab": dECEr, "dECEr_lo": dECEr_lo, "dECEr_hi": dECEr_hi,
            "mcnemar_b": b_mc, "mcnemar_c": c_mc, "mcnemar_p": p_mc, **dl})
        if r == 0:
            np.savetxt(os.path.join(RES, "expS_reliability_curve.csv"),
                       mt.reliability_curve(Pcr, y), delimiter=",",
                       header="conf,acc,frac", comments="")
        # hard-condition subsets, all metrics, same replications (no
        # single-seed subset reporting): radio-silent drones, and far-range
        # scenarios, under abstract / calibrated / calib+reliability
        silent = np.array([s["cls"] == 2 and s["emit"] == 1 for s in test])
        far = np.array([s["d"] == 0 for s in test])
        for subname, mask in [("silent-drone", silent), ("far-range", far),
                              ("far-silent", silent & far)]:
            if mask.sum() < 30:
                continue
            for cfgname, Pm in [("abstract", Pa), ("calibrated", Pc),
                                ("calib+reliab", Pcr)]:
                ys = y[mask]; Pmm = Pm[mask]
                fus_rows.append({
                    "rep": r, "config": f"_subset_{subname}_{cfgname}",
                    "n_sub": int(mask.sum()),
                    "acc": float((Pmm.argmax(1) == ys).mean()),
                    "macroAUC": macro_auc(Pmm, ys), "ece": mt.ece(Pmm, ys),
                    "brier": mt.brier(Pmm, ys),
                    "cwece": mt.classwise_ece(Pmm, ys)})

        # ---- ablation (Table 5-style block; N=80) ----------------------
        dataA = dataio.generate_dataset(gt, ABLATION_N, seed=r)
        abl_variants = [
            ("Proposed (full)",  "Proposed",                est, alpha),
            ("-ICI (full table)", "cMAP-full",              est, alpha),
            ("-constraints",     "Proposed-noconstraints",  est, alpha),
            ("-prior (MLE)",     "B1-MLE",                  est, alpha),
            ("-reliability",     "Proposed",                est, None),
        ]
        for name, meth, sc, al in abl_variants:
            cptsA, _ = pl.ground_internal(meth, dataA, r, gt=gt)
            met, _ = mt.evaluate(cptsA, test, sensor_cpts=sc, alpha=al)
            abl_rows.append({"rep": r, "variant": name, **met})

        # ---- calibration-set size --------------------------------------
        for cn in CALIB_GRID:
            if cn == 0:
                met, _ = mt.evaluate(cpts, test, sensor_mode="abstract")
                src = "abstract"
            elif cn == -1:
                met, _ = mt.evaluate(cpts, test, sensor_cpts=None, alpha=None)
                src = "oracle-marginal"
            else:
                est_n = dataio.estimate_confusion(
                    dataio.generate_dataset(gt, cn, seed=5000 + r))
                met, _ = mt.evaluate(cpts, test, sensor_cpts=est_n, alpha=None)
                src = "estimated"
            cs_rows.append({"rep": r, "calib_n": cn, "source": src, **met})

        # ---- abstract-diagonal sweep -----------------------------------
        for dgv in DIAG_GRID:
            src_cpts = dataio.abstract_sensor_cpts_diag(dgv)
            met, _ = mt.evaluate(cpts, test, sensor_cpts=src_cpts, alpha=None)
            dg_rows.append({"rep": r, "diag": dgv, **met})

        print(f"[expS] rep {r+1}/{reps} done ({time.time()-t0:.0f}s) "
              f"solver={g.cmap_report()}", flush=True)
        pd.DataFrame(sc_rows).to_csv(os.path.join(RES, "expS_scarcity_raw.csv"), index=False)
        pd.DataFrame(fus_rows).to_csv(os.path.join(RES, "expS_fusion_raw.csv"), index=False)
        pd.DataFrame(abl_rows).to_csv(os.path.join(RES, "expS_ablation_raw.csv"), index=False)
        pd.DataFrame(cs_rows).to_csv(os.path.join(RES, "expS_calibsize_raw.csv"), index=False)
        pd.DataFrame(dg_rows).to_csv(os.path.join(RES, "expS_absweep_raw.csv"), index=False)

    with open(os.path.join(RES, "expS_solver.txt"), "w") as f:
        f.write(str(g.cmap_report()) + "\n")
    summarize(reps, B)
    fig3()
    print(f"[expS] ALL DONE in {time.time()-t0:.0f}s; solver {g.cmap_report()}",
          flush=True)


# ---------------------------------------------------------------- outputs
def fig3():
    sca = pd.read_csv(os.path.join(RES, "expS_scarcity_raw.csv"))
    sca = sca[~sca.method.str.startswith("_")]
    show = ["B0-heuristic", "B1-MLE", "B2-expert", "Proposed"]
    metrics_to_plot = [("acc", "Threat accuracy"), ("macroAUC", "Macro AUC"),
                       ("brier", "Brier score (lower=better)"),
                       ("ece", "ECE (lower=better)")]
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.6))
    for ax, (key, title) in zip(axes.ravel(), metrics_to_plot):
        for m in show:
            gdf = sca[sca.method == m].groupby("N")[key]
            mean = gdf.mean()
            sem = gdf.std() / np.sqrt(gdf.count())
            h = 2.093 * sem
            ax.plot(mean.index, mean.values, marker="o", ms=3.5, label=m)
            ax.fill_between(mean.index, mean - h, mean + h, alpha=0.18)
        ax.set_xscale("log")
        ax.set_xlabel("training scenarios $N$")
        ax.set_title(title, fontsize=10)
        ax.grid(alpha=0.3)
    axes.ravel()[0].legend(fontsize=8, loc="lower right")
    fig.suptitle("CPT grounding under data scarcity "
                 "(mean ±95% CI over 20 independent replications)", fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(RES, "exp1_scarcity.png"), dpi=300)
    plt.close(fig)


def summarize(reps, B):
    sca = pd.read_csv(os.path.join(RES, "expS_scarcity_raw.csv"))
    fus = pd.read_csv(os.path.join(RES, "expS_fusion_raw.csv"))
    abl = pd.read_csv(os.path.join(RES, "expS_ablation_raw.csv"))
    cs = pd.read_csv(os.path.join(RES, "expS_calibsize_raw.csv"))
    dg = pd.read_csv(os.path.join(RES, "expS_absweep_raw.csv"))
    lines = [f"# Exp-S v2 (S={reps}, B={B}, test n={TEST_N})", ""]
    tab_stats = []

    def pm(d, k):
        m, lo, hi = t_ci(d[k].values)
        return m, (hi - lo) / 2

    lines.append("## class distribution / majority baseline")
    mj = sca[sca.method == "Proposed"]["majority"].mean()
    lines.append(f"- majority-class baseline accuracy: {mj:.3f}")

    lines.append("\n## Table 3 (acc/ECE/AUC mean +- 95%CI half)")
    for N in TABLE3_NS:
        row = [f"N={N}"]
        for m in ["B0-heuristic", "B1-MLE", "B2-expert", "Proposed"]:
            d = sca[(sca.N == N) & (sca.method == m)]
            a, ah = pm(d, "acc"); e, eh = pm(d, "ece"); u, uh = pm(d, "macroAUC")
            row.append(f"{m}: {a:.3f}+-{ah:.3f}/{e:.3f}+-{eh:.3f}/{u:.3f}+-{uh:.3f}")
            tab_stats.append({"table": "T3", "cell": f"N{N}-{m}", "acc": a,
                              "acc_h": ah, "ece": e, "ece_h": eh,
                              "auc": u, "auc_h": uh,
                              "far": pm(d, "far@pd0.9")[0],
                              "cwece": pm(d, "cwece")[0]})
        lines.append("- " + " | ".join(row))
    lines.append(f"\n## ESS: fixed a priori at alpha0={sca['ess'].iloc[0]:g} "
                 "(no validation labels consumed by the main pipeline)")
    for ref in ("B1-MLE", "B2-expert"):
        lines.append(f"\n## Proposed vs {ref} paired deltas")
        for N in TABLE3_NS:
            dd = sca[(sca.N == N) & (sca.method == f"_delta_vs_{ref}")]
            rA = across_rep_tests(dd["dAUC"].values)
            rE = across_rep_tests(dd["dECE"].values)
            mcp = dd["mcnemar_p"].values
            lines.append(
                f"- N={N}: dAUC {rA['mean']:+.4f} [{rA['lo']:+.4f},{rA['hi']:+.4f}] "
                f"(t p={rA['t_p']:.1e}, W p={rA['wilcoxon_p']:.1e}); "
                f"dECE {rE['mean']:+.4f} [{rE['lo']:+.4f},{rE['hi']:+.4f}] "
                f"(t p={rE['t_p']:.1e}); McNemar sig {(mcp < 0.05).sum()}/{len(mcp)}")

    lines.append("\n## Table 4 (acc/F1/AUC/Brier/ECE/cwECE/FAR mean +- CIhalf)")
    for cfg in fus[~fus.config.str.startswith("_")].config.unique():
        d = fus[fus.config == cfg]
        vals = []
        cell = {"table": "T4", "cell": cfg}
        for k in ["acc", "macroF1", "macroAUC", "brier", "ece", "cwece",
                  "far@pd0.9"]:
            m, h = pm(d, k)
            vals.append(f"{k} {m:.3f}+-{h:.3f}")
            cell[k] = m; cell[k + "_h"] = h
        tab_stats.append(cell)
        lines.append(f"- {cfg}: " + ", ".join(vals))
    lines.append("\n## hard-condition subsets (all metrics, all replications)")
    for sub in ("silent-drone", "far-range", "far-silent"):
        for cfgname in ("abstract", "calibrated", "calib+reliab"):
            d = fus[fus.config == f"_subset_{sub}_{cfgname}"]
            if len(d) == 0:
                continue
            vals = []
            for k in ["acc", "macroAUC", "brier", "ece", "cwece"]:
                m, h = pm(d, k)
                vals.append(f"{k} {m:.3f}+-{h:.3f}")
            lines.append(f"- {sub:12s} {cfgname:12s} (n~{d['n_sub'].mean():.0f}): "
                         + ", ".join(vals))
    ps = fus[fus.config == "_paired_stats"]
    lines.append("\n## fusion paired stats (calibrated - abstract)")
    for key, label in [("dAUC", "delta macro-AUC"), ("dECE", "delta ECE"),
                       ("dECE_reliab", "delta ECE (reliab-calib)")]:
        rr = across_rep_tests(ps[key].values)
        lines.append(f"- {label}: {rr['mean']:+.4f} [{rr['lo']:+.4f},{rr['hi']:+.4f}] "
                     f"t p={rr['t_p']:.1e}, W p={rr['wilcoxon_p']:.1e}")
    for k in range(4):
        pk = ps[f"delong_p_c{k}"].values
        lines.append(f"- DeLong class {k}: median p={np.nanmedian(pk):.1e}, "
                     f"sig {(pk < 0.05).sum()}/{len(pk)}")
    mp = ps["mcnemar_p"].values
    lines.append(f"- McNemar: sig {(mp < 0.05).sum()}/{len(mp)}; "
                 f"boot dAUC CI excl 0 in "
                 f"{((ps['dAUC_lo'] > 0) | (ps['dAUC_hi'] < 0)).sum()}/{len(ps)}; "
                 f"dECE in {((ps['dECE_lo'] > 0) | (ps['dECE_hi'] < 0)).sum()}/{len(ps)}")

    lines.append("\n## ablation (N=80) vs full")
    full = abl[abl.variant == "Proposed (full)"].set_index("rep")
    for v in abl.variant.unique():
        d = abl[abl.variant == v]
        cell = {"table": "T5abl", "cell": v}
        for k in ["acc", "macroAUC", "ece"]:
            m, h = pm(d, k)
            cell[k] = m; cell[k + "_h"] = h
        tab_stats.append(cell)
        if v == "Proposed (full)":
            lines.append(f"- {v}: acc {cell['acc']:.3f}, AUC {cell['macroAUC']:.3f}, "
                         f"ECE {cell['ece']:.3f}")
            continue
        dv = d.set_index("rep")
        common = full.index.intersection(dv.index)
        rE = across_rep_tests((dv.loc[common, "ece"] - full.loc[common, "ece"]).values)
        rA = across_rep_tests((dv.loc[common, "macroAUC"] - full.loc[common, "macroAUC"]).values)
        lines.append(f"- {v}: dECE {rE['mean']:+.4f} [{rE['lo']:+.4f},{rE['hi']:+.4f}] "
                     f"(p={rE['t_p']:.1e}); dAUC {rA['mean']:+.4f} "
                     f"[{rA['lo']:+.4f},{rA['hi']:+.4f}] (p={rA['t_p']:.1e})")

    lines.append("\n## calibration-set size")
    for cn in CALIB_GRID:
        d = cs[cs.calib_n == cn]
        vals = []
        cell = {"table": "T5cs", "cell": str(cn)}
        for k in ["acc", "macroF1", "macroAUC", "brier", "ece"]:
            m, h = pm(d, k)
            vals.append(f"{k} {m:.3f}+-{h:.3f}")
            cell[k] = m; cell[k + "_h"] = h
        tab_stats.append(cell)
        lines.append(f"- calib_n={cn} ({d['source'].iloc[0]}): " + ", ".join(vals))

    lines.append("\n## abstract-diagonal sweep (calibrated-fusion reference above)")
    for dgv in DIAG_GRID:
        d = dg[dg.diag == dgv]
        u, uh = pm(d, "macroAUC"); e, eh = pm(d, "ece")
        lines.append(f"- diag={dgv:.2f}: AUC {u:.3f}+-{uh:.3f}, ECE {e:.3f}+-{eh:.3f}")

    pd.DataFrame(tab_stats).to_csv(os.path.join(RES, "expS_table_stats.csv"),
                                   index=False)
    txt = "\n".join(lines)
    with open(os.path.join(RES, "expS_summary.md"), "w", encoding="utf-8") as f:
        f.write(txt + "\n")
    print(txt.encode(sys.stdout.encoding or "utf-8", errors="replace")
             .decode(sys.stdout.encoding or "utf-8", errors="replace"))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=20)
    ap.add_argument("--boot", type=int, default=1000)
    ap.add_argument("--summary-only", action="store_true")
    a = ap.parse_args()
    if a.summary_only:
        summarize(a.reps, a.boot)
        fig3()
    else:
        run(a.reps, a.boot)
