"""
Exp-E v2: sensitivity to prior quality and ESS; when constraints help.

(R1-5, R1-6, R1-11; revised for this resubmission: 20 independent
seeds throughout, regime maps for BOTH alpha0=8 and alpha0=50, text and
figures generated from the same files.)

Part (i)  prior quality: adversarially biased priors via ordinal-effect
          reversal (lambda: 0 elicited, 0.5 no trend, 1 fully reversed),
          MAP vs constrained (cMAP-full) at ESS in {8, 50}, N in
          {10,50,200,400}; prior-free MLE reference; plus the proposed
          ordinal-ICI threat node (monotone by construction) at a reduced
          grid to show its behaviour under a reversed prior.
Part (ii) ESS range: alpha0 in {1,2,5,10,20,50,100} x N for the Proposed
          method, and the held-out log-loss selection tracked against the
          per-N ECE optimum.
Part (iii) regime maps: ECE(MAP)-ECE(cMAP) on the lambda x N grid at both
          ESS values.

Outputs (results/): expE_prior_raw.csv, expE_prop_raw.csv, expE_ess_raw.csv,
  expE_summary.md, fig_expE_ess.png, fig_expE_regime.png
Run:  python exp_prior_ess.py [--smoke]
"""
from __future__ import annotations
import os, sys, time, argparse
import numpy as np
import pandas as pd
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
from exp_constraints import mono_violations

RES = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(RES, exist_ok=True)
TEST_N = 2500

BASE_W = {
    "H": ({"v": 0.9, "rdot": 1.0, "d": 1.1}, 0.05, ms.TAU_HC),
    "N": ({"cls": 1.4, "emit": 0.9}, -0.10, ms.TAU_HC),
    "C": ({"d": 1.1, "z": 1.0}, 0.05, ms.TAU_HC),
    "T": ({"H": 1.1, "N": 1.0, "C": 1.0}, 0.0, ms.TAU_T),
}


def biased_prior_cpts(rng, lam, noise=0.18):
    """Adversarially biased prior means: each parent's believed ordinal effect
    is interpolated toward its reverse, x' = (1-lam)x + lam(1-x)."""
    from itertools import product as ip
    out = {}
    for node, (weights, bias, tau) in BASE_W.items():
        pa = ms.PARENTS[node]
        K = ms.CARD[node]
        w = {p: weights[p] * (1 + noise * rng.standard_normal())
             for p in weights}
        b = bias + noise * 0.5 * rng.standard_normal()
        wsum = sum(abs(x) for x in w.values()) or 1.0
        shape = tuple(ms.CARD[p] for p in pa) + (K,)
        table = np.zeros(shape)
        for cfg in ip(*[range(ms.CARD[p]) for p in pa]) if pa else [()]:
            acc = 0.0
            for p, val in zip(pa, cfg):
                x = val / max(ms.CARD[p] - 1, 1)
                acc += w.get(p, 0.0) * ((1 - lam) * x + lam * (1 - x))
            mu = b + acc / wsum
            center = np.clip(mu, 0, 1) * (K - 1)
            ks = np.arange(K)
            logits = -((ks - center) ** 2) / (2 * tau ** 2)
            row = np.exp(logits - logits.max())
            table[cfg] = row / row.sum()
        out[node] = ms.Factor(tuple(pa) + (node,), table)
    return out


def ground(method, data, prior, ess, seed=0):
    out = {}
    for node in ms.INTERNAL:
        if method == "MAP":
            out[node] = g.map_cpt(node, data, prior[node], ess)
        elif method == "cMAP":
            out[node] = g.cmap_cpt(node, data, prior[node], ess)
        elif method == "MLE":
            out[node] = g.mle_cpt(node, data)
        elif method == "Proposed":
            out[node] = (g.ordinal_ici_cpt(node, data, prior[node], ess,
                                           seed=seed)
                         if node == "T"
                         else g.cmap_cpt(node, data, prior[node], ess))
        else:
            raise ValueError(method)
    return out


def part_prior(lams, Ns, esses, seeds, test, est, alpha):
    gt = ms.ground_truth_bn()
    rows, prop_rows = [], []
    t0 = time.time()
    for seed in range(seeds):
        for N in Ns:
            data = dataio.generate_dataset(gt, N, seed=seed)
            cp = ground("MLE", data, None, 0)
            met, _ = mt.evaluate(cp, test, sensor_cpts=est, alpha=alpha)
            rows.append({"seed": seed, "N": N, "lam": np.nan, "ess": np.nan,
                         "method": "MLE",
                         "T_viol": mono_violations(cp["T"]), **met})
            for lam in lams:
                rng = np.random.default_rng(1000 + seed)
                prior = biased_prior_cpts(rng, lam)
                for ess in esses:
                    for method in ["MAP", "cMAP"]:
                        cp = ground(method, data, prior, ess)
                        met, _ = mt.evaluate(cp, test, sensor_cpts=est,
                                             alpha=alpha)
                        rows.append({"seed": seed, "N": N, "lam": lam,
                                     "ess": ess, "method": method,
                                     "T_viol": mono_violations(cp["T"]),
                                     **met})
                    # Proposed (structurally monotone T) at the reduced grid
                    if lam in (0.0, 1.0) and ess == max(esses) \
                            and N in (50, 200):
                        cp = ground("Proposed", data, prior, ess, seed=seed)
                        met, _ = mt.evaluate(cp, test, sensor_cpts=est,
                                             alpha=alpha)
                        prop_rows.append({"seed": seed, "N": N, "lam": lam,
                                          "ess": ess,
                                          "T_viol": mono_violations(cp["T"]),
                                          **met})
        print(f"[expE-i] seed {seed+1}/{seeds} ({time.time()-t0:.0f}s)",
              flush=True)
        pd.DataFrame(rows).to_csv(os.path.join(RES, "expE_prior_raw.csv"),
                                  index=False)
        pd.DataFrame(prop_rows).to_csv(os.path.join(RES, "expE_prop_raw.csv"),
                                       index=False)
    return pd.DataFrame(rows), pd.DataFrame(prop_rows)


def part_ess(alphas, Ns, seeds, test, est, alpha):
    gt = ms.ground_truth_bn()
    rows = []
    t0 = time.time()
    for seed in range(seeds):
        val = dataio.generate_dataset(gt, pl.VAL_N, seed=7000 + seed)
        for N in Ns:
            data = dataio.generate_dataset(gt, N, seed=seed)
            rng = np.random.default_rng(1000 + seed)
            prior = ms.expert_prior_cpts(rng)
            sel = pl.select_ess(data, prior, val, est, alpha,
                                grid=tuple(alphas))
            for a0 in alphas:
                cp = ground("Proposed", data, prior, a0, seed=seed)
                met, _ = mt.evaluate(cp, test, sensor_cpts=est, alpha=alpha)
                rows.append({"seed": seed, "N": N, "alpha0": a0,
                             "selected": sel, **met})
        print(f"[expE-ii] seed {seed+1}/{seeds} ({time.time()-t0:.0f}s)",
              flush=True)
        pd.DataFrame(rows).to_csv(os.path.join(RES, "expE_ess_raw.csv"),
                                  index=False)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- figures
def fig_ess(df, out):
    Ns = sorted(df.N.unique())
    alphas = sorted(df.alpha0.unique())
    M = np.zeros((len(alphas), len(Ns)))
    for i, a in enumerate(alphas):
        for j, N in enumerate(Ns):
            M[i, j] = df[(df.alpha0 == a) & (df.N == N)]["ece"].mean()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.4, 3.9))
    im = ax1.imshow(M, aspect="auto", cmap="viridis_r", origin="lower")
    ax1.set_xticks(range(len(Ns))); ax1.set_xticklabels(Ns)
    ax1.set_yticks(range(len(alphas))); ax1.set_yticklabels(alphas)
    ax1.set_xlabel("training scenarios $N$"); ax1.set_ylabel(r"ESS $\alpha_0$")
    ax1.set_title("held-out ECE (lower=better)", fontsize=10)
    fig.colorbar(im, ax=ax1, fraction=0.046)
    for j, N in enumerate(Ns):
        i_opt = int(np.argmin(M[:, j]))
        ax1.plot(j, i_opt, "w*", ms=13, mec="k")
        sel_mode = df[df.N == N]["selected"].mode().iloc[0]
        ax1.plot(j, alphas.index(sel_mode), "ro", ms=7, mfc="none", mew=2)
    ax1.plot([], [], "w*", ms=10, mec="k", label="per-N ECE optimum")
    ax1.plot([], [], "ro", ms=6, mfc="none", mew=2,
             label="held-out log-loss pick (mode)")
    ax1.legend(fontsize=7, loc="upper right")
    for N in Ns:
        d = df[df.N == N].groupby("alpha0")["ece"]
        mean = d.mean(); sem = d.std() / np.sqrt(d.count())
        ax2.errorbar(mean.index, mean.values, yerr=2.093 * sem, marker="o",
                     ms=3, capsize=2, label=f"N={N}")
    ax2.set_xscale("log"); ax2.set_xlabel(r"ESS $\alpha_0$")
    ax2.set_ylabel("ECE")
    ax2.grid(alpha=0.3); ax2.legend(fontsize=8)
    ax2.set_title("knowledge-vs-data trade-off (mean ±95% CI)", fontsize=10)
    fig.tight_layout(); fig.savefig(out, dpi=300); plt.close(fig)


def fig_regime(df, esses, out):
    lams = sorted(df[~df.lam.isna()].lam.unique())
    Ns = sorted(df.N.unique())
    fig, axes = plt.subplots(1, len(esses), figsize=(5.6 * len(esses), 4.0))
    if len(esses) == 1:
        axes = [axes]
    for ax, ess in zip(axes, esses):
        M = np.zeros((len(lams), len(Ns)))
        for i, lam in enumerate(lams):
            for j, N in enumerate(Ns):
                dm = df[(df.method == "MAP") & (df.N == N) & (df.lam == lam) &
                        (df.ess == ess)]["ece"].mean()
                dc = df[(df.method == "cMAP") & (df.N == N) & (df.lam == lam) &
                        (df.ess == ess)]["ece"].mean()
                M[i, j] = dm - dc
        v = np.nanmax(np.abs(M))
        im = ax.imshow(M, aspect="auto", cmap="RdBu_r", origin="lower",
                       vmin=-v, vmax=v)
        ax.set_xticks(range(len(Ns))); ax.set_xticklabels(Ns)
        ax.set_yticks(range(len(lams))); ax.set_yticklabels(lams)
        ax.set_xlabel("training scenarios $N$")
        ax.set_ylabel(r"prior bias $\lambda$")
        ax.set_title(f"ECE(MAP) $-$ ECE(cMAP), $\\alpha_0={ess:g}$\n"
                     "(red $=$ constraints help)", fontsize=10)
        for i in range(len(lams)):
            for j in range(len(Ns)):
                ax.text(j, i, f"{M[i,j]:+.3f}", ha="center", va="center",
                        fontsize=7.5)
        fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout(); fig.savefig(out, dpi=300); plt.close(fig)


def summarize(dfp, dfprop, dfe, esses):
    lines = ["# Exp-E v2: prior quality, ESS, when constraints help", ""]
    Ns = sorted(dfp.N.unique())
    lines.append("## (i) adversarial prior lam=1 (mean over seeds)")
    for ess in esses:
        for N in Ns:
            d1 = dfp[(dfp.lam == 1.0) & (dfp.N == N) & (dfp.ess == ess)]
            d0 = dfp[(dfp.lam == 0.0) & (dfp.N == N) & (dfp.ess == ess)]
            mle = dfp[(dfp.method == "MLE") & (dfp.N == N)]
            for meth in ["MAP", "cMAP"]:
                lines.append(
                    f"- ESS={ess:g} N={N} {meth}: "
                    f"acc {d0[d0.method==meth]['acc'].mean():.3f}->"
                    f"{d1[d1.method==meth]['acc'].mean():.3f}, "
                    f"ECE {d0[d0.method==meth]['ece'].mean():.3f}->"
                    f"{d1[d1.method==meth]['ece'].mean():.3f}, "
                    f"AUC->{d1[d1.method==meth]['macroAUC'].mean():.3f} "
                    f"(MLE ref acc {mle['acc'].mean():.3f} ECE "
                    f"{mle['ece'].mean():.3f})")
    lines.append("\n## (i) T-CPT violation rate under lam=1")
    for ess in esses:
        for N in Ns:
            d1 = dfp[(dfp.lam == 1.0) & (dfp.N == N) & (dfp.ess == ess)]
            lines.append(f"- ESS={ess:g} N={N}: MAP "
                         f"{d1[d1.method=='MAP']['T_viol'].mean():.3f} vs "
                         f"cMAP {d1[d1.method=='cMAP']['T_viol'].mean():.3f}")
    if len(dfprop):
        lines.append("\n## (i) Proposed (ordinal-ICI T) under reversed prior")
        for _, r in dfprop.groupby(["lam", "N"]).mean(numeric_only=True) \
                          .reset_index().iterrows():
            lines.append(f"- lam={r['lam']:g} N={int(r['N'])}: "
                         f"acc {r['acc']:.3f}, ECE {r['ece']:.3f}, "
                         f"AUC {r['macroAUC']:.3f}, viol {r['T_viol']:.3f}")
    lines.append("\n## (ii) ESS selection quality")
    alphas = sorted(dfe.alpha0.unique())
    for N in sorted(dfe.N.unique()):
        d = dfe[dfe.N == N]
        eces = d.groupby("alpha0")["ece"].mean()
        opt_a = eces.idxmin()
        gaps = []
        for seed in d.seed.unique():
            sel = d[d.seed == seed]["selected"].iloc[0]
            gaps.append(eces[sel] - eces[opt_a])
        lines.append(f"- N={N}: ECE-opt a0={opt_a:g} ({eces[opt_a]:.4f}); "
                     f"mean selection gap {np.mean(gaps):+.4f} "
                     f"(max {np.max(gaps):+.4f})")
    lines.append("\n## (iii) regime ECE(MAP)-ECE(cMAP) by lam x N")
    for ess in esses:
        lines.append(f"### ESS={ess:g}")
        for lam in sorted(dfp[~dfp.lam.isna()].lam.unique()):
            vals = []
            for N in Ns:
                dm = dfp[(dfp.method == "MAP") & (dfp.N == N) &
                         (dfp.lam == lam) & (dfp.ess == ess)]["ece"].mean()
                dc = dfp[(dfp.method == "cMAP") & (dfp.N == N) &
                         (dfp.lam == lam) & (dfp.ess == ess)]["ece"].mean()
                vals.append(f"N{N}:{dm-dc:+.4f}")
            lines.append(f"- lam={lam:g}: " + "  ".join(vals))
    txt = "\n".join(lines)
    with open(os.path.join(RES, "expE_summary.md"), "w", encoding="utf-8") as f:
        f.write(txt + "\n")
    print(txt.encode(sys.stdout.encoding or "utf-8", errors="replace")
             .decode(sys.stdout.encoding or "utf-8", errors="replace"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--summary-only", action="store_true")
    a = ap.parse_args()
    lams = (0.0, 0.25, 0.5, 0.75, 1.0)
    Ns = (10, 50, 200, 400)
    esses = (8.0, 50.0)
    alphas = (1, 2, 5, 10, 20, 50, 100)
    seeds = 20
    if a.smoke:
        lams = (0.0, 1.0); Ns = (50, 200); alphas = (1, 10, 100); seeds = 2
    if a.summary_only:
        dfp = pd.read_csv(os.path.join(RES, "expE_prior_raw.csv"))
        dfprop = pd.read_csv(os.path.join(RES, "expE_prop_raw.csv"))
        dfe = pd.read_csv(os.path.join(RES, "expE_ess_raw.csv"))
        summarize(dfp, dfprop, dfe, sorted(dfp[~dfp.ess.isna()].ess.unique()))
        fig_ess(dfe, os.path.join(RES, "fig_expE_ess.png"))
        fig_regime(dfp, sorted(dfp[~dfp.ess.isna()].ess.unique()),
                   os.path.join(RES, "fig_expE_regime.png"))
        return
    gt = ms.ground_truth_bn()
    test = dataio.generate_dataset(gt, TEST_N, seed=999)
    calib = dataio.generate_dataset(gt, 300, seed=777)
    est = dataio.estimate_confusion(calib)
    alpha, _ = dataio.fit_reliability(calib)
    t0 = time.time()
    dfp, dfprop = part_prior(lams, Ns, esses, seeds, test, est, alpha)
    dfe = part_ess(alphas, Ns, seeds, test, est, alpha)
    fig_ess(dfe, os.path.join(RES, "fig_expE_ess.png"))
    fig_regime(dfp, esses, os.path.join(RES, "fig_expE_regime.png"))
    summarize(dfp, dfprop, dfe, esses)
    print(f"[expE] ALL DONE in {time.time()-t0:.0f}s; solver {g.cmap_report()}",
          flush=True)


if __name__ == "__main__":
    main()
