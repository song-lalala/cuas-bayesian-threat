"""
Regenerate the manuscript's multi-panel figures at double-column width with
legible fonts (>= 8 pt at print size), from the existing results/ CSVs.
Kept separate from the experiment scripts so a pure re-plot does not trip the
freshness gate.
"""
from __future__ import annotations
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({"font.family": "serif", "mathtext.fontset": "stix",
                     "font.size": 12})

RES = os.path.join(os.path.dirname(__file__), "results")


def fig_ess():
    df = pd.read_csv(os.path.join(RES, "expE_ess_raw.csv"))
    Ns = sorted(df.N.unique())
    alphas = sorted(df.alpha0.unique())
    M = np.zeros((len(alphas), len(Ns)))
    for i, a in enumerate(alphas):
        for j, N in enumerate(Ns):
            M[i, j] = df[(df.alpha0 == a) & (df.N == N)]["ece"].mean()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.2, 4.6))
    im = ax1.imshow(M, aspect="auto", cmap="viridis_r", origin="lower")
    ax1.set_xticks(range(len(Ns))); ax1.set_xticklabels(Ns)
    ax1.set_yticks(range(len(alphas))); ax1.set_yticklabels(alphas)
    ax1.set_xlabel("training scenarios $N$")
    ax1.set_ylabel(r"ESS $\alpha_0$")
    ax1.set_title("held-out ECE (lower$=$better)", fontsize=12)
    fig.colorbar(im, ax=ax1, fraction=0.046)
    for j, N in enumerate(Ns):
        i_opt = int(np.argmin(M[:, j]))
        ax1.plot(j, i_opt, "w*", ms=15, mec="k")
        sel_mode = df[df.N == N]["selected"].mode().iloc[0]
        ax1.plot(j, alphas.index(sel_mode), "ro", ms=9, mfc="none", mew=2)
    ax1.plot([], [], "w*", ms=11, mec="k", label="per-$N$ ECE optimum")
    ax1.plot([], [], "ro", ms=8, mfc="none", mew=2,
             label="held-out log-loss pick (mode)")
    ax1.legend(fontsize=9, loc="upper right")
    for N in Ns:
        d = df[df.N == N].groupby("alpha0")["ece"]
        mean = d.mean(); sem = d.std() / np.sqrt(d.count())
        ax2.errorbar(mean.index, mean.values, yerr=2.093 * sem, marker="o",
                     ms=4, capsize=3, label=f"$N$={N}")
    ax2.set_xscale("log"); ax2.set_xlabel(r"ESS $\alpha_0$")
    ax2.set_ylabel("ECE")
    ax2.grid(alpha=0.3); ax2.legend(fontsize=10)
    ax2.set_title("knowledge-vs-data trade-off (mean $\\pm$95% CI)",
                  fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(RES, "fig_expE_ess.png"), dpi=300)
    plt.close(fig)


def fig_regime():
    """2x2 regime maps: rows = metric (top-label ECE, Brier), cols = ESS.
    Positive (red) = the constrained estimator helps on that metric."""
    from scipy import stats as st
    df = pd.read_csv(os.path.join(RES, "expE_prior_raw.csv"))
    esses = sorted(df[~df.ess.isna()].ess.unique())
    lams = sorted(df[~df.lam.isna()].lam.unique())
    Ns = sorted(df.N.unique())
    metrics = [("ece", "top-label ECE"), ("brier", "Brier")]
    # one shared colour bar per metric row: frees panel width, so the in-cell
    # numbers can be set large enough to read at print size.
    fig, axes = plt.subplots(len(metrics), len(esses), figsize=(13.2, 8.2),
                             layout="constrained")
    for r, (met, mlab) in enumerate(metrics):
        mats, sigs = [], []
        for ess in esses:
            M = np.zeros((len(lams), len(Ns)))
            S = np.zeros((len(lams), len(Ns)), dtype=bool)
            for i, lam in enumerate(lams):
                for j, N in enumerate(Ns):
                    a = df[(df.method == "MAP") & (df.N == N) & (df.lam == lam)
                           & (df.ess == ess)].set_index("seed")[met]
                    b = df[(df.method == "cMAP") & (df.N == N) & (df.lam == lam)
                           & (df.ess == ess)].set_index("seed")[met]
                    ix = a.index.intersection(b.index)
                    d = (a.loc[ix] - b.loc[ix]).values
                    M[i, j] = d.mean()
                    S[i, j] = st.ttest_1samp(d, 0).pvalue < 0.05
            mats.append(M); sigs.append(S)
        v = max(np.nanmax(np.abs(M)) for M in mats)
        for c, ess in enumerate(esses):
            ax, M, S = axes[r][c], mats[c], sigs[c]
            im = ax.imshow(M, aspect="auto", cmap="RdBu_r", origin="lower",
                           vmin=-v, vmax=v)
            ax.set_xticks(range(len(Ns)))
            ax.set_xticklabels(Ns, fontsize=12)
            ax.set_yticks(range(len(lams)))
            ax.set_yticklabels(lams, fontsize=12)
            ax.set_xlabel("training scenarios $N$", fontsize=12)
            if c == 0:
                ax.set_ylabel(r"prior bias $\lambda$", fontsize=12)
            ax.set_title(r"$\Delta$ " + mlab + r",  $\alpha_0 = $" +
                         f"{ess:g}", fontsize=13)
            for i in range(len(lams)):
                for j in range(len(Ns)):
                    ax.text(j, i, f"{M[i,j]:+.3f}" + ("*" if S[i, j] else ""),
                            ha="center", va="center", fontsize=13)
        fig.colorbar(im, ax=list(axes[r]), fraction=0.03, pad=0.02)
    fig.suptitle("MAP $-$ cMAP by metric: red $=$ constraints help "
                 "($*$: paired $t$-test $p<0.05$)", fontsize=13)
    fig.savefig(os.path.join(RES, "fig_expE_regime.png"), dpi=300)
    plt.close(fig)


def fig_corr():
    df = pd.read_csv(os.path.join(RES, "expF_corr_raw.csv"))
    FUSIONS = ["naive-CI", "W-observed", "W-latent", "W-noisy"]
    styles = {"naive-CI": ("--o", "C0"), "W-observed": ("-s", "C1"),
              "W-latent": ("-^", "C2"), "W-noisy": ("-v", "C3")}
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 4.8))
    for ax, met, ttl in [(axes[0], "ece_adv",
                          "adverse-subset ECE (lower$=$better)"),
                         (axes[1], "auc_adv", "adverse-subset macro-AUC")]:
        for fus in FUSIONS:
            gdf = df[df.fusion == fus].groupby("severity")[met]
            mean = gdf.mean()
            sem = gdf.std() / np.sqrt(gdf.count())
            ls, col = styles[fus]
            ax.errorbar(mean.index, mean.values, yerr=2.093 * sem, fmt=ls,
                        color=col, ms=5, capsize=3, label=fus)
        ax.set_xlabel("correlated-degradation severity $s$")
        ax.set_title(ttl, fontsize=12)
        ax.grid(alpha=0.3)
    rho = df.groupby("severity")["rho"].mean()
    ax2 = axes[0].twiny()
    ax2.set_xlim(axes[0].get_xlim())
    ax2.set_xticks(rho.index)
    ax2.set_xticklabels([f"{v:.2f}" for v in rho.values], fontsize=9)
    ax2.set_xlabel(r"induced error correlation $\rho$ (cls$=$drone)",
                   fontsize=10)
    axes[0].legend(fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(RES, "fig_expF_corr.png"), dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    fig_ess()
    fig_regime()
    fig_corr()
    print("replotted fig_expE_ess / fig_expE_regime / fig_expF_corr "
          "at double-column size")
