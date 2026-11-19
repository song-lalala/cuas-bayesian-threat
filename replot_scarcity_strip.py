"""
One-row version of the scarcity figure. It draws the four metrics of
exp_significance.fig3 from the same file (results/expS_scarcity_raw.csv: same methods, same means, same 95% bands), with line styles
and markers that survive black-and-white printing. No experiment is run and no result file changes.

Writes results/exp1_scarcity_strip4.png; the manuscript uses a copy of it in paper/ieeeaccess.

Run:  python replot_scarcity_strip.py [--selftest]
"""
import os, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(HERE, "results", "expS_scarcity_raw.csv")
OUT = os.path.join(HERE, "results", "exp1_scarcity_strip4.png")
SHOW = [("B0-heuristic", "B0 heuristic", "#7f7f7f", ":", "s"), ("B1-MLE", "B1 MLE", "#e8820c", "--", "^"),
        ("B2-expert", "B2 expert", "#2ca02c", "-.", "D"), ("Proposed", "Proposed", "#6a3fbf", "-", "o")]
METRICS = [("acc", "Threat accuracy"), ("macroAUC", "Macro-AUC"), ("brier", "Brier score (lower is better)"), ("ece", "ECE (lower is better)")]
CI_T = 2.093   # the factor of exp_significance.fig3 (t, 19 df, 95%)


def series(sca, method, key):
    g = sca[sca.method == method].groupby("N")[key]
    mean, sem = g.mean(), g.std() / np.sqrt(g.count())
    return mean, CI_T * sem


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    sca = pd.read_csv(RAW)
    sca = sca[~sca.method.str.startswith("_")]
    assert sca.rep.nunique() == 20 and sorted(sca.N.unique()) == [10, 20, 50, 100, 200, 400]
    n = 0
    for m, *_ in SHOW:
        for key, _t in METRICS:
            mean, h = series(sca, m, key)
            assert len(mean) == 6 and not mean.isna().any() and not h.isna().any()
            n += 1
    print("series checked:", n)
    assert n == 16
    if "--selftest" in sys.argv:
        return
    plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "Times", "DejaVu Serif"], "mathtext.fontset": "stix",
                         "font.size": 9, "axes.titlesize": 9, "axes.labelsize": 9, "xtick.labelsize": 8, "ytick.labelsize": 8,
                         "legend.fontsize": 8.5, "lines.linewidth": 1.1, "lines.markersize": 3, "axes.linewidth": 0.6})
    fig, axes = plt.subplots(1, 4, figsize=(7.16, 1.85))
    hs = None
    for ax, (key, title) in zip(axes, METRICS):
        for m, lab, col, ls, mk in SHOW:
            mean, h = series(sca, m, key)
            ax.plot(mean.index, mean.values, color=col, linestyle=ls, marker=mk, markerfacecolor="white", markeredgewidth=0.7, label=lab)
            ax.fill_between(mean.index, mean - h, mean + h, color=col, alpha=0.16, linewidth=0)
        ax.set_xscale("log")
        ax.set_xticks([10, 100, 400])
        ax.set_xticklabels(["10", "100", "400"])
        ax.minorticks_off()
        ax.set_xlabel("training scenarios $N$", labelpad=1)
        ax.set_title(title, pad=2)
        ax.grid(alpha=0.3, linewidth=0.4)
        ax.tick_params(length=2, pad=1.5)
        hs = ax.get_legend_handles_labels()
    fig.legend(*hs, loc="lower center", ncol=4, frameon=False, handlelength=2.6, columnspacing=1.2, borderaxespad=0.0)
    fig.tight_layout(rect=[0, 0.08, 1, 1], pad=0.3, w_pad=0.6)
    fig.savefig(OUT, dpi=300)
    plt.close(fig)
    print("written", OUT)


if __name__ == "__main__":
    main()
