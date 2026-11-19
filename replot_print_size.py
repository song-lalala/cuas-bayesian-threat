"""
Print-size versions of two manuscript figures: the original files were drawn at about twice the printed width,
so their labels came out near 6 pt and 3.5 pt on the page. These are drawn at the printed width with labels of 7 to 8 pt.
No experiment is run and no result file changes; the old figure files stay.

- Correlated failures: results/expF_corr_raw.csv, same computation as replot_figs.fig_corr (mean and 2.093 SEM per severity,
  mean rho on the top axis). Writes results/fig_expF_corr_print.png.
- Tornado: the fifteen rows and the baseline of results/expT_summary.md, written by exp_sensitivity.py in the same run as
  fig_expT_tornado.png, in the same order. That file keeps three decimals, which is finer than a printed bar can show.
  Writes results/fig_expT_tornado_print.png.

Run:  python replot_print_size.py [--selftest]
"""
import os, re, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
FUSIONS = ["naive-CI", "W-observed", "W-latent", "W-noisy"]
STYLES = {"naive-CI": ("--o", "C0"), "W-observed": ("-s", "C1"), "W-latent": ("-^", "C2"), "W-noisy": ("-v", "C3")}
CI_T = 2.093   # the factor of replot_figs.fig_corr (t, 19 df, 95%)
ROW = re.compile(r"^- ([A-Z]\([0-9, ]+\)k[0-9]): theta=[0-9.]+, P in \[([0-9.]+), ([0-9.]+)\]")
BASE = re.compile(r"^- baseline P\(T=high\|e\*\) = ([0-9.]+)")


def tornado_rows():
    rows, base = [], None
    with open(os.path.join(RES, "expT_summary.md"), encoding="utf-8") as f:
        for line in f:
            m = BASE.match(line)
            if m:
                base = float(m.group(1))
            m = ROW.match(line)
            if m:
                rows.append((m.group(1), float(m.group(2)), float(m.group(3))))
    print("tornado rows read:", len(rows))
    assert len(rows) == 15 and base is not None
    return rows, base


def corr_series():
    df = pd.read_csv(os.path.join(RES, "expF_corr_raw.csv"))
    out = {}
    for met in ("ece_adv", "auc_adv"):
        for fus in FUSIONS:
            g = df[df.fusion == fus].groupby("severity")[met]
            out[(met, fus)] = (g.mean(), CI_T * g.std() / np.sqrt(g.count()))
    print("correlated-failure series read:", len(out))
    assert len(out) == 8 and all(len(m) == 5 and not m.isna().any() and not h.isna().any() for m, h in out.values())
    return out, df.groupby("severity")["rho"].mean()


def fig_corr(series, rho):
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.35))
    for ax, met in [(axes[0], "ece_adv"), (axes[1], "auc_adv")]:
        for fus in FUSIONS:
            mean, h = series[(met, fus)]
            ls, col = STYLES[fus]
            ax.errorbar(mean.index, mean.values, yerr=h.values, fmt=ls, color=col, ms=3, capsize=1.8, elinewidth=0.7, label=fus)
        ax.set_xlabel("correlated-degradation severity $s$", labelpad=1)
        ax.grid(alpha=0.3, linewidth=0.4)
        ax.tick_params(length=2, pad=1.5)
    axes[1].set_title("adverse-subset macro-AUC", pad=3)
    top = axes[0].twiny()
    top.set_xlim(axes[0].get_xlim())
    top.set_xticks(rho.index)
    top.set_xticklabels([f"{v:.2f}" for v in rho.values])
    top.tick_params(length=2, pad=1)
    top.set_xlabel("adverse-subset ECE (lower is better)\n" + r"top axis: induced error correlation $\rho$ (cls$=$drone)", labelpad=3, linespacing=1.05)
    axes[0].legend(frameon=True, framealpha=0.9, borderpad=0.3, labelspacing=0.25, handlelength=2.2)
    fig.tight_layout(pad=0.3, w_pad=1.0)
    fig.savefig(os.path.join(RES, "fig_expF_corr_print.png"), dpi=300)
    plt.close(fig)


def fig_tornado(rows, base):
    fig, ax = plt.subplots(figsize=(3.1, 2.3))
    ypos = np.arange(len(rows))[::-1]
    for yy, (_, lo, hi) in zip(ypos, rows):
        ax.barh(yy, hi - lo, left=lo, height=0.62, color="#4878CF", alpha=0.85)
    ax.axvline(base, color="k", lw=0.9, ls="--", label=f"baseline {base:.2f}")
    ax.set_yticks(ypos)
    ax.set_yticklabels([r[0] for r in rows], fontsize=7)
    ax.set_xlabel(r"$P(T=\mathrm{high}\mid e^{*})$ over $\theta \pm 0.1$", labelpad=1)
    ax.legend(loc="lower right", borderpad=0.3, handlelength=2.0)
    ax.grid(alpha=0.3, axis="x", linewidth=0.4)
    ax.tick_params(length=2, pad=1.5)
    fig.tight_layout(pad=0.3)
    fig.savefig(os.path.join(RES, "fig_expT_tornado_print.png"), dpi=300)
    plt.close(fig)


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    rows, base = tornado_rows()
    series, rho = corr_series()
    if "--selftest" in sys.argv:
        return
    plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "Times", "DejaVu Serif"], "mathtext.fontset": "stix",
                         "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
                         "legend.fontsize": 7.5, "lines.linewidth": 1.0, "axes.linewidth": 0.6})
    fig_corr(series, rho)
    fig_tornado(rows, base)
    print("written fig_expF_corr_print.png, fig_expT_tornado_print.png")


if __name__ == "__main__":
    main()
