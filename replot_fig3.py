"""
Regenerate results/exp1_scarcity.png (CPT grounding under scarcity) from the
saved exp1 summary, with publication styling: wider aspect, larger fonts,
distinct line styles + markers per method (black-and-white distinguishable),
and a single shared figure legend. Fast (no experiment re-run).
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(__file__)
RES = os.path.join(HERE, "results")
summ = pd.read_csv(os.path.join(RES, "exp1_summary.csv"))

# oracle values (true CPTs) — recompute for accuracy
import model_spec as ms, dataio, metrics as mt
gt = ms.ground_truth_bn()
test = dataio.generate_dataset(gt, 2500, seed=999)
est = dataio.estimate_confusion(dataio.generate_dataset(gt, 300, seed=777))
orc, _ = mt.evaluate({k: gt.cpts[k] for k in ms.INTERNAL}, test,
                     sensor_cpts=est)

# per-method style: (label, color, linestyle, marker)
STYLE = [
    ("B0-heuristic",  "B0 heuristic",  "#7f7f7f", ":",             "s"),
    ("B1-MLE",        "B1 MLE",        "#e8820c", "--",            "^"),
    ("B2-expert",     "B2 expert",     "#2ca02c", "-.",            "D"),
    ("MAP",           "MAP",           "#d62728", (0, (4, 1.5)),   "o"),
    ("Proposed-cMAP", "Proposed-cMAP", "#6a3fbf", "-",             "X"),
]
PANELS = [("acc", "Threat accuracy"), ("macroAUC", "Macro AUC"),
          ("brier", "Brier score (lower = better)"),
          ("ece", "ECE (lower = better)")]

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "mathtext.fontset": "stix",          # Times-like math
    "font.size": 15, "axes.titlesize": 17, "axes.labelsize": 15,
    "xtick.labelsize": 13, "ytick.labelsize": 13, "legend.fontsize": 14,
    "lines.linewidth": 2.4, "lines.markersize": 9, "axes.linewidth": 1.0,
})

fig, axes = plt.subplots(2, 2, figsize=(15.5, 7.0))
handles = None
for ax, (key, title) in zip(axes.ravel(), PANELS):
    for mkey, label, color, ls, mk in STYLE:
        d = summ[summ.method == mkey].sort_values("N")
        ax.plot(d["N"], d[key], color=color, linestyle=ls, marker=mk,
                markerfacecolor="white", markeredgecolor=color,
                markeredgewidth=1.6, label=label)
    ax.axhline(orc[key], ls=(0, (1, 1)), color="black", lw=1.6, label="oracle")
    ax.set_xscale("log")
    ax.set_xlabel("training scenarios $N$")
    ax.set_title(title)
    ax.grid(alpha=0.3)
    if handles is None:
        handles, labels = ax.get_legend_handles_labels()

# single shared legend below all panels
fig.legend(handles, labels, loc="lower center", ncol=6,
           frameon=True, bbox_to_anchor=(0.5, -0.02))
fig.tight_layout(rect=[0, 0.06, 1, 1.0])

out = os.path.join(RES, "exp1_scarcity.png")
fig.savefig(out, dpi=150, bbox_inches="tight")
plt.close(fig)
print("regenerated exp1_scarcity.png with distinct line styles/markers + shared legend")
print("oracle:", {k: round(v, 3) for k, v in orc.items()})
