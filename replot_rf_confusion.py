"""
RF-node confusion matrix of the paper's Fig. 4: the held-out test matrix of the segment-level split, row-normalized to P(report | true). It reads the cached forward
passes of exp_rf_e2e.py (results/expG_cnn.npz), so it needs neither the DroneRF data nor a GPU. No experiment is run. rf_confusion.py trains its own CNN on its own split
and does not give this matrix.

Writes results/rf_confusion.png, or the file named by --out.

Run:  python replot_rf_confusion.py [--selftest] [--out PATH]
"""
import argparse, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
DISPLAY = ["background", "Bebop", "AR", "Phantom"]


def matrix():
    z = np.load(os.path.join(RES, "expG_cnn.npz"), allow_pickle=True)
    y, p = z["test_y"], z["test_softmax"].argmax(1)
    cm = np.zeros((4, 4), int)
    for a, b in zip(y, p):
        cm[a, b] += 1
    print("test segments counted:", int(cm.sum()), "| accuracy:", round(float((y == p).mean()), 4))
    assert cm.sum() == len(y) and (cm.sum(1) > 0).all()
    return cm / cm.sum(1, keepdims=True)


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--out", default=os.path.join(RES, "rf_confusion.png"))
    a = ap.parse_args()
    cmn = matrix()
    if a.selftest:
        return
    plt.rcParams.update({"font.family": "serif", "mathtext.fontset": "stix", "font.size": 12})
    fig, ax = plt.subplots(figsize=(4.0, 3.4))
    im = ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(4)); ax.set_yticks(range(4))
    ax.set_xticklabels(DISPLAY, rotation=30, ha="right"); ax.set_yticklabels(DISPLAY)
    ax.set_xlabel("RF report (predicted class)"); ax.set_ylabel("true class")
    for i in range(4):
        for j in range(4):
            ax.text(j, i, f"{cmn[i, j]:.2f}", ha="center", va="center", color="white" if cmn[i, j] > 0.5 else "black", fontsize=11)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label(r"$P(\mathrm{report}\mid\mathrm{true})$")
    fig.tight_layout()
    fig.savefig(a.out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("written", a.out)


if __name__ == "__main__":
    main()
