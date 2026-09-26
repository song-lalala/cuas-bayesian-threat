"""
Exp-T: one-way sensitivity (tornado) and credal intervals on the
linear-fractional posterior (C4 executed).

Any BN posterior is a linear-fractional function of a single CPT entry
P(x) = (a x + b) / (c x + d) under proportional co-variation of the rest of
the column. For the worked-track evidence e* we (1) recover the
linear-fractional coefficients of P(T=high | e*) for every learned internal
CPT entry from three exact evaluations, (2) rank parameters by the posterior
swing over x in [theta-0.1, theta+0.1] (clipped to [0.001, 0.999]) --
the tornado -- and (3) wrap the top-5 parameters in +-0.1 credal intervals,
reporting the one-at-a-time posterior bounds.

Run:  python exp_sensitivity.py       (~1 min)
Outputs: results/expT_summary.md, results/fig_expT_tornado.png
"""
from __future__ import annotations
import os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({"font.family": "serif", "mathtext.fontset": "stix",
                     "font.size": 10})

import model_spec as ms
import dataio
import metrics as mt
import pipeline as pl
from cuas_bn import Factor

RES = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(RES, exist_ok=True)
DELTA = 0.10
EPS = 1e-3

# worked-track evidence (Sec. VII-A): fast, closing, near range, no-fly zone;
# radar/EO-IR/acoustic report 'drone', RF reports 'none' (radio-silent)
EVIDENCE = {"v": 2, "rdot": 2, "d": 1, "z": 1,
            "Rr": 0, "Re": 0, "Ra": 0, "Rf": 2}


def posterior_T_high(cpts, est, alpha, ev=EVIDENCE):
    bn = dataio.build_inference_bn(cpts, ms.SENSORS, ev["d"],
                                   sensor_cpts=est, alpha=alpha)
    return float(bn.query(["T"], ev).table[3])


def perturbed(cpts, node, j_idx, k, x):
    """Set theta_{node, j, k} = x with proportional co-variation of the rest
    of the column; return a new cpts dict."""
    new = {n: Factor(f.vars, f.table.copy()) for n, f in cpts.items()}
    tab = new[node].table
    row = tab[j_idx].copy()
    rest = 1.0 - row[k]
    row_new = row * ((1.0 - x) / max(rest, 1e-12))
    row_new[k] = x
    tab[j_idx] = row_new / row_new.sum()
    return new


def linfrac_coeffs(f0, f1, f2, x0, x1, x2):
    """Fit P(x) = (a x + b) / (c x + 1) through three exact evaluations."""
    A = np.array([[x0, 1.0, -f0 * x0],
                  [x1, 1.0, -f1 * x1],
                  [x2, 1.0, -f2 * x2]])
    rhs = np.array([f0, f1, f2])
    try:
        a, b, c = np.linalg.solve(A, rhs)
    except np.linalg.LinAlgError:
        return None
    return a, b, c


def main():
    gt = ms.ground_truth_bn()
    calib = dataio.generate_dataset(gt, 300, seed=777)
    est = dataio.estimate_confusion(calib)
    alpha, _ = dataio.fit_reliability(calib)
    data = dataio.generate_dataset(gt, 200, seed=0)
    cpts, ess = pl.ground_internal("Proposed", data, 0, gt=gt)
    p0 = posterior_T_high(cpts, est, alpha)
    rows = []
    for node in ms.INTERNAL:
        tab = cpts[node].table
        Jshape = tab.shape[:-1]
        K = tab.shape[-1]
        from itertools import product as ip
        for j_idx in ip(*[range(s) for s in Jshape]):
            for k in range(K):
                th = float(tab[j_idx + (k,)])
                lo = max(EPS, th - DELTA)
                hi = min(1 - EPS, th + DELTA)
                xm = 0.5 * (lo + hi)
                f_lo = posterior_T_high(perturbed(cpts, node, j_idx, k, lo),
                                        est, alpha)
                f_hi = posterior_T_high(perturbed(cpts, node, j_idx, k, hi),
                                        est, alpha)
                f_md = posterior_T_high(perturbed(cpts, node, j_idx, k, xm),
                                        est, alpha)
                swing = f_hi - f_lo
                coeffs = linfrac_coeffs(f_lo, f_md, f_hi, lo, xm, hi)
                rows.append({"node": node, "j": j_idx, "k": k, "theta": th,
                             "p_lo": f_lo, "p_hi": f_hi, "swing": swing,
                             "linfrac_ok": coeffs is not None})
    rows.sort(key=lambda r: -abs(r["swing"]))
    top = rows[:15]

    lines = ["# Exp-T: tornado sensitivity of P(T=high | e*) "
             f"(worked-track evidence; ESS={ess})", "",
             f"- baseline P(T=high|e*) = {p0:.3f}",
             f"- parameters examined: {len(rows)} (all learned internal CPT "
             f"entries; +-{DELTA} proportional co-variation)",
             f"- linear-fractional fit succeeded for "
             f"{sum(r['linfrac_ok'] for r in rows)}/{len(rows)} entries", ""]
    lines.append("## top-15 by posterior swing")
    labels, lows, highs = [], [], []
    for r in top:
        lab = f"{r['node']}{r['j']}k{r['k']}"
        labels.append(lab)
        lows.append(r["p_lo"])
        highs.append(r["p_hi"])
        lines.append(f"- {lab}: theta={r['theta']:.3f}, "
                     f"P in [{min(r['p_lo'], r['p_hi']):.3f}, "
                     f"{max(r['p_lo'], r['p_hi']):.3f}] "
                     f"(swing {r['swing']:+.3f})")
    lines.append("\n## credal bounds (top-5, one-at-a-time +-0.1)")
    w_lo = min(min(r["p_lo"], r["p_hi"]) for r in rows[:5])
    w_hi = max(max(r["p_lo"], r["p_hi"]) for r in rows[:5])
    lines.append(f"- one-at-a-time envelope over the top-5 parameters: "
                 f"P(T=high|e*) in [{w_lo:.3f}, {w_hi:.3f}] "
                 f"(baseline {p0:.3f}) -- the threat decision "
                 f"{'is' if w_lo > 0.5 else 'is NOT'} invariant over the "
                 f"credal set at the 0.5 decision level")

    fig, ax = plt.subplots(figsize=(7.0, 5.2))
    ypos = np.arange(len(top))[::-1]
    for yy, r in zip(ypos, top):
        lo, hi = sorted((r["p_lo"], r["p_hi"]))
        ax.barh(yy, hi - lo, left=lo, height=0.62, color="#4878CF",
                alpha=0.85)
    ax.axvline(p0, color="k", lw=1.2, ls="--", label=f"baseline {p0:.2f}")
    ax.set_yticks(ypos)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel(r"$P(T=\mathrm{high}\mid e^{*})$ over $\theta \pm 0.1$")
    ax.set_title("One-way sensitivity (tornado): worked-track evidence",
                 fontsize=11)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, axis="x")
    fig.tight_layout()
    fig.savefig(os.path.join(RES, "fig_expT_tornado.png"), dpi=300)
    plt.close(fig)

    txt = "\n".join(lines)
    with open(os.path.join(RES, "expT_summary.md"), "w", encoding="utf-8") as f:
        f.write(txt + "\n")
    print(txt.encode(sys.stdout.encoding or "utf-8", errors="replace")
             .decode(sys.stdout.encoding or "utf-8", errors="replace"))


if __name__ == "__main__":
    main()
