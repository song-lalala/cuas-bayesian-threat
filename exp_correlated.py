"""
Exp-F v2: correlated sensor errors and reliability miscalibration.

(R1-7, R1-8, R1-10; revised for this resubmission: 20 seeds, adverse-subset
metrics with CIs shown in the figure, a LATENT-W and a noisy-W variant in
addition to observed W, reliability weights actually fitted by logistic
regression, and the far-heavy transfer test run in a world that really has
range-dependent degradation.)

Part (i) correlated failures. On top of world v2 (range-degraded sensors), a
  shared nuisance root W in {benign, adverse} (P(adverse)=0.35) mixes the
  environment-sensitive sensors' d-conditional tables toward a common
  low-SNR miss profile with severity s (induced pairwise error correlation
  up to rho ~ 0.3). Fusions compared on the same draws:
    naive-CI   : deployed model, W unmodeled (marginal matrices + fitted
                 reliability from a W-mixed calibration split).
    W-observed : explicit nuisance parent, W-conditional matrices, W known
                 at run time (weather is metered).
    W-latent   : same model, W never observed -- marginalized mixture
                 likelihood (tests the option without the free observation).
    W-noisy    : W observed through a 90%-accurate indicator.
Part (ii) reliability transfer / miscalibration with the FITTED alpha:
  (a) alpha scaled x{0.5..1.5}; (b) far-heavy deployment context
  P(d)=(0.70,0.20,0.10) with alpha trained in the nominal context;
  (c) P(e) monitoring (clean 5th-percentile threshold; adverse-subset flag
  rate reported separately).

Outputs (results/): expF_corr_raw.csv, expF_rel_raw.csv, expF_summary.md,
  fig_expF_corr.png
Run:  python exp_correlated.py [--smoke] [--part corr|rel|all]
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

from cuas_bn import Factor, BayesNet, sample_ancestral, factor_reduce, \
    factor_product, factor_marginalize
import model_spec as ms
import dataio
import grounding as g
import metrics as mt
import pipeline as pl

RES = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(RES, exist_ok=True)
TEST_N = 2500
GROUND_N = 200
W_SENSORS = ["Rr", "Re", "Ra"]
P_ADVERSE = 0.35
GAMMA = {"Re": 0.90, "Ra": 0.90, "Rr": 0.60}   # adverse mixing strengths
W_OBS_ACC = 0.90                                # noisy-W indicator accuracy

# Common low-SNR miss profile (rows follow cls order clutter,bird,drone): a
# weak-signature drone is read as bird/clutter/none, which makes the degraded
# sensors' errors conditionally DEPENDENT given the true state.
ADVERSE_PROFILE = {
    "Rr": np.array([[0.05, 0.15, 0.80],
                    [0.08, 0.52, 0.40],
                    [0.10, 0.20, 0.70]]),
    "Re": np.array([[0.04, 0.12, 0.84],
                    [0.06, 0.58, 0.36],
                    [0.12, 0.44, 0.44]]),
    "Ra": np.array([[0.05, 0.95],
                    [0.08, 0.92],
                    [0.12, 0.88]]),
}


# ---------------------------------------------------------------- W-worlds
def world_with_W(severity: float) -> BayesNet:
    gt = ms.ground_truth_bn()
    card = dict(ms.CARD); card["W"] = 2; card["Wobs"] = 2
    parents = {k: list(v) for k, v in ms.PARENTS.items()}
    parents["W"] = []
    parents["Wobs"] = ["W"]
    cpts = dict(gt.cpts)
    cpts["W"] = Factor(("W",), np.array([1 - P_ADVERSE, P_ADVERSE]))
    cpts["Wobs"] = Factor(("W", "Wobs"),
                          np.array([[W_OBS_ACC, 1 - W_OBS_ACC],
                                    [1 - W_OBS_ACC, W_OBS_ACC]]))
    for s in W_SENSORS:
        base = gt.cpts[s].table                 # (cls, d, R)
        gam = GAMMA[s] * severity
        prof = ADVERSE_PROFILE[s][:, None, :]   # broadcast over d
        adverse = (1 - gam) * base + gam * prof
        tab = np.stack([base, adverse], axis=-2)   # (cls, d, W, R)
        parents[s] = list(ms.PARENTS[s]) + ["W"]
        cpts[s] = Factor(tuple(ms.PARENTS[s]) + ("W", s), tab)
    return BayesNet(card, parents, cpts)


def estimate_confusion_W(calib, smoothing=0.5):
    """W-conditional, range-marginal confusion estimates for the W-aware
    fusion (parents = marginal parents + W)."""
    out = {}
    for s in ms.SENSORS:
        pa = dataio.MARGINAL_PARENTS[s] + (["W"] if s in W_SENSORS else [])
        K = ms.CARD[s]
        shape = tuple((2 if p == "W" else ms.CARD[p]) for p in pa) + (K,)
        n = np.zeros(shape)
        for row in calib:
            idx = tuple(row[p] for p in pa) + (row[s],)
            n[idx] += 1
        n = n + smoothing
        out[s] = Factor(tuple(pa) + (s,),
                        n / n.sum(axis=-1, keepdims=True))
    return out


def build_bn_W(internal_cpts, sensor_cpts, w_mode, alpha=None, d_value=0):
    """w_mode: 'none' (plain deployed model), 'observed'/'latent'/'noisy'
    (nuisance parent present; W observed, marginalized, or seen through
    Wobs)."""
    gt = ms.ground_truth_bn()
    cpts = dict(gt.cpts)
    parents = {k: list(v) for k, v in ms.PARENTS.items()}
    for k in ms.INTERNAL:
        cpts[k] = internal_cpts[k]
    for s in ms.SENSORS:
        cpts.pop(s, None)
        parents.pop(s, None)
    card = dict(ms.CARD)
    if w_mode != "none":
        card["W"] = 2
        parents["W"] = []
        cpts["W"] = Factor(("W",), np.array([1 - P_ADVERSE, P_ADVERSE]))
        if w_mode == "noisy":
            card["Wobs"] = 2
            parents["Wobs"] = ["W"]
            cpts["Wobs"] = Factor(("W", "Wobs"),
                                  np.array([[W_OBS_ACC, 1 - W_OBS_ACC],
                                            [1 - W_OBS_ACC, W_OBS_ACC]]))
    for s in ms.SENSORS:
        c = sensor_cpts[s]
        if alpha is not None:
            c = dataio._discount_cpt(c, float(alpha[s][d_value]))
        cpts[s] = c
        parents[s] = list(c.vars[:-1])
    net_parents = {k: parents[k] for k in cpts}
    net_card = {k: card[k] for k in cpts}
    return BayesNet(net_card, net_parents, cpts)


def posteriors_W(internal_cpts, sensor_cpts, test, w_mode, alpha=None,
                 collect_pe=False):
    P = np.zeros((len(test), ms.CARD["T"]))
    y = np.zeros(len(test), dtype=int)
    pes = np.zeros(len(test))
    cache = {}
    for i, scn in enumerate(test):
        key = scn["d"]
        if key not in cache:
            cache[key] = build_bn_W(internal_cpts, sensor_cpts, w_mode,
                                    alpha=alpha, d_value=scn["d"])
        bn = cache[key]
        ev = {k: scn[k] for k in ms.OBSERVED}
        for s in ms.SENSORS:
            ev[s] = scn[s]
        if w_mode == "observed":
            ev["W"] = scn["W"]
        elif w_mode == "noisy":
            ev["Wobs"] = scn["Wobs"]
        P[i] = bn.query(["T"], ev).table
        y[i] = scn["T"]
        if collect_pe:
            pes[i] = evidence_prob(bn, ev)
    return (P, y, pes) if collect_pe else (P, y)


def evidence_prob(bn, evidence):
    factors = []
    scalar = 1.0
    for v in bn.order:
        f = factor_reduce(bn.cpts[v], evidence)
        if f.vars:
            factors.append(f)
        else:
            scalar *= float(f.table)
    live = list(factors)
    order = [v for v in bn.order if any(v in f.vars for f in live)]
    for v in order:
        involved = [f for f in live if v in f.vars]
        rest = [f for f in live if v not in f.vars]
        if not involved:
            continue
        prod = involved[0]
        for f in involved[1:]:
            prod = factor_product(prod, f, bn.card)
        prod = factor_marginalize(prod, v)
        if prod.vars:
            rest.append(prod)
        else:
            scalar *= float(prod.table)
        live = rest
    for f in live:
        scalar *= float(f.table.sum())
    return scalar


def macro_auc(P, y):
    from sklearn.metrics import roc_auc_score
    try:
        return float(roc_auc_score(np.eye(4)[y], P, multi_class="ovr",
                                   average="macro", labels=[0, 1, 2, 3]))
    except Exception:
        return np.nan


def error_correlation(test):
    sub = [s for s in test if s["cls"] == 2]
    errs = {s: np.array([scn[s] != 0 for scn in sub], float)
            for s in W_SENSORS}
    cors = []
    ss = W_SENSORS
    for i in range(len(ss)):
        for j in range(i + 1, len(ss)):
            a, b = errs[ss[i]], errs[ss[j]]
            if a.std() > 0 and b.std() > 0:
                cors.append(np.corrcoef(a, b)[0, 1])
    return float(np.mean(cors)) if cors else np.nan


def extra_params():
    tot = 0
    for s in W_SENSORS:
        cfgs = int(np.prod([ms.CARD[p] for p in dataio.MARGINAL_PARENTS[s]]))
        tot += cfgs * (ms.CARD[s] - 1)
    return tot


FUSIONS = ["naive-CI", "W-observed", "W-latent", "W-noisy"]


def part_corr(severities, seeds):
    rows = []
    t0 = time.time()
    for sev in severities:
        world = world_with_W(sev)
        for seed in range(seeds):
            test = sample_ancestral(world, np.random.default_rng(9000 + seed),
                                    TEST_N)
            calib = sample_ancestral(world, np.random.default_rng(5000 + seed),
                                     300)
            train = sample_ancestral(world, np.random.default_rng(seed),
                                     GROUND_N)
            est_n = dataio.estimate_confusion(calib)
            alpha, _ = dataio.fit_reliability(calib)
            cpts, _ = pl.ground_internal("Proposed", train, seed,
                                         gt=ms.ground_truth_bn())
            est_w = estimate_confusion_W(calib)
            rho = error_correlation(test)
            adv = np.array([s["W"] == 1 for s in test])
            for fus in FUSIONS:
                if fus == "naive-CI":
                    P, y = posteriors_W(cpts, est_n, test, "none", alpha=alpha)
                else:
                    mode = {"W-observed": "observed", "W-latent": "latent",
                            "W-noisy": "noisy"}[fus]
                    P, y = posteriors_W(cpts, est_w, test, mode, alpha=alpha)
                rows.append({
                    "severity": sev, "seed": seed, "fusion": fus, "rho": rho,
                    "acc": float((P.argmax(1) == y).mean()),
                    "macroAUC": macro_auc(P, y), "ece": mt.ece(P, y),
                    "brier": mt.brier(P, y), "cwece": mt.classwise_ece(P, y),
                    "acc_adv": float((P.argmax(1) == y)[adv].mean()),
                    "ece_adv": mt.ece(P[adv], y[adv]),
                    "auc_adv": macro_auc(P[adv], y[adv])})
        print(f"[expF-i] severity={sev} done ({time.time()-t0:.0f}s)",
              flush=True)
        pd.DataFrame(rows).to_csv(os.path.join(RES, "expF_corr_raw.csv"),
                                  index=False)
    return pd.DataFrame(rows)


def shifted_world(pd_far=(0.70, 0.20, 0.10)) -> BayesNet:
    gt = ms.ground_truth_bn()
    cpts = dict(gt.cpts)
    cpts["d"] = Factor(("d",), np.asarray(pd_far, float))
    return BayesNet(ms.CARD, ms.PARENTS, cpts)


def part_rel(mults, seeds):
    gt = ms.ground_truth_bn()
    rows = []
    t0 = time.time()
    for seed in range(seeds):
        test = dataio.generate_dataset(gt, TEST_N, seed=9000 + seed)
        test_far = sample_ancestral(shifted_world(),
                                    np.random.default_rng(9500 + seed), TEST_N)
        calib = dataio.generate_dataset(gt, 300, seed=5000 + seed)
        est = dataio.estimate_confusion(calib)
        alpha, _ = dataio.fit_reliability(calib)
        train = dataio.generate_dataset(gt, GROUND_N, seed=seed)
        cpts, _ = pl.ground_internal("Proposed", train, seed, gt=gt)
        for m in mults:
            am = {s: np.clip(np.asarray(a) * m, 0.02, 0.995)
                  for s, a in alpha.items()}
            met, _ = mt.evaluate(cpts, test, sensor_cpts=est, alpha=am)
            rows.append({"part": "miscal", "mult": m, "world": "nominal",
                         "seed": seed, **met})
        met, _ = mt.evaluate(cpts, test, sensor_cpts=est, alpha=None)
        rows.append({"part": "miscal", "mult": np.nan, "world": "nominal-noRel",
                     "seed": seed, **met})
        for al, tag in [(alpha, "far-heavy"), (None, "far-heavy-noRel")]:
            met, _ = mt.evaluate(cpts, test_far, sensor_cpts=est, alpha=al)
            rows.append({"part": "shift", "mult": 1.0 if al else np.nan,
                         "world": tag, "seed": seed, **met})
        # (c) P(e) monitoring
        _, _, pe_clean = posteriors_W(cpts, est, test[:800], "none",
                                      alpha=alpha, collect_pe=True)
        thr = np.percentile(pe_clean, 5)
        world_adv = world_with_W(1.0)
        test_adv = sample_ancestral(world_adv,
                                    np.random.default_rng(9900 + seed), 800)
        _, _, pe_adv = posteriors_W(cpts, est, test_adv, "none", alpha=alpha,
                                    collect_pe=True)
        _, _, pe_far = posteriors_W(cpts, est, test_far[:800], "none",
                                    alpha=alpha, collect_pe=True)
        wmask = np.array([s["W"] == 1 for s in test_adv])
        rows.append({"part": "pe", "world": "flag-rates", "seed": seed,
                     "pe_thr": thr,
                     "flag_clean": float((pe_clean < thr).mean()),
                     "flag_adverse": float((pe_adv < thr).mean()),
                     "flag_adverse_subset": float((pe_adv[wmask] < thr).mean()),
                     "flag_farheavy": float((pe_far < thr).mean())})
        print(f"[expF-ii] seed {seed+1}/{seeds} ({time.time()-t0:.0f}s)",
              flush=True)
        pd.DataFrame(rows).to_csv(os.path.join(RES, "expF_rel_raw.csv"),
                                  index=False)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- figure
def fig_corr(df, out):
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.0))
    styles = {"naive-CI": ("--o", "C0"), "W-observed": ("-s", "C1"),
              "W-latent": ("-^", "C2"), "W-noisy": ("-v", "C3")}
    for ax, met, ttl in [(axes[0], "ece_adv",
                          "adverse-subset ECE (lower=better)"),
                         (axes[1], "auc_adv", "adverse-subset macro-AUC")]:
        for fus in FUSIONS:
            gdf = df[df.fusion == fus].groupby("severity")[met]
            mean = gdf.mean()
            sem = gdf.std() / np.sqrt(gdf.count())
            ls, col = styles[fus]
            ax.errorbar(mean.index, mean.values, yerr=2.093 * sem, fmt=ls,
                        color=col, ms=4, capsize=2, label=fus)
        ax.set_xlabel("correlated-degradation severity $s$")
        ax.set_title(ttl, fontsize=10)
        ax.grid(alpha=0.3)
    rho = df.groupby("severity")["rho"].mean()
    ax2 = axes[0].twiny()
    ax2.set_xlim(axes[0].get_xlim())
    ax2.set_xticks(rho.index)
    ax2.set_xticklabels([f"{v:.2f}" for v in rho.values], fontsize=7)
    ax2.set_xlabel(r"induced error correlation $\rho$ (cls=drone)", fontsize=8)
    axes[0].legend(fontsize=8)
    fig.suptitle("Correlated sensor failures: adverse-subset metrics "
                 "(mean ±95% CI over 20 seeds)", fontsize=11)
    fig.tight_layout()
    fig.savefig(out, dpi=300)
    plt.close(fig)


def summarize(dfc, dfr):
    lines = ["# Exp-F v2: correlated failures and reliability miscalibration",
             ""]
    lines.append(f"W-aware nuisance layer adds {extra_params()} free "
                 f"parameters (W parent on {W_SENSORS}).")
    lines.append("\n## (i) correlated failures (means over seeds; "
                 "overall | adverse-subset)")
    for sev in sorted(dfc.severity.unique()):
        d = dfc[dfc.severity == sev]
        rho = d["rho"].mean()
        parts = []
        for fus in FUSIONS:
            x = d[d.fusion == fus]
            parts.append(f"{fus}: ECE {x['ece'].mean():.3f}|"
                         f"{x['ece_adv'].mean():.3f}, AUC "
                         f"{x['macroAUC'].mean():.3f}|{x['auc_adv'].mean():.3f}")
        lines.append(f"- s={sev:g} (rho={rho:.2f}): " + "  ||  ".join(parts))
    ref = dfr[dfr.world == "nominal-noRel"]
    lines.append("\n## (ii-a) reliability miscalibration (nominal world)")
    lines.append(f"- reliability OFF: ECE {ref['ece'].mean():.3f}, "
                 f"acc {ref['acc'].mean():.3f}")
    for m in sorted(dfr[dfr.part == "miscal"].dropna(subset=["mult"])
                    .mult.unique()):
        d = dfr[(dfr.part == "miscal") & (dfr.mult == m)]
        lines.append(f"- alpha x{m:g}: ECE {d['ece'].mean():.3f}, "
                     f"acc {d['acc'].mean():.3f}, AUC "
                     f"{d['macroAUC'].mean():.3f}")
    lines.append("\n## (ii-b) far-heavy context shift (alpha as fitted "
                 "in the nominal context)")
    for tag in ["far-heavy", "far-heavy-noRel"]:
        d = dfr[dfr.world == tag]
        lines.append(f"- {tag}: ECE {d['ece'].mean():.3f}, acc "
                     f"{d['acc'].mean():.3f}, AUC {d['macroAUC'].mean():.3f}, "
                     f"cwECE {d['cwece'].mean():.3f}")
    d = dfr[dfr.part == "pe"]
    lines.append("\n## (ii-c) P(e) monitoring (threshold = clean 5th pct)")
    lines.append(f"- flag rate: clean {d['flag_clean'].mean():.3f}, "
                 f"adverse world {d['flag_adverse'].mean():.3f}, "
                 f"adverse-only tracks {d['flag_adverse_subset'].mean():.3f}, "
                 f"far-heavy {d['flag_farheavy'].mean():.3f}")
    txt = "\n".join(lines)
    with open(os.path.join(RES, "expF_summary.md"), "w", encoding="utf-8") as f:
        f.write(txt + "\n")
    print(txt.encode(sys.stdout.encoding or "utf-8", errors="replace")
             .decode(sys.stdout.encoding or "utf-8", errors="replace"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--summary-only", action="store_true")
    ap.add_argument("--part", choices=["all", "corr", "rel"], default="all")
    a = ap.parse_args()
    sevs = (0.0, 0.25, 0.5, 0.75, 1.0)
    mults = (0.5, 0.75, 1.0, 1.25, 1.5)
    seeds = 20
    if a.smoke:
        sevs = (0.0, 1.0); mults = (0.5, 1.5); seeds = 2
    if a.summary_only:
        dfc = pd.read_csv(os.path.join(RES, "expF_corr_raw.csv"))
        dfr = pd.read_csv(os.path.join(RES, "expF_rel_raw.csv"))
        summarize(dfc, dfr)
        fig_corr(dfc, os.path.join(RES, "fig_expF_corr.png"))
        return
    t0 = time.time()
    if a.part in ("all", "corr"):
        dfc = part_corr(sevs, seeds)
    else:
        dfc = pd.read_csv(os.path.join(RES, "expF_corr_raw.csv"))
    if a.part in ("all", "rel"):
        dfr = part_rel(mults, seeds)
    else:
        dfr = pd.read_csv(os.path.join(RES, "expF_rel_raw.csv"))
    fig_corr(dfc, os.path.join(RES, "fig_expF_corr.png"))
    summarize(dfc, dfr)
    print(f"[expF] ALL DONE in {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
