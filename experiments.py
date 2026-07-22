"""
Run the experimental study and emit results (CSV + markdown tables + figures)
to results/. Three experiments:
  Exp1  scarcity sweep: CPT-grounding methods vs training size N
  Exp2  sensor layer: fusion vs single, calibrated vs abstract, reliability,
        and the radio-silent-drone case
  Exp3  ablation at fixed N
Optional: Exp1b MAP-EM (latent H,N,C), Exp4 sensitivity tornado.
"""
from __future__ import annotations
import os, time, json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import model_spec as ms
import dataio
import grounding as g
import metrics as mt

RES = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(RES, exist_ok=True)
ESS = 8.0
TEST_N = 2500
TEST_SEED = 999
CALIB_N = 300         # labelled scenarios used to ESTIMATE sensor confusion matrices
CALIB_SEED = 777


def calib_confusion(n=CALIB_N, seed=CALIB_SEED):
    """Estimate sensor confusion matrices from a labelled calibration split of
    the same world (the real procedure; not the true CPTs)."""
    gt = ms.ground_truth_bn()
    return dataio.estimate_confusion(dataio.generate_dataset(gt, n, seed))


def ground_internal(method, data, seed):
    """Return internal CPTs {H,N,C,T} for a method name."""
    rng = np.random.default_rng(1000 + seed)
    prior = ms.expert_prior_cpts(rng)
    if method == "B0-heuristic":
        return ms.heuristic_cpts()
    if method == "B2-expert":
        return prior
    out = {}
    for node in ms.INTERNAL:
        if method == "B1-MLE":
            out[node] = g.mle_cpt(node, data)
        elif method == "MAP":
            out[node] = g.map_cpt(node, data, prior[node], ESS)
        elif method == "Proposed-cMAP":
            out[node] = g.cmap_cpt(node, data, prior[node], ESS)
        else:
            raise ValueError(method)
    return out


# ------------------------------------------------------------------ Exp1
def exp1(N_grid=(10, 20, 50, 100, 200, 400), seeds=6):
    gt = ms.ground_truth_bn()
    test = dataio.generate_dataset(gt, TEST_N, seed=TEST_SEED)
    est = calib_confusion()          # estimated confusion matrices (one world)
    methods = ["B0-heuristic", "B1-MLE", "B2-expert", "MAP", "Proposed-cMAP"]
    rows = []
    t0 = time.time()
    for N in N_grid:
        for seed in range(seeds):
            data = dataio.generate_dataset(gt, N, seed=seed)  # fully observed
            for m in methods:
                cpts = ground_internal(m, data, seed)
                met, _ = mt.evaluate(cpts, test, reliability=True,
                                     sensor_mode="calibrated", use_soft_eoir=True,
                                     sensor_cpts=est)
                rows.append({"N": N, "seed": seed, "method": m, **met})
            print(f"  [Exp1] N={N} seed={seed} done ({time.time()-t0:.0f}s)")
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RES, "exp1_raw.csv"), index=False)
    agg = df.groupby(["method", "N"]).agg(["mean", "std"])
    # tidy summary
    summ = df.groupby(["method", "N"]).mean(numeric_only=True).reset_index()
    summ.to_csv(os.path.join(RES, "exp1_summary.csv"), index=False)
    # oracle line
    orc, _ = mt.evaluate({k: gt.cpts[k] for k in ms.INTERNAL}, test)
    _plot_exp1(summ, orc)
    return df, summ, orc


def _plot_exp1(summ, orc):
    metrics_to_plot = [("acc", "Threat accuracy", False),
                       ("macroAUC", "Macro AUC", False),
                       ("brier", "Brier score (lower=better)", True),
                       ("ece", "ECE (lower=better)", True)]
    order = ["B0-heuristic", "B1-MLE", "B2-expert", "MAP", "Proposed-cMAP"]
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    for ax, (key, title, lower) in zip(axes.ravel(), metrics_to_plot):
        for m in order:
            d = summ[summ.method == m].sort_values("N")
            ax.plot(d["N"], d[key], marker="o", label=m)
        ax.axhline(orc[key], ls="--", color="k", lw=1, alpha=0.6, label="oracle")
        ax.set_xscale("log"); ax.set_xlabel("training scenarios N"); ax.set_title(title)
        ax.grid(alpha=0.3)
    axes.ravel()[0].legend(fontsize=8, loc="lower right")
    fig.suptitle("Exp1 — CPT grounding under data scarcity", fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(RES, "exp1_scarcity.png"), dpi=130)
    plt.close(fig)


# ------------------------------------------------------------------ Exp2
def exp2(N=200, seed=0):
    gt = ms.ground_truth_bn()
    test = dataio.generate_dataset(gt, TEST_N, seed=TEST_SEED)
    data = dataio.generate_dataset(gt, N, seed=seed)
    cpts = ground_internal("Proposed-cMAP", data, seed)  # fix grounding
    est = calib_confusion()
    configs = [
        ("radar only",            ["Rr"],                 True,  "calibrated"),
        ("RF only",               ["Rf"],                 True,  "calibrated"),
        ("EO/IR only",            ["Re"],                 True,  "calibrated"),
        ("acoustic only",         ["Ra"],                 True,  "calibrated"),
        ("all (abstract model)",  ms.SENSORS,             False, "abstract"),
        ("all (calibrated)",      ms.SENSORS,             False, "calibrated"),
        ("all (calib+reliab)",    ms.SENSORS,             True,  "calibrated"),
    ]
    rows = []
    for name, son, rel, mode in configs:
        sc = est if mode == "calibrated" else None
        met, _ = mt.evaluate(cpts, test, sensors_on=son, reliability=rel,
                             sensor_mode=mode, use_soft_eoir=True, sensor_cpts=sc)
        rows.append({"config": name, **met})
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RES, "exp2_sensors.csv"), index=False)

    # radio-silent-drone case: subset + reliability on/off + far-range subset
    silent = [s for s in test if s["cls"] == 2 and s["emit"] == 1]
    farsilent = [s for s in silent if s["d"] == 0]
    sub_rows = []
    for label, subset in [("silent-drone (all ranges)", silent),
                          ("silent-drone (far range)", farsilent)]:
        if len(subset) < 20:
            continue
        for rel in [False, True]:
            met, _ = mt.evaluate(cpts, subset, sensors_on=ms.SENSORS,
                                 reliability=rel, sensor_mode="calibrated",
                                 use_soft_eoir=True, sensor_cpts=est)
            sub_rows.append({"subset": label, "n": len(subset),
                             "reliability": rel, **met})
    dfs = pd.DataFrame(sub_rows)
    dfs.to_csv(os.path.join(RES, "exp2_silent.csv"), index=False)
    _plot_exp2(df)
    return df, dfs


def _plot_exp2(df):
    fig, ax = plt.subplots(figsize=(9, 4.5))
    d = df.set_index("config")
    d[["acc", "macroAUC"]].plot(kind="bar", ax=ax)
    ax.set_ylabel("score"); ax.set_title("Exp2 — sensor fusion & reliability")
    ax.set_ylim(0, 1); ax.grid(alpha=0.3, axis="y")
    plt.setp(ax.get_xticklabels(), rotation=25, ha="right", fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(RES, "exp2_sensors.png"), dpi=130)
    plt.close(fig)


# ------------------------------------------------------------------ Exp3 ablation
def exp3(N=80, seeds=6):
    gt = ms.ground_truth_bn()
    test = dataio.generate_dataset(gt, TEST_N, seed=TEST_SEED)
    variants = {
        "Proposed (full)":       dict(method="Proposed-cMAP", rel=True,  soft=True),
        "-constraints (MAP)":    dict(method="MAP",           rel=True,  soft=True),
        "-prior (MLE)":          dict(method="B1-MLE",        rel=True,  soft=True),
        "-reliability":          dict(method="Proposed-cMAP", rel=False, soft=True),
        "-virtual evid. (hard)": dict(method="Proposed-cMAP", rel=True,  soft=False),
    }
    est = calib_confusion()
    rows = []
    for seed in range(seeds):
        data = dataio.generate_dataset(gt, N, seed=seed)
        for name, cfg in variants.items():
            cpts = ground_internal(cfg["method"], data, seed)
            met, _ = mt.evaluate(cpts, test, reliability=cfg["rel"],
                                 sensor_mode="calibrated", use_soft_eoir=cfg["soft"],
                                 sensor_cpts=est)
            rows.append({"variant": name, "seed": seed, **met})
    df = pd.DataFrame(rows)
    summ = df.groupby("variant").mean(numeric_only=True).drop(columns=["seed"]).reset_index()
    df.to_csv(os.path.join(RES, "exp3_ablation_raw.csv"), index=False)
    summ.to_csv(os.path.join(RES, "exp3_ablation.csv"), index=False)
    return summ


# --------------------------------------------- Exp2b sensor-calibration size
def exp2b(N=200, seed=0, calib_grid=(30, 100, 300, 1000)):
    """How much labelled calibration data is needed to estimate the sensor
    confusion matrices? Estimate from increasing calib sizes and evaluate full
    fusion; compare to the abstract model and the oracle (true matrices)."""
    gt = ms.ground_truth_bn()
    test = dataio.generate_dataset(gt, TEST_N, seed=TEST_SEED)
    data = dataio.generate_dataset(gt, N, seed=seed)
    cpts = ground_internal("Proposed-cMAP", data, seed)
    rows = []
    # abstract baseline
    m0, _ = mt.evaluate(cpts, test, sensor_mode="abstract", reliability=False,
                        use_soft_eoir=True)
    rows.append({"calib_n": 0, "source": "abstract", **m0})
    for cn in calib_grid:
        est = calib_confusion(n=cn, seed=CALIB_SEED)
        m, _ = mt.evaluate(cpts, test, sensor_mode="calibrated", reliability=False,
                           use_soft_eoir=True, sensor_cpts=est)
        rows.append({"calib_n": cn, "source": "estimated", **m})
    # oracle (true matrices) upper bound
    mo, _ = mt.evaluate(cpts, test, sensor_mode="calibrated", reliability=False,
                        use_soft_eoir=True, sensor_cpts=None)
    rows.append({"calib_n": -1, "source": "oracle-true-CM", **mo})
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RES, "exp2b_calibsize.csv"), index=False)
    return df


def _md_table(df, floatfmt=3):
    cols = list(df.columns)
    out = ["| " + " | ".join(cols) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    for _, r in df.iterrows():
        vals = []
        for c in cols:
            v = r[c]
            vals.append(f"{v:.{floatfmt}f}" if isinstance(v, (float, np.floating)) else str(v))
        out.append("| " + " | ".join(vals) + " |")
    return "\n".join(out)


if __name__ == "__main__":
    t0 = time.time()
    print("== Exp1 scarcity sweep ==")
    df1, summ1, orc = exp1()
    print("== Exp2 sensors ==")
    df2, dfs2 = exp2()
    print("== Exp2b sensor-calibration size ==")
    df2b = exp2b()
    print("== Exp3 ablation ==")
    summ3 = exp3()

    # write a consolidated markdown report
    with open(os.path.join(RES, "RESULTS.md"), "w", encoding="utf-8") as f:
        f.write("# Experimental results (synthetic digital-twin study)\n\n")
        f.write(f"Test set: {TEST_N} held-out scenarios (seed {TEST_SEED}); ESS={ESS}. "
                "Oracle = true CPTs.\n\n")
        f.write("## Oracle (upper bound)\n\n")
        f.write(_md_table(pd.DataFrame([{**{'model':'oracle'}, **{k:round(v,3) for k,v in orc.items()}}])) + "\n\n")
        f.write("## Exp1 — CPT grounding under scarcity (mean over seeds)\n\n")
        piv = summ1.pivot(index="N", columns="method", values="ece")
        f.write("**ECE by N (lower is better):**\n\n" + _md_table(piv.reset_index()) + "\n\n")
        piv2 = summ1.pivot(index="N", columns="method", values="acc")
        f.write("**Accuracy by N:**\n\n" + _md_table(piv2.reset_index()) + "\n\n")
        f.write("See `exp1_scarcity.png` for accuracy/AUC/Brier/ECE curves.\n\n")
        f.write("Sensor confusion matrices are ESTIMATED from a labelled "
                f"calibration split of the same world (n={CALIB_N}); not the true CPTs.\n\n")
        f.write("## Exp2 — sensor fusion & reliability\n\n")
        f.write(_md_table(df2) + "\n\n")
        f.write("**Radio-silent-drone subset (reliability on/off):**\n\n")
        f.write(_md_table(dfs2) + "\n\n")
        f.write("## Exp2b — sensor-calibration-set size (estimated vs abstract vs oracle)\n\n")
        f.write(_md_table(df2b) + "\n\n")
        f.write("## Exp3 — ablation (N=80)\n\n")
        f.write(_md_table(summ3) + "\n\n")
    print(f"ALL DONE in {time.time()-t0:.0f}s -> results/RESULTS.md")
