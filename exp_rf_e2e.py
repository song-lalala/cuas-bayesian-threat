"""
Exp-G v2: end-to-end effect of the DroneRF-measured RF channel, with REAL
samples passing through the pipeline at test time.

Revised design (addresses the circularity critique of the earlier
matrix-resampling version):
  * The DroneRF segments are split by RECORDING (source CSV file), stratified
    by class: 60% train / 15% validation / 25% test. No segment of a test
    recording is seen in training or validation (no segment-level leakage).
  * The compact CNN of Sec. VII-C is trained on the train split; the
    checkpoint is selected on the VALIDATION split (never the test split).
  * Mixed world: scenarios come from the synthetic generative proxy; the RF
    channel is realized by drawing an actual held-out spectrogram of the
    mapped DroneRF class (emitting drone -> a random drone model; silent
    drone and non-drone targets -> background) and passing it through the
    CNN at test time.
  * Fusion variants on the same scenario draws:
      (1) abstract      : CNN hard report + over-confident identity model;
      (2) calib-hard    : CNN hard report + confusion matrix estimated by the
                          deployed pipeline on a labeled mixed-world
                          calibration split whose RF samples come from the
                          VALIDATION pool (disjoint from the test pool);
      (3) calib-soft    : the CNN softmax enters as virtual likelihood
                          evidence with the train-prior scaling
                          p(x|r) ∝ s_r / pi_r  (C3's soft-score mechanism,
                          evaluated with a real classifier) -- the
                          (-virtual-evidence) comparison is (3) vs (2).
  * 20 replications: the CNN is trained once (deterministic seed); each
    replication redraws the scenario world, the calibration split, and the
    spectrogram assignments.

Inputs: droneRF_four.npz (see README "DroneRF data").
Run:  python exp_rf_e2e.py --train /path/droneRF_four.npz     (once, ~15 min)
      python exp_rf_e2e.py [--reps 20]                        (main study)
Outputs: results/expG_cnn.npz (cached forward passes), expG_rf_e2e_raw.csv,
         expG_summary.md
"""
from __future__ import annotations
import os, sys, time, argparse
import numpy as np
import pandas as pd

from cuas_bn import Factor, BayesNet
import model_spec as ms
import dataio
import metrics as mt
import pipeline as pl

RES = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(RES, exist_ok=True)
CNN_CACHE = os.path.join(RES, "expG_cnn.npz")
TEST_N = 2500
GROUND_N = 200
DRONE_CLASSES = [1, 2, 3]                      # bebop, ar_drone, phantom
SEED = 42


# ---------------------------------------------------------------- training
def session_key(fname):
    """DroneRF source files come in high/low-band PAIRS per capture session
    (e.g. 00000H_13.csv and 00000L_13.csv). Grouping by the session key
    (BUI code + capture index, band letter dropped) keeps both bands of a
    session in the same split -- the leakage unit is the capture, not the
    file."""
    import re
    m = re.match(r"^(\d+)[HL]_(\d+)\.csv$", fname)
    return f"{m.group(1)}_{m.group(2)}" if m else fname


def group_split(files, y, rng, frac_train=0.60, frac_val=0.15):
    """Session-level stratified split: assign capture SESSIONS (H/L file
    pairs) to train/val/test within each class."""
    sessions = np.array([session_key(f) for f in files])
    part = {}
    for c in np.unique(y):
        ss = sorted(set(sessions[y == c]))
        ss = list(rng.permutation(ss))
        n_tr = int(round(frac_train * len(ss)))
        n_va = int(round(frac_val * len(ss)))
        for s in ss[:n_tr]:
            part[s] = "train"
        for s in ss[n_tr:n_tr + n_va]:
            part[s] = "val"
        for s in ss[n_tr + n_va:]:
            part[s] = "test"
    return np.array([part[s] for s in sessions])


def train_and_cache(npz_path):
    import torch, torch.nn as nn
    from sklearn.metrics import accuracy_score
    torch.manual_seed(SEED); np.random.seed(SEED)
    d = np.load(npz_path, allow_pickle=True)
    X = d["X"].astype(np.float32)
    y = d["y"].astype(np.int64)
    files = np.array([m[0] for m in d["meta"]])
    names = [str(s) for s in d["class_names"]]
    X = (X - X.mean(axis=(1, 2), keepdims=True)) / \
        (X.std(axis=(1, 2), keepdims=True) + 1e-6)
    split = group_split(files, y, np.random.default_rng(SEED))
    itr, iva, ite = (split == "train"), (split == "val"), (split == "test")
    n_sess = len(set(session_key(f) for f in files))
    print(f"[expG] session-level split ({n_sess} capture sessions, H/L pairs "
          f"kept together): train {itr.sum()} segs / val {iva.sum()} / "
          f"test {ite.sum()}")
    from rf_confusion import CNN
    Xtr = torch.from_numpy(X[itr]).unsqueeze(1)
    ytr = torch.from_numpy(y[itr])
    Xva = torch.from_numpy(X[iva]).unsqueeze(1)
    net = CNN(len(names))
    opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=35)
    w = torch.tensor([1.0 / c for c in np.bincount(y[itr])],
                     dtype=torch.float32)
    lossf = nn.CrossEntropyLoss(weight=w / w.sum() * len(names))
    bs, best, best_state = 64, -1.0, None
    for ep in range(35):
        net.train(); perm = torch.randperm(len(Xtr))
        for i in range(0, len(Xtr), bs):
            idx = perm[i:i + bs]; opt.zero_grad()
            lossf(net(Xtr[idx]), ytr[idx]).backward(); opt.step()
        sched.step(); net.eval()
        with torch.no_grad():
            va = accuracy_score(y[iva], net(Xva).argmax(1).numpy())
        if va > best:
            best, best_state = va, {k: v.clone()
                                    for k, v in net.state_dict().items()}
        if (ep + 1) % 5 == 0:
            print(f"  epoch {ep+1:2d} val_acc={va:.4f} (best {best:.4f})")
    net.load_state_dict(best_state); net.eval()
    with torch.no_grad():
        sm_va = torch.softmax(net(Xva), 1).numpy()
        sm_te = torch.softmax(net(torch.from_numpy(X[ite]).unsqueeze(1)),
                              1).numpy()
    acc_te = float((sm_te.argmax(1) == y[ite]).mean())
    print(f"[expG] checkpoint val_acc={best:.4f}; recording-level "
          f"test_acc={acc_te:.4f}")
    np.savez_compressed(
        CNN_CACHE,
        val_softmax=sm_va, val_y=y[iva],
        test_softmax=sm_te, test_y=y[ite],
        train_prior=np.bincount(y[itr], minlength=4) / itr.sum(),
        val_acc=best, test_acc=acc_te, class_names=np.array(names))


# ---------------------------------------------------------------- BN side
def rf4_abstract_L():
    """Over-confident identity model in the DroneRF report space:
    L[c,e,report]."""
    L = np.zeros((ms.CARD["cls"], ms.CARD["emit"], 4))
    for c in range(ms.CARD["cls"]):
        for e in range(ms.CARD["emit"]):
            if c == 2:
                L[c, e] = [0.06, 0.94 / 3, 0.94 / 3, 0.94 / 3]
            else:
                L[c, e] = [0.94, 0.02, 0.02, 0.02]
    return L


def map_class(scn, rng):
    """Scenario -> DroneRF true class (emitting drone: random model;
    silent drone & non-drone: background)."""
    if scn["cls"] == 2 and scn["emit"] == 0:
        return int(rng.choice(DRONE_CLASSES))
    return 0


def draw_pool_index(pools, c, rng):
    idx = pools[c]
    return int(idx[rng.integers(0, len(idx))])


def posteriors_rf4(internal_cpts, test, rf_soft, rf_hard, est, mode,
                   L_hard=None, train_prior=None, alpha=None):
    """Threat posteriors with the RF channel realized by real CNN outputs.
    mode 'abstract'/'calib-hard': hard report through L_hard[c,e,report].
    mode 'calib-soft': scaled-likelihood virtual evidence
    L(c,e) = sum_r P_gen(r|c,e) * s_r / pi_r with the deterministic generator
    mapping (emitting drone: uniform over models; else background)."""
    gt = ms.ground_truth_bn()
    cpts = dict(gt.cpts)
    parents = {k: list(v) for k, v in ms.PARENTS.items()}
    for k in ms.INTERNAL:
        cpts[k] = internal_cpts[k]
    for s in ms.SENSORS:
        cpts.pop(s, None); parents.pop(s, None)
    for s in ["Rr", "Re", "Ra"]:
        c = est[s]
        if alpha is not None:
            pass  # discounting applied per-d below via cache
        cpts[s] = c
        parents[s] = list(c.vars[:-1])
    card = {k: ms.CARD[k] for k in cpts}
    base_bn = BayesNet(card, {k: parents[k] for k in cpts}, cpts)
    cache = {}
    P = np.zeros((len(test), ms.CARD["T"]))
    y = np.zeros(len(test), dtype=int)
    for i, scn in enumerate(test):
        dv = scn["d"]
        if dv not in cache:
            if alpha is not None:
                c2 = dict(cpts)
                for s in ["Rr", "Re", "Ra"]:
                    c2[s] = dataio._discount_cpt(est[s],
                                                 float(alpha[s][dv]))
                cache[dv] = BayesNet(card, {k: parents[k] for k in cpts}, c2)
            else:
                cache[dv] = base_bn
        bn = cache[dv]
        ev = {k: scn[k] for k in ms.OBSERVED}
        for s in ["Rr", "Re", "Ra"]:
            ev[s] = scn[s]
        if mode == "calib-soft":
            s_vec = rf_soft[i]
            sc = s_vec / np.maximum(train_prior, 1e-9)
            Lce = np.zeros((ms.CARD["cls"], ms.CARD["emit"]))
            for c in range(ms.CARD["cls"]):
                for e in range(ms.CARD["emit"]):
                    if c == 2 and e == 0:
                        Lce[c, e] = np.mean([sc[m] for m in DRONE_CLASSES])
                    else:
                        Lce[c, e] = sc[0]
        else:
            r = rf_hard[i]
            Lce = L_hard[:, :, r]
        P[i] = query_with_ce_virtual(bn, ev, Lce)
        y[i] = scn["T"]
    return P, y


def query_with_ce_virtual(bn, ev, Lce):
    """Query P(T | ev) with an extra likelihood factor over (cls, emit),
    folded in by rewriting the cls/emit roots."""
    joint = np.outer(bn.cpts["cls"].table, bn.cpts["emit"].table) * Lce
    p_emit = joint.sum(axis=0)
    p_cls_g_emit = joint / np.maximum(joint.sum(axis=0, keepdims=True), 1e-300)
    cpts = dict(bn.cpts)
    parents = {k: list(v) for k, v in bn.parents.items()}
    parents["cls"] = ["emit"]
    cpts["cls"] = Factor(("emit", "cls"), p_cls_g_emit.T)
    cpts["emit"] = Factor(("emit",), p_emit / max(p_emit.sum(), 1e-300))
    bn2 = BayesNet(bn.card, parents, cpts)
    return bn2.query(["T"], ev).table


def macro_auc(P, y):
    from sklearn.metrics import roc_auc_score
    try:
        return float(roc_auc_score(np.eye(4)[y], P, multi_class="ovr",
                                   average="macro", labels=[0, 1, 2, 3]))
    except Exception:
        return np.nan


# ---------------------------------------------------------------- study
def run(reps=20):
    if not os.path.exists(CNN_CACHE):
        sys.exit("CNN cache not found; first run\n"
                 "  python exp_rf_e2e.py --train /path/droneRF_four.npz")
    cc = np.load(CNN_CACHE, allow_pickle=True)
    sm_te, y_te = cc["test_softmax"], cc["test_y"]
    sm_va, y_va = cc["val_softmax"], cc["val_y"]
    # The CNN is trained with class-weighted cross-entropy, so its effective
    # training prior is (near-)uniform; dividing by the empirical class
    # frequencies would double-correct. Use the uniform prior for the
    # scaled-likelihood conversion.
    train_prior = np.full(4, 0.25)
    pools_te = {c: np.where(y_te == c)[0] for c in range(4)}
    pools_va = {c: np.where(y_va == c)[0] for c in range(4)}
    gt = ms.ground_truth_bn()
    rows = []
    t0 = time.time()
    for r in range(reps):
        rng = np.random.default_rng(41000 + r)
        test = dataio.generate_dataset(gt, TEST_N, seed=9000 + r)
        calib_syn = dataio.generate_dataset(gt, 300, seed=5000 + r)
        est = dataio.estimate_confusion(calib_syn)
        alpha, _ = dataio.fit_reliability(calib_syn)
        train = dataio.generate_dataset(gt, GROUND_N, seed=r)
        cpts, _ = pl.ground_internal("Proposed", train, r, gt=gt)
        # realize RF channel for the TEST scenarios from the TEST pool
        rf_soft = np.zeros((len(test), 4))
        rf_hard = np.zeros(len(test), dtype=int)
        for i, scn in enumerate(test):
            c = map_class(scn, rng)
            j = draw_pool_index(pools_te, c, rng)
            rf_soft[i] = sm_te[j]
            rf_hard[i] = int(sm_te[j].argmax())
        # deployed calibration split: mixed-world scenarios whose RF samples
        # come from the VALIDATION pool (disjoint from the test pool)
        cnt = np.zeros((ms.CARD["cls"], ms.CARD["emit"], 4)) + 0.5
        for scn in calib_syn:
            c = map_class(scn, rng)
            j = draw_pool_index(pools_va, c, rng)
            rhat = int(sm_va[j].argmax())
            cnt[scn["cls"], scn["emit"], rhat] += 1
        L_calib = cnt / cnt.sum(axis=-1, keepdims=True)
        variants = [
            ("abstract",   "abstract",   rf4_abstract_L()),
            ("calib-hard", "calib-hard", L_calib),
            ("calib-soft", "calib-soft", None),
        ]
        for name, mode, Lh in variants:
            P, y = posteriors_rf4(cpts, test, rf_soft, rf_hard, est, mode,
                                  L_hard=Lh, train_prior=train_prior,
                                  alpha=alpha)
            rows.append({"rep": r, "rf_likelihood": name,
                         "acc": float((P.argmax(1) == y).mean()),
                         "macroAUC": macro_auc(P, y), "ece": mt.ece(P, y),
                         "brier": mt.brier(P, y),
                         "cwece": mt.classwise_ece(P, y)})
        print(f"[expG] rep {r+1}/{reps} ({time.time()-t0:.0f}s)", flush=True)
        pd.DataFrame(rows).to_csv(os.path.join(RES, "expG_rf_e2e_raw.csv"),
                                  index=False)
    summarize()
    print(f"[expG] ALL DONE in {time.time()-t0:.0f}s", flush=True)


def summarize():
    from scipy import stats as st
    cc = np.load(CNN_CACHE, allow_pickle=True)
    df = pd.read_csv(os.path.join(RES, "expG_rf_e2e_raw.csv"))
    reps = df.rep.nunique()
    lines = [f"# Exp-G v2: real-sample end-to-end RF study (reps={reps})", "",
             f"- CNN (recording-level split): val_acc {float(cc['val_acc']):.4f}, "
             f"test_acc {float(cc['test_acc']):.4f}"]

    def ci(d, k):
        x = d[k].values
        h = x.std(ddof=1) / np.sqrt(len(x)) * st.t.ppf(0.975, len(x) - 1)
        return x.mean(), h
    for name in ["abstract", "calib-hard", "calib-soft"]:
        d = df[df.rf_likelihood == name]
        parts = []
        for k in ["acc", "macroAUC", "ece", "cwece"]:
            m, h = ci(d, k)
            parts.append(f"{k} {m:.3f}+-{h:.3f}")
        lines.append(f"- {name}: " + ", ".join(parts))
    for a, b, lab in [("calib-hard", "abstract", "measured vs abstract"),
                      ("calib-soft", "calib-hard", "virtual soft vs hard")]:
        da = df[df.rf_likelihood == a].set_index("rep")
        db = df[df.rf_likelihood == b].set_index("rep")
        for k in ["macroAUC", "ece"]:
            delta = (da[k] - db[k]).values
            tt = st.ttest_1samp(delta, 0.0)
            lines.append(f"- {lab} d{k}: {delta.mean():+.4f} (t p={tt.pvalue:.1e})")
    txt = "\n".join(lines)
    with open(os.path.join(RES, "expG_summary.md"), "w", encoding="utf-8") as f:
        f.write(txt + "\n")
    print(txt.encode(sys.stdout.encoding or "utf-8", errors="replace")
             .decode(sys.stdout.encoding or "utf-8", errors="replace"))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", metavar="NPZ")
    ap.add_argument("--reps", type=int, default=20)
    ap.add_argument("--summary-only", action="store_true")
    a = ap.parse_args()
    if a.train:
        train_and_cache(a.train)
    elif a.summary_only:
        summarize()
    else:
        run(a.reps)
