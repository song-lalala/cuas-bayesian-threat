"""
Exp-I: the RF channel on independent recordings, and real-data high-threat
calibration. Core files and
exp_rf_e2e.py are not modified (its functions are imported); results go to
results/expI_*. The data files are only read; the DroneDetect spectrogram
cache is written to a folder outside this repository (--cache).

Data and model
  DroneRF: droneRF_four.npz (see README "DroneRF data"; path from --dronerf
    or the DRONERF_NPZ environment variable, default ./droneRF_four.npz),
    4,540 chunks (128x128 log-spectrograms) from 227 segments of ten
    continuous captures (BUI = drone x mode); group of a chunk =
    "BUI_segment" (H/L band files together), per-chunk z-norm.
  DroneDetect V2 (Swinney and Woods, IEEE DataPort 2021,
    doi:10.21227/5jjj-1m32; file DroneDetect_V2.zip): 390 recordings, the
    first ten 10^6-sample complex64 chunks of each; seven drone models,
    conditions CLEAN/BLUE/WIFI/BOTH, no background class. Spectrograms:
    two-sided complex STFT (n_fft 512, hop 256, Hann), 20 log10(|z|/sum w +
    1e-8), antialiased bilinear resize to 128x128, per-chunk z-norm, built
    once from the zip into the cache folder (dronedetect_spec128.npy, 0.26
    GB, and dronedetect_meta.npz; command below).
  Classifier: the compact CNN of Sec. VII-C (rf_confusion.CNN) with the
    recipe of exp_rf_e2e.train_and_cache: 35 epochs, Adam (lr 1e-3, weight
    decay 1e-4), cosine schedule, batch 64, class-weighted cross-entropy,
    checkpoint by validation accuracy; torch seed 42 and 4 torch threads,
    both set at the start of every training task.
  Splits: group_split, time_block_split and cap_train_groups are copied from
    the split module (src/splits.py) of the author's DroneRF benchmark
    repository drone-rf-id (github.com/song-lalala/drone-rf-id), called T3 below,
    and the DroneDetect recording split from its
    src/dronedetect_v2.py; nothing is imported from that repository.
    SPLIT_SEEDS = (42, 123, 2024, 7, 99). --check-t3 <drone-rf-id folder>
    optionally compares the copies with the original module (35 split
    comparisons) and the DroneDetect metadata with the original's.

FIXED BEFORE THE RUN (in this docstring; not registered externally)

[P] DroneRF protocols (4 classes).
  P0  the manuscript's split (exp_rf_e2e: segment split 60/15/25, seed 42);
      its CNN outputs are results/expG_cnn.npz (not retrained).
  P1  segment-group split (T3 group_split, 70/15/15), five split seeds:
      segments of one capture fall on both sides (within-recording).
  P2  time-block split (T3 time_block_split, guard 1, 70/15/15), five
      seeds: contiguous blocks of each capture, one guard segment dropped at
      every block boundary.
  P1c P1 with its training set reduced to P2's training size for the same
      seed (T3 cap_train_groups, seeded by the split seed): size control.
  P3  leave-one-flight-mode-out for Bebop and AR, four folds (mode code 00,
      01, 10, 11 of the BUI): both classes' captures in that mode are the
      test set (independent recordings); their other three modes give
      train/validation by time-block split (85/15, guard 1, seed of the
      fold = SPLIT_SEEDS[fold]). Background and Phantom have one capture each,
      so they cannot be held out by recording; they are split by time block
      (70/15/15, guard 1, same seed) and stay within-capture.
  Node metrics on each test set: accuracy, macro-F1, per-class recall,
  confusion matrix; for P3 also the Bebop+AR accuracy on the held-out modes.

[N] Node-level verdict ("within-recording optimism").
  d = acc(P1) - acc(P2), paired over the five seeds; the same for
  d_c = acc(P1c) - acc(P2) (size controlled). "large" if mean d > 0.05 and
  paired t-test p < 0.05; "small" if mean d <= 0.02 or p >= 0.05;
  "moderate" otherwise. For P3: Bebop+AR accuracy under P2 (time-block test,
  five seeds) minus under P3 (held-out modes, four folds), Welch t-test,
  same thresholds.

[E] Threat level (mixed world of Sec. VII-H, exp_rf_e2e.run unchanged: 20
  replications, scenario seeds 9000+r, calibration 5000+r, training r, RF
  draws rng(41000+r); variants abstract, calib-hard, calib-soft) run with
  each protocol's CNN outputs; replication r uses split r mod 5 (P1, P1c,
  P2) or fold r mod 4 (P3); P0 must reproduce expG_rf_e2e_raw.csv bit for
  bit. An added CPT-oracle variant uses the true internal CPTs, the true
  range-marginal radar/EO-IR/acoustic tables (no discount) and, for RF, the
  CNN's empirical confusion on that split's test pool (hard report).
  (E1) The Sec. VII-H claim holds under a protocol if calib-hard minus
       abstract has macro-AUC > 0 and ECE < 0, paired t-test p < 0.05.
  (E2) Within-recording optimism propagates materially if, for calib-hard,
       metric(P1) - metric(P2) paired by replication has |d AUC| >= 0.010 or
       |d ECE| >= 0.0165 (half of the +0.020 / -0.033 of Sec. VII-H) with
       p < 0.05; the same for P0 - P2 and P0 - P3.

[H] Real-data high-threat reliability (Table 9 format): P(T = high) of
  calib-hard and of the CPT oracle pooled over the 20 replications, bins
  (0,.10], (.10,.25], (.25,.50], (.50,.75], (.75,1]; mean predicted,
  observed rate, 95% Clopper-Pearson interval, bins whose prediction lies
  outside the interval. Operational: an alarm at P(T = high) >= 0.5,
  precision and recall pooled over the replications. Descriptive.

[DD] DroneDetect (7 classes; independent recordings for every class).
  (a) T3's recording-level CLEAN split (70/15/15 stratified by drone,
      np.random.RandomState(seed)), five seeds; trained on CLEAN only.
      Evaluated on the CLEAN test recordings and on all BLUE, WIFI, BOTH
      recordings: chunk accuracy, macro-F1, confusion, recording-level
      majority-vote accuracy.
  (b) Reliability with the interference condition as context. Recordings
      not used for training (CLEAN validation and test; all of BLUE, WIFI,
      BOTH) are halved per (condition, drone) into calibration and
      evaluation recordings with np.random.RandomState(seed + 1) (calibration
      gets floor(n/2)). On the calibration half: logistic regression (C =
      10, as dataio.fit_reliability) of chunk correctness on the condition
      one-hot (CLEAN reference), p(cond) its fitted probability, Eq. (10)
      alpha = clip(0.98 p / max p, 0.05, 0.98); Eq. (11)'s chance-corrected
      omega = (p - 1/7)/(max p - 1/7) reported for reference. Transfer:
      context {clean, interference}, fitted on CLEAN plus one interference
      condition and applied to the other two. Score on the evaluation half:
      mean log-likelihood (nats) of the observed CNN report under the
      discounted CLEAN confusion matrix (calibration half, smoothing 0.5)
      for alpha = 1 (no discount), alpha own (condition fitted on its own
      calibration half) and alpha transferred. "Useful" if own alpha beats
      alpha = 1 in that condition; "transfers" if the transferred alpha
      beats alpha = 1 in the held-out condition (mean over seeds, paired
      t-test p < 0.05 over the five seeds).
  (c) No end-to-end run with DroneDetect: in the network the RF evidence
      enters through (cls, emit), so which of seven drone models is reported
      cannot change P(T); what can, detection of emission against no
      emission, is what DroneDetect cannot measure (no background
      recordings), and borrowing a false-alarm rate from DroneRF would mix
      receivers (BladeRF against USRP, different bands).

Run (CACHE: a folder outside this repository; the CUAS_RF_CACHE environment
variable can replace --cache):
  1) once, about 2 min: DroneDetect spectrogram cache from the zip
     python exp_rf_independent.py --cache CACHE --build-dd-zip path/DroneDetect_V2.zip
  2) 24 CNNs, threat level, DroneDetect reliability, summary (about 60 min
     on 7 worker processes of 4 threads)
     python exp_rf_independent.py --cache CACHE --dronerf path/droneRF_four.npz --workers 7
  3) from the saved CNN outputs only (no data files needed)
     python exp_rf_independent.py --only e2e,dd
     python exp_rf_independent.py --summary-only
  optional: add --check-t3 path/drone-rf-id to step 2
"""
from __future__ import annotations
import os
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "4")
import sys, time, argparse, re
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
P = "expI"
DRONERF_NPZ = os.environ.get("DRONERF_NPZ", "droneRF_four.npz")
DD_SPEC = "dronedetect_spec128.npy"         # in the cache folder
DD_META = "dronedetect_meta.npz"            # in the cache folder
SPLIT_SEEDS = (42, 123, 2024, 7, 99)
TORCH_SEED = 42
THREADS = 4
REPS = 20
BINS = [0.0, 0.10, 0.25, 0.50, 0.75, 1.0]
DD_CHUNK = 1_000_000
DD_PER_REC = 10                             # chunks per recording
DD_N = 3900
DD_CLASSES = ("AIR", "DIS", "INS", "MIN", "MP1", "MP2", "PHA")
MODES = ("00", "01", "10", "11")


# ----------------------------------------------------- splits (copied) ----
# group_split, cap_train_groups and time_block_split are copies of the
# functions of the same names in src/splits.py of the drone-rf-id repository
# (github.com/song-lalala/drone-rf-id). dd_split below copies split() in that
# repository's src/dronedetect_v2.py. Option --check-t3 compares them with
# the originals; nothing else depends on that repository.
def group_split(groups, y, seed, ratios=(0.7, 0.15, 0.15)):
    """drone-rf-id src/splits.py group_split (copy)."""
    uniq, ginv = np.unique(groups, return_inverse=True)
    grp_label = np.full(len(uniq), -1, dtype=np.int64)
    for g in range(len(uniq)):
        labs = np.unique(y[ginv == g])
        assert len(labs) == 1
        grp_label[g] = labs[0]
    rng = np.random.RandomState(seed)
    tr_g, va_g, te_g = [], [], []
    for c in np.unique(grp_label):
        gidx = np.where(grp_label == c)[0]
        rng.shuffle(gidx)
        n = len(gidx)
        n_tr = int(round(n * ratios[0]))
        n_va = int(round(n * ratios[1]))
        tr_g += gidx[:n_tr].tolist()
        va_g += gidx[n_tr:n_tr + n_va].tolist()
        te_g += gidx[n_tr + n_va:].tolist()

    def to_idx(gl):
        return np.where(np.isin(ginv, gl))[0]
    return to_idx(tr_g), to_idx(va_g), to_idx(te_g)


def cap_train_groups(tr, groups, y, cap, seed):
    """drone-rf-id src/splits.py cap_train_groups (copy)."""
    if cap <= 0 or len(tr) <= cap:
        return tr
    g_tr = groups[tr]
    uniq = np.unique(g_tr)
    size = {g: int((g_tr == g).sum()) for g in uniq}
    lab = {g: int(y[tr][g_tr == g][0]) for g in uniq}
    per_chunk = len(tr) / len(uniq)
    n_drop = int(np.ceil((len(tr) - cap) / per_chunk))
    by_class = {c: np.array(sorted(g for g in uniq if lab[g] == c))
                for c in sorted({lab[g] for g in uniq})}
    share = {c: n_drop * len(v) / len(uniq) for c, v in by_class.items()}
    take = {c: int(np.floor(s)) for c, s in share.items()}
    for c in sorted(share, key=lambda k: (-(share[k] - take[k]), k)):
        if sum(take.values()) >= n_drop:
            break
        take[c] += 1
    rng = np.random.RandomState(seed)
    drop = set()
    for c, gs in by_class.items():
        gs = gs.copy()
        rng.shuffle(gs)
        drop.update(gs[:take[c]].tolist())
    keep = tr[~np.isin(g_tr, list(drop))]
    assert len(keep) <= cap
    assert len(keep) == len(tr) - sum(size[g] for g in drop)
    return keep


def time_block_split(groups, y, seed, ratios=(0.7, 0.15, 0.15), guard=1):
    """drone-rf-id src/splits.py time_block_split (copy)."""
    uniq, ginv = np.unique(groups, return_inverse=True)
    bui = np.array([g.split("_")[0] for g in uniq])
    seg = np.array([int(g.split("_")[1]) for g in uniq])
    rng = np.random.RandomState(seed)
    assign = {}
    for b in np.unique(bui):
        gsel = np.where(bui == b)[0]
        order_time = gsel[np.argsort(seg[gsel])]
        n = len(order_time)
        rot = rng.randint(n)
        rolled = np.roll(order_time, -rot)
        n_tr = int(round(n * ratios[0]))
        n_va = int(round(n * ratios[1]))
        for gi in rolled[:n_tr]:
            assign[gi] = "tr"
        for gi in rolled[n_tr:n_tr + n_va]:
            assign[gi] = "va"
        for gi in rolled[n_tr + n_va:]:
            assign[gi] = "te"
        orig = [assign[gi] for gi in order_time]
        for i in range(n - 1):
            if orig[i] != orig[i + 1]:
                for k in range(guard):
                    if i + 1 + k < n:
                        assign[order_time[i + 1 + k]] = "drop"
    sets = {"tr": [], "va": [], "te": []}
    for gi, tag in assign.items():
        if tag in sets:
            sets[tag] += np.where(ginv == gi)[0].tolist()
    return (np.array(sorted(sets["tr"]), dtype=np.int64),
            np.array(sorted(sets["va"]), dtype=np.int64),
            np.array(sorted(sets["te"]), dtype=np.int64))


def check_against_t3(t3_dir, cache):
    """Optional check (--check-t3): import drone-rf-id's src/splits.py
    read-only (no bytecode written into that folder) and compare every
    DroneRF split used here; if both exist, compare the DroneDetect metadata
    in the cache with that repository's data/processed/dronedetect_meta.npz."""
    import importlib.util
    X, y, groups = load_dronerf()
    del X
    sys.dont_write_bytecode = True
    spec = importlib.util.spec_from_file_location(
        "t3_splits", os.path.join(t3_dir, "src", "splits.py"))
    t3 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(t3)
    assert tuple(t3.SPLIT_SEEDS) == SPLIT_SEEDS
    n = 0
    for s in SPLIT_SEEDS:
        for a, b in zip(group_split(groups, y, s), t3.group_split(groups, y, s)):
            assert np.array_equal(a, b); n += 1
        tb = time_block_split(groups, y, s)
        for a, b in zip(tb, t3.time_block_split(groups, y, s)):
            assert np.array_equal(a, b); n += 1
        tr1 = group_split(groups, y, s)[0]
        assert np.array_equal(cap_train_groups(tr1, groups, y, len(tb[0]), s),
                              t3.cap_train_groups(tr1, groups, y, len(tb[0]), s))
        n += 1
    assert n == 35
    print(f"[expI check-t3] split functions identical to drone-rf-id's: {n} comparisons",
          flush=True)
    m_t3 = os.path.join(t3_dir, "data", "processed", "dronedetect_meta.npz")
    m_mine = os.path.join(cache, DD_META) if cache else ""
    if os.path.exists(m_t3) and m_mine and os.path.exists(m_mine):
        a, b = np.load(m_t3), np.load(m_mine)
        assert sorted(a.files) == sorted(b.files)
        for k in a.files:
            assert np.array_equal(a[k], b[k]), k
        print(f"[expI check-t3] DroneDetect metadata identical: {len(a.files)} arrays, "
              f"{len(b['y'])} chunks", flush=True)


# ------------------------------------------------------------- DroneRF ----
def session_key(fname):
    m = re.match(r"^(\d+)[HL]_(\d+)\.csv$", fname)
    return f"{m.group(1)}_{m.group(2)}" if m else fname


def load_dronerf():
    d = np.load(DRONERF_NPZ, allow_pickle=True)
    X = d["X"].astype(np.float32)
    X = (X - X.mean(axis=(1, 2), keepdims=True)) / \
        (X.std(axis=(1, 2), keepdims=True) + 1e-6)
    y = d["y"].astype(np.int64)
    groups = np.array([session_key(m[0]) for m in d["meta"]])
    return X, y, groups


def lofmo_split(groups, y, fold):
    """P3: leave one flight mode out for Bebop (BUI 100mm) and AR (101mm)."""
    m = MODES[fold]
    seed = SPLIT_SEEDS[fold]
    bui = np.array([g.split("_")[0] for g in groups])
    test_bui = {"100" + m, "101" + m}
    te = np.where(np.isin(bui, list(test_bui)))[0]
    ba_train = np.where(((y == 1) | (y == 2)) & ~np.isin(bui, list(test_bui)))[0]
    tr_a, va_a, te_a = time_block_split(groups[ba_train], y[ba_train], seed,
                                        ratios=(0.85, 0.15, 0.0))
    assert len(te_a) == 0
    other = np.where((y == 0) | (y == 3))[0]
    tr_b, va_b, te_b = time_block_split(groups[other], y[other], seed)
    tr = np.sort(np.concatenate([ba_train[tr_a], other[tr_b]]))
    va = np.sort(np.concatenate([ba_train[va_a], other[va_b]]))
    te = np.sort(np.concatenate([te, other[te_b]]))
    return tr, va, te


def dronerf_instances(groups, y):
    inst = []
    for s in SPLIT_SEEDS:
        p1 = group_split(groups, y, s)
        p2 = time_block_split(groups, y, s)
        p1c = (cap_train_groups(p1[0], groups, y, len(p2[0]), s), p1[1], p1[2])
        inst += [("P1", s, p1), ("P2", s, p2), ("P1c", s, p1c)]
    for f in range(4):
        inst.append(("P3", f, lofmo_split(groups, y, f)))
    return inst


# ---------------------------------------------------------------- CNN -----
def train_cnn(Xtr, ytr, Xva, yva, n_classes, Xeval):
    """exp_rf_e2e.train_and_cache's recipe; returns (val softmax, eval
    softmax list, best val acc, train prior)."""
    import torch, torch.nn as nn
    from rf_confusion import CNN
    torch.set_num_threads(THREADS)
    torch.manual_seed(TORCH_SEED); np.random.seed(TORCH_SEED)
    Xt = torch.from_numpy(Xtr).unsqueeze(1)
    yt = torch.from_numpy(ytr)
    Xv = torch.from_numpy(Xva).unsqueeze(1)
    net = CNN(n_classes)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=35)
    cnt = np.bincount(ytr, minlength=n_classes).astype(float)
    w = torch.tensor([1.0 / c if c > 0 else 0.0 for c in cnt], dtype=torch.float32)
    lossf = nn.CrossEntropyLoss(weight=w / w.sum() * n_classes)
    bs, best, best_state = 64, -1.0, None
    for ep in range(35):
        net.train(); perm = torch.randperm(len(Xt))
        for i in range(0, len(Xt), bs):
            idx = perm[i:i + bs]; opt.zero_grad()
            lossf(net(Xt[idx]), yt[idx]).backward(); opt.step()
        sched.step(); net.eval()
        with torch.no_grad():
            va = float((net(Xv).argmax(1).numpy() == yva).mean())
        if va > best:
            best, best_state = va, {k: v.clone() for k, v in net.state_dict().items()}
    net.load_state_dict(best_state); net.eval()
    out = []
    with torch.no_grad():
        sm_va = torch.softmax(net(Xv), 1).numpy()
        for Xe in Xeval:
            sm = []
            for i in range(0, len(Xe), 512):
                sm.append(torch.softmax(net(torch.from_numpy(Xe[i:i + 512]).unsqueeze(1)), 1).numpy())
            out.append(np.concatenate(sm) if sm else np.zeros((0, n_classes)))
    return sm_va, out, best, cnt / cnt.sum()


_DATA = {}


def task_dronerf(t):
    proto, key, (tr, va, te) = t
    if "rf" not in _DATA:
        _DATA["rf"] = load_dronerf()
    X, y, groups = _DATA["rf"]
    t0 = time.time()
    sm_va, (sm_te,), best, prior = train_cnn(X[tr], y[tr], X[va], y[va], 4, [X[te]])
    return {"kind": "rf", "proto": proto, "key": key, "tr": tr, "va": va, "te": te,
            "val_softmax": sm_va, "test_softmax": sm_te, "val_acc": best,
            "train_prior": prior, "secs": time.time() - t0}


def dd_split(seed, meta):
    """drone-rf-id src/dronedetect_v2.py split(seed) (copy): recording-level
    CLEAN split."""
    y_all, cond, group = meta["y"], meta["cond"], meta["group"]
    clean_rows = np.where(cond == "CLEAN")[0]
    clean_groups = np.unique(group[clean_rows])
    g_drone = {g: y_all[np.where(group == g)[0][0]] for g in clean_groups}
    rng = np.random.RandomState(seed)
    tr_g, va_g, te_g = [], [], []
    for d in range(len(DD_CLASSES)):
        gs = np.array(sorted(g for g in clean_groups if g_drone[g] == d))
        rng.shuffle(gs)
        n_tr = int(round(len(gs) * 0.7))
        n_va = int(round(len(gs) * 0.15))
        tr_g += gs[:n_tr].tolist()
        va_g += gs[n_tr:n_tr + n_va].tolist()
        te_g += gs[n_tr + n_va:].tolist()

    def rows(gl):
        return clean_rows[np.isin(group[clean_rows], gl)]
    return rows(tr_g), rows(va_g), rows(te_g)


def task_dd(t):
    _, seed, cache = t
    meta = np.load(os.path.join(cache, DD_META))
    S = np.load(os.path.join(cache, DD_SPEC), mmap_mode="r")
    y = meta["y"]
    tr, va, te = dd_split(seed, meta)
    rest = np.setdiff1d(np.arange(DD_N), tr)
    t0 = time.time()
    sm_va, (sm_rest,), best, prior = train_cnn(np.asarray(S[tr]), y[tr], np.asarray(S[va]),
                                               y[va], 7, [np.asarray(S[rest])])
    return {"kind": "dd", "proto": "DD", "key": seed, "tr": tr, "va": va, "te": te,
            "rest": rest, "rest_softmax": sm_rest, "val_acc": best,
            "train_prior": prior, "secs": time.time() - t0}


def run_task(t):
    return task_dronerf(t) if t[0] != "DD" else task_dd(t)


# ------------------------------------------------------ DroneDetect spec --
def _read_recording(zip_path, member):
    """First DD_PER_REC chunks of one recording (float32-interleaved IQ)."""
    import zipfile
    nbytes = DD_PER_REC * DD_CHUNK * 8
    with zipfile.ZipFile(zip_path) as zf, zf.open(member) as f:
        buf = f.read(nbytes)
    assert len(buf) == nbytes, member
    return np.frombuffer(buf, dtype=np.float32).view(np.complex64).reshape(DD_PER_REC, DD_CHUNK)


def build_dd_cache(zip_path, cache):
    """Spectrogram cache and metadata from DroneDetect_V2.zip: members in
    sorted order, metadata from the member path DroneDetect_V2/COND/DR_MODE/
    stem.dat (as drone-rf-id's src/dronedetect_v2.py build), spectrograms in
    batches of 50 chunks with 8 torch threads."""
    import zipfile
    from collections import deque
    from concurrent.futures import ThreadPoolExecutor
    import torch, torch.nn.functional as F
    out, meta_p = os.path.join(cache, DD_SPEC), os.path.join(cache, DD_META)
    if os.path.exists(out) or os.path.exists(meta_p):
        print(f"[expI dd-cache] {cache} already holds a cache; not rebuilt", flush=True)
        return
    with zipfile.ZipFile(zip_path) as zf:
        members = sorted(n for n in zf.namelist() if n.endswith(".dat"))
    assert len(members) * DD_PER_REC == DD_N, len(members)
    cond, drone, mode, group = [], [], [], []
    for m in members:
        parts = m.split("/")
        c, dm, fn = parts[1], parts[2], parts[3]
        d, md = dm.split("_")
        for _ in range(DD_PER_REC):
            cond.append(c); drone.append(d); mode.append(md); group.append(fn[:-4])
    y = np.array([DD_CLASSES.index(d) for d in drone], dtype=np.int64)
    torch.set_num_threads(8)
    win = torch.hann_window(512, periodic=True)
    ws = float(win.sum())
    S = np.lib.format.open_memmap(out + ".part", mode="w+", dtype=np.float32,
                                  shape=(DD_N, 128, 128))
    per = 50 // DD_PER_REC
    t0 = time.time()
    with ThreadPoolExecutor(4) as ex:
        todo, futs = iter(members), deque()

        def fill():
            while len(futs) < 3 * per:
                m = next(todo, None)
                if m is None:
                    return
                futs.append(ex.submit(_read_recording, zip_path, m))
        fill()
        for i in range(0, DD_N, 50):
            xs = []
            for _ in range(per):
                xs.append(futs.popleft().result()); fill()
            x = torch.from_numpy(np.concatenate(xs))
            z = torch.stft(x, n_fft=512, hop_length=256, win_length=512, window=win,
                           center=False, onesided=False, return_complex=True)
            s = 20.0 * torch.log10(z.abs() / ws + 1e-8)
            s = F.interpolate(s.unsqueeze(1), size=(128, 128), mode="bilinear",
                              align_corners=False, antialias=True)
            m = s.mean(dim=(1, 2, 3), keepdim=True)
            sd = s.std(dim=(1, 2, 3), keepdim=True) + 1e-8
            S[i:i + 50] = ((s - m) / sd)[:, 0].numpy()
            if i % 500 == 0:
                print(f"[expI dd-cache] {i}/{DD_N} ({time.time()-t0:.0f}s)", flush=True)
    S.flush(); del S
    with open(meta_p + ".part", "wb") as f:
        np.savez(f, cond=np.array(cond), drone=np.array(drone), mode=np.array(mode),
                 group=np.array(group), y=y, class_names=np.array(DD_CLASSES))
    os.replace(out + ".part", out)
    os.replace(meta_p + ".part", meta_p)
    print(f"[expI dd-cache] built {out} and {meta_p} ({time.time()-t0:.0f}s)", flush=True)


# ------------------------------------------------------------ metrics -----
def node_metrics(sm, yt, k):
    pred = sm.argmax(1)
    conf = np.zeros((k, k), dtype=int)
    np.add.at(conf, (yt, pred), 1)
    tp = np.diag(conf).astype(float)
    col, row = conf.sum(0).astype(float), conf.sum(1).astype(float)
    prec = np.divide(tp, col, out=np.zeros_like(tp), where=col > 0)
    rec = np.divide(tp, row, out=np.zeros_like(tp), where=row > 0)
    f1 = np.divide(2 * prec * rec, prec + rec, out=np.zeros_like(tp), where=(prec + rec) > 0)
    return {"acc": float(tp.sum() / conf.sum()), "macroF1": float(f1.mean()),
            "recall": rec, "confusion": conf}


def cp_interval(k, n):
    from scipy import stats as st
    lo = st.beta.ppf(0.025, k, n - k + 1) if k > 0 else 0.0
    hi = st.beta.ppf(0.975, k + 1, n - k) if k < n else 1.0
    return float(lo), float(hi)


# --------------------------------------------------------------- study ----
def run_cnns(cache, workers):
    from multiprocessing import Pool
    for f in (DD_SPEC, DD_META):
        assert os.path.exists(os.path.join(cache, f)), \
            f"{f} not in {cache}; build it first with --build-dd-zip (see docstring)"
    X, y, groups = load_dronerf()
    inst = dronerf_instances(groups, y)
    tasks = inst + [("DD", s, cache) for s in SPLIT_SEEDS]
    del X
    t0 = time.time()
    outs = []
    with Pool(workers) as pool:
        for i, o in enumerate(pool.imap_unordered(run_task, tasks), 1):
            outs.append(o)
            print(f"[expI cnn] {i}/{len(tasks)} {o['proto']} {o['key']} val {o['val_acc']:.3f} "
                  f"({o['secs']:.0f}s; {time.time()-t0:.0f}s total)", flush=True)
    order = {t[:2]: i for i, t in enumerate(tasks)}     # saved in task order
    outs.sort(key=lambda o: order[(o["proto"], o["key"])])
    rf = {}
    for o in outs:
        if o["kind"] == "rf":
            k = f"{o['proto']}_{o['key']}"
            for f in ("tr", "va", "te", "val_softmax", "test_softmax", "train_prior"):
                rf[f"{k}__{f}"] = o[f]
            rf[f"{k}__val_acc"] = np.array(o["val_acc"])
            rf[f"{k}__va_y"], rf[f"{k}__te_y"] = y[o["va"]], y[o["te"]]
    np.savez_compressed(os.path.join(RES, f"{P}_cnn_dronerf.npz"), **rf)
    meta = np.load(os.path.join(cache, DD_META))
    dd = {f"meta__{f}": meta[f] for f in ("y", "cond", "group")}
    for o in outs:
        if o["kind"] == "dd":
            k = f"DD_{o['key']}"
            for f in ("tr", "va", "te", "rest", "rest_softmax", "train_prior"):
                dd[f"{k}__{f}"] = o[f]
            dd[f"{k}__val_acc"] = np.array(o["val_acc"])
    np.savez_compressed(os.path.join(RES, f"{P}_cnn_dronedetect.npz"), **dd)


def rf_instances_loaded():
    z = np.load(os.path.join(RES, f"{P}_cnn_dronerf.npz"))
    out = {}
    for k in z.files:
        inst, f = k.split("__")
        out.setdefault(inst, {})[f] = z[k]
    return out


def node_table():
    X, y, groups = load_dronerf()
    del X
    rf = rf_instances_loaded()
    rows = []
    cc = np.load(os.path.join(RES, "expG_cnn.npz"), allow_pickle=True)
    m = node_metrics(cc["test_softmax"], cc["test_y"], 4)
    import exp_rf_e2e as eg
    files = np.array([mm[0] for mm in np.load(DRONERF_NPZ, allow_pickle=True)["meta"]])
    sp0 = eg.group_split(files, y, np.random.default_rng(eg.SEED))
    assert (sp0 == "test").sum() == len(cc["test_y"]) and (sp0 == "val").sum() == len(cc["val_y"])
    rows.append({"proto": "P0", "key": 42, "n_train": int((sp0 == "train").sum()),
                 "n_val": len(cc["val_y"]),
                 "n_test": len(cc["test_y"]), "acc": m["acc"], "macroF1": m["macroF1"],
                 **{f"recall_{c}": m["recall"][c] for c in range(4)},
                 "acc_bebop_ar": float((cc["test_softmax"].argmax(1) == cc["test_y"])[np.isin(cc["test_y"], [1, 2])].mean()),
                 "confusion": " ".join(map(str, m["confusion"].ravel()))})
    for inst, d in rf.items():
        proto, key = inst.split("_")
        te = d["te"]
        m = node_metrics(d["test_softmax"], y[te], 4)
        ba = np.isin(y[te], [1, 2])
        rows.append({"proto": proto, "key": int(key), "n_train": len(d["tr"]),
                     "n_val": len(d["va"]), "n_test": len(te), "acc": m["acc"],
                     "macroF1": m["macroF1"], **{f"recall_{c}": m["recall"][c] for c in range(4)},
                     "acc_bebop_ar": float((d["test_softmax"].argmax(1) == y[te])[ba].mean()),
                     "confusion": " ".join(map(str, m["confusion"].ravel()))})
    df = pd.DataFrame(rows).sort_values(["proto", "key"])
    df.to_csv(os.path.join(RES, f"{P}_node_dronerf.csv"), index=False)
    return df


def e2e_protocol(proto, sources, reps):
    """exp_rf_e2e.run with the protocol's CNN outputs (sources: list of
    (val_softmax, val_y, test_softmax, test_y)), plus the CPT oracle."""
    import exp_rf_e2e as eg
    import model_spec as ms, dataio, metrics as mt, pipeline as pl
    gt = ms.ground_truth_bn()
    gt_int = {k: gt.cpts[k] for k in ms.INTERNAL}
    true_marg = dataio.true_marginal_confusion()
    train_prior = np.full(4, 0.25)
    rows, bins_rows, alarm_rows = [], [], []
    for r in range(reps):
        sm_va, y_va, sm_te, y_te = sources[r % len(sources)]
        pools_te = {c: np.where(y_te == c)[0] for c in range(4)}
        pools_va = {c: np.where(y_va == c)[0] for c in range(4)}
        rng = np.random.default_rng(41000 + r)
        test = dataio.generate_dataset(gt, eg.TEST_N, seed=9000 + r)
        calib_syn = dataio.generate_dataset(gt, 300, seed=5000 + r)
        est = dataio.estimate_confusion(calib_syn)
        alpha, _ = dataio.fit_reliability(calib_syn)
        train = dataio.generate_dataset(gt, eg.GROUND_N, seed=r)
        cpts, _ = pl.ground_internal("Proposed", train, r, gt=gt)
        rf_soft = np.zeros((len(test), 4))
        rf_hard = np.zeros(len(test), dtype=int)
        for i, scn in enumerate(test):
            c = eg.map_class(scn, rng)
            j = eg.draw_pool_index(pools_te, c, rng)
            rf_soft[i] = sm_te[j]
            rf_hard[i] = int(sm_te[j].argmax())
        cnt = np.zeros((ms.CARD["cls"], ms.CARD["emit"], 4)) + 0.5
        for scn in calib_syn:
            c = eg.map_class(scn, rng)
            j = eg.draw_pool_index(pools_va, c, rng)
            cnt[scn["cls"], scn["emit"], int(sm_va[j].argmax())] += 1
        L_calib = cnt / cnt.sum(axis=-1, keepdims=True)
        conf_te = np.zeros((4, 4))
        np.add.at(conf_te, (y_te, sm_te.argmax(1)), 1)
        row_te = conf_te / conf_te.sum(axis=1, keepdims=True)
        L_or = np.zeros((ms.CARD["cls"], ms.CARD["emit"], 4))
        for c in range(ms.CARD["cls"]):
            for e in range(ms.CARD["emit"]):
                L_or[c, e] = (row_te[eg.DRONE_CLASSES].mean(axis=0)
                              if (c == 2 and e == 0) else row_te[0])
        variants = [("abstract", cpts, est, "abstract", eg.rf4_abstract_L(), alpha),
                    ("calib-hard", cpts, est, "calib-hard", L_calib, alpha),
                    ("calib-soft", cpts, est, "calib-soft", None, alpha),
                    ("CPT-oracle", gt_int, true_marg, "calib-hard", L_or, None)]
        for name, cp, es_, mode, Lh, al in variants:
            Pm, y = eg.posteriors_rf4(cp, test, rf_soft, rf_hard, es_, mode, L_hard=Lh,
                                      train_prior=train_prior, alpha=al)
            rows.append({"proto": proto, "rep": r, "rf_likelihood": name,
                         "acc": float((Pm.argmax(1) == y).mean()),
                         "macroAUC": eg.macro_auc(Pm, y), "ece": mt.ece(Pm, y),
                         "brier": mt.brier(Pm, y), "cwece": mt.classwise_ece(Pm, y)})
            if name in ("calib-hard", "CPT-oracle"):
                ph, hit = Pm[:, 3], (y == 3)
                for lo, hi in zip(BINS[:-1], BINS[1:]):
                    msk = (ph > lo) & (ph <= hi)        # Table 9 convention
                    bins_rows.append({"proto": proto, "rep": r, "variant": name,
                                      "bin_lo": lo, "bin_hi": hi, "n": int(msk.sum()),
                                      "sum_pred": float(ph[msk].sum()), "hits": int(hit[msk].sum())})
                al_ = ph >= 0.5
                alarm_rows.append({"proto": proto, "rep": r, "variant": name,
                                   "tp": int((al_ & hit).sum()), "fp": int((al_ & ~hit).sum()),
                                   "fn": int((~al_ & hit).sum()), "tn": int((~al_ & ~hit).sum())})
    return rows, bins_rows, alarm_rows


def run_e2e(reps=REPS):
    rf = rf_instances_loaded()
    cc = np.load(os.path.join(RES, "expG_cnn.npz"), allow_pickle=True)
    srcs = {"P0": [(cc["val_softmax"], cc["val_y"], cc["test_softmax"], cc["test_y"])]}
    for proto, keys in (("P1", SPLIT_SEEDS), ("P1c", SPLIT_SEEDS), ("P2", SPLIT_SEEDS),
                        ("P3", (0, 1, 2, 3))):
        srcs[proto] = [(rf[f"{proto}_{k}"]["val_softmax"], rf[f"{proto}_{k}"]["va_y"],
                        rf[f"{proto}_{k}"]["test_softmax"], rf[f"{proto}_{k}"]["te_y"])
                       for k in keys]
    R, Bn, A = [], [], []
    t0 = time.time()
    for proto, s in srcs.items():
        r_, b_, a_ = e2e_protocol(proto, s, reps)
        R += r_; Bn += b_; A += a_
        print(f"[expI e2e] {proto} done ({time.time()-t0:.0f}s)", flush=True)
    pd.DataFrame(R).to_csv(os.path.join(RES, f"{P}_e2e_raw.csv"), index=False)
    pd.DataFrame(Bn).to_csv(os.path.join(RES, f"{P}_highthreat_bins.csv"), index=False)
    pd.DataFrame(A).to_csv(os.path.join(RES, f"{P}_alarm.csv"), index=False)


def run_dd_reliability():
    from sklearn.linear_model import LogisticRegression
    z = np.load(os.path.join(RES, f"{P}_cnn_dronedetect.npz"))
    y, cond, group = z["meta__y"], z["meta__cond"], z["meta__group"]
    node, rel = [], []
    conds = ("CLEAN", "BLUE", "WIFI", "BOTH")
    for seed in SPLIT_SEEDS:
        k = f"DD_{seed}"
        tr, va, te, rest = (z[f"{k}__{f}"] for f in ("tr", "va", "te", "rest"))
        sm = z[f"{k}__rest_softmax"]
        pred_all = np.full(DD_N, -1)
        pred_all[rest] = sm.argmax(1)
        # (a) node level
        for c in conds:
            rows_c = te if c == "CLEAN" else np.where(cond == c)[0]
            pr = pred_all[rows_c]
            m = node_metrics(np.eye(7)[pr], y[rows_c], 7)
            gs = group[rows_c]
            votes_ok = []
            for g in np.unique(gs):
                msk = gs == g
                votes_ok.append(np.bincount(pr[msk], minlength=7).argmax() == y[rows_c][msk][0])
            node.append({"seed": seed, "cond": c, "n_chunks": len(rows_c), "acc": m["acc"],
                         "macroF1": m["macroF1"], "rec_acc": float(np.mean(votes_ok)),
                         "n_recordings": len(votes_ok),
                         "confusion": " ".join(map(str, m["confusion"].ravel()))})
        # (b) reliability with the condition as context
        elig = np.concatenate([va, te, np.where(cond != "CLEAN")[0]])
        key = np.char.add(np.char.add(cond.astype(str), "/"), group.astype(str))
        rng = np.random.RandomState(seed + 1)
        cal_g, ev_g = [], []
        for c in conds:
            for d in range(7):
                gs = np.array(sorted(set(key[elig][(cond[elig] == c) & (y[elig] == d)])))
                rng.shuffle(gs)
                h = len(gs) // 2
                cal_g += gs[:h].tolist(); ev_g += gs[h:].tolist()
        cal = elig[np.isin(key[elig], cal_g)]
        ev = elig[np.isin(key[elig], ev_g)]
        assert len(np.intersect1d(cal, ev)) == 0
        ok = (pred_all == y).astype(float)

        def fit_alpha(rows, ctx_of):
            # the clean context is the reference level (as range "far" is in
            # dataio.fit_reliability)
            levels = sorted(set(ctx_of(cond[rows])),
                            key=lambda v: (v not in ("CLEAN", "clean"), v))
            Xc = np.array([[1.0 if v == lv else 0.0 for lv in levels[1:]] for v in ctx_of(cond[rows])])
            lr = LogisticRegression(C=10.0).fit(Xc, ok[rows])
            p = {lv: float(lr.predict_proba(np.array([[1.0 if lv == l2 else 0.0 for l2 in levels[1:]]]))[0, 1])
                 for lv in levels}
            mx = max(p.values())
            a = {lv: float(np.clip(p[lv] / mx * 0.98, 0.05, 0.98)) for lv in levels}
            om = {lv: float(np.clip((p[lv] - 1 / 7) / (mx - 1 / 7), 0.0, 1.0)) for lv in levels}
            return p, a, om
        p4, a4, om4 = fit_alpha(cal, lambda v: v)
        cl = cal[cond[cal] == "CLEAN"]
        Cm = np.zeros((7, 7)) + 0.5
        np.add.at(Cm, (y[cl], pred_all[cl]), 1)
        Cm = Cm / Cm.sum(axis=1, keepdims=True)

        def ll(rows, a):
            Pd = a * Cm + (1 - a) / 7
            return float(np.mean(np.log(Pd[y[rows], pred_all[rows]])))
        for c in conds:
            er = ev[cond[ev] == c]
            rel.append({"seed": seed, "fit": "4-condition", "cond": c, "p_correct": p4[c],
                        "alpha": a4[c], "omega": om4[c], "acc_eval": float(ok[er].mean()),
                        "ll_none": ll(er, 1.0), "ll_own": ll(er, a4[c]), "n_eval": len(er)})
        for B in ("BLUE", "WIFI", "BOTH"):
            rows_t = cal[np.isin(cond[cal], ["CLEAN", B])]
            pb, ab, _ = fit_alpha(rows_t, lambda v: np.where(v == "CLEAN", "clean", "interference"))
            for c in [x for x in ("BLUE", "WIFI", "BOTH") if x != B]:
                er = ev[cond[ev] == c]
                rel.append({"seed": seed, "fit": f"clean+{B}", "cond": c,
                            "p_correct": pb["interference"], "alpha": ab["interference"],
                            "omega": np.nan, "acc_eval": float(ok[er].mean()),
                            "ll_none": ll(er, 1.0), "ll_own": ll(er, a4[c]),
                            "ll_transfer": ll(er, ab["interference"]), "n_eval": len(er)})
    pd.DataFrame(node).to_csv(os.path.join(RES, f"{P}_dd_node.csv"), index=False)
    pd.DataFrame(rel).to_csv(os.path.join(RES, f"{P}_dd_reliability.csv"), index=False)


# ------------------------------------------------------------ summary -----
def _paired(a, b):
    from scipy import stats as st
    d = np.asarray(a, float) - np.asarray(b, float)
    return float(d.mean()), (float(st.ttest_1samp(d, 0.0).pvalue) if len(d) > 1 else np.nan)


def verdict_n(mean, p):
    if mean > 0.05 and p < 0.05:
        return "large"
    if mean <= 0.02 or p >= 0.05:
        return "small"
    return "moderate"


def summarize():
    from scipy import stats as st
    nd = pd.read_csv(os.path.join(RES, f"{P}_node_dronerf.csv"))
    e2 = pd.read_csv(os.path.join(RES, f"{P}_e2e_raw.csv"), float_precision="round_trip")
    hb = pd.read_csv(os.path.join(RES, f"{P}_highthreat_bins.csv"))
    al = pd.read_csv(os.path.join(RES, f"{P}_alarm.csv"))
    L = ["# Exp-I: RF channel on independent recordings and real-data high-threat "
         "calibration", ""]
    g0 = pd.read_csv(os.path.join(RES, "expG_rf_e2e_raw.csv"), float_precision="round_trip")
    mine = e2[(e2.proto == "P0") & (e2.rf_likelihood != "CPT-oracle")].reset_index(drop=True)
    n_cmp = bad = 0
    for k in ("acc", "macroAUC", "ece", "brier", "cwece"):
        for name in ("abstract", "calib-hard", "calib-soft"):
            a = mine[mine.rf_likelihood == name].sort_values("rep")[k].values
            b = g0[g0.rf_likelihood == name].sort_values("rep")[k].values
            n_cmp += len(a); bad += int(not np.array_equal(a, b))
    L.append(f"- P0 reproduces expG_rf_e2e_raw.csv: {n_cmp} values compared, "
             f"{bad} metric columns differing")
    L.append("\n## DroneRF node level (test sets)")
    L.append("| protocol | splits | n_train (mean) | n_test (mean) | accuracy | macro-F1 | "
             "Bebop+AR acc | recall bg/Bebop/AR/Phantom |")
    L.append("|---|---|---|---|---|---|---|---|")
    for proto in ("P0", "P1", "P1c", "P2", "P3"):
        q = nd[nd.proto == proto]
        rec = "/".join(f"{q[f'recall_{c}'].mean():.3f}" for c in range(4))
        L.append(f"| {proto} | {len(q)} | {q.n_train.mean():.0f} | {q.n_test.mean():.0f} | "
                 f"{q.acc.mean():.4f} (sd {q.acc.std(ddof=1) if len(q) > 1 else 0:.4f}) | "
                 f"{q.macroF1.mean():.4f} | {q.acc_bebop_ar.mean():.4f} | {rec} |")
    L.append("\n## [N] within-recording optimism (node level)")
    for a, b in (("P1", "P2"), ("P1c", "P2")):
        qa = nd[nd.proto == a].set_index("key").acc
        qb = nd[nd.proto == b].set_index("key").acc
        m, p = _paired(qa.loc[list(SPLIT_SEEDS)], qb.loc[list(SPLIT_SEEDS)])
        L.append(f"- acc({a}) - acc({b}): {m:+.4f} (paired p {p:.2g}) -> {verdict_n(m, p)}")
    ba2 = nd[nd.proto == "P2"].acc_bebop_ar.values
    ba3 = nd[nd.proto == "P3"].acc_bebop_ar.values
    tt = st.ttest_ind(ba2, ba3, equal_var=False)
    m3 = float(ba2.mean() - ba3.mean())
    L.append(f"- Bebop+AR acc: P2 {ba2.mean():.4f} vs P3 held-out modes {ba3.mean():.4f}: "
             f"{m3:+.4f} (Welch p {tt.pvalue:.2g}) -> {verdict_n(m3, tt.pvalue)}")
    for f, d in nd[nd.proto == "P3"].iterrows():
        L.append(f"  - P3 fold {int(d.key)} (mode {MODES[int(d.key)]}): acc {d.acc:.4f}, "
                 f"Bebop recall {d.recall_1:.3f}, AR recall {d.recall_2:.3f}")
    L.append("\n## threat level (mixed world, 20 replications)")
    L.append("| protocol | variant | acc | macro-AUC | Brier | ECE |")
    L.append("|---|---|---|---|---|---|")
    for proto in ("P0", "P1", "P1c", "P2", "P3"):
        for v in ("abstract", "calib-hard", "calib-soft", "CPT-oracle"):
            q = e2[(e2.proto == proto) & (e2.rf_likelihood == v)]
            L.append(f"| {proto} | {v} | {q.acc.mean():.4f} | {q.macroAUC.mean():.4f} | "
                     f"{q.brier.mean():.4f} | {q.ece.mean():.4f} |")
    L.append("\n## [E1] calib-hard minus abstract (Sec. VII-H claim)")
    for proto in ("P0", "P1", "P1c", "P2", "P3"):
        h = e2[(e2.proto == proto) & (e2.rf_likelihood == "calib-hard")].set_index("rep")
        a = e2[(e2.proto == proto) & (e2.rf_likelihood == "abstract")].set_index("rep")
        ma, pa = _paired(h.macroAUC, a.macroAUC)
        me, pe = _paired(h.ece, a.ece)
        ok = ma > 0 and pa < 0.05 and me < 0 and pe < 0.05
        L.append(f"- {proto}: dAUC {ma:+.4f} (p {pa:.1e}), dECE {me:+.4f} (p {pe:.1e}) -> "
                 f"{'holds' if ok else 'does not hold'}")
    L.append("\n## [E2] optimism carried to the threat posterior (calib-hard)")
    for a, b in (("P1", "P2"), ("P0", "P2"), ("P0", "P3")):
        qa = e2[(e2.proto == a) & (e2.rf_likelihood == "calib-hard")].set_index("rep")
        qb = e2[(e2.proto == b) & (e2.rf_likelihood == "calib-hard")].set_index("rep")
        ma, pa = _paired(qa.macroAUC, qb.macroAUC)
        me, pe = _paired(qa.ece, qb.ece)
        mat = (abs(ma) >= 0.010 and pa < 0.05) or (abs(me) >= 0.0165 and pe < 0.05)
        L.append(f"- {a} - {b}: dAUC {ma:+.4f} (p {pa:.1e}), dECE {me:+.4f} (p {pe:.1e}) -> "
                 f"{'material' if mat else 'not material'}")
    L.append("\n## [H] real-data P(T = high) reliability (pooled over 20 replications)")
    for proto in ("P0", "P2", "P3"):
        L.append(f"\n### {proto}")
        L.append("| bin | n | predicted | observed [95% CP] | oracle predicted | oracle observed |")
        L.append("|---|---|---|---|---|---|")
        out = {"calib-hard": 0, "CPT-oracle": 0}
        for lo, hi in zip(BINS[:-1], BINS[1:]):
            cells = {}
            for v in ("calib-hard", "CPT-oracle"):
                q = hb[(hb.proto == proto) & (hb.variant == v) & (hb.bin_lo == lo)]
                n, sp, k = int(q.n.sum()), float(q.sum_pred.sum()), int(q.hits.sum())
                lo_, hi_ = cp_interval(k, n) if n else (np.nan, np.nan)
                pm, om = (sp / n, k / n) if n else (np.nan, np.nan)
                out[v] += int(n > 0 and not (lo_ <= pm <= hi_))
                cells[v] = (n, pm, om, lo_, hi_)
            n, pm, om, lo_, hi_ = cells["calib-hard"]
            _, opm, oom, _, _ = cells["CPT-oracle"]
            L.append(f"| ({lo:.2f},{hi:.2f}] | {n} | {pm:.3f} | {om:.3f} [{lo_:.3f}, {hi_:.3f}] | "
                     f"{opm:.3f} | {oom:.3f} |")
        L.append(f"- bins outside the 95% interval: calib-hard {out['calib-hard']}/5, "
                 f"CPT oracle {out['CPT-oracle']}/5")
        for v in ("calib-hard", "CPT-oracle"):
            q = al[(al.proto == proto) & (al.variant == v)]
            tp, fp, fn = int(q.tp.sum()), int(q.fp.sum()), int(q.fn.sum())
            L.append(f"- alarm P(T=high) >= 0.5 ({v}): precision {tp / max(tp + fp, 1):.3f}, "
                     f"recall {tp / max(tp + fn, 1):.3f} (tp {tp}, fp {fp}, fn {fn})")
    dn = os.path.join(RES, f"{P}_dd_node.csv")
    if os.path.exists(dn):
        dd = pd.read_csv(dn)
        rl = pd.read_csv(os.path.join(RES, f"{P}_dd_reliability.csv"))
        L.append("\n## DroneDetect node level (5 seeds; CLEAN-trained)")
        L.append("| condition | chunk accuracy | macro-F1 | recording accuracy | recordings |")
        L.append("|---|---|---|---|---|")
        for c in ("CLEAN", "BLUE", "WIFI", "BOTH"):
            q = dd[dd.cond == c]
            L.append(f"| {c} | {q.acc.mean():.4f} (sd {q.acc.std(ddof=1):.4f}) | {q.macroF1.mean():.4f} | "
                     f"{q.rec_acc.mean():.4f} | {q.n_recordings.mean():.0f} |")
        L.append("\n## DroneDetect reliability (condition as context)")
        L.append("| condition | p(correct) | alpha Eq.(10) | omega Eq.(11) | eval acc | "
                 "LL none | LL own alpha | own - none (p) |")
        L.append("|---|---|---|---|---|---|---|---|")
        for c in ("CLEAN", "BLUE", "WIFI", "BOTH"):
            q = rl[(rl.fit == "4-condition") & (rl.cond == c)]
            m, p = _paired(q.ll_own, q.ll_none)
            L.append(f"| {c} | {q.p_correct.mean():.3f} | {q.alpha.mean():.3f} | {q.omega.mean():.3f} | "
                     f"{q.acc_eval.mean():.3f} | {q.ll_none.mean():.3f} | {q.ll_own.mean():.3f} | "
                     f"{m:+.3f} ({p:.1e}) -> {'useful' if (m > 0 and p < 0.05) else 'not shown'} |")
        L.append("\n### transfer (context clean vs interference)")
        L.append("| fitted on | applied to | alpha transferred | alpha own | LL none | LL transfer | "
                 "transfer - none (p) |")
        L.append("|---|---|---|---|---|---|---|")
        for B in ("BLUE", "WIFI", "BOTH"):
            for c in [x for x in ("BLUE", "WIFI", "BOTH") if x != B]:
                q = rl[(rl.fit == f"clean+{B}") & (rl.cond == c)]
                own = rl[(rl.fit == "4-condition") & (rl.cond == c)].alpha.mean()
                m, p = _paired(q.ll_transfer, q.ll_none)
                L.append(f"| CLEAN+{B} | {c} | {q.alpha.mean():.3f} | {own:.3f} | {q.ll_none.mean():.3f} | "
                         f"{q.ll_transfer.mean():.3f} | {m:+.3f} ({p:.1e}) -> "
                         f"{'transfers' if (m > 0 and p < 0.05) else 'not shown'} |")
    txt = "\n".join(L)
    with open(os.path.join(RES, f"{P}_summary.md"), "w", encoding="utf-8") as f:
        f.write(txt + "\n")
    print(txt)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default=os.environ.get("CUAS_RF_CACHE", ""),
                    help="folder outside the repository for the DroneDetect cache")
    ap.add_argument("--dronerf", default="", help="path of droneRF_four.npz")
    ap.add_argument("--build-dd-zip", default="", help="path of DroneDetect_V2.zip")
    ap.add_argument("--check-t3", default="", help="drone-rf-id folder (optional check)")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--only", default="", help="cnn,e2e,dd")
    ap.add_argument("--summary-only", action="store_true")
    a = ap.parse_args()
    if a.dronerf:
        DRONERF_NPZ = os.environ["DRONERF_NPZ"] = a.dronerf     # workers inherit it
    if a.cache:
        assert not os.path.normcase(os.path.abspath(a.cache)).startswith(os.path.normcase(HERE)), \
            "--cache must be a folder outside this repository"
    if a.build_dd_zip:
        assert a.cache, "--build-dd-zip needs --cache"
        os.makedirs(a.cache, exist_ok=True)
        build_dd_cache(a.build_dd_zip, a.cache)
        sys.exit(0)
    if a.check_t3:
        check_against_t3(a.check_t3, a.cache)
    if not a.summary_only:
        parts = a.only.split(",") if a.only else ["cnn", "e2e", "dd"]
        if "cnn" in parts:
            assert a.cache, "the CNN stage needs --cache (or CUAS_RF_CACHE)"
            run_cnns(a.cache, a.workers)
            node_table()
        if "e2e" in parts:
            run_e2e()
        if "dd" in parts:
            run_dd_reliability()
    summarize()
