"""
Exp-O: the EO/IR channel on real thermal-infrared clips, and the mixed world
with two real channels (RF + EO/IR). Core files, exp_rf_e2e.py and exp_rf_independent.py are not
modified (their functions are imported); results go to results/expO_*. The
data are only read, from a folder outside this repository (--data), and are
checked against a SHA-256 manifest at start.

Data
  Svanstrom, Alonso-Fernandez, Englund, "A dataset for multi-sensor drone
  detection", Data in Brief 39 (2021) 107521; GitHub
  DroneDetectionThesis/Drone-detection-dataset, tag v1.0.0 (CC0-1.0; Zenodo
  doi:10.5281/zenodo.5500576). The 365 thermal-infrared clips (Video_IR,
  320x256, 30 fps, about 10 s): AIRPLANE 74, BIRD 79, DRONE 157, HELICOPTER
  55; distance bins CLOSE/MEDIUM/DISTANT from Video_dataset_description.xlsx,
  which are class-specific distance ranges at which a nominal-size target is
  15 or 5 pixels wide (DRI; drone < 20 / 20-60 / > 60 m, bird < 40 / 40-120 /
  > 120 m, helicopter < 500 / 500-1500 / > 1500 m, airplane < 1000 /
  1000-3000 / > 3000 m). The labelled boxes do not follow these pixel
  widths (G1), so the bins are not read as apparent-size classes. Boxes:
  exported once from the MATLAB groundTruth label files to a CSV (--boxes;
  export script export_boxes.m; columns file,row,time_s,label,box_idx,x,y,
  w,h in MATLAB 1-based [x y w h]). Frames are matched by row: the frame of
  a box is its LabelData row - 1 (G1: the rows are 1:1 with the decoded
  frames; time_s starts at 0 s in 349 clips and at 1/30 s in 16, as the
  video streams do, so time_s is not used for the match).
  Network mapping: DRONE -> cls drone, report drone; BIRD -> cls bird, report
  bird; AIRPLANE and HELICOPTER -> cls clutter, report other (the EO/IR node
  Re = {drone, bird, other}); bin CLOSE/MEDIUM/DISTANT -> d imminent/near/far.

FIXED BEFORE THE RUN (in this docstring, before G2 is run; not registered externally)

[S] Sessions: consecutive clips of one class whose boundary mean absolute
    difference (last frame of a clip against the first frame of the next,
    grey levels 0-255) is at most 10 form one session.
[X] Inputs: for every clip, the first frame with a box in each window of 30
    frames (frames 30k .. 30k+29, k = 0..9), so at most 10 frames per clip;
    clips without such frames are dropped and counted. The box of a frame is
    the largest box whose label equals the clip's class (G1: 146 BIRD boxes
    in three DRONE clips of one session are not used). Patch: the square of
    side s = max(64, ceil(max(w, h))) centred on the box centre (padding with
    the frame median at the border); s = 64 keeps native resolution, larger
    squares (boxes wider or taller than 64 pixels) are resized to 64x64 with
    antialiased bilinear interpolation; the share of such boxes is reported
    per class and bin (c3; G1: 37 of 3,650 sampled frames, 33 of them in
    HELICOPTER CLOSE, 22% of that cell). Per-patch z-norm.
[P] Split: every session gets its dominant bin (the bin of most of its
    clips; ties go to the farther bin). Per class, the strata of dominant
    bins are taken in the order DISTANT, CLOSE, MEDIUM; within a stratum the
    sessions are shuffled with one np.random.RandomState(42), stably sorted
    by size (largest first), and each goes to the fold of 4 with the fewest
    sessions of that stratum, ties to the fold with the fewest clips of the
    class, then to the lowest index. Fold k is the test fold, fold (k+1) mod 4
    the validation fold, the other two the training folds; every clip is
    tested once.
[C] Classifier: rf_confusion.CNN(4) with the recipe of
    exp_rf_e2e.train_and_cache (35 epochs, Adam lr 1e-3, weight decay 1e-4,
    cosine schedule, batch 64, class-weighted cross-entropy, checkpoint by
    validation accuracy), torch seed 42 and 4 torch threads set in every
    task; one model per test fold (exp_rf_independent.train_cnn).
[G2] Confound gate (c1, c2). Mask size M = 40, fixed before G2 from the G1
    box sizes only, by the rule: the smallest multiple of 8 (so that the
    mask edge falls on the 4-pixel pooling grid of check (a)) for which at
    least 90% of the sampled frames have a box side max(w, h) of at most
    M - 4 (G1, share left out: 11.7% at M = 32, 5.0% at M = 40). A patch
    enters G2 if its box fits inside the central M x M square with a margin
    of 2 pixels on each side (max(w, h) <= M - 4). The square is set to 0
    and the remaining background is re-standardized by its own mean and
    standard deviation, so that the target reaches the masked input neither
    directly nor through the patch z-norm (its brightness and size); the
    mask is the same for every patch and carries no size information. The
    share of patches left out is reported per class and bin (G1 at M = 40:
    CLOSE airplane 32%, drone 18%, helicopter 63%; every other cell at most
    3%); the background of the large targets left out is not tested by G2.
    Two checks on these patches, both required:
      (a) leave-one-session-out 1-nearest-neighbour on the masked patches
          mean-pooled to 16x16;
      (b) the CNN of [C] trained and tested on the masked patches with the
          4-fold session split of [P].
    Chance = share of the most frequent class among the evaluated test
    patches (pooled; per-fold values reported). Pass if both accuracies are
    at most chance + 0.10. If G2 fails, no mixed-world run is made: the study
    goes to the "not feasible" branch and the node-level masked-versus-
    unmasked results serve only as evidence.
    Amendment (2026-11-06 10:49, before the G2 results were
    seen; SHA-256 of this docstring as it stood when G2 was started:
    6bf85fc1e3c2ca5f384e36f500d0b9772b9d9b1ca871feb8f47af6eb714c3cc4): the majority share is not the no-information level of either
    check (for (a) that level is the sum of squared class shares; for the
    class-weighted CNN it lies between 0.25 and the majority share), so a
    second condition is added and G2 passes only if both checks meet both
    (1) accuracy <= majority share + 0.10 (as above) and (2) balanced
    accuracy (mean per-class recall; 0.25 under no information) <= 0.35.
    Reported with them: per-class recall and the 4x4 confusion of both
    checks, the no-information accuracy of (a) (sum of squared class
    shares, descriptive), and per-fold values.
[G3] Test sessions per (class, bin) and classes per fold are reported; cells
    with fewer than 2 test sessions are marked thin and their reliability is
    not interpreted on its own.
[N] Node metrics on the unmasked patches, test folds pooled: accuracy,
    macro-F1, 4x4 confusion, 3x3 report confusion {drone, bird, other},
    per-bin accuracy and drone recall. Descriptive.
[R] Reliability with range context, per fold: on the validation fold,
    logistic regression (C = 10) of report correctness on the d one-hot (far
    = reference, as dataio.fit_reliability; the L2 penalty shrinks the near
    and imminent levels toward far); alpha(d) = Eq. (10); omega(d) =
    Eq. (11) for reference. On the test fold: mean log-likelihood (nats per
    report) of the observed 3-way report under the discounted range-marginal
    confusion (validation fold, smoothing 0.5) against alpha = 1, per bin.
    "Useful" if the DISTANT-bin gain is positive with paired p < 0.05 over the
    4 folds (low power stated).
[E] Mixed world (exp_rf_e2e world and seeds: test 9000+r, calibration 5000+r,
    training r; 20 replications; replication r uses fold r mod 4). RF as in
    Sec. VII-H (expG_cnn.npz, draws rng(41000+r), calibration-split
    likelihood from the validation pool). EO/IR real: EO draws use
    rng(51000+r); for a scenario (cls, d) draw a test-fold patch of DRONE
    (cls drone), BIRD (bird), or AIRPLANE or HELICOPTER with probability 1/2
    each (clutter), in the bin of d; report = CNN argmax mapped to {drone,
    bird, other}. The 300 calibration scenarios draw validation-fold patches
    the same way: range-marginal confusion (smoothing 0.5) and the logistic
    p(d) of report correctness. If a (class, bin) pool of the fold is empty,
    the draw comes from the same class in the nearest non-empty bin of that
    fold (for CLOSE or DISTANT: MEDIUM, then the other end; for MEDIUM:
    CLOSE, then DISTANT); such draws are counted per (class, bin) and fold
    and reported with the thin cells of G3. Radar and
    acoustic stay synthetic with the deployed estimate and discount. EO/IR
    likelihood variants:
      (i-a) abstract: dataio.abstract_sensor_cpts() EO/IR table, no discount
            (the comparison of E1-O);
      (i-b) synthetic-world calibration: the EO/IR table and discount the
            deployed pipeline estimates from the synthetic calibration split
            (descriptive: synthetic against real calibration);
      (ii)  real calibration: real range-marginal confusion discounted by
            alpha(d) of Eq. (10);
      (iii) real calibration without discount (alpha = 1);
      (iv)  CPT oracle: true internal CPTs, true range-marginal radar and
            acoustic tables, the test-pool RF confusion (as Sec. VII-H), and
            the test-fold EO/IR confusion per (cls, d);
      (v)   chance-corrected layer on the same real range-marginal confusion:
            reference matrix by Eq. (12) (floored-simplex projection) weighted
            by omega(d) of Eq. (11) (descriptive).
    (E1-O) Real EO/IR calibration improves the threat posterior if (ii) minus
           (i-a) has macro-AUC > 0 and ECE < 0, paired t-test p < 0.05.
    (E2-O) The range discount matters if |ECE(ii) - ECE(iii)| >= 0.005 with
           p < 0.05 (sign reported).
    Descriptive: (i-b) and (v) against (ii) on Brier score and ECE (paired p,
    sign as found). Check: synthetic EO/IR evidence through this code path
    reproduces the calib-hard metrics of expG_rf_e2e_raw.csv within 1e-9.
[H] Table 9 decomposition of P(T = high) for (ii) and (iv) (bins (0,.10],
    (.10,.25], (.25,.50], (.50,.75], (.75,1]; 95% Clopper-Pearson); alarms at
    P(T = high) >= 0.5 and > 0.25. Descriptive.
Limitations stated in advance: the two real channels come from different
recordings and are drawn independently given (cls, d, emit), which is the
network's own assumption; co-registered data would be needed to test it.
The map CLOSE/MEDIUM/DISTANT -> imminent/near/far matches levels of
detection difficulty (the bins equalize the nominal apparent size across
classes), not metric range: near is 20-60 m for a drone but 1-3 km for an
airplane.

Outcome and node-level run (added 2026-11-06 11:33, after the G2 result was
known, so outside the rule fixed in advance): G2 failed, so [R], [E] and [H] are
not run and stage_main stops.
--stage node runs an unmasked node-level comparison, a descriptive check only:
the CNN of [C] on all 3,650 unmasked patches with the split of [P], scored on
all patches and on the 3,466 patches of G2, and the leave-one-session-out
1-nearest-neighbour of G2 (a) on the same 3,466 patches, unmasked (per-patch
z-norm, mean-pooled to 16x16). The session bootstrap of the balanced accuracy
(sessions resampled within class, predictions fixed, 2,000 resamples, seed 0)
is descriptive, for G2 and for this run.

Run (DATA = .../Drone-detection-dataset-1.0.0/Data, outside this repository):
  python exp_eo_independent.py --data DATA --boxes BOXES.csv --manifest MANIFEST.csv --stage g1
  python exp_eo_independent.py --data DATA --boxes BOXES.csv --manifest MANIFEST.csv --stage g2
  python exp_eo_independent.py --data DATA --boxes BOXES.csv --manifest MANIFEST.csv --stage main
  python exp_eo_independent.py --data DATA --boxes BOXES.csv --manifest MANIFEST.csv --stage node
  python exp_eo_independent.py --summary-only
"""
from __future__ import annotations
import os
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "4")
import sys, time, argparse, hashlib, re
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
if HERE not in sys.path:                  # imports of this repository under python -P
    sys.path.insert(0, HERE)
P = "expO"
CLASSES = ("AIRPLANE", "BIRD", "DRONE", "HELICOPTER")
BINS = ("CLOSE", "MEDIUM", "DISTANT")
D_OF_BIN = {"CLOSE": 2, "MEDIUM": 1, "DISTANT": 0}         # model_spec d: far 0, near 1, imminent 2
BIN_OF_D = {v: k for k, v in D_OF_BIN.items()}
CLS_OF = {"AIRPLANE": 0, "BIRD": 1, "DRONE": 2, "HELICOPTER": 0}   # cls: clutter 0, bird 1, drone 2
REP_OF = {"AIRPLANE": 2, "BIRD": 1, "DRONE": 0, "HELICOPTER": 2}   # Re: drone 0, bird 1, other 2
SESSION_T = 10.0
WIN = 30
MAX_FRAMES = 10
PATCH = 64
MASK_M = 40              # fixed from the G1 box sizes before G2 (rule in [G2])
MASK_MARGIN = 2
MASK_CAND = (16, 24, 32, 40, 48)
NFOLD = 4
SEED = 42
REPS = 20
TBINS = [0.0, 0.10, 0.25, 0.50, 0.75, 1.0]


# ------------------------------------------------------------- data ---------
def check_manifest(data, manifest):
    """Every file of the manifest (paths relative to the Data folder) must
    have the listed SHA-256."""
    m = pd.read_csv(manifest)
    n = 0
    for _, r in m.iterrows():
        p = os.path.join(data, *r.path.split("/"))
        h = hashlib.sha256(open(p, "rb").read()).hexdigest()
        assert h == r.sha256, f"SHA-256 mismatch: {r.path}"
        n += 1
    assert n == 1391, n
    return n


def load_description(data):
    import openpyxl
    ws = openpyxl.load_workbook(os.path.join(data, "Video_dataset_description.xlsx"),
                                read_only=True, data_only=True).worksheets[0]
    rows = list(ws.iter_rows(values_only=True))
    x = pd.DataFrame(rows[1:], columns=["sensor", "cls", "num", "bin", "inbin", "src"])
    x = x[x.sensor == "IR"].copy()
    x["bin"] = x["bin"].str.strip()
    x["num"] = x["num"].astype(int)
    return x[["cls", "num", "bin", "src"]].reset_index(drop=True)


def clip_path(data, cls, num):
    return os.path.join(data, "Video_IR", f"IR_{cls}_{num:03d}.mp4")


def decode(path):
    import av
    with av.open(path) as c:
        return [f.to_ndarray(format="gray") for f in c.decode(c.streams.video[0])]


def sessions(data, desc):
    """[S]: boundary differences between consecutive clips of a class."""
    first, last, nfr = {}, {}, {}
    for _, r in desc.iterrows():
        fr = decode(clip_path(data, r.cls, r.num))
        first[(r.cls, r.num)] = fr[0].astype(np.float32)
        last[(r.cls, r.num)] = fr[-1].astype(np.float32)
        nfr[(r.cls, r.num)] = len(fr)
    sid, s, bnd = {}, 0, {}
    for c in CLASSES:
        nums = sorted(desc[desc.cls == c].num)
        for i, n in enumerate(nums):
            if i == 0 or bnd[(c, nums[i - 1])] > SESSION_T:
                s += 1
            sid[(c, n)] = s
            if i + 1 < len(nums):
                bnd[(c, n)] = float(np.mean(np.abs(first[(c, nums[i + 1])] - last[(c, n)])))
    return sid, bnd, nfr


def load_boxes(path):
    """Largest box of the clip's own class per labelled frame; the frame
    index (0-based) is the LabelData row - 1 (rows are 1:1 with the decoded
    frames; checked in G1)."""
    b = pd.read_csv(path)
    b = b[b.label == b.file.str.extract(r"IR_([A-Z]+)_")[0]].copy()
    b["area"] = b.w * b.h
    b["frame"] = (b.row - 1).astype(int)
    b = b.sort_values(["file", "frame", "area"], ascending=[True, True, False])
    return b.groupby(["file", "frame"], as_index=False).first()


def sampled_frames(desc, boxes, nfr):
    """[X]: per clip, the first boxed frame in each window of WIN frames (at
    most MAX_FRAMES per clip); shared by G1, G2 and the main run."""
    by_file = {f: g.set_index("frame") for f, g in boxes.groupby("file")}
    rows = []
    for _, r in desc.iterrows():
        bb = by_file.get(f"IR_{r.cls}_{r.num:03d}_LABELS.mat")
        if bb is None:
            continue
        n = nfr[(r.cls, r.num)]
        for k in range(MAX_FRAMES):
            cand = [i for i in range(WIN * k, min(WIN * (k + 1), n)) if i in bb.index]
            if cand:
                q = bb.loc[cand[0]]
                rows.append({"cls": r.cls, "num": int(r.num), "bin": r.bin, "window": k, "frame": cand[0],
                             "x": float(q.x), "y": float(q.y), "w": float(q.w), "h": float(q.h)})
    return pd.DataFrame(rows)


def patch_of(frame, box):
    """[X]: square of side max(64, ceil(max(w, h))) centred on the box centre,
    padded with the frame median, resized to 64x64 only when larger."""
    import torch, torch.nn.functional as F
    x, y, w, h = box
    cx, cy = x - 1 + w / 2.0, y - 1 + h / 2.0                 # 0-based centre
    s = max(PATCH, int(np.ceil(max(w, h))))
    H, W = frame.shape
    med = float(np.median(frame))
    canvas = np.full((H + 2 * s, W + 2 * s), med, dtype=np.float32)
    canvas[s:s + H, s:s + W] = frame
    x0 = int(round(cx - s / 2.0)) + s
    y0 = int(round(cy - s / 2.0)) + s
    p = canvas[y0:y0 + s, x0:x0 + s]
    if s > PATCH:
        t = torch.from_numpy(np.ascontiguousarray(p))[None, None]
        p = F.interpolate(t, size=(PATCH, PATCH), mode="bilinear", align_corners=False,
                          antialias=True)[0, 0].numpy()
    p = (p - p.mean()) / (p.std() + 1e-6)
    return p.astype(np.float32), s > PATCH


def build_patches(data, desc, boxes, sid, nfr):
    sf = sampled_frames(desc, boxes, nfr)
    X, meta = [], []
    for (c, n), g in sf.groupby(["cls", "num"], sort=False):
        frames = decode(clip_path(data, c, n))
        assert len(frames) == nfr[(c, n)]
        for _, q in g.iterrows():
            p, big = patch_of(frames[int(q.frame)].astype(np.float32), (q.x, q.y, q.w, q.h))
            X.append(p)
            meta.append({"cls": c, "num": int(n), "bin": q["bin"], "session": sid[(c, n)],
                         "frame": int(q.frame), "w": float(q.w), "h": float(q.h), "big": bool(big)})
    dropped = len(desc) - sf.drop_duplicates(["cls", "num"]).shape[0]
    return np.stack(X), pd.DataFrame(meta), dropped


def dominant_bin(clips):
    """Bin of most clips of a session; ties go to the farther bin."""
    cnt = clips["bin"].value_counts()
    best = cnt.max()
    for b in ("DISTANT", "MEDIUM", "CLOSE"):
        if cnt.get(b, 0) == best:
            return b


def assign_folds(meta):
    """[P]: sessions dealt to folds within (class, dominant-bin) strata."""
    fold_of = {}
    rs = np.random.RandomState(SEED)
    for c in CLASSES:
        clips = meta[meta.cls == c].drop_duplicates(["num"])
        size = clips.groupby("session").size()
        dom = {s: dominant_bin(g) for s, g in clips.groupby("session")}
        nclip = [0] * NFOLD
        for b in ("DISTANT", "CLOSE", "MEDIUM"):
            ss = np.array(sorted(s for s in size.index if dom[s] == b))
            rs.shuffle(ss)
            order = sorted(ss, key=lambda s: -size[s])        # stable: ties keep the shuffled order
            nsess = [0] * NFOLD
            for s in order:
                f = min(range(NFOLD), key=lambda k: (nsess[k], nclip[k], k))
                fold_of[s] = f
                nsess[f] += 1
                nclip[f] += int(size[s])
    return meta.session.map(fold_of).values


def masked(X, meta):
    """[G2]: patches whose box fits inside the central M x M square with a
    margin of MASK_MARGIN pixels; the square is set to 0 and the background
    is re-standardized by its own mean and standard deviation, so that the
    target's brightness and size do not reach it through the patch z-norm."""
    m = MASK_M
    keep = np.maximum(meta.w.values, meta.h.values) <= m - 2 * MASK_MARGIN
    Xm = X[keep].copy()
    a = (PATCH - m) // 2
    bg = np.ones((PATCH, PATCH), dtype=bool)
    bg[a:a + m, a:a + m] = False
    v = Xm[:, bg]
    Xm[:, bg] = (v - v.mean(1, keepdims=True)) / (v.std(1, keepdims=True) + 1e-6)
    Xm[:, ~bg] = 0.0
    return Xm, keep


# ------------------------------------------------------------- CNN ----------
def _task(t):
    import exp_rf_independent as eri
    k, Xtr, ytr, Xva, yva, Xte = t
    t0 = time.time()
    sm_va, (sm_te,), best, prior = eri.train_cnn(Xtr, ytr, Xva, yva, 4, [Xte])
    return k, sm_va, sm_te, best, time.time() - t0


def run_folds(X, y, fold, workers):
    from multiprocessing import Pool
    tasks = []
    for k in range(NFOLD):
        te, va = fold == k, fold == (k + 1) % NFOLD
        tr = ~(te | va)
        tasks.append((k, X[tr], y[tr].astype(np.int64), X[va], y[va].astype(np.int64), X[te]))
    out = {}
    with Pool(min(workers, NFOLD)) as pool:
        for k, sm_va, sm_te, best, secs in pool.imap_unordered(_task, tasks):
            out[k] = (sm_va, sm_te, best)
            print(f"[expO cnn] fold {k} val {best:.3f} ({secs:.0f}s)", flush=True)
    return out


def pooled_test(out, y, fold):
    pred = np.full(len(y), -1)
    sm = np.zeros((len(y), 4), dtype=np.float32)
    for k in range(NFOLD):
        te = np.where(fold == k)[0]
        sm[te] = out[k][1]
        pred[te] = out[k][1].argmax(1)
    return pred, sm


# ------------------------------------------------------------- stages -------
def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def stream_start(path):
    """Start time (s) of the video stream (no decoding)."""
    import av
    with av.open(path) as c:
        st = c.streams.video[0]
        return float(st.start_time * st.time_base) if st.start_time is not None else 0.0


def _cell_order(df):
    df = df.reset_index()
    df["_b"] = df["bin"].map({b: i for i, b in enumerate(BINS)})
    return df.sort_values(["cls", "_b"]).drop(columns="_b").set_index(["cls", "bin"])


def stage_g1(data, boxes_path, manifest, desc, sid, nfr):
    """G1: provenance and consistency of the box export, box coverage of the
    sampled windows (criterion >= 0.95), and the box sizes of the sampled
    frames (c3), from which alone the mask M of G2 is fixed. Also writes the
    per-label-file provenance table (c6)."""
    raw = pd.read_csv(boxes_path)
    raw["ccls"] = raw.file.str.extract(r"IR_([A-Z]+)_")[0]
    files = {f"IR_{r.cls}_{r.num:03d}_LABELS.mat": (r.cls, int(r.num)) for _, r in desc.iterrows()}
    assert set(raw.file) == set(files), "label files of the CSV differ from the description"
    short = lambda f: f.replace("IR_", "").replace("_LABELS.mat", "")
    L = [f"box CSV {os.path.basename(boxes_path)}: SHA-256 {sha256_file(boxes_path)}, {len(raw)} box rows, "
         f"{raw.file.nunique()} label files, labels {sorted(raw.label.unique())}"]
    L.append("boxes by clip class (rows) and label (columns):\n" + pd.crosstab(raw.ccls, raw.label).to_string())
    off = raw[raw.label != raw.ccls].groupby("file").size()
    L.append(f"boxes whose label differs from the clip class (ignored): {int(off.sum())}; "
             + ", ".join(f"{short(f)} {n} (bin {desc.set_index(['cls', 'num']).loc[files[f], 'bin']}, "
                         f"session {sid[files[f]]})" for f, n in off.items()))
    rows = raw.drop_duplicates(["file", "row"])
    per_row = raw[raw.label == raw.ccls].groupby(["file", "row"]).size()
    L.append(f"rows with several boxes of the clip class (largest used): {int((per_row > 1).sum())}")
    last = rows.groupby("file").row.max()
    diff = {f: int(last[f]) - nfr[k] for f, k in files.items()}
    assert all(v <= 0 for v in diff.values())
    L.append(f"decoded frames, all clips: {sum(nfr.values())}; last boxed row = decoded frame count in "
             f"{sum(v == 0 for v in diff.values())}/{len(files)} clips; others (last boxed row - frames): "
             + (", ".join(f"{short(f)} {v:+d}" for f, v in diff.items() if v != 0) or "none"))
    c = rows.time_s * 30 - (rows.row - 1)
    cmin, cmax = c.groupby(rows.file).min(), c.groupby(rows.file).max()
    st = {f: stream_start(clip_path(data, *k)) for f, k in files.items()}
    tmatch = [f for f in files if abs(cmin[f] - 30 * st[f]) < 1e-3 and abs(cmax[f] - 30 * st[f]) < 1e-3]
    late = [f for f in files if abs(30 * st[f] - 1) < 1e-3]
    L.append(f"label time = (row - 1)/30 + video stream start time in {len(tmatch)}/{len(files)} clips; stream "
             f"start 0 s in {sum(abs(v) < 1e-9 for v in st.values())} clips and 1/30 s in {len(late)}; frames are "
             f"matched by row (frame = row - 1); round(time_s * 30) would be one frame late in those {len(late)}")
    boxes = load_boxes(boxes_path)
    sf = sampled_frames(desc, boxes, nfr)
    tot = sum(min(MAX_FRAMES, -(-nfr[k] // WIN)) for k in files.values())
    cov = len(sf) / tot
    L.append(f"G1 criterion: sampled windows with a box of the clip class {len(sf)}/{tot} = {cov:.4f} "
             f"(>= 0.95: {'pass' if cov >= 0.95 else 'FAIL'}); the sampled frame is the first frame of its "
             f"window in {int((sf.frame == WIN * sf.window).sum())}/{len(sf)}")
    sf["side"] = np.maximum(sf.w, sf.h)
    g = sf.groupby(["cls", "bin"])
    tab = pd.DataFrame({"n": g.size(), "w_median": g.w.median(), "w<5": g.w.apply(lambda s: int((s < 5).sum())),
                        "w>=15": g.w.apply(lambda s: float((s >= 15).mean())), "side_median": g.side.median(),
                        "side_p90": g.side.quantile(0.9), "side_p95": g.side.quantile(0.95),
                        "side_max": g.side.max(), f"side>{PATCH}": g.side.apply(lambda s: int((s > PATCH).sum()))})
    L.append("box sizes of the sampled frames (pixels; side = max(w, h)), by class and bin:\n"
             + _cell_order(tab).round(3).to_string())
    big = sf.side > PATCH
    L.append(f"c3: sampled frames with side > {PATCH} (resized to {PATCH}x{PATCH}): {int(big.sum())}/{len(sf)} = "
             f"{big.mean():.4f}; " + ", ".join(f"{c}/{b} {int(v.sum())} ({v.mean():.3f} of the cell)"
                                              for (c, b), v in big.groupby([sf.cls, sf["bin"]]) if v.sum() > 0))
    lo = pd.DataFrame({f"M={m}": g.side.apply(lambda s, m=m: float((s > m - 2 * MASK_MARGIN).mean()))
                       for m in MASK_CAND})
    allrow = pd.DataFrame({f"M={m}": [float((sf.side > m - 2 * MASK_MARGIN).mean())] for m in MASK_CAND},
                          index=pd.MultiIndex.from_tuples([("ALL", "")], names=["cls", "bin"]))
    L.append(f"share of sampled frames left out of G2 for a mask of side M (box side > M - {2 * MASK_MARGIN}):\n"
             + pd.concat([_cell_order(lo), allrow]).round(3).to_string())
    man = pd.read_csv(manifest).set_index("path")
    nb = raw.groupby("file").size()
    pv = pd.DataFrame([{"file": f, "label_sha256": man.loc[f"Video_IR/{f}", "sha256"],
                        "label_bytes": int(man.loc[f"Video_IR/{f}", "bytes"]), "cls": k[0], "num": k[1],
                        "session": sid[k], "decoded_frames": nfr[k], "last_boxed_row": int(last[f]),
                        "box_rows": int(nb[f]), "boxes_other_label": int(off.get(f, 0)),
                        "stream_start_s": st[f], "sampled_frames": int(((sf.cls == k[0]) & (sf.num == k[1])).sum())}
                       for f, k in files.items()])
    pv.to_csv(os.path.join(RES, f"{P}_g1_label_files.csv"), index=False)
    L.append(f"per-label-file provenance (manifest SHA-256 of the label file, frames, boxes): "
             f"{P}_g1_label_files.csv ({len(pv)} rows)")
    return "\n".join(L)


def g3_table(meta):
    """[G3]: test sessions per (class, bin), overall and per fold."""
    rows = []
    for c in CLASSES:
        for b in BINS:
            q = meta[(meta.cls == c) & (meta["bin"] == b)]
            per = [int(q[q.fold == k].session.nunique()) for k in range(NFOLD)]
            rows.append({"cls": c, "bin": b, "clips": int(q.drop_duplicates(["num"]).shape[0]),
                         "patches": len(q), "sessions": int(q.session.nunique()),
                         **{f"sessions_fold{k}": per[k] for k in range(NFOLD)},
                         "thin": int(q.session.nunique()) < 2})
    return pd.DataFrame(rows)


def g3_folds(meta):
    """[G3]: classes per test fold (clips and sessions per class)."""
    rows = []
    for k in range(NFOLD):
        q = meta[meta.fold == k]
        row = {"fold": k, "classes": int(q.cls.nunique()), "clips": int(q.drop_duplicates(["cls", "num"]).shape[0]),
               "sessions": int(q.session.nunique()), "patches": len(q)}
        for c in CLASSES:
            qc = q[q.cls == c]
            row[f"{c}_clips"] = int(qc.drop_duplicates(["num"]).shape[0])
            row[f"{c}_sessions"] = int(qc.session.nunique())
        rows.append(row)
    return pd.DataFrame(rows)


def nn_loso(Z, y, sess):
    """Leave-one-session-out 1-nearest-neighbour accuracy (squared L2)."""
    sq = (Z ** 2).sum(1)
    D = sq[:, None] + sq[None, :] - 2 * Z @ Z.T
    pred = np.empty(len(y), dtype=int)
    for i in range(len(y)):
        d = np.where(sess != sess[i], D[i], np.inf)
        pred[i] = y[int(np.argmin(d))]
    return pred


def cls_metrics(y, pred):
    """4x4 confusion (rows true), per-class recall (nan for an absent class)
    and balanced accuracy (mean recall over the classes present)."""
    conf = np.zeros((4, 4), dtype=int)
    np.add.at(conf, (y, pred), 1)
    n = conf.sum(1)
    rec = np.divide(np.diag(conf), n, out=np.full(4, np.nan), where=n > 0)
    return conf, rec, float(np.nanmean(rec))


G2_BALANCED_LIMIT = 0.35          # amendment of 2026-11-06 10:49
BOOT_B = 2000                     # session bootstrap (descriptive), number of resamples
BOOT_SEED = 0                     # session bootstrap (descriptive), generator seed


def stage_g2(a, desc, X, meta):
    Xm, keep = masked(X, meta)
    ex = meta.assign(left_out=~keep).groupby(["cls", "bin"]).left_out.agg(["mean", "sum", "size"]).reset_index()
    ex.columns = ["cls", "bin", "share_left_out", "left_out", "patches"]
    ex.to_csv(os.path.join(RES, f"{P}_g2_left_out.csv"), index=False)
    mk = meta[keep].reset_index(drop=True)
    y = mk.y.values
    share = np.bincount(y, minlength=4) / len(y)
    chance, noinfo = float(share.max()), float((share ** 2).sum())
    Z = Xm.reshape(len(Xm), 16, 4, 16, 4).mean(axis=(2, 4)).reshape(len(Xm), -1)
    pred_nn = nn_loso(Z.astype(np.float64), y, mk.session.values)
    out = run_folds(Xm, y, mk.fold.values, a.workers)
    pred_cnn, sm = pooled_test(out, y, mk.fold.values)
    rows, per_fold = [], []
    for name, pred in (("1-NN masked (LOSO)", pred_nn), ("CNN masked (4-fold sessions)", pred_cnn)):
        conf, rec, bacc = cls_metrics(y, pred)
        acc = float((pred == y).mean())
        ok1, ok2 = acc <= chance + 0.10, bacc <= G2_BALANCED_LIMIT
        rows.append({"check": name, "n": len(y), "acc": acc, "chance": chance, "limit": chance + 0.10,
                     "pass_acc": ok1, "balanced_acc": bacc, "limit_balanced": G2_BALANCED_LIMIT,
                     "pass_balanced": ok2, "pass": ok1 and ok2,
                     "no_info_acc": noinfo if name.startswith("1-NN") else np.nan,
                     **{f"recall_{c}": float(rec[i]) for i, c in enumerate(CLASSES)},
                     "confusion4": " ".join(map(str, conf.ravel()))})
        for k in range(NFOLD):
            t = mk.fold.values == k
            if not t.any():
                continue
            cf, rc, ba = cls_metrics(y[t], pred[t])
            per_fold.append({"check": name, "fold": k, "n": int(t.sum()), "acc": float((pred[t] == y[t]).mean()),
                             "chance": float(np.bincount(y[t], minlength=4).max() / t.sum()), "balanced_acc": ba,
                             "classes": int((cf.sum(1) > 0).sum()),
                             **{f"recall_{c}": float(rc[i]) for i, c in enumerate(CLASSES)}})
    g2 = pd.DataFrame(rows)
    ok = bool(g2["pass"].all())
    g2.to_csv(os.path.join(RES, f"{P}_g2.csv"), index=False)
    pd.DataFrame(per_fold).to_csv(os.path.join(RES, f"{P}_g2_folds.csv"), index=False)
    mk.assign(pred_nn=pred_nn, pred_cnn=pred_cnn)[["cls", "num", "bin", "session", "fold", "frame", "w", "h", "y",
                                                   "pred_nn", "pred_cnn"]].to_csv(
        os.path.join(RES, f"{P}_g2_predictions.csv"), index=False)
    np.savez_compressed(os.path.join(RES, f"{P}_g2_cnn.npz"), keep=keep, test_softmax=sm,
                        **{f"fold{k}__val_acc": np.array(out[k][2]) for k in range(NFOLD)})
    return g2, pd.DataFrame(per_fold), ok, float(keep.mean())


# ------------------------------------------------------- reliability --------
def logistic_p(d, correct):
    """P(correct | d) from logistic regression on the d one-hot, far = reference
    (as dataio.fit_reliability, C = 10)."""
    from sklearn.linear_model import LogisticRegression
    Xd = np.array([[1.0 if v == k else 0.0 for k in (1, 2)] for v in d])
    if correct.min() == correct.max():
        return np.full(3, float(correct.mean()))
    lr = LogisticRegression(C=10.0).fit(Xd, correct)
    return lr.predict_proba(np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]))[:, 1]


def alpha_eq10(p):
    return np.clip(p / max(p.max(), 1e-6) * 0.98, 0.05, 0.98)


def omega_eq11(p):
    import reliability_cc as rcc
    return rcc.kappa_from_p(p, 3)


def marg_confusion(cls_idx, rep, smoothing=0.5):
    C = np.zeros((3, 3)) + smoothing
    np.add.at(C, (cls_idx, rep), 1)
    return C / C.sum(1, keepdims=True), np.bincount(cls_idx, minlength=3)


def discount(C, a):
    return a * C + (1 - a) / C.shape[-1]


def ref_eq12(C, kappa, pd_, n_row, smoothing=0.5):
    """Eq. (12): reference matrix recovered from the range-marginal one, rows
    projected onto the floored simplex (reliability_cc)."""
    import reliability_cc as rcc
    kbar = float(np.asarray(pd_) @ np.asarray(kappa))
    raw = (C - (1.0 - kbar) / 3) / kbar
    out = raw.copy()
    for i in range(3):
        eps = smoothing / (n_row[i] + 3 * smoothing)
        if (raw[i] < eps).any():
            out[i] = rcc.project_floored_simplex(raw[i], eps)
    return out


def reliability_folds(meta, rep_val, rep_te):
    """[R] per fold; rep_* hold the 3-way reports of each fold model."""
    rows = []
    for k in range(NFOLD):
        va, te = rep_val[k], rep_te[k]
        mv, mt_ = meta.iloc[va["idx"]], meta.iloc[te["idx"]]
        cv = mv.cls.map(CLS_OF).values
        dv = mv["bin"].map(D_OF_BIN).values
        ok = (va["rep"] == np.array([dataio_correct(c) for c in cv])).astype(float)
        p = logistic_p(dv, ok)
        a, om = alpha_eq10(p), omega_eq11(p)
        C, _ = marg_confusion(cv, va["rep"])
        ct = mt_.cls.map(CLS_OF).values
        dt = mt_["bin"].map(D_OF_BIN).values
        for d in (0, 1, 2):
            s = dt == d
            if not s.any():
                continue
            ll_none = float(np.mean(np.log(C[ct[s], te["rep"][s]])))
            ll_a = float(np.mean(np.log(discount(C, a[d])[ct[s], te["rep"][s]])))
            rows.append({"fold": k, "bin": BIN_OF_D[d], "n": int(s.sum()), "p_correct": float(p[d]),
                         "alpha": float(a[d]), "omega": float(om[d]),
                         "acc": float((te["rep"][s] == np.array([dataio_correct(c) for c in ct[s]])).mean()),
                         "ll_none": ll_none, "ll_alpha": ll_a, "gain": ll_a - ll_none})
    return pd.DataFrame(rows)


def dataio_correct(c):
    import dataio
    return dataio.CORRECT_REPORT["Re"][int(c)]


# ------------------------------------------------------------ mixed world ---
def eo_pools(meta, idx, rep):
    """{(class, bin): array of 3-way reports} for the patches idx."""
    m = meta.iloc[idx]
    pools = {}
    for c in CLASSES:
        for b in BINS:
            s = ((m.cls == c) & (m["bin"] == b)).values
            pools[(c, b)] = rep[s]
    return pools


def draw_eo(pools, cls, d, rng, counter):
    """One EO/IR report for a scenario (cls, d); fallback draws are counted in
    counter[(class, requested bin)]."""
    if cls == 2:
        c = "DRONE"
    elif cls == 1:
        c = "BIRD"
    else:
        c = "AIRPLANE" if rng.random() < 0.5 else "HELICOPTER"
    b = BIN_OF_D[d]
    pool = pools[(c, b)]
    if len(pool) == 0:
        order = {"CLOSE": ("MEDIUM", "DISTANT"), "MEDIUM": ("CLOSE", "DISTANT"),
                 "DISTANT": ("MEDIUM", "CLOSE")}[b]
        for b2 in order:
            if len(pools[(c, b2)]):
                pool = pools[(c, b2)]
                counter[(c, b)] = counter.get((c, b), 0) + 1
                break
    return int(pool[rng.integers(0, len(pool))]), c


def oracle_eo(pools):
    """Test-pool EO/IR confusion per (cls, d); clutter = 1/2 airplane +
    1/2 helicopter. Empty cells use the fallback order of draw_eo."""
    L = np.zeros((3, 3, 3))                                   # d, cls, report
    for d in (0, 1, 2):
        b = BIN_OF_D[d]
        order = (b,) + {"CLOSE": ("MEDIUM", "DISTANT"), "MEDIUM": ("CLOSE", "DISTANT"),
                        "DISTANT": ("MEDIUM", "CLOSE")}[b]

        def freq(c):
            for b2 in order:
                if len(pools[(c, b2)]):
                    return np.bincount(pools[(c, b2)], minlength=3) / len(pools[(c, b2)])
            raise ValueError(c)
        L[d, 2] = freq("DRONE")
        L[d, 1] = freq("BIRD")
        L[d, 0] = 0.5 * freq("AIRPLANE") + 0.5 * freq("HELICOPTER")
    return L


def e2e(meta, rep_val, rep_te, reps=REPS):
    import exp_rf_e2e as eg
    import model_spec as ms, dataio, metrics as mt, pipeline as pl
    from cuas_bn import BayesNet
    gt = ms.ground_truth_bn()
    gt_int = {k: gt.cpts[k] for k in ms.INTERNAL}
    true_marg = dataio.true_marginal_confusion()
    abstract_eo = dataio.abstract_sensor_cpts()["Re"].table          # (cls, Re)
    cc = np.load(os.path.join(RES, "expG_cnn.npz"), allow_pickle=True)
    sm_va, y_va, sm_te, y_te = cc["val_softmax"], cc["val_y"], cc["test_softmax"], cc["test_y"]
    pools_te_rf = {c: np.where(y_te == c)[0] for c in range(4)}
    pools_va_rf = {c: np.where(y_va == c)[0] for c in range(4)}
    rows, bins_rows, alarm_rows, fb_rows = [], [], [], []
    for r in range(reps):
        k = r % NFOLD
        pte = eo_pools(meta, rep_te[k]["idx"], rep_te[k]["rep"])
        pva = eo_pools(meta, rep_val[k]["idx"], rep_val[k]["rep"])
        rng = np.random.default_rng(41000 + r)
        rng_eo = np.random.default_rng(51000 + r)
        test = dataio.generate_dataset(gt, eg.TEST_N, seed=9000 + r)
        calib_syn = dataio.generate_dataset(gt, 300, seed=5000 + r)
        est = dataio.estimate_confusion(calib_syn)
        alpha_syn, _ = dataio.fit_reliability(calib_syn)
        train = dataio.generate_dataset(gt, eg.GROUND_N, seed=r)
        cpts, _ = pl.ground_internal("Proposed", train, r, gt=gt)
        # RF channel exactly as Sec. VII-H (same rng order: test draws, then calibration)
        rf_hard = np.zeros(len(test), dtype=int)
        for i, scn in enumerate(test):
            c = eg.map_class(scn, rng)
            j = eg.draw_pool_index(pools_te_rf, c, rng)
            rf_hard[i] = int(sm_te[j].argmax())
        cnt = np.zeros((ms.CARD["cls"], ms.CARD["emit"], 4)) + 0.5
        for scn in calib_syn:
            c = eg.map_class(scn, rng)
            j = eg.draw_pool_index(pools_va_rf, c, rng)
            cnt[scn["cls"], scn["emit"], int(sm_va[j].argmax())] += 1
        L_calib = cnt / cnt.sum(axis=-1, keepdims=True)
        conf_te = np.zeros((4, 4))
        np.add.at(conf_te, (y_te, sm_te.argmax(1)), 1)
        row_te = conf_te / conf_te.sum(axis=1, keepdims=True)
        L_or = np.zeros((ms.CARD["cls"], ms.CARD["emit"], 4))
        for c in range(ms.CARD["cls"]):
            for e in range(ms.CARD["emit"]):
                L_or[c, e] = row_te[eg.DRONE_CLASSES].mean(axis=0) if (c == 2 and e == 0) else row_te[0]
        # EO/IR channel: test draws, then calibration draws (own rng)
        fb_t, fb_c = {}, {}
        eo_rep = np.array([draw_eo(pte, scn["cls"], scn["d"], rng_eo, fb_t)[0] for scn in test])
        cal = [(scn["cls"], scn["d"], draw_eo(pva, scn["cls"], scn["d"], rng_eo, fb_c)[0]) for scn in calib_syn]
        cal_cls = np.array([x[0] for x in cal]); cal_d = np.array([x[1] for x in cal])
        cal_rep = np.array([x[2] for x in cal])
        okc = (cal_rep == np.array([dataio_correct(c) for c in cal_cls])).astype(float)
        p_eo = logistic_p(cal_d, okc)
        a_eo, k_eo = alpha_eq10(p_eo), omega_eq11(p_eo)
        C_marg, n_row = marg_confusion(cal_cls, cal_rep)
        pd_cal = np.bincount(cal_d, minlength=3) / len(cal_d)
        C_ref = ref_eq12(C_marg, k_eo, pd_cal, n_row)
        L_ob = oracle_eo(pte)
        for phase, cnt_fb in (("test", fb_t), ("calibration", fb_c)):
            for (c_, b_), v in sorted(cnt_fb.items()):
                fb_rows.append({"rep": r, "fold": k, "phase": phase, "cls": c_, "bin": b_, "draws": v})
        Leo = {
            "(i-a) abstract": lambda d: abstract_eo,
            "(i-b) synthetic calibration": lambda d: discount(est["Re"].table, float(alpha_syn["Re"][d])),
            "(ii) real calibration": lambda d: discount(C_marg, float(a_eo[d])),
            "(iii) real, no discount": lambda d: C_marg,
            "(iv) CPT oracle": lambda d: L_ob[d],
            "(v) chance-corrected": lambda d: discount(C_ref, float(k_eo[d])),
        }
        variants = [("syn-check", cpts, est, alpha_syn, L_calib, None)] + \
                   [(n, gt_int if n.startswith("(iv)") else cpts,
                     true_marg if n.startswith("(iv)") else est,
                     None if n.startswith("(iv)") else alpha_syn,
                     L_or if n.startswith("(iv)") else L_calib, f) for n, f in Leo.items()]
        for name, cp, es_, al, Lrf, feo in variants:
            Pm, y = posteriors_two(cp, test, rf_hard, eo_rep, es_, al, Lrf, feo)
            rows.append({"rep": r, "fold": k, "variant": name, "acc": float((Pm.argmax(1) == y).mean()),
                         "macroAUC": eg.macro_auc(Pm, y), "ece": mt.ece(Pm, y),
                         "brier": mt.brier(Pm, y), "cwece": mt.classwise_ece(Pm, y)})
            if name.startswith(("(ii)", "(iv)")):
                ph, hit = Pm[:, 3], (y == 3)
                for lo, hi in zip(TBINS[:-1], TBINS[1:]):
                    msk = (ph > lo) & (ph <= hi)
                    bins_rows.append({"rep": r, "variant": name, "bin_lo": lo, "bin_hi": hi,
                                      "n": int(msk.sum()), "sum_pred": float(ph[msk].sum()),
                                      "hits": int(hit[msk].sum())})
                for thr, rule in ((0.5, ph >= 0.5), (0.25, ph > 0.25)):
                    alarm_rows.append({"rep": r, "variant": name, "threshold": thr,
                                       "tp": int((rule & hit).sum()), "fp": int((rule & ~hit).sum()),
                                       "fn": int((~rule & hit).sum()), "tn": int((~rule & ~hit).sum())})
        print(f"[expO e2e] rep {r} fold {k} done (fallback draws test {sum(fb_t.values())}, "
              f"calibration {sum(fb_c.values())})", flush=True)
    pd.DataFrame(rows).to_csv(os.path.join(RES, f"{P}_e2e_raw.csv"), index=False)
    pd.DataFrame(bins_rows).to_csv(os.path.join(RES, f"{P}_highthreat_bins.csv"), index=False)
    pd.DataFrame(alarm_rows).to_csv(os.path.join(RES, f"{P}_alarm.csv"), index=False)
    pd.DataFrame(fb_rows, columns=["rep", "fold", "phase", "cls", "bin", "draws"]).to_csv(
        os.path.join(RES, f"{P}_fallback.csv"), index=False)


def posteriors_two(internal_cpts, test, rf_hard, eo_rep, est, alpha, L_rf, eo_like):
    """Threat posteriors with RF (hard report through L_rf[c, e, r]) and, if
    eo_like is given, the EO/IR channel realized by real reports through
    eo_like(d)[c, r]; otherwise the synthetic EO/IR report enters as in
    exp_rf_e2e.posteriors_rf4 (used for the reproduction check)."""
    import exp_rf_e2e as eg
    import model_spec as ms, dataio
    from cuas_bn import BayesNet
    gt = ms.ground_truth_bn()
    cpts = dict(gt.cpts)
    parents = {k: list(v) for k, v in ms.PARENTS.items()}
    for k in ms.INTERNAL:
        cpts[k] = internal_cpts[k]
    for s in ms.SENSORS:
        cpts.pop(s, None); parents.pop(s, None)
    for s in ["Rr", "Re", "Ra"]:
        cpts[s] = est[s]
        parents[s] = list(est[s].vars[:-1])
    card = {k: ms.CARD[k] for k in cpts}
    base = BayesNet(card, {k: parents[k] for k in cpts}, cpts)
    cache = {}
    P = np.zeros((len(test), ms.CARD["T"]))
    y = np.zeros(len(test), dtype=int)
    for i, scn in enumerate(test):
        dv = scn["d"]
        if dv not in cache:
            if alpha is not None:
                c2 = dict(cpts)
                for s in ["Rr", "Re", "Ra"]:
                    c2[s] = dataio._discount_cpt(est[s], float(alpha[s][dv]))
                cache[dv] = BayesNet(card, {k: parents[k] for k in cpts}, c2)
            else:
                cache[dv] = base
        bn = cache[dv]
        ev = {k: scn[k] for k in ms.OBSERVED}
        ev["Rr"], ev["Ra"] = scn["Rr"], scn["Ra"]
        Lce = L_rf[:, :, rf_hard[i]]
        if eo_like is None:
            ev["Re"] = scn["Re"]
        else:
            Lce = Lce * eo_like(dv)[:, eo_rep[i]][:, None]
        P[i] = eg.query_with_ce_virtual(bn, ev, Lce)
        y[i] = scn["T"]
    return P, y


def macro_f1(conf):
    tp = np.diag(conf).astype(float)
    col, row = conf.sum(0).astype(float), conf.sum(1).astype(float)
    prec = np.divide(tp, col, out=np.zeros_like(tp), where=col > 0)
    rec = np.divide(tp, row, out=np.zeros_like(tp), where=row > 0)
    f1 = np.divide(2 * prec * rec, prec + rec, out=np.zeros_like(tp), where=(prec + rec) > 0)
    return float(f1.mean()), rec


def stage_main(a, X, meta):
    g2 = pd.read_csv(os.path.join(RES, f"{P}_g2.csv"))
    if not bool(g2["pass"].all()):
        raise SystemExit("G2 failed: no node-level main run and no mixed world (not-feasible branch)")
    y = meta.y.values
    fold = meta.fold.values
    out = run_folds(X, y, fold, a.workers)
    rep_val, rep_te, save = {}, {}, {}
    for k in range(NFOLD):
        vi, ti = np.where(fold == (k + 1) % NFOLD)[0], np.where(fold == k)[0]
        rv = np.array([REP_OF[CLASSES[j]] for j in out[k][0].argmax(1)])
        rt = np.array([REP_OF[CLASSES[j]] for j in out[k][1].argmax(1)])
        rep_val[k], rep_te[k] = {"idx": vi, "rep": rv}, {"idx": ti, "rep": rt}
        save.update({f"fold{k}__val_idx": vi, f"fold{k}__test_idx": ti,
                     f"fold{k}__val_softmax": out[k][0], f"fold{k}__test_softmax": out[k][1],
                     f"fold{k}__val_acc": np.array(out[k][2])})
    np.savez_compressed(os.path.join(RES, f"{P}_cnn.npz"), **save)
    meta.to_csv(os.path.join(RES, f"{P}_patches_meta.csv"), index=False)
    node_and_reliability(meta, rep_val, rep_te, out)
    e2e(meta, rep_val, rep_te)


def load_saved():
    meta = pd.read_csv(os.path.join(RES, f"{P}_patches_meta.csv"))
    z = np.load(os.path.join(RES, f"{P}_cnn.npz"))
    rep_val, rep_te, out = {}, {}, {}
    for k in range(NFOLD):
        rv = np.array([REP_OF[CLASSES[j]] for j in z[f"fold{k}__val_softmax"].argmax(1)])
        rt = np.array([REP_OF[CLASSES[j]] for j in z[f"fold{k}__test_softmax"].argmax(1)])
        rep_val[k] = {"idx": z[f"fold{k}__val_idx"], "rep": rv}
        rep_te[k] = {"idx": z[f"fold{k}__test_idx"], "rep": rt}
        out[k] = (z[f"fold{k}__val_softmax"], z[f"fold{k}__test_softmax"], float(z[f"fold{k}__val_acc"]))
    return meta, rep_val, rep_te, out


def node_and_reliability(meta, rep_val, rep_te, out):
    y = meta.y.values
    pred, _ = pooled_test(out, y, meta.fold.values)
    conf = np.zeros((4, 4), dtype=int)
    np.add.at(conf, (y, pred), 1)
    f1, rec = macro_f1(conf)
    tr = meta.cls.map(REP_OF).values
    pr = np.array([REP_OF[CLASSES[j]] for j in pred])
    conf3 = np.zeros((3, 3), dtype=int)
    np.add.at(conf3, (tr, pr), 1)
    rows = [{"scope": "all", "n": len(y), "acc": float((pred == y).mean()), "macroF1": f1,
             "acc3": float((pr == tr).mean()),
             "confusion4": " ".join(map(str, conf.ravel())), "confusion3": " ".join(map(str, conf3.ravel())),
             **{f"recall_{c}": float(rec[i]) for i, c in enumerate(CLASSES)}}]
    for b in BINS:
        s = (meta["bin"] == b).values
        dr = s & (meta.cls == "DRONE").values
        rows.append({"scope": b, "n": int(s.sum()), "acc": float((pred[s] == y[s]).mean()),
                     "acc3": float((pr[s] == tr[s]).mean()),
                     "recall_DRONE": float((pred[dr] == y[dr]).mean()) if dr.any() else np.nan})
    pd.DataFrame(rows).to_csv(os.path.join(RES, f"{P}_node.csv"), index=False)
    reliability_folds(meta, rep_val, rep_te).to_csv(os.path.join(RES, f"{P}_reliability.csv"), index=False)


# ------------------------------------------------------------- node level ---
def session_bootstrap(y, preds, sess, base=None, b=BOOT_B, seed=BOOT_SEED):
    """Session-cluster bootstrap of the balanced accuracy, descriptive and not
    fixed in advance: sessions are resampled within class with the predictions
    held fixed; one generator runs through preds in order. With base (same
    keys) the interval is for the difference from base on the same resamples."""
    rng = np.random.default_rng(seed)
    by_cls = {k: pd.unique(sess[y == k]) for k in range(4)}
    idx = {s: np.where(sess == s)[0] for s in pd.unique(sess)}
    rows = []
    for name, p in preds.items():
        q = None if base is None else base[name]
        est = cls_metrics(y, p)[2] - (0.0 if q is None else cls_metrics(y, q)[2])
        bs = np.empty(b)
        for i in range(b):
            ii = np.concatenate([idx[s] for k in range(4)
                                 for s in rng.choice(by_cls[k], len(by_cls[k]), replace=True)])
            bs[i] = cls_metrics(y[ii], p[ii])[2] - (0.0 if q is None else cls_metrics(y[ii], q[ii])[2])
        lo, hi = np.percentile(bs, [2.5, 97.5])
        rows.append({"name": name, "n": len(y), "sessions": len(idx), "estimate": est, "lo": float(lo), "hi": float(hi),
                     "share_le_0.35": float(np.mean(bs <= G2_BALANCED_LIMIT)) if q is None else np.nan,
                     "share_le_0.25": float(np.mean(bs <= 0.25)) if q is None else np.nan,
                     "share_le_0": float(np.mean(bs <= 0.0)) if q is not None else np.nan,
                     "B": b, "seed": seed})
    return pd.DataFrame(rows)


def g2_bootstrap():
    """Interval of the G2 balanced accuracies from the saved predictions."""
    pr = pd.read_csv(os.path.join(RES, f"{P}_g2_predictions.csv"))
    assert len(pr) > 0
    bs = session_bootstrap(pr.y.values, {"1-NN masked (LOSO)": pr.pred_nn.values,
                                         "CNN masked (4-fold sessions)": pr.pred_cnn.values}, pr.session.values)
    bs.to_csv(os.path.join(RES, f"{P}_g2_bootstrap.csv"), index=False)
    return bs


def node_rows(model, patches, mm, pred):
    """[N] metrics of one classifier on one set of patches: overall, per bin,
    per fold of [P]."""
    rows = []

    def one(scope, s):
        y, p = mm.y.values[s], pred[s]
        conf, rec, ba = cls_metrics(y, p)
        tr = mm.cls.map(REP_OF).values[s]
        pr = np.array([REP_OF[CLASSES[j]] for j in p])
        conf3 = np.zeros((3, 3), dtype=int)
        np.add.at(conf3, (tr, pr), 1)
        rows.append({"model": model, "patches": patches, "scope": scope, "n": int(s.sum()),
                     "acc": float((p == y).mean()), "balanced_acc": ba, "macroF1": macro_f1(conf)[0],
                     "acc3": float((pr == tr).mean()), **{f"recall_{c}": float(rec[i]) for i, c in enumerate(CLASSES)},
                     "confusion4": " ".join(map(str, conf.ravel())), "confusion3": " ".join(map(str, conf3.ravel()))})
    one("all", np.ones(len(mm), dtype=bool))
    for b in BINS:
        one(b, (mm["bin"] == b).values)
    for k in range(NFOLD):
        one(f"fold{k}", mm.fold.values == k)
    return rows


def stage_node(a, X, meta):
    """Node-level run after the G2 failure (see the docstring), a
    descriptive check only: unmasked CNN and unmasked 1-NN; no [R], [E] or [H]."""
    g2 = pd.read_csv(os.path.join(RES, f"{P}_g2.csv"))
    assert not bool(g2["pass"].all()), "this stage belongs to the branch where G2 failed"
    _, keep = masked(X, meta)
    assert np.array_equal(keep, np.load(os.path.join(RES, f"{P}_g2_cnn.npz"))["keep"]), "not the patches of G2"
    mk = meta[keep].reset_index(drop=True)
    pr = pd.read_csv(os.path.join(RES, f"{P}_g2_predictions.csv"))
    for col in ("cls", "num", "frame", "session", "fold", "y"):
        assert np.array_equal(pr[col].values, mk[col].values), f"G2 predictions differ in {col}"
    n = int(keep.sum())
    Z = X[keep].reshape(n, 16, 4, 16, 4).mean(axis=(2, 4)).reshape(n, -1)
    pred_nn = nn_loso(Z.astype(np.float64), mk.y.values, mk.session.values)
    y, fold = meta.y.values, meta.fold.values
    out = run_folds(X, y, fold, a.workers)
    pred_cnn, sm = pooled_test(out, y, fold)
    pd.DataFrame(node_rows("CNN unmasked", "all", meta, pred_cnn) + node_rows("CNN unmasked", "G2", mk, pred_cnn[keep])
                 + node_rows("1-NN unmasked", "G2", mk, pred_nn)).to_csv(os.path.join(RES, f"{P}_node.csv"), index=False)
    pp = meta[["cls", "num", "bin", "session", "fold", "frame", "w", "h", "big", "y"]].assign(
        in_g2=keep, pred_cnn=pred_cnn, pred_nn_g2=-1)
    pp.loc[keep, "pred_nn_g2"] = pred_nn
    pp.to_csv(os.path.join(RES, f"{P}_node_predictions.csv"), index=False)
    np.savez_compressed(os.path.join(RES, f"{P}_node_cnn.npz"), keep=keep, test_softmax=sm,
                        **{f"fold{k}__val_acc": np.array(out[k][2]) for k in range(NFOLD)})
    yk, sk = mk.y.values, mk.session.values
    pd.concat([
        session_bootstrap(yk, {"1-NN unmasked, G2 patches": pred_nn, "CNN unmasked, G2 patches": pred_cnn[keep]}, sk),
        session_bootstrap(y, {"CNN unmasked, all patches": pred_cnn}, meta.session.values),
        session_bootstrap(yk, {"1-NN unmasked minus masked": pred_nn, "CNN unmasked minus masked": pred_cnn[keep]}, sk,
                          base={"1-NN unmasked minus masked": pr.pred_nn.values,
                                "CNN unmasked minus masked": pr.pred_cnn.values}),
    ]).to_csv(os.path.join(RES, f"{P}_node_bootstrap.csv"), index=False)


def _paired(a, b):
    from scipy import stats as st
    d = np.asarray(a, float) - np.asarray(b, float)
    return float(d.mean()), float(st.ttest_1samp(d, 0.0).pvalue)


def cp_interval(k, n):
    from scipy import stats as st
    lo = st.beta.ppf(0.025, k, n - k + 1) if k > 0 else 0.0
    hi = st.beta.ppf(0.975, k + 1, n - k) if k < n else 1.0
    return float(lo), float(hi)


def summarize():
    L = ["# Exp-O: EO/IR channel on real thermal-infrared clips, mixed world with two real channels", ""]
    g1 = os.path.join(RES, f"{P}_g1.txt")
    if os.path.exists(g1):
        L += ["## G1 (boxes)", "```", open(g1, encoding="utf-8").read().strip(), "```"]
    g2p = os.path.join(RES, f"{P}_g2.csv")
    if os.path.exists(g2p):
        g2 = pd.read_csv(g2p)
        L += ["", f"## G2 (confound gate, mask M = {MASK_M}; both checks must meet both conditions)"]
        for _, r in g2.iterrows():
            L.append(f"- {r.check} (n {int(r.n)}): accuracy {r.acc:.3f} vs limit {r.limit:.3f} (majority share "
                     f"{r.chance:.3f}) -> {'pass' if r.pass_acc else 'FAIL'}; balanced accuracy {r.balanced_acc:.3f} "
                     f"vs limit {r.limit_balanced:.2f} -> {'pass' if r.pass_balanced else 'FAIL'}; overall "
                     f"{'pass' if r['pass'] else 'FAIL'}")
            L.append("  recall " + ", ".join(f"{c} {r[f'recall_{c}']:.3f}" for c in CLASSES)
                     + f"; 4x4 confusion (rows true {', '.join(CLASSES)}): {r.confusion4}"
                     + (f"; no-information accuracy (sum of squared shares) {r.no_info_acc:.3f}"
                        if pd.notna(r.no_info_acc) else ""))
        L.append(f"- G2 overall: {'PASS' if bool(g2['pass'].all()) else 'FAIL'}")
        gf = pd.read_csv(os.path.join(RES, f"{P}_g2_folds.csv"))
        for chk, z in gf.groupby("check", sort=False):
            L.append(f"- per fold, {chk}: " + "; ".join(
                f"fold {int(r.fold)} n {int(r.n)} acc {r.acc:.3f} (majority {r.chance:.3f}) balanced {r.balanced_acc:.3f}"
                for _, r in z.iterrows()))
        lo = pd.read_csv(os.path.join(RES, f"{P}_g2_left_out.csv"))
        lo = lo.assign(_b=lo["bin"].map({b: i for i, b in enumerate(BINS)})).sort_values(["cls", "_b"])
        L.append(f"- patches left out of G2 (box side > M - {2 * MASK_MARGIN}), share by class and bin: "
                 + "; ".join(f"{r.cls}/{r['bin']} {r.share_left_out:.2f} ({int(r.left_out)}/{int(r.patches)})"
                             for _, r in lo.iterrows()))
    g3p = os.path.join(RES, f"{P}_g3.csv")
    if os.path.exists(g3p):
        g3 = pd.read_csv(g3p)
        L += ["", "## G3 (test sessions per class and bin)", "| class | bin | clips | patches | sessions | per fold | thin |",
              "|---|---|---|---|---|---|---|"]
        for _, r in g3.iterrows():
            L.append(f"| {r.cls} | {r['bin']} | {r.clips} | {r.patches} | {r.sessions} | "
                     f"{'/'.join(str(r[f'sessions_fold{k}']) for k in range(NFOLD))} | {'thin' if r.thin else ''} |")
        gfp = os.path.join(RES, f"{P}_g3_folds.csv")
        if os.path.exists(gfp):
            for _, r in pd.read_csv(gfp).iterrows():
                L.append(f"- test fold {int(r.fold)}: {int(r.classes)} classes, {int(r.clips)} clips, "
                         f"{int(r.sessions)} sessions, {int(r.patches)} patches; clips (sessions) "
                         + ", ".join(f"{c} {int(r[f'{c}_clips'])} ({int(r[f'{c}_sessions'])})" for c in CLASSES))
    gbp = os.path.join(RES, f"{P}_g2_bootstrap.csv")
    if os.path.exists(gbp):
        L += ["", "## G2 session bootstrap (descriptive, not part of the decision)"]
        for _, r in pd.read_csv(gbp).iterrows():
            L.append(f"- {r['name']}: balanced accuracy {r.estimate:.3f}, 95% interval [{r.lo:.3f}, {r.hi:.3f}], share of "
                     f"resamples <= 0.35 {r['share_le_0.35']:.3f}, <= 0.25 {r['share_le_0.25']:.3f} "
                     f"(B {int(r.B)}, seed {int(r.seed)}, {int(r.sessions)} sessions)")
    nd = os.path.join(RES, f"{P}_node.csv")
    q = pd.read_csv(nd) if os.path.exists(nd) else None
    if q is not None and "model" in q.columns:
        L += ["", "## [N] node level after the G2 failure (a descriptive check only; [R], [E], [H] not run)"]
        for (m_, pt), z in q.groupby(["model", "patches"], sort=False):
            r0 = z[z.scope == "all"].iloc[0]
            L.append(f"- {m_}, {pt} patches (n {int(r0.n)}): accuracy {r0.acc:.3f}, balanced accuracy {r0.balanced_acc:.3f}, "
                     f"macro-F1 {r0.macroF1:.3f}, 3-way report accuracy {r0.acc3:.3f}")
            L.append("  recall " + ", ".join(f"{c} {r0[f'recall_{c}']:.3f}" for c in CLASSES)
                     + f"; 4x4 confusion (rows true {', '.join(CLASSES)}): {r0.confusion4}"
                     + f"; 3x3 report confusion (rows true drone, bird, other): {r0.confusion3}")
            L.append("  per bin: " + "; ".join(
                f"{b} n {int(r.n)} acc {r.acc:.3f} balanced {r.balanced_acc:.3f} drone recall {r.recall_DRONE:.3f}"
                for b in BINS for _, r in z[z.scope == b].iterrows()))
            L.append("  balanced accuracy per fold: " + ", ".join(
                f"fold {k} {z[z.scope == f'fold{k}'].iloc[0].balanced_acc:.3f}" for k in range(NFOLD)))
        g2m = pd.read_csv(g2p).set_index("check")
        un = q[(q.patches == "G2") & (q.scope == "all")].set_index("model")
        L.append("- masked against unmasked on the same G2 patches: " + "; ".join(
            f"{tag}: accuracy {g2m.loc[chk].acc:.3f} -> {un.loc[mod].acc:.3f}, balanced accuracy "
            f"{g2m.loc[chk].balanced_acc:.3f} -> {un.loc[mod].balanced_acc:.3f}"
            for tag, chk, mod in (("1-NN", "1-NN masked (LOSO)", "1-NN unmasked"),
                                  ("CNN", "CNN masked (4-fold sessions)", "CNN unmasked"))))
        for _, r in pd.read_csv(os.path.join(RES, f"{P}_node_bootstrap.csv")).iterrows():
            L.append(f"- session bootstrap, {r['name']}: {r.estimate:+.3f}, 95% interval [{r.lo:.3f}, {r.hi:.3f}] "
                     f"(B {int(r.B)}, seed {int(r.seed)}, {int(r.sessions)} sessions)"
                     + (f", share of resamples <= 0: {r.share_le_0:.3f}" if pd.notna(r.share_le_0) else ""))
    elif q is not None:
        a0 = q[q.scope == "all"].iloc[0]
        L += ["", "## [N] node level (test folds pooled)",
              f"- accuracy {a0.acc:.4f}, macro-F1 {a0.macroF1:.4f}, 3-way report accuracy {a0.acc3:.4f}",
              "- recall " + ", ".join(f"{c} {a0[f'recall_{c}']:.3f}" for c in CLASSES),
              f"- 4x4 confusion (rows true {CLASSES}): {a0.confusion4}", f"- 3x3 report confusion: {a0.confusion3}"]
        for b in BINS:
            r = q[q.scope == b].iloc[0]
            L.append(f"- {b}: n {int(r.n)}, accuracy {r.acc:.3f}, 3-way {r.acc3:.3f}, drone recall {r.recall_DRONE:.3f}")
        rl = pd.read_csv(os.path.join(RES, f"{P}_reliability.csv"))
        L += ["", "## [R] reliability with range context (per fold, then mean)",
              "| bin | alpha Eq.(10) | omega Eq.(11) | acc | LL none | LL alpha | gain (p over folds) |", "|---|---|---|---|---|---|---|"]
        for b in BINS:
            z = rl[rl["bin"] == b]
            m, pv = _paired(z.ll_alpha, z.ll_none) if len(z) > 1 else (np.nan, np.nan)
            L.append(f"| {b} | {z.alpha.mean():.3f} | {z.omega.mean():.3f} | {z.acc.mean():.3f} | {z.ll_none.mean():.3f} | "
                     f"{z.ll_alpha.mean():.3f} | {m:+.4f} ({pv:.2g}) |")
    ep = os.path.join(RES, f"{P}_e2e_raw.csv")
    if os.path.exists(ep):
        e = pd.read_csv(ep, float_precision="round_trip")
        g0 = pd.read_csv(os.path.join(RES, "expG_rf_e2e_raw.csv"), float_precision="round_trip")
        sc = e[e.variant == "syn-check"].sort_values("rep")
        gh = g0[g0.rf_likelihood == "calib-hard"].sort_values("rep")
        dev = max(float(np.max(np.abs(sc[k].values - gh[k].values))) for k in ("acc", "macroAUC", "ece", "brier"))
        L += ["", "## [E] mixed world with real RF and real EO/IR (20 replications)",
              f"- check: synthetic EO/IR through this code path vs expG calib-hard: max |difference| {dev:.2e} "
              f"-> {'reproduced' if dev <= 1e-9 else 'NOT reproduced'}",
              "| variant | acc | macro-AUC | Brier | ECE |", "|---|---|---|---|---|"]
        for v in e.variant.unique():
            z = e[e.variant == v]
            L.append(f"| {v} | {z.acc.mean():.4f} | {z.macroAUC.mean():.4f} | {z.brier.mean():.4f} | {z.ece.mean():.4f} |")
        V = lambda n: e[e.variant == n].sort_values("rep")
        ma, pa = _paired(V("(ii) real calibration").macroAUC, V("(i-a) abstract").macroAUC)
        me, pe = _paired(V("(ii) real calibration").ece, V("(i-a) abstract").ece)
        L.append(f"- (E1-O) (ii) - (i-a): dAUC {ma:+.4f} (p {pa:.1e}), dECE {me:+.4f} (p {pe:.1e}) -> "
                 f"{'holds' if (ma > 0 and pa < 0.05 and me < 0 and pe < 0.05) else 'does not hold'}")
        m2, p2 = _paired(V("(ii) real calibration").ece, V("(iii) real, no discount").ece)
        L.append(f"- (E2-O) ECE(ii) - ECE(iii): {m2:+.4f} (p {p2:.1e}) -> "
                 f"{'matters' if (abs(m2) >= 0.005 and p2 < 0.05) else 'not shown'}")
        for n in ("(i-b) synthetic calibration", "(v) chance-corrected"):
            mb, pb = _paired(V(n).brier, V("(ii) real calibration").brier)
            mc, pc = _paired(V(n).ece, V("(ii) real calibration").ece)
            L.append(f"- descriptive {n} - (ii): dBrier {mb:+.4f} (p {pb:.1e}), dECE {mc:+.4f} (p {pc:.1e})")
        fbk = pd.read_csv(os.path.join(RES, f"{P}_fallback.csv"))
        if len(fbk):
            agg = fbk.groupby(["phase", "cls", "bin", "fold"]).draws.sum().reset_index()
            L.append("- fallback draws (empty (class, bin) pool of the fold), summed over replications: "
                     + "; ".join(f"{r.phase} {r.cls}/{r['bin']} fold {int(r.fold)}: {int(r.draws)}" for _, r in agg.iterrows()))
        else:
            L.append("- fallback draws: none")
        hb = pd.read_csv(os.path.join(RES, f"{P}_highthreat_bins.csv"))
        al = pd.read_csv(os.path.join(RES, f"{P}_alarm.csv"))
        for v in ("(ii) real calibration", "(iv) CPT oracle"):
            L += ["", f"### [H] {v}", "| bin | n | predicted | observed [95% CP] | outside |", "|---|---|---|---|---|"]
            for lo in TBINS[:-1]:
                z = hb[(hb.variant == v) & (hb.bin_lo == lo)]
                n, k, sp = int(z.n.sum()), int(z.hits.sum()), float(z.sum_pred.sum())
                if n == 0:
                    continue
                l_, h_ = cp_interval(k, n)
                L.append(f"| ({lo:.2f}] | {n} | {sp / n:.3f} | {k / n:.3f} [{l_:.3f}, {h_:.3f}] | "
                         f"{'*' if not (l_ <= sp / n <= h_) else ''} |")
            for thr in (0.5, 0.25):
                z = al[(al.variant == v) & (al.threshold == thr)]
                tp, fp, fn = int(z.tp.sum()), int(z.fp.sum()), int(z.fn.sum())
                L.append(f"- alarm at {thr}: precision {tp / max(tp + fp, 1):.3f}, recall {tp / max(tp + fn, 1):.3f}")
    txt = "\n".join(L)
    open(os.path.join(RES, f"{P}_summary.md"), "w", encoding="utf-8").write(txt + "\n")
    print(txt)


def prepare(a):
    """Manifest check, description, sessions, patches and folds (shared by
    the g2 and main stages; deterministic)."""
    n = check_manifest(a.data, a.manifest)
    print(f"[expO] manifest: {n} files match their SHA-256", flush=True)
    desc = load_description(a.data)
    assert len(desc) == 365
    sid, bnd, nfr = sessions(a.data, desc)
    print(f"[expO] boxes {os.path.basename(a.boxes)} SHA-256 {sha256_file(a.boxes)}", flush=True)
    boxes = load_boxes(a.boxes)
    X, meta, dropped = build_patches(a.data, desc, boxes, sid, nfr)
    meta["fold"] = assign_folds(meta)
    meta["y"] = meta.cls.map({c: i for i, c in enumerate(CLASSES)}).astype(int)
    print(f"[expO] patches {len(X)} from {meta.drop_duplicates(['cls', 'num']).shape[0]} clips "
          f"({dropped} clips without a box dropped); sessions {meta.session.nunique()}", flush=True)
    return desc, sid, nfr, X, meta, dropped


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.environ.get("CUAS_EO_DATA", ""))
    ap.add_argument("--boxes", default=os.environ.get("CUAS_EO_BOXES", ""))
    ap.add_argument("--manifest", default=os.environ.get("CUAS_EO_MANIFEST", ""))
    ap.add_argument("--stage", default="", help="g1, g2, main or node")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--summary-only", action="store_true")
    a = ap.parse_args()
    if a.stage in ("g1", "g2", "main", "node"):
        for p in (a.data, a.boxes, a.manifest):
            assert p and not os.path.normcase(os.path.abspath(p)).startswith(os.path.normcase(HERE)), \
                "data, boxes and manifest must be outside this repository"
    if a.stage == "g1":
        n = check_manifest(a.data, a.manifest)
        desc = load_description(a.data)
        assert len(desc) == 365
        sid, _, nfr = sessions(a.data, desc)
        txt = f"manifest: {n} files match their SHA-256\n" + stage_g1(a.data, a.boxes, a.manifest, desc, sid, nfr)
        open(os.path.join(RES, f"{P}_g1.txt"), "w", encoding="utf-8").write(txt + "\n")
        print(txt)
        return
    if a.stage in ("g2", "main", "node"):
        assert MASK_M is not None, "MASK_M must be fixed from the G1 box sizes before G2"
        desc, sid, nfr, X, meta, dropped = prepare(a)
        if a.stage == "g2":
            g3_table(meta).to_csv(os.path.join(RES, f"{P}_g3.csv"), index=False)
            g3_folds(meta).to_csv(os.path.join(RES, f"{P}_g3_folds.csv"), index=False)
            g2, gf, ok, share = stage_g2(a, desc, X, meta)
            print(f"[expO] G2 uses {share:.3f} of the patches (box side <= {MASK_M}); pass: {ok}", flush=True)
        elif a.stage == "node":
            g2_bootstrap()
            stage_node(a, X, meta)
        else:
            stage_main(a, X, meta)
    summarize()


if __name__ == "__main__":
    main()
