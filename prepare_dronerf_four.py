"""
Build droneRF_four.npz, the input of rf_confusion.py, exp_rf_e2e.py and
exp_rf_independent.py, from the DroneRF recordings (Allahham et al., Data in
Brief 26:104313, 2019; https://data.mendeley.com/datasets/f4c2b4n755, CC BY
4.0). The recordings are not redistributed with this repository.

Each CSV file of the dataset holds one band (H or L) of one segment, named
<BUI><band>_<segment>.csv. The BUI code gives the class:
  00000                         background
  10000, 10001, 10010, 10011    Bebop (four flight modes)
  10100, 10101, 10110, 10111    AR
  11000                         Phantom
Other codes in the dataset are not used.

Each file is cut into up to 10 chunks of 1,000,000 samples. Each chunk becomes
one spectrogram: scipy.signal.stft with nperseg 512, a step of 2048 samples
(noverlap = 512 - 2048, so the Hann windows do not touch) and a one-sided
spectrum, 20 log10(|Z| + 1e-8), bilinear resize to 128 x 128 (scipy.ndimage.zoom,
order 1), then zero mean and unit standard deviation per spectrogram.

Output arrays: X (N, 128, 128) float32, y (N,) int64, meta (N, 2) with the
source file name and the chunk index, class_names (4,). Rows follow the file
data_derived/droneRF_four_order.csv when it is given with --order, which is the
row order of the file behind the reported results (it came from parallel
completion); the CNN's data split depends on the row order. Without --order
they are sorted by file name and chunk.

Run:  python prepare_dronerf_four.py --raw /path/to/DroneRF --out droneRF_four.npz
          [--order data_derived/droneRF_four_order.csv] [--workers 8]
      python prepare_dronerf_four.py --compare built.npz other.npz
"""
from __future__ import annotations
import argparse
import csv
import os
import re
import sys
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from scipy.ndimage import zoom
from scipy.signal import stft

CLASS_OF_BUI = {"00000": 0, "10000": 1, "10001": 1, "10010": 1, "10011": 1,
                "10100": 2, "10101": 2, "10110": 2, "10111": 2, "11000": 3}
CLASS_NAMES = ["background", "bebop", "ar_drone", "phantom"]
FILE_NAME = re.compile(r"^(\d{5})([HL])_(\d+)\.csv$", re.IGNORECASE)
CHUNK_SAMPLES = 1_000_000
MAX_CHUNKS = 10
NFFT = 512
HOP = 2048
SIZE = 128


def spectrogram(chunk):
    _, _, z = stft(chunk, nperseg=NFFT, noverlap=NFFT - HOP, window="hann", return_onesided=True)
    db = 20.0 * np.log10(np.abs(z).astype(np.float32) + 1e-8)
    h, w = db.shape
    s = zoom(db, (SIZE / h, SIZE / w), order=1).astype(np.float32)
    return ((s - s.mean()) / (s.std() + 1e-8)).astype(np.float32)


def one_file(path):
    sig = np.fromfile(path, sep=",", dtype=np.float32)
    n = min(MAX_CHUNKS, sig.size // CHUNK_SAMPLES)
    if n == 0:
        raise ValueError(f"{path}: shorter than one chunk ({sig.size} samples)")
    return np.stack([spectrogram(sig[k * CHUNK_SAMPLES:(k + 1) * CHUNK_SAMPLES]) for k in range(n)])


def list_files(raw, limit=-1):
    out = []
    for d, _, fs in os.walk(raw):
        for f in fs:
            m = FILE_NAME.match(f)
            if m and m.group(1) in CLASS_OF_BUI:
                out.append(os.path.join(d, f))
    out.sort(key=lambda p: os.path.basename(p))
    return out[:limit] if limit > 0 else out


def build(raw, out, order=None, workers=8, limit=-1):
    files = list_files(raw, limit)
    assert files, f"no DroneRF file under {raw}"
    names = [os.path.basename(p) for p in files]
    assert len(set(names)) == len(names), "two files with one name"
    print(f"{len(files)} files, {workers} workers", flush=True)
    specs = {}
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for i, (name, x) in enumerate(zip(names, ex.map(one_file, files))):
            specs[name] = x
            if (i + 1) % 50 == 0:
                print(f"{i + 1}/{len(files)}", flush=True)
    rows = [(n, k) for n in names for k in range(len(specs[n]))]
    if order:
        with open(order, newline="", encoding="utf-8") as f:
            wanted = [(r["file"], int(r["chunk"])) for r in csv.DictReader(f)]
        if limit <= 0:
            assert sorted(wanted) == sorted(rows), "the order file does not list these rows"
            rows = wanted
    X = np.stack([specs[n][k] for n, k in rows]).astype(np.float32)
    y = np.array([CLASS_OF_BUI[FILE_NAME.match(n).group(1)] for n, _ in rows], dtype=np.int64)
    meta = np.array(rows, dtype=object)
    np.savez_compressed(out, X=X, y=y, meta=meta, class_names=np.array(CLASS_NAMES))
    print(f"{out}: X {X.shape}, classes {dict(zip(*np.unique(y, return_counts=True)))}", flush=True)


def compare(a, b):
    """Rows matched by (file, chunk); the order is reported apart."""
    A, B = np.load(a, allow_pickle=True), np.load(b, allow_pickle=True)
    Xa, Xb, ya, yb = A["X"], B["X"], A["y"], B["y"]
    ka = [(str(x), int(k)) for x, k in A["meta"]]
    kb = [(str(x), int(k)) for x, k in B["meta"]]
    ib = {k: i for i, k in enumerate(kb)}
    common = [k for k in ka if k in ib]
    print(f"rows {len(ka)} and {len(kb)}, in both {len(common)}; same order: {ka == kb}")
    assert common, "no row in common"
    ia = {k: i for i, k in enumerate(ka)}
    d = np.array([np.abs(Xa[ia[k]] - Xb[ib[k]]).max() for k in common])
    same = int((d == 0).sum())
    y_same = all(ya[ia[k]] == yb[ib[k]] for k in common)
    print(f"spectrograms identical bit for bit {same} of {len(common)}; largest |difference| {d.max():.3e}; labels the same: {y_same}; class names the same: {list(A['class_names']) == list(B['class_names'])}")
    return same == len(common) and y_same


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw")
    ap.add_argument("--out", default="droneRF_four.npz")
    ap.add_argument("--order", default=None)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=-1)
    ap.add_argument("--compare", nargs=2, default=None)
    a = ap.parse_args()
    if a.compare:
        sys.exit(0 if compare(*a.compare) else 1)
    build(a.raw, a.out, a.order, a.workers, a.limit)
