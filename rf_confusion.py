"""
Real-data RF-node calibration on the DroneRF benchmark.

Trains a compact CNN on the 4-class DroneRF spectrogram dataset
(background + Bebop / AR / Phantom) and reports the held-out confusion matrix,
row-normalized into the class-conditional report distribution P(report | class)
that the C-UAS Bayesian network consumes as the RF sensor likelihood.

Data: the processed DroneRF file `droneRF_four.npz` with arrays
  X: (N, 128, 128) float32 log-spectrograms, y: (N,) int64, class_names: (4,).
It is derived from the public DroneRF dataset (Allahham et al., Data in Brief
26:104313, 2019; https://data.mendeley.com/datasets/f4c2b4n755). Set the path
with the DRONERF_NPZ environment variable, or pass it as the first argument.

Run:  python rf_confusion.py [path/to/droneRF_four.npz]
Outputs (results/): rf_confusion_matrix.csv, rf_confusion.png, and the console
accuracy / macro-F1. Deterministic given the seed.
"""
import os, sys, numpy as np, torch, torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score

SEED = 42
torch.manual_seed(SEED); np.random.seed(SEED)
HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results"); os.makedirs(RES, exist_ok=True)
NPZ = (sys.argv[1] if len(sys.argv) > 1
       else os.environ.get("DRONERF_NPZ", "droneRF_four.npz"))


def load():
    d = np.load(NPZ, allow_pickle=True)
    X = d["X"].astype(np.float32)
    y = d["y"].astype(np.int64)
    names = [str(s) for s in d["class_names"]]
    # per-sample standardization (spectrograms are already log-scaled)
    X = (X - X.mean(axis=(1, 2), keepdims=True)) / (X.std(axis=(1, 2), keepdims=True) + 1e-6)
    return X, y, names


class CNN(nn.Module):
    def __init__(self, nc=4):
        super().__init__()
        self.f = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),   # 64
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),  # 32
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),  # 16
            nn.AdaptiveAvgPool2d(4))
        self.c = nn.Sequential(nn.Flatten(), nn.Dropout(0.3),
                               nn.Linear(64 * 16, 64), nn.ReLU(), nn.Linear(64, nc))

    def forward(self, x):
        return self.c(self.f(x))


def main():
    X, y, names = load()
    print("data:", X.shape, "classes:", names, "counts:", np.bincount(y).tolist())
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.30, stratify=y, random_state=SEED)
    Xtr = torch.from_numpy(Xtr).unsqueeze(1); Xte = torch.from_numpy(Xte).unsqueeze(1)
    ytr = torch.from_numpy(ytr); yte = torch.from_numpy(yte)

    net = CNN(len(names))
    opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=35)
    w = torch.tensor([1.0 / c for c in np.bincount(ytr.numpy())], dtype=torch.float32)
    lossf = nn.CrossEntropyLoss(weight=w / w.sum() * len(names))
    bs, best, best_state = 64, 0.0, None
    for ep in range(35):
        net.train(); perm = torch.randperm(len(Xtr))
        for i in range(0, len(Xtr), bs):
            idx = perm[i:i + bs]; opt.zero_grad()
            lossf(net(Xtr[idx]), ytr[idx]).backward(); opt.step()
        sched.step(); net.eval()
        with torch.no_grad():
            acc = accuracy_score(yte.numpy(), net(Xte).argmax(1).numpy())
        if acc > best:
            best, best_state = acc, {k: v.clone() for k, v in net.state_dict().items()}
        if (ep + 1) % 5 == 0:
            print(f"epoch {ep+1:2d}  test_acc={acc:.4f}  (best {best:.4f})")
    net.load_state_dict(best_state); net.eval()
    with torch.no_grad():
        pred = net(Xte).argmax(1).numpy()
    yt = yte.numpy()
    cm = confusion_matrix(yt, pred)
    cmn = cm / cm.sum(1, keepdims=True)
    acc = accuracy_score(yt, pred); mf1 = f1_score(yt, pred, average="macro")
    print("\ntest accuracy:", round(acc, 4), " macro-F1:", round(mf1, 4))
    print("row-normalized P(report | true):"); np.set_printoptions(precision=4, suppress=True); print(cmn)

    # save the row-normalized confusion matrix as CSV (rows=true, cols=report)
    import csv
    with open(os.path.join(RES, "rf_confusion_matrix.csv"), "w", newline="") as fh:
        wtr = csv.writer(fh); wtr.writerow(["true\\report"] + names)
        for i, r in enumerate(cmn):
            wtr.writerow([names[i]] + [f"{v:.4f}" for v in r])

    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        disp = ["background", "Bebop", "AR", "Phantom"]
        plt.rcParams.update({"font.family": "serif", "mathtext.fontset": "stix", "font.size": 12})
        fig, ax = plt.subplots(figsize=(4.0, 3.4))
        im = ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
        ax.set_xticks(range(4)); ax.set_yticks(range(4))
        ax.set_xticklabels(disp, rotation=30, ha="right"); ax.set_yticklabels(disp)
        ax.set_xlabel("RF report (predicted class)"); ax.set_ylabel("true class")
        for i in range(4):
            for j in range(4):
                ax.text(j, i, f"{cmn[i,j]:.2f}", ha="center", va="center",
                        color="white" if cmn[i, j] > 0.5 else "black", fontsize=11)
        cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cb.set_label(r"$P(\mathrm{report}\mid\mathrm{true})$")
        fig.tight_layout(); fig.savefig(os.path.join(RES, "rf_confusion.png"), dpi=300, bbox_inches="tight")
        print("saved results/rf_confusion.png")
    except Exception as e:
        print("(figure skipped:", e, ")")


if __name__ == "__main__":
    main()
