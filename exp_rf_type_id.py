"""
AR-versus-Bebop identification on DroneRF from the saved CNN outputs of Exp-I, in
the metric of Shulman (arXiv:2607.01025, pooled out-of-fold macro-F1 on two labels).
Added after the first results of exp_base_rule.py; not fixed in advance.
exp_rf_independent.py and results/expI_* are only read.

For each split of Exp-I the test segments of Bebop (class 1) and AR (class 2) are
pooled over the split's instances. "Two-way" decides between the two classes only,
as in the preprint; "four-way" is the node's own decision over the four classes.
The splits differ from the preprint's: P3 holds out one flight mode of both drones
at once and keeps the other classes in training, the preprint holds out one recording
at a time; its inputs are hand-crafted features and its classifier a random forest.

Run:  python exp_rf_type_id.py      (writes results/expI_type_id.csv)
"""
from __future__ import annotations
import os, sys
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
SPLITS = ("P1", "P1c", "P2", "P3")


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    z = np.load(os.path.join(RES, "expI_cnn_dronerf.npz"), allow_pickle=True)
    inst = sorted({k.split("__")[0] for k in z.files if "__" in k})
    assert inst, "no instances in expI_cnn_dronerf.npz"
    rows = []
    for proto in SPLITS:
        ys, ss = [], []
        for k in inst:
            if k.rsplit("_", 1)[0] != proto:
                continue
            y, sm = z[f"{k}__te_y"], z[f"{k}__test_softmax"]
            m = np.isin(y, (1, 2))
            ys.append(y[m])
            ss.append(sm[m])
            two = 1 + sm[m][:, 1:3].argmax(1)
            rows.append({"split": proto, "instance": k, "n": int(m.sum()),
                         "macro_f1_two_way": float(f1_score(y[m], two, labels=[1, 2], average="macro")),
                         "acc_two_way": float((two == y[m]).mean())})
        assert ys, proto
        y, sm = np.concatenate(ys), np.concatenate(ss)
        two, four = 1 + sm[:, 1:3].argmax(1), sm.argmax(1)
        rows.append({"split": proto, "instance": "pooled", "n": len(y),
                     "macro_f1_two_way": float(f1_score(y, two, labels=[1, 2], average="macro")),
                     "macro_f1_four_way": float(f1_score(y, four, labels=[1, 2], average="macro")),
                     "acc_two_way": float((two == y).mean()), "acc_four_way": float((four == y).mean())})
    d = pd.DataFrame(rows)
    d.to_csv(os.path.join(RES, "expI_type_id.csv"), index=False)
    print(d[d.instance == "pooled"].round(4).to_string(index=False))
    print(d[d.split == "P3"].round(3)[["instance", "n", "macro_f1_two_way", "acc_two_way"]].to_string(index=False))


if __name__ == "__main__":
    main()
