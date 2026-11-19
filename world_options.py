"""
Generative-world options for the double-discounting study (item A4 of
exp_reliability_variants.py). The DEFAULT world is model_spec.ground_truth_bn() itself, returned
unchanged; nothing in model_spec is modified (the consistency gate's freshness
guard would otherwise invalidate every published result file).

'structural' world. Range degradation moves probability mass onto ONE
specific wrong report instead of mixing toward uniform, so the
uniform-mixing form shared by the standard world and the reliability layer
(alpha*C + (1-alpha)/K) no longer matches the truth. Exact row definitions,
with gamma(d) = model_spec.SENSOR_DEGRADE (unchanged schedule: radar
.12/.03/0, EO/IR .45/.08/0, acoustic .50/.12/0 at far/near/imminent) and
base(c) = the standard world's best-condition row (its gamma = 0 slice):

  listed rows      P(R | c, d) = (1 - gamma(d)) base(c) + gamma(d) onehot(target(c))
     radar    : drone -> 'clutter',  bird -> 'clutter'
     EO/IR    : drone -> 'bird',     bird -> 'other'
     acoustic : drone -> 'none'
  other rows       P(R | c, d) = (1 - gamma(d)) base(c) + gamma(d) / K
     (radar clutter, EO/IR clutter, acoustic clutter and bird keep the
      standard world's uniform mixing, byte for byte, so only the listed
      rows change)
  RF               unchanged (no range dependence).

Everything else -- roots, H, N, C, T, RF -- is byte-identical between the two
worlds (`identity_report` checks it). This matters because
dataio.build_inference_bn takes the CPTs it does not learn (the roots) from
model_spec.ground_truth_bn(), i.e. from the standard world; that is correct
in the structural world only because those CPTs coincide. The sensor-oracle
tables must be supplied explicitly per world (`true_marginal_tables`,
`true_conditional_tables`): dataio.true_marginal_confusion() is tied to the
standard world.
"""
from __future__ import annotations
import numpy as np
from cuas_bn import Factor, BayesNet
import model_spec as ms

WORLDS = ("standard", "structural")
DEGRADED = ("Rr", "Re", "Ra")
STRUCTURAL_TARGET = {
    "Rr": {"drone": "clutter", "bird": "clutter"},
    "Re": {"drone": "bird", "bird": "other"},
    "Ra": {"drone": "none"},
}


def _base_rows(gt, s):
    """Best-condition rows: the gamma = 0 (imminent) slice of the standard
    table, which equals model_spec's base matrix exactly."""
    gam = ms.SENSOR_DEGRADE[s]
    assert gam[2] == 0.0, (s, gam)
    f = gt.cpts[s]
    assert f.vars == ("cls", "d", s), f.vars
    return f.table[:, 2, :].copy()


def structural_sensor_table(s, gt=None):
    gt = gt or ms.ground_truth_bn()
    base = _base_rows(gt, s)
    K = base.shape[-1]
    gam = ms.SENSOR_DEGRADE[s]
    tab = np.empty((ms.CARD["cls"], ms.CARD["d"], K))
    for c, cname in enumerate(ms.STATES["cls"]):
        tgt = STRUCTURAL_TARGET[s].get(cname)
        for di, g in enumerate(gam):
            if tgt is None:                       # same expression as ms
                tab[c, di] = (1 - g) * base[c] + g * (1.0 / K)
            else:
                e = np.zeros(K)
                e[ms.STATES[s].index(tgt)] = 1.0
                tab[c, di] = (1 - g) * base[c] + g * e
    return Factor(("cls", "d", s), tab)


def world_bn(world="standard"):
    gt = ms.ground_truth_bn()
    if world == "standard":
        return gt
    if world != "structural":
        raise ValueError(world)
    cpts = dict(gt.cpts)
    for s in DEGRADED:
        cpts[s] = structural_sensor_table(s, gt)
    return BayesNet(ms.CARD, ms.PARENTS, cpts)


def identity_report(world="structural"):
    """Names of the nodes whose CPT (vars or bytes) differs from the standard
    world, and, per degraded sensor, the rows that differ."""
    a, b = ms.ground_truth_bn(), world_bn(world)
    assert list(a.card) == list(b.card) and a.parents == b.parents
    differ = [v for v in a.card
              if a.cpts[v].vars != b.cpts[v].vars
              or a.cpts[v].table.tobytes() != b.cpts[v].table.tobytes()]
    rows = {}
    for s in differ:
        ta, tb = a.cpts[s].table, b.cpts[s].table
        rows[s] = sorted({ms.STATES["cls"][c] for c in range(ta.shape[0])
                          for di in range(ta.shape[1])
                          if ta[c, di].tobytes() != tb[c, di].tobytes()})
    return differ, rows


def true_marginal_tables(bn):
    """Range-marginal of bn's sensor CPTs under bn's P(d) -- the sensor oracle
    of the deployed structure, for any world (same operations as
    dataio.true_marginal_confusion, which is fixed to the standard world)."""
    pd = bn.cpts["d"].table
    out = {}
    for s in ms.SENSORS:
        f = bn.cpts[s]
        if "d" in f.vars:
            ax = f.vars.index("d")
            tab = np.tensordot(f.table, pd, axes=([ax], [0]))
            out[s] = Factor(tuple(v for v in f.vars if v != "d"), tab)
        else:
            out[s] = Factor(f.vars, f.table.copy())
    return out


def true_conditional_tables(bn):
    return {s: bn.cpts[s] for s in ms.SENSORS}


if __name__ == "__main__":
    import sys
    import dataio
    sys.stdout.reconfigure(encoding="utf-8")
    differ, rows = identity_report()
    print("structural world differs from standard in:", differ, rows)
    assert differ == ["Rr", "Re", "Ra"], differ
    assert rows == {"Rr": ["bird", "drone"], "Re": ["bird", "drone"],
                    "Ra": ["drone"]}, rows
    assert world_bn("standard") is not None
    tm_std = true_marginal_tables(ms.ground_truth_bn())
    tm_ref = dataio.true_marginal_confusion()
    for s in ms.SENSORS:
        assert tm_std[s].vars == tm_ref[s].vars
        assert tm_std[s].table.tobytes() == tm_ref[s].table.tobytes(), s
    st = world_bn("structural")
    for s in DEGRADED:
        assert np.allclose(st.cpts[s].table.sum(-1), 1.0)
        print(s, "far rows:\n", np.round(st.cpts[s].table[:, 0, :], 3))
    print("world_options OK")
