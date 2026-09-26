"""
Parameter-count reporting helpers (not part of the estimation path).

Kept out of the core modules so that editing the prose here never invalidates
the results-freshness gate in check_manuscript_numbers.py.
"""
from __future__ import annotations
import numpy as np
import model_spec as ms


def full_table_params(node):
    """Free parameters of the full tabular CPT for `node`."""
    cfgs = int(np.prod([ms.CARD[p] for p in ms.PARENTS[node]]))
    return cfgs * (ms.CARD[node] - 1)


def ordinal_ici_params(node, identifiable=True):
    """Parameters of the ordinal cumulative-logit ICI response.

    The optimizer carries sum_p (r_p - 1) parent effects + (K - 1) thresholds
    + 1 sharpness. The map (s, theta, beta) -> (c*s, theta/c, beta/c) leaves
    the induced table invariant, so one degree of freedom is NOT identifiable:
    the manuscript quotes the identifiable count (eight for the threat node),
    which is what this returns by default.
    """
    pa_cards = [ms.CARD[p] for p in ms.PARENTS[node]]
    raw = sum(r - 1 for r in pa_cards) + (ms.CARD[node] - 1) + 1
    return raw - 1 if identifiable else raw


def noisy_max_params(node):
    pa_cards = [ms.CARD[p] for p in ms.PARENTS[node]]
    return (sum(pa_cards) + 1) * (ms.CARD[node] - 1)


def weighted_sum_params(node):
    pa_cards = [ms.CARD[p] for p in ms.PARENTS[node]]
    return (len(pa_cards) - 1) + sum(pa_cards) * (ms.CARD[node] - 1)


if __name__ == "__main__":
    for node in ms.INTERNAL:
        print(f"{node}: full table {full_table_params(node)}, "
              f"ordinal ICI {ordinal_ici_params(node)} identifiable "
              f"({ordinal_ici_params(node, False)} carried), "
              f"Noisy-MAX {noisy_max_params(node)}, "
              f"weighted-sum {weighted_sum_params(node)}")
