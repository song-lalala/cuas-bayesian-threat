# Exp-TS: worked track under other training seeds; cells that move its posterior

- true world: drone 0.970, P(T>=medium) 0.987, P(T=high) 0.758
- seed 0: drone 0.959, P(T>=medium) 0.965, P(T=high) 0.633
- seeds 0-19, P(T=high): mean 0.730, range 0.633 to 0.832, below 0.5: 0, above the true world's: 7
- seeds 0-19, P(T>=medium): range 0.955 to 0.991
- seeds 0-19, drone posterior: range 0.959 to 0.959
- cells: 183; |swing| > 1e-12: 62 (H 3, N 12, C 3, T 44); |swing| > 1e-3: 25 (H 3, N 8, C 3, T 11)
- cells per table: H 81, N 12, C 18, T 72

## the RF report on this evidence (none), with and without it
| model | seed | RF report | P(drone) | P(silent) | P(T>=medium) | P(T=high) |
|---|---|---|---|---|---|---|
| true world | -1 | with | 0.970 | 0.890 | 0.987 | 0.758 |
| true world | -1 | without | 0.985 | 0.450 | 0.955 | 0.604 |
| calibrated | 0 | with | 0.959 | 0.826 | 0.965 | 0.633 |
| calibrated | 0 | without | 0.977 | 0.450 | 0.918 | 0.519 |
| abstract | 0 | with | 0.996 | 0.450 | 0.921 | 0.525 |
| abstract | 0 | without | 1.000 | 0.450 | 0.921 | 0.526 |

- seeds 0-19, rise of P(T=high) from the RF report: +0.086 to +0.207; below zero: 0
- P(Rf = none | drone, silent) in the estimated matrix (calibration seed 777): 0.9175
