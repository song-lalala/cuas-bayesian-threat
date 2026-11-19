# Exp-EM: identifiability of the unlabeled case and quality of the latent posteriors

## Jacobian rank at the true tables
| what | parameters | rank | singular value at the rank | next singular value |
|---|---|---|---|---|
| Q, 126 free parameters | 126 | 112 | 0.0185 | 2.67e-15 |
| Q, T known, 72 free parameters | 72 | 72 | 0.0512 | 0 |
| P(T, reports | inputs), 126 free parameters | 126 | 112 | 0.00156 | 4.82e-16 |
| P(T, reports | inputs), T known, 72 free parameters | 72 | 72 | 0.00535 | 0 |
| recoding tangents | 14 | 14 | nan | 4.39e-16 |

Last row: the next column gives the largest entry of J times the recoding tangents.

## reach of a recoding near the true tables (SLSQP from R = I, lower bounds; permutations excluded)
searches that ended infeasible and count as no move: 12 of 366
| table | largest move of any entry |
|---|---|
| H | 0.039 |
| N | 0.006 |
| C | 0.048 |
| T | 0.038 |

entries searched, both directions: 366

the recoding with the largest move: entries move by up to H 0.018, N 0.006, C 0.048, T 0.027; Q changes by at most 3.3e-16 and P(T, reports | inputs) by at most 5.6e-17

## latent posteriors on test scenarios (means over 20 replications)
| method | N | H acc | N acc | C acc | H log loss | N log loss | C log loss |
|---|---|---|---|---|---|---|---|
| prior only | 0 | 0.787 | 0.739 | 0.846 | 0.579 | 0.531 | 0.356 |
| true CPTs | 0 | 0.883 | 0.748 | 0.923 | 0.262 | 0.499 | 0.194 |
| Proposed | 50 | 0.835 | 0.741 | 0.915 | 0.404 | 0.511 | 0.221 |
| MAP-EM | 50 | 0.789 | 0.734 | 0.856 | 0.552 | 0.542 | 0.311 |
| Proposed | 200 | 0.868 | 0.744 | 0.923 | 0.318 | 0.502 | 0.209 |
| MAP-EM | 200 | 0.804 | 0.737 | 0.888 | 0.498 | 0.523 | 0.266 |

## MAP-EM minus Proposed (paired t over 20 replications)
| N | node | d acc (p) | d log loss (p) |
|---|---|---|---|
| 50 | H | -0.045 (0.00068) | +0.148 (0.0017) |
| 50 | N | -0.006 (0.13) | +0.031 (0.00037) |
| 50 | C | -0.060 (0.00095) | +0.090 (3.6e-05) |
| 200 | H | -0.063 (0.00042) | +0.180 (0.0011) |
| 200 | N | -0.006 (0.11) | +0.021 (0.00091) |
| 200 | C | -0.036 (0.0054) | +0.057 (5.9e-05) |
