# Exp-W-add: additivity diagnostic [A] and adaptive rules (20 replications)

- [A] tests: 220; ICI fit identical to the deployed fit in 220; max |cMAP-full - deployed| 0.0e+00; solver: ECOS 0, SCS 4, MAP fallback 0 (all tests, all others CLARABEL)
- fresh-path check: 32 of 32 cells bit-identical to expW_raw.csv

## [A] flag rate (p_T < 0.05)
| world | N | flagged | median p |
|---|---|---|---|
| standard | 50 | 0/20 | 0.512 |
| standard | 200 | 0/20 | 0.453 |
| standard | 1000 | 1/20 | 0.613 |
| nonadd-0.5 | 50 | 7/20 | 0.092 |
| nonadd-0.5 | 200 | 20/20 | 0.005 |
| nonadd-0.5 | 1000 | 20/20 | 0.005 |
| nonadd-1.0 | 50 | 18/20 | 0.010 |
| nonadd-1.0 | 200 | 20/20 | 0.005 |
| nonadd-1.0 | 1000 | 20/20 | 0.005 |
| H-v-U | 50 | 0/20 | 0.483 |
| H-v-U | 200 | 1/20 | 0.498 |

## [D] in S1 (p_X < 0.0125)
| world | N | any node | per node H/N/C/T |
|---|---|---|---|
| nonadd-0.5 | 50 | 0/20 | 0/0/0/0 |
| nonadd-0.5 | 200 | 0/20 | 0/0/0/0 |
| nonadd-1.0 | 50 | 0/20 | 0/0/0/0 |
| nonadd-1.0 | 200 | 0/20 | 0/0/0/0 |

## adaptive selection (counts over 20 replications)
| world | N | Proposed | cMAP-full | Proposed-Hfree | fresh (combination) |
|---|---|---|---|---|---|
| standard | 50 | 20 | 0 | 0 | 0 |
| standard | 200 | 20 | 0 | 0 | 0 |
| nonadd-0.5 | 50 | 13 | 7 | 0 | 0 |
| nonadd-0.5 | 200 | 0 | 20 | 0 | 0 |
| nonadd-1.0 | 50 | 2 | 18 | 0 | 0 |
| nonadd-1.0 | 200 | 0 | 20 | 0 | 0 |
| H-v-U | 50 | 3 | 0 | 17 | 0 |
| H-v-U | 200 | 0 | 0 | 19 | 1 (T full+relax H) |

## Brier and ECE: Proposed-adaptive against the fixed estimators
| world | N | adaptive Brier | Proposed | cMAP-full | H-free | adaptive - Proposed (p) | adaptive - alternative (p) | adaptive ECE | ECE: adaptive - Proposed (p) |
|---|---|---|---|---|---|---|---|---|---|
| standard | 50 | 0.5506 | 0.5506 | 0.5566 | - | +0.0000 (nan) | -0.0061 (1.5e-03) | 0.0513 | +0.0000 (nan) |
| standard | 200 | 0.5280 | 0.5280 | 0.5308 | - | +0.0000 (nan) | -0.0028 (5.3e-04) | 0.0301 | +0.0000 (nan) |
| nonadd-0.5 | 50 | 0.5786 | 0.5792 | 0.5730 | - | -0.0006 (7.2e-01) | +0.0055 (7.0e-03) | 0.0523 | +0.0102 (2.4e-02) |
| nonadd-0.5 | 200 | 0.5330 | 0.5479 | 0.5330 | - | -0.0149 (8.4e-10) | +0.0000 (nan) | 0.0299 | -0.0094 (1.0e-02) |
| nonadd-1.0 | 50 | 0.5358 | 0.5586 | 0.5329 | - | -0.0229 (1.1e-05) | +0.0029 (1.7e-01) | 0.0979 | -0.0001 (9.9e-01) |
| nonadd-1.0 | 200 | 0.4472 | 0.4788 | 0.4472 | - | -0.0316 (1.2e-16) | +0.0000 (nan) | 0.0545 | -0.0157 (3.8e-05) |
| H-v-U | 50 | 0.5628 | 0.5665 | 0.5727 | 0.5623 | -0.0037 (4.0e-04) | +0.0005 (9.4e-02) | 0.0458 | +0.0001 (9.5e-01) |
| H-v-U | 200 | 0.5295 | 0.5422 | 0.5445 | 0.5291 | -0.0126 (5.4e-13) | +0.0004 (3.3e-01) | 0.0295 | +0.0015 (6.1e-01) |
