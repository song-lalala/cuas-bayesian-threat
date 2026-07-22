# Experimental results (synthetic digital-twin study)

Test set: 2500 held-out scenarios (seed 999); ESS=8.0. Oracle = true CPTs.

## Oracle (upper bound)

| model | acc | macroF1 | macroAUC | brier | ece | far@pd0.9 |
|---|---|---|---|---|---|---|
| oracle | 0.604 | 0.501 | 0.827 | 0.509 | 0.008 | 0.448 |

## Exp1 — CPT grounding under scarcity (mean over seeds)

**ECE by N (lower is better):**

| N | B0-heuristic | B1-MLE | B2-expert | MAP | Proposed-cMAP |
|---|---|---|---|---|---|
| 10.000 | 0.103 | 0.107 | 0.110 | 0.092 | 0.092 |
| 20.000 | 0.103 | 0.107 | 0.110 | 0.078 | 0.078 |
| 50.000 | 0.103 | 0.081 | 0.110 | 0.046 | 0.046 |
| 100.000 | 0.103 | 0.053 | 0.110 | 0.039 | 0.039 |
| 200.000 | 0.103 | 0.032 | 0.110 | 0.028 | 0.029 |
| 400.000 | 0.103 | 0.026 | 0.110 | 0.027 | 0.030 |

**Accuracy by N:**

| N | B0-heuristic | B1-MLE | B2-expert | MAP | Proposed-cMAP |
|---|---|---|---|---|---|
| 10.000 | 0.456 | 0.420 | 0.500 | 0.514 | 0.515 |
| 20.000 | 0.456 | 0.456 | 0.500 | 0.527 | 0.526 |
| 50.000 | 0.456 | 0.548 | 0.500 | 0.559 | 0.559 |
| 100.000 | 0.456 | 0.562 | 0.500 | 0.570 | 0.570 |
| 200.000 | 0.456 | 0.583 | 0.500 | 0.583 | 0.583 |
| 400.000 | 0.456 | 0.588 | 0.500 | 0.591 | 0.589 |

See `exp1_scarcity.png` for accuracy/AUC/Brier/ECE curves.

Sensor confusion matrices are ESTIMATED from a labelled calibration split of the same world (n=300); not the true CPTs.

## Exp2 — sensor fusion & reliability

| config | acc | macroF1 | macroAUC | brier | ece | far@pd0.9 |
|---|---|---|---|---|---|---|
| radar only | 0.544 | 0.409 | 0.784 | 0.561 | 0.039 | 0.509 |
| RF only | 0.525 | 0.379 | 0.754 | 0.589 | 0.044 | 0.588 |
| EO/IR only | 0.558 | 0.465 | 0.794 | 0.549 | 0.034 | 0.490 |
| acoustic only | 0.539 | 0.386 | 0.775 | 0.571 | 0.036 | 0.583 |
| all (abstract model) | 0.573 | 0.494 | 0.799 | 0.548 | 0.057 | 0.500 |
| all (calibrated) | 0.590 | 0.494 | 0.815 | 0.527 | 0.030 | 0.468 |
| all (calib+reliab) | 0.590 | 0.499 | 0.814 | 0.528 | 0.028 | 0.473 |

**Radio-silent-drone subset (reliability on/off):**

| subset | n | reliability | acc | macroF1 | macroAUC | brier | ece | far@pd0.9 |
|---|---|---|---|---|---|---|---|---|
| silent-drone (all ranges) | 389 | False | 0.617 | 0.420 | 0.722 | 0.497 | 0.054 | 0.259 |
| silent-drone (all ranges) | 389 | True | 0.604 | 0.406 | 0.712 | 0.521 | 0.062 | 0.296 |
| silent-drone (far range) | 120 | False | 0.483 | 0.292 | 0.794 | 0.558 | 0.149 | 0.500 |
| silent-drone (far range) | 120 | True | 0.475 | 0.286 | 0.741 | 0.603 | 0.100 | 0.478 |

## Exp2b — sensor-calibration-set size (estimated vs abstract vs oracle)

| calib_n | source | acc | macroF1 | macroAUC | brier | ece | far@pd0.9 |
|---|---|---|---|---|---|---|---|
| 0 | abstract | 0.573 | 0.494 | 0.799 | 0.548 | 0.057 | 0.500 |
| 30 | estimated | 0.576 | 0.477 | 0.810 | 0.533 | 0.031 | 0.474 |
| 100 | estimated | 0.580 | 0.486 | 0.812 | 0.530 | 0.033 | 0.476 |
| 300 | estimated | 0.590 | 0.494 | 0.815 | 0.527 | 0.030 | 0.468 |
| 1000 | estimated | 0.588 | 0.496 | 0.815 | 0.527 | 0.030 | 0.475 |
| -1 | oracle-true-CM | 0.591 | 0.503 | 0.816 | 0.526 | 0.030 | 0.469 |

## Exp3 — ablation (N=80)

| variant | acc | macroF1 | macroAUC | brier | ece | far@pd0.9 |
|---|---|---|---|---|---|---|
| -constraints (MAP) | 0.568 | 0.478 | 0.808 | 0.550 | 0.042 | 0.470 |
| -prior (MLE) | 0.554 | 0.471 | 0.787 | 0.566 | 0.066 | 0.521 |
| -reliability | 0.570 | 0.480 | 0.809 | 0.549 | 0.044 | 0.471 |
| -virtual evid. (hard) | 0.565 | 0.472 | 0.807 | 0.552 | 0.043 | 0.479 |
| Proposed (full) | 0.568 | 0.477 | 0.809 | 0.550 | 0.042 | 0.469 |

