# Exp-A: ESS sensitivity of the sample-efficiency claim, and the two reference baselines

## reference baselines (means over replications)
- class-prior constant: acc 0.434, macroAUC 0.500, ece 0.008, brier 0.654, far@pd0.9 1.000
- CPT-oracle: acc 0.593, macroAUC 0.824, ece 0.017, brier 0.516, far@pd0.9 0.455

## per-(N, alpha0) means
- N=10 MLE: acc 0.457, macroAUC 0.661, ece 0.111, brier 0.696
    Proposed a0=1: acc 0.535, macroAUC 0.792, ece 0.087, brier 0.587
    Proposed a0=2: acc 0.534, macroAUC 0.793, ece 0.077, brier 0.586
    Proposed a0=5: acc 0.527, macroAUC 0.789, ece 0.079, brier 0.595
- N=50 MLE: acc 0.533, macroAUC 0.761, ece 0.107, brier 0.602
    Proposed a0=1: acc 0.558, macroAUC 0.806, ece 0.056, brier 0.550
    Proposed a0=2: acc 0.559, macroAUC 0.806, ece 0.051, brier 0.551
    Proposed a0=5: acc 0.553, macroAUC 0.803, ece 0.051, brier 0.560
- N=100 MLE: acc 0.554, macroAUC 0.792, ece 0.074, brier 0.562
    Proposed a0=1: acc 0.566, macroAUC 0.812, ece 0.043, brier 0.537
    Proposed a0=2: acc 0.567, macroAUC 0.811, ece 0.040, brier 0.538
    Proposed a0=5: acc 0.564, macroAUC 0.809, ece 0.041, brier 0.545
- N=200 MLE: acc 0.571, macroAUC 0.810, ece 0.044, brier 0.536
    Proposed a0=1: acc 0.576, macroAUC 0.817, ece 0.032, brier 0.527
    Proposed a0=2: acc 0.577, macroAUC 0.817, ece 0.030, brier 0.528
    Proposed a0=5: acc 0.573, macroAUC 0.815, ece 0.031, brier 0.533
- N=400 MLE: acc 0.578, macroAUC 0.816, ece 0.032, brier 0.526
    Proposed a0=1: acc 0.583, macroAUC 0.820, ece 0.025, brier 0.522
    Proposed a0=2: acc 0.582, macroAUC 0.820, ece 0.024, brier 0.522
    Proposed a0=5: acc 0.580, macroAUC 0.819, ece 0.026, brier 0.524

## Proposed(N=10) - MLE(N=50), paired by replication
- a0=1: acc +0.003 (p=0.81), macroAUC +0.032 (p=0.00), ece -0.020 (p=0.16), brier -0.015 (p=0.28)
- a0=2: acc +0.001 (p=0.93), macroAUC +0.032 (p=0.00), ece -0.030 (p=0.05), brier -0.016 (p=0.31)
- a0=5: acc -0.005 (p=0.67), macroAUC +0.028 (p=0.00), ece -0.027 (p=0.08), brier -0.007 (p=0.66)

## Proposed(N=50) vs MLE at larger N, paired (a0=2, the reported setting)
- vs MLE N=100: acc +0.006 (p=0.30), macroAUC +0.014 (p=0.00), ece -0.023 (p=0.00), brier -0.011 (p=0.07)
- vs MLE N=200: acc -0.011 (p=0.02), macroAUC -0.004 (p=0.05), ece +0.007 (p=0.22), brier +0.014 (p=0.01)
