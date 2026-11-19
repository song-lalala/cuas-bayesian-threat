# Exp-C: interpretation and consistency of the cMAP and MAP-EM estimators (ESS 2, 20 replications)

## 2.1 complete data, standard world: column-weighted KL to the truth (mean over replications)

### node H
| N | cMAP | MAP | cMLE | maxabs cMAP | max|cMAP-cMLE| |
|---|---|---|---|---|---|
| 50 | 1.45e-01 | 1.48e-01 | 1.40e+00 | 0.5055 | 5.83e-01 |
| 200 | 5.64e-02 | 5.71e-02 | 3.56e-01 | 0.3076 | 2.10e-01 |
| 1000 | 2.05e-02 | 2.12e-02 | 7.36e-02 | 0.1409 | 3.32e-02 |
| 5000 | 5.21e-03 | 5.38e-03 | 1.46e-02 | 0.0644 | 6.36e-03 |
| 20000 | 1.51e-03 | 1.56e-03 | 2.58e-03 | 0.0352 | 1.55e-03 |
- log-log slope vs N: max|cMAP-cMLE| -1.01; cMAP KL -0.76

### node N
| N | cMAP | MAP | cMLE | maxabs cMAP | max|cMAP-cMLE| |
|---|---|---|---|---|---|
| 50 | 3.85e-02 | 3.84e-02 | 2.61e-01 | 0.1982 | 7.40e-02 |
| 200 | 1.81e-02 | 1.85e-02 | 7.74e-02 | 0.1137 | 1.42e-02 |
| 1000 | 4.73e-03 | 4.72e-03 | 1.28e-02 | 0.0417 | 3.07e-03 |
| 5000 | 8.56e-04 | 8.55e-04 | 1.32e-03 | 0.0202 | 6.55e-04 |
| 20000 | 1.87e-04 | 1.87e-04 | 1.87e-04 | 0.0107 | 1.61e-04 |
- log-log slope vs N: max|cMAP-cMLE| -1.01; cMAP KL -0.90

### node C
| N | cMAP | MAP | cMLE | maxabs cMAP | max|cMAP-cMLE| |
|---|---|---|---|---|---|
| 50 | 2.87e-02 | 2.87e-02 | 2.05e-01 | 0.1243 | 1.02e-01 |
| 200 | 1.58e-02 | 1.58e-02 | 5.35e-02 | 0.0813 | 2.74e-02 |
| 1000 | 7.45e-03 | 7.43e-03 | 2.16e-02 | 0.0361 | 5.39e-03 |
| 5000 | 1.30e-03 | 1.29e-03 | 2.50e-03 | 0.0168 | 1.06e-03 |
| 20000 | 3.03e-04 | 2.95e-04 | 4.03e-04 | 0.0074 | 2.64e-04 |
- log-log slope vs N: max|cMAP-cMLE| -1.00; cMAP KL -0.76

### node T
| N | cMAP | MAP | cMLE | maxabs cMAP | max|cMAP-cMLE| |
|---|---|---|---|---|---|
| 50 | 1.49e-01 | 1.53e-01 | 1.49e+00 | 0.4790 | 5.84e-01 |
| 200 | 6.77e-02 | 6.84e-02 | 3.51e-01 | 0.3852 | 4.51e-01 |
| 1000 | 1.65e-02 | 1.66e-02 | 5.82e-02 | 0.2606 | 1.94e-01 |
| 5000 | 6.05e-03 | 6.05e-03 | 1.39e-02 | 0.1328 | 4.59e-02 |
| 20000 | 2.04e-03 | 2.07e-03 | 3.50e-03 | 0.0581 | 1.17e-02 |
- log-log slope vs N: max|cMAP-cMLE| -0.67; cMAP KL -0.72

## 2.2 non-monotone world (H-v-U), node H
- KL(truth || KL projection onto M): H-v-U 2.0818e-01; standard 6.91e-09 (truth already monotone)
| N | cMAP->truth | cMAP->pseudo-true | MAP->truth | cMLE->pseudo-true |
|---|---|---|---|---|
| 50 | 4.914e-01 | 2.047e-01 | 4.508e-01 | 1.082e+00 |
| 200 | 2.842e-01 | 6.252e-02 | 1.215e-01 | 1.974e-01 |
| 1000 | 2.283e-01 | 1.774e-02 | 2.705e-02 | 4.050e-02 |
| 5000 | 2.132e-01 | 4.450e-03 | 6.225e-03 | 6.413e-03 |
| 20000 | 2.098e-01 | 1.391e-03 | 2.146e-03 | 1.829e-03 |

## 2.3 MAP-EM (latent H, N, C)
- final-only N=200: decreases 0/1000 steps (max drop -0.0e+00); relative logpost change iter 12 -> 50: mean 1.85e-03, max 3.76e-03; relative step at 12: 2.34e-04
- final-only N=1000: decreases 0/1000 steps (max drop -0.0e+00); relative logpost change iter 12 -> 50: mean 2.12e-03, max 5.80e-03; relative step at 12: 1.94e-04
- final-only N=5000: decreases 0/1000 steps (max drop -0.0e+00); relative logpost change iter 12 -> 50: mean 1.59e-03, max 5.99e-03; relative step at 12: 1.60e-04
- every-iteration N=200: decreases 0/980 steps (max drop -0.0e+00); relative logpost change iter 12 -> 50: mean 7.34e-04, max 1.52e-03; relative step at 12: 1.18e-04
- every-iteration N=1000: decreases 0/980 steps (max drop -0.0e+00); relative logpost change iter 12 -> 50: mean 1.32e-03, max 4.62e-03; relative step at 12: 1.60e-04
- every-iteration N=5000: decreases 0/980 steps (max drop -0.0e+00); relative logpost change iter 12 -> 50: mean 1.43e-03, max 5.52e-03; relative step at 12: 1.56e-04
- final-only N=200: logpost(unconstrained iterate) - logpost(final constrained step) at 12: mean +8.577e-01 (max +2.654e+00); at 50: mean +2.646e+00
- final-only N=1000: logpost(unconstrained iterate) - logpost(final constrained step) at 12: mean +9.560e-01 (max +3.641e+00); at 50: mean +4.798e+00
- final-only N=5000: logpost(unconstrained iterate) - logpost(final constrained step) at 12: mean +3.596e-01 (max +3.366e+00); at 50: mean +3.843e+00

| N | variant | iters | acc | AUC | ECE | KL H | KL N | KL C | KL T | labels ok |
|---|---|---|---|---|---|---|---|---|---|---|
| 200 | final-only | 12 | 0.5642 | 0.8044 | 0.0457 | 0.2361 | 0.0480 | 0.0738 | 0.1370 | 60/60 |
| 200 | final-only | 50 | 0.5641 | 0.8032 | 0.0518 | 0.2335 | 0.0506 | 0.0819 | 0.1528 | 60/60 |
| 200 | every-iteration | 12 | 0.5637 | 0.8044 | 0.0439 | 0.2376 | 0.0485 | 0.0726 | 0.1373 | 60/60 |
| 200 | every-iteration | 50 | 0.5646 | 0.8038 | 0.0473 | 0.2371 | 0.0505 | 0.0759 | 0.1440 | 60/60 |
| 1000 | final-only | 12 | 0.5812 | 0.8149 | 0.0266 | 0.1636 | 0.0315 | 0.0486 | 0.0747 | 60/60 |
| 1000 | final-only | 50 | 0.5812 | 0.8151 | 0.0293 | 0.1303 | 0.0319 | 0.0367 | 0.0865 | 60/60 |
| 1000 | every-iteration | 12 | 0.5814 | 0.8150 | 0.0262 | 0.1652 | 0.0314 | 0.0485 | 0.0748 | 60/60 |
| 1000 | every-iteration | 50 | 0.5819 | 0.8154 | 0.0292 | 0.1237 | 0.0310 | 0.0370 | 0.0803 | 60/60 |
| 5000 | final-only | 12 | 0.5875 | 0.8194 | 0.0210 | 0.1187 | 0.0235 | 0.0468 | 0.0571 | 60/60 |
| 5000 | final-only | 50 | 0.5893 | 0.8208 | 0.0218 | 0.0531 | 0.0161 | 0.0225 | 0.0482 | 60/60 |
| 5000 | every-iteration | 12 | 0.5876 | 0.8194 | 0.0210 | 0.1189 | 0.0235 | 0.0468 | 0.0572 | 60/60 |
| 5000 | every-iteration | 50 | 0.5892 | 0.8209 | 0.0219 | 0.0542 | 0.0158 | 0.0224 | 0.0479 | 60/60 |

### paired differences over replications (t p)
- N=200 acc: final-only 50-12 -0.00008 (p 9.3e-01); every-iteration - final-only at 12 -0.00050 (p 2.8e-01)
- N=200 macroAUC: final-only 50-12 -0.00115 (p 8.6e-02); every-iteration - final-only at 12 +0.00003 (p 8.1e-01)
- N=200 ece: final-only 50-12 +0.00607 (p 5.4e-03); every-iteration - final-only at 12 -0.00180 (p 1.3e-03)
- N=200 wkl_H: final-only 50-12 -0.00257 (p 8.0e-01); every-iteration - final-only at 12 +0.00154 (p 1.7e-01)
- N=200 wkl_N: final-only 50-12 +0.00254 (p 5.9e-01); every-iteration - final-only at 12 +0.00049 (p 2.3e-01)
- N=200 wkl_C: final-only 50-12 +0.00807 (p 2.1e-01); every-iteration - final-only at 12 -0.00119 (p 1.6e-01)
- N=200 wkl_T: final-only 50-12 +0.01578 (p 2.9e-02); every-iteration - final-only at 12 +0.00022 (p 8.4e-01)
- N=1000 acc: final-only 50-12 +0.00000 (p 1.0e+00); every-iteration - final-only at 12 +0.00016 (p 2.5e-01)
- N=1000 macroAUC: final-only 50-12 +0.00012 (p 7.4e-01); every-iteration - final-only at 12 +0.00003 (p 7.7e-02)
- N=1000 ece: final-only 50-12 +0.00269 (p 9.4e-02); every-iteration - final-only at 12 -0.00041 (p 5.4e-01)
- N=1000 wkl_H: final-only 50-12 -0.03336 (p 1.2e-01); every-iteration - final-only at 12 +0.00161 (p 4.4e-02)
- N=1000 wkl_N: final-only 50-12 +0.00033 (p 9.5e-01); every-iteration - final-only at 12 -0.00012 (p 9.1e-02)
- N=1000 wkl_C: final-only 50-12 -0.01189 (p 3.6e-02); every-iteration - final-only at 12 -0.00012 (p 1.7e-01)
- N=1000 wkl_T: final-only 50-12 +0.01183 (p 1.6e-01); every-iteration - final-only at 12 +0.00015 (p 6.9e-01)
- N=5000 acc: final-only 50-12 +0.00178 (p 1.5e-02); every-iteration - final-only at 12 +0.00006 (p 3.3e-01)
- N=5000 macroAUC: final-only 50-12 +0.00148 (p 2.4e-03); every-iteration - final-only at 12 +0.00001 (p 4.2e-01)
- N=5000 ece: final-only 50-12 +0.00076 (p 6.8e-01); every-iteration - final-only at 12 +0.00004 (p 7.9e-01)
- N=5000 wkl_H: final-only 50-12 -0.06568 (p 1.8e-02); every-iteration - final-only at 12 +0.00012 (p 6.4e-02)
- N=5000 wkl_N: final-only 50-12 -0.00736 (p 1.3e-01); every-iteration - final-only at 12 +0.00002 (p 4.3e-01)
- N=5000 wkl_C: final-only 50-12 -0.02430 (p 5.2e-04); every-iteration - final-only at 12 -0.00002 (p 2.8e-01)
- N=5000 wkl_T: final-only 50-12 -0.00889 (p 1.1e-01); every-iteration - final-only at 12 +0.00008 (p 2.2e-01)

## 2.4 ordinal ICI threat node vs full-table cMAP (standard world)
- infinite-data ICI fit: KL(truth || ICI family) 1.8132e-04, max abs 0.0026
| N | ICI KL | full cMAP KL | ICI KL to its limit |
|---|---|---|---|
| 50 | 8.8357e-02 | 1.4900e-01 | 8.8136e-02 |
| 200 | 2.9221e-02 | 6.7696e-02 | 2.9027e-02 |
| 1000 | 4.7532e-03 | 1.6489e-02 | 4.5708e-03 |
| 5000 | 1.1051e-03 | 6.0549e-03 | 9.2342e-04 |
| 20000 | 3.8174e-04 | 2.0404e-03 | 2.0037e-04 |
