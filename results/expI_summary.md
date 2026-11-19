# Exp-I: RF channel on independent recordings and real-data high-threat calibration

- P0 reproduces expG_rf_e2e_raw.csv: 300 values compared, 0 metric columns differing

## DroneRF node level (test sets)
| protocol | splits | n_train (mean) | n_test (mean) | accuracy | macro-F1 | Bebop+AR acc | recall bg/Bebop/AR/Phantom |
|---|---|---|---|---|---|---|---|
| P0 | 1 | 2740 | 1120 | 0.8143 (sd 0.0000) | 0.8345 | 0.7671 | 0.990/0.714/0.823/0.850 |
| P1 | 5 | 3200 | 660 | 0.8321 (sd 0.0130) | 0.8546 | 0.7846 | 0.983/0.713/0.857/0.910 |
| P1c | 5 | 3056 | 660 | 0.8321 (sd 0.0152) | 0.8529 | 0.7871 | 0.980/0.727/0.847/0.897 |
| P2 | 5 | 3056 | 460 | 0.8129 (sd 0.0212) | 0.8302 | 0.7585 | 0.956/0.688/0.839/0.880 |
| P3 | 4 | 2850 | 970 | 0.7603 (sd 0.0605) | 0.7758 | 0.7291 | 0.950/0.705/0.748/0.894 |

## [N] within-recording optimism (node level)
- acc(P1) - acc(P2): +0.0192 (paired p 0.11) -> small
- acc(P1c) - acc(P2): +0.0192 (paired p 0.11) -> small
- Bebop+AR acc: P2 0.7585 vs P3 held-out modes 0.7291: +0.0294 (Welch p 0.5) -> small
  - P3 fold 0 (mode 00): acc 0.7469, Bebop recall 0.552, AR recall 0.888
  - P3 fold 1 (mode 01): acc 0.7867, Bebop recall 0.733, AR recall 0.790
  - P3 fold 2 (mode 10): acc 0.8245, Bebop recall 0.767, AR recall 0.840
  - P3 fold 3 (mode 11): acc 0.6830, Bebop recall 0.767, AR recall 0.472

## threat level (mixed world, 20 replications)
| protocol | variant | acc | macro-AUC | Brier | ECE |
|---|---|---|---|---|---|
| P0 | abstract | 0.5475 | 0.7983 | 0.5569 | 0.0629 |
| P0 | calib-hard | 0.5782 | 0.8184 | 0.5261 | 0.0295 |
| P0 | calib-soft | 0.5793 | 0.8187 | 0.5256 | 0.0296 |
| P0 | CPT-oracle | 0.5947 | 0.8254 | 0.5139 | 0.0167 |
| P1 | abstract | 0.5482 | 0.7981 | 0.5567 | 0.0619 |
| P1 | calib-hard | 0.5785 | 0.8184 | 0.5258 | 0.0294 |
| P1 | calib-soft | 0.5795 | 0.8188 | 0.5253 | 0.0299 |
| P1 | CPT-oracle | 0.5954 | 0.8255 | 0.5134 | 0.0175 |
| P1c | abstract | 0.5478 | 0.7977 | 0.5571 | 0.0623 |
| P1c | calib-hard | 0.5780 | 0.8182 | 0.5260 | 0.0302 |
| P1c | calib-soft | 0.5794 | 0.8187 | 0.5254 | 0.0301 |
| P1c | CPT-oracle | 0.5949 | 0.8253 | 0.5137 | 0.0169 |
| P2 | abstract | 0.5460 | 0.7966 | 0.5579 | 0.0622 |
| P2 | calib-hard | 0.5768 | 0.8176 | 0.5270 | 0.0289 |
| P2 | calib-soft | 0.5791 | 0.8187 | 0.5257 | 0.0305 |
| P2 | CPT-oracle | 0.5946 | 0.8248 | 0.5145 | 0.0170 |
| P3 | abstract | 0.5472 | 0.7974 | 0.5568 | 0.0616 |
| P3 | calib-hard | 0.5765 | 0.8176 | 0.5271 | 0.0296 |
| P3 | calib-soft | 0.5788 | 0.8186 | 0.5258 | 0.0297 |
| P3 | CPT-oracle | 0.5950 | 0.8249 | 0.5144 | 0.0174 |

## [E1] calib-hard minus abstract (Sec. VII-H claim)
- P0: dAUC +0.0202 (p 8.7e-15), dECE -0.0334 (p 2.5e-08) -> holds
- P1: dAUC +0.0203 (p 3.7e-14), dECE -0.0325 (p 6.3e-09) -> holds
- P1c: dAUC +0.0205 (p 9.6e-14), dECE -0.0321 (p 1.3e-08) -> holds
- P2: dAUC +0.0210 (p 1.4e-14), dECE -0.0333 (p 2.7e-08) -> holds
- P3: dAUC +0.0202 (p 3.5e-15), dECE -0.0319 (p 2.0e-09) -> holds

## [E2] optimism carried to the threat posterior (calib-hard)
- P1 - P2: dAUC +0.0008 (p 3.8e-02), dECE +0.0006 (p 5.5e-01) -> not material
- P0 - P2: dAUC +0.0008 (p 1.1e-02), dECE +0.0006 (p 5.7e-01) -> not material
- P0 - P3: dAUC +0.0009 (p 2.1e-03), dECE -0.0002 (p 9.0e-01) -> not material

## [H] real-data P(T = high) reliability (pooled over 20 replications)

### P0
| bin | n | predicted | observed [95% CP] | oracle predicted | oracle observed |
|---|---|---|---|---|---|
| (0.00,0.10] | 36008 | 0.021 | 0.019 [0.018, 0.021] | 0.019 | 0.018 |
| (0.10,0.25] | 8076 | 0.166 | 0.179 [0.170, 0.187] | 0.168 | 0.174 |
| (0.25,0.50] | 4582 | 0.350 | 0.339 [0.325, 0.353] | 0.348 | 0.360 |
| (0.50,0.75] | 1039 | 0.600 | 0.567 [0.536, 0.597] | 0.573 | 0.586 |
| (0.75,1.00] | 295 | 0.848 | 0.885 [0.843, 0.919] | 0.858 | 0.883 |
- bins outside the 95% interval: calib-hard 3/5, CPT oracle 0/5
- alarm P(T=high) >= 0.5 (calib-hard): precision 0.637, recall 0.188 (tp 850, fp 484, fn 3683)
- alarm P(T=high) >= 0.5 (CPT-oracle): precision 0.653, recall 0.195 (tp 886, fp 471, fn 3647)

### P2
| bin | n | predicted | observed [95% CP] | oracle predicted | oracle observed |
|---|---|---|---|---|---|
| (0.00,0.10] | 35967 | 0.021 | 0.019 [0.018, 0.020] | 0.019 | 0.018 |
| (0.10,0.25] | 8098 | 0.166 | 0.178 [0.170, 0.186] | 0.168 | 0.174 |
| (0.25,0.50] | 4600 | 0.349 | 0.341 [0.328, 0.355] | 0.348 | 0.356 |
| (0.50,0.75] | 1040 | 0.598 | 0.560 [0.529, 0.590] | 0.577 | 0.595 |
| (0.75,1.00] | 295 | 0.847 | 0.871 [0.828, 0.907] | 0.863 | 0.882 |
- bins outside the 95% interval: calib-hard 3/5, CPT oracle 0/5
- alarm P(T=high) >= 0.5 (calib-hard): precision 0.628, recall 0.185 (tp 839, fp 496, fn 3694)
- alarm P(T=high) >= 0.5 (CPT-oracle): precision 0.658, recall 0.193 (tp 877, fp 455, fn 3656)

### P3
| bin | n | predicted | observed [95% CP] | oracle predicted | oracle observed |
|---|---|---|---|---|---|
| (0.00,0.10] | 35949 | 0.021 | 0.019 [0.018, 0.021] | 0.019 | 0.018 |
| (0.10,0.25] | 8074 | 0.166 | 0.178 [0.170, 0.187] | 0.168 | 0.174 |
| (0.25,0.50] | 4635 | 0.350 | 0.336 [0.322, 0.350] | 0.347 | 0.356 |
| (0.50,0.75] | 1054 | 0.600 | 0.568 [0.538, 0.598] | 0.577 | 0.598 |
| (0.75,1.00] | 288 | 0.850 | 0.875 [0.831, 0.911] | 0.867 | 0.888 |
- bins outside the 95% interval: calib-hard 4/5, CPT oracle 0/5
- alarm P(T=high) >= 0.5 (calib-hard): precision 0.634, recall 0.188 (tp 851, fp 491, fn 3682)
- alarm P(T=high) >= 0.5 (CPT-oracle): precision 0.659, recall 0.195 (tp 884, fp 457, fn 3649)

## DroneDetect node level (5 seeds; CLEAN-trained)
| condition | chunk accuracy | macro-F1 | recording accuracy | recordings |
|---|---|---|---|---|
| CLEAN | 0.7988 (sd 0.0545) | 0.7775 | 0.8235 | 17 |
| BLUE | 0.7653 (sd 0.0252) | 0.7629 | 0.8253 | 95 |
| WIFI | 0.6666 (sd 0.0307) | 0.6646 | 0.7240 | 100 |
| BOTH | 0.5670 (sd 0.0413) | 0.5631 | 0.6060 | 100 |

## DroneDetect reliability (condition as context)
| condition | p(correct) | alpha Eq.(10) | omega Eq.(11) | eval acc | LL none | LL own alpha | own - none (p) |
|---|---|---|---|---|---|---|---|
| CLEAN | 0.808 | 0.980 | 1.000 | 0.804 | -0.942 | -0.941 | +0.001 (8.4e-01) -> not shown |
| BLUE | 0.747 | 0.907 | 0.909 | 0.782 | -1.062 | -1.056 | +0.006 (6.2e-01) -> not shown |
| WIFI | 0.668 | 0.811 | 0.790 | 0.665 | -1.410 | -1.339 | +0.071 (7.3e-03) -> useful |
| BOTH | 0.579 | 0.702 | 0.656 | 0.556 | -1.573 | -1.454 | +0.119 (7.3e-03) -> useful |

### transfer (context clean vs interference)
| fitted on | applied to | alpha transferred | alpha own | LL none | LL transfer | transfer - none (p) |
|---|---|---|---|---|---|---|
| CLEAN+BLUE | WIFI | 0.905 | 0.811 | -1.410 | -1.355 | +0.055 (1.5e-02) -> transfers |
| CLEAN+BLUE | BOTH | 0.905 | 0.702 | -1.573 | -1.504 | +0.069 (6.3e-03) -> transfers |
| CLEAN+WIFI | BLUE | 0.810 | 0.907 | -1.062 | -1.070 | -0.008 (7.6e-01) -> not shown |
| CLEAN+WIFI | BOTH | 0.810 | 0.702 | -1.573 | -1.473 | +0.100 (1.3e-04) -> transfers |
| CLEAN+BOTH | BLUE | 0.702 | 0.907 | -1.062 | -1.103 | -0.041 (3.1e-01) -> not shown |
| CLEAN+BOTH | WIFI | 0.702 | 0.811 | -1.410 | -1.332 | +0.078 (7.8e-02) -> not shown |
