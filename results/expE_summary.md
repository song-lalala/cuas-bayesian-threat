# Exp-E v2: prior quality, ESS, when constraints help

## (i) adversarial prior lam=1 (mean over seeds)
- ESS=8 N=10 MAP: acc 0.525->0.488, ECE 0.088->0.081, AUC->0.731 (MLE ref acc 0.453 ECE 0.116)
- ESS=8 N=10 cMAP: acc 0.525->0.399, ECE 0.088->0.043, AUC->0.645 (MLE ref acc 0.453 ECE 0.116)
- ESS=8 N=50 MAP: acc 0.545->0.423, ECE 0.058->0.062, AUC->0.627 (MLE ref acc 0.531 ECE 0.109)
- ESS=8 N=50 cMAP: acc 0.545->0.409, ECE 0.058->0.049, AUC->0.699 (MLE ref acc 0.531 ECE 0.109)
- ESS=8 N=200 MAP: acc 0.572->0.480, ECE 0.032->0.054, AUC->0.695 (MLE ref acc 0.571 ECE 0.045)
- ESS=8 N=200 cMAP: acc 0.572->0.538, ECE 0.032->0.107, AUC->0.798 (MLE ref acc 0.571 ECE 0.045)
- ESS=8 N=400 MAP: acc 0.579->0.534, ECE 0.024->0.049, AUC->0.770 (MLE ref acc 0.580 ECE 0.031)
- ESS=8 N=400 cMAP: acc 0.579->0.572, ECE 0.024->0.092, AUC->0.813 (MLE ref acc 0.580 ECE 0.031)
- ESS=50 N=10 MAP: acc 0.515->0.500, ECE 0.098->0.102, AUC->0.758 (MLE ref acc 0.453 ECE 0.116)
- ESS=50 N=10 cMAP: acc 0.515->0.401, ECE 0.098->0.038, AUC->0.613 (MLE ref acc 0.453 ECE 0.116)
- ESS=50 N=50 MAP: acc 0.523->0.499, ECE 0.084->0.072, AUC->0.742 (MLE ref acc 0.531 ECE 0.109)
- ESS=50 N=50 cMAP: acc 0.523->0.395, ECE 0.084->0.031, AUC->0.630 (MLE ref acc 0.531 ECE 0.109)
- ESS=50 N=200 MAP: acc 0.542->0.441, ECE 0.058->0.053, AUC->0.671 (MLE ref acc 0.571 ECE 0.045)
- ESS=50 N=200 cMAP: acc 0.542->0.397, ECE 0.058->0.050, AUC->0.677 (MLE ref acc 0.571 ECE 0.045)
- ESS=50 N=400 MAP: acc 0.557->0.420, ECE 0.041->0.059, AUC->0.620 (MLE ref acc 0.580 ECE 0.031)
- ESS=50 N=400 cMAP: acc 0.557->0.434, ECE 0.041->0.087, AUC->0.734 (MLE ref acc 0.580 ECE 0.031)

## (i) T-CPT violation rate under lam=1
- ESS=8 N=10: MAP 0.841 vs cMAP 0.000
- ESS=8 N=50: MAP 0.733 vs cMAP 0.000
- ESS=8 N=200: MAP 0.608 vs cMAP 0.000
- ESS=8 N=400: MAP 0.551 vs cMAP 0.000
- ESS=50 N=10: MAP 0.866 vs cMAP 0.000
- ESS=50 N=50: MAP 0.791 vs cMAP 0.000
- ESS=50 N=200: MAP 0.711 vs cMAP 0.000
- ESS=50 N=400: MAP 0.655 vs cMAP 0.000

## (i) Proposed (ordinal-ICI T) under reversed prior
- lam=0 N=50: acc 0.524, ECE 0.084, AUC 0.790, viol 0.000
- lam=0 N=200: acc 0.543, ECE 0.059, AUC 0.801, viol 0.000
- lam=1 N=50: acc 0.408, ECE 0.022, AUC 0.702, viol 0.000
- lam=1 N=200: acc 0.408, ECE 0.019, AUC 0.745, viol 0.000

## (ii) ESS selection quality
- N=10: ECE-opt a0=2 (0.0780); mean selection gap +0.0083 (max +0.0213)
- N=50: ECE-opt a0=5 (0.0468); mean selection gap +0.0100 (max +0.0447)
- N=200: ECE-opt a0=2 (0.0251); mean selection gap +0.0082 (max +0.0489)
- N=400: ECE-opt a0=2 (0.0231); mean selection gap +0.0032 (max +0.0342)

## (iii) regime ECE(MAP)-ECE(cMAP) by lam x N
### ESS=8
- lam=0: N10:-0.0000  N50:+0.0002  N200:-0.0001  N400:+0.0003
- lam=0.25: N10:-0.0006  N50:-0.0003  N200:-0.0010  N400:-0.0007
- lam=0.5: N10:-0.0173  N50:-0.0265  N200:-0.0082  N400:+0.0015
- lam=0.75: N10:+0.0427  N50:+0.0360  N200:+0.0219  N400:-0.0062
- lam=1: N10:+0.0382  N50:+0.0130  N200:-0.0534  N400:-0.0430
### ESS=50
- lam=0: N10:-0.0001  N50:-0.0000  N200:-0.0000  N400:-0.0001
- lam=0.25: N10:-0.0001  N50:-0.0001  N200:+0.0001  N400:-0.0001
- lam=0.5: N10:-0.0041  N50:-0.0113  N200:-0.0501  N400:-0.0324
- lam=0.75: N10:+0.0628  N50:+0.0473  N200:+0.0374  N400:+0.0310
- lam=1: N10:+0.0642  N50:+0.0415  N200:+0.0032  N400:-0.0278
