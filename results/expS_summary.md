# Exp-S v2 (S=20, B=1000, test n=2500)

## class distribution / majority baseline
- majority-class baseline accuracy: 0.434

## Table 3 (acc/ECE/AUC mean +- 95%CI half)
- N=10 | B0-heuristic: 0.450+-0.004/0.098+-0.004/0.784+-0.002 | B1-MLE: 0.457+-0.025/0.111+-0.014/0.661+-0.023 | B2-expert: 0.512+-0.028/0.103+-0.030/0.780+-0.012 | Proposed: 0.534+-0.021/0.077+-0.026/0.793+-0.009
- N=50 | B0-heuristic: 0.450+-0.004/0.098+-0.004/0.784+-0.002 | B1-MLE: 0.533+-0.011/0.107+-0.018/0.761+-0.008 | B2-expert: 0.512+-0.028/0.103+-0.030/0.780+-0.012 | Proposed: 0.559+-0.009/0.051+-0.013/0.806+-0.004
- N=200 | B0-heuristic: 0.450+-0.004/0.098+-0.004/0.784+-0.002 | B1-MLE: 0.571+-0.007/0.044+-0.005/0.810+-0.002 | B2-expert: 0.512+-0.028/0.103+-0.030/0.780+-0.012 | Proposed: 0.577+-0.008/0.030+-0.005/0.817+-0.003
- N=400 | B0-heuristic: 0.450+-0.004/0.098+-0.004/0.784+-0.002 | B1-MLE: 0.578+-0.007/0.032+-0.006/0.816+-0.003 | B2-expert: 0.512+-0.028/0.103+-0.030/0.780+-0.012 | Proposed: 0.582+-0.007/0.024+-0.003/0.820+-0.003

## ESS: fixed a priori at alpha0=2 (no validation labels consumed by the main pipeline)

## Proposed vs B1-MLE paired deltas
- N=10: dAUC +0.1311 [+0.1068,+0.1554] (t p=7.2e-10, W p=1.9e-06); dECE -0.0341 [-0.0632,-0.0050] (t p=2.4e-02); McNemar sig 19/20
- N=50: dAUC +0.0454 [+0.0371,+0.0537] (t p=5.4e-10, W p=1.9e-06); dECE -0.0554 [-0.0728,-0.0381] (t p=2.1e-06); McNemar sig 15/20
- N=200: dAUC +0.0066 [+0.0045,+0.0087] (t p=2.5e-06, W p=1.9e-06); dECE -0.0138 [-0.0177,-0.0099] (t p=5.0e-07); McNemar sig 4/20
- N=400: dAUC +0.0036 [+0.0025,+0.0047] (t p=2.4e-06, W p=5.7e-06); dECE -0.0075 [-0.0115,-0.0036] (t p=7.9e-04); McNemar sig 2/20

## Proposed vs B2-expert paired deltas
- N=10: dAUC +0.0125 [+0.0067,+0.0183] (t p=2.3e-04, W p=9.5e-06); dECE -0.0261 [-0.0436,-0.0086] (t p=5.7e-03); McNemar sig 10/20
- N=50: dAUC +0.0261 [+0.0168,+0.0355] (t p=1.3e-05, W p=1.9e-06); dECE -0.0516 [-0.0763,-0.0268] (t p=3.3e-04); McNemar sig 13/20
- N=200: dAUC +0.0368 [+0.0256,+0.0481] (t p=1.6e-06, W p=1.9e-06); dECE -0.0727 [-0.1008,-0.0446] (t p=3.2e-05); McNemar sig 16/20
- N=400: dAUC +0.0399 [+0.0285,+0.0513] (t p=6.0e-07, W p=1.9e-06); dECE -0.0785 [-0.1074,-0.0497] (t p=1.7e-05); McNemar sig 17/20

## Table 4 (acc/F1/AUC/Brier/ECE/cwECE/FAR mean +- CIhalf)
- radar only: acc 0.555+-0.008, macroF1 0.445+-0.014, macroAUC 0.797+-0.003, brier 0.550+-0.005, ece 0.034+-0.005, cwece 0.031+-0.003, far@pd0.9 0.506+-0.012
- RF only: acc 0.532+-0.008, macroF1 0.387+-0.009, macroAUC 0.770+-0.003, brier 0.576+-0.004, ece 0.031+-0.006, cwece 0.031+-0.003, far@pd0.9 0.593+-0.013
- EO/IR only: acc 0.555+-0.008, macroF1 0.452+-0.013, macroAUC 0.798+-0.002, brier 0.550+-0.004, ece 0.030+-0.003, cwece 0.032+-0.003, far@pd0.9 0.522+-0.015
- acoustic only: acc 0.544+-0.007, macroF1 0.420+-0.013, macroAUC 0.785+-0.002, brier 0.562+-0.003, ece 0.029+-0.005, cwece 0.031+-0.003, far@pd0.9 0.576+-0.014
- all (abstract model): acc 0.565+-0.007, macroF1 0.481+-0.008, macroAUC 0.808+-0.003, brier 0.541+-0.006, ece 0.043+-0.008, cwece 0.035+-0.004, far@pd0.9 0.470+-0.011
- all (calibrated): acc 0.576+-0.008, macroF1 0.487+-0.007, macroAUC 0.816+-0.002, brier 0.529+-0.005, ece 0.032+-0.005, cwece 0.031+-0.003, far@pd0.9 0.466+-0.012
- all (calib+reliab): acc 0.577+-0.008, macroF1 0.487+-0.008, macroAUC 0.817+-0.003, brier 0.528+-0.004, ece 0.030+-0.005, cwece 0.031+-0.003, far@pd0.9 0.467+-0.011

## hard-condition subsets (all metrics, all replications)
- silent-drone abstract     (n~392): acc 0.503+-0.022, macroAUC 0.781+-0.014, brier 0.611+-0.020, ece 0.091+-0.013, cwece 0.139+-0.010
- silent-drone calibrated   (n~392): acc 0.593+-0.017, macroAUC 0.811+-0.014, brier 0.527+-0.013, ece 0.065+-0.009, cwece 0.100+-0.008
- silent-drone calib+reliab (n~392): acc 0.588+-0.016, macroAUC 0.813+-0.013, brier 0.532+-0.014, ece 0.063+-0.010, cwece 0.103+-0.009
- far-range    abstract     (n~858): acc 0.559+-0.016, macroAUC 0.755+-0.010, brier 0.549+-0.011, ece 0.059+-0.014, cwece 0.048+-0.008
- far-range    calibrated   (n~858): acc 0.567+-0.016, macroAUC 0.765+-0.009, brier 0.539+-0.009, ece 0.051+-0.011, cwece 0.044+-0.006
- far-range    calib+reliab (n~858): acc 0.570+-0.017, macroAUC 0.767+-0.009, brier 0.536+-0.009, ece 0.045+-0.010, cwece 0.041+-0.007
- far-silent   abstract     (n~135): acc 0.438+-0.028, macroAUC 0.682+-0.042, brier 0.676+-0.027, ece 0.156+-0.024, cwece 0.176+-0.013
- far-silent   calibrated   (n~135): acc 0.545+-0.029, macroAUC 0.718+-0.045, brier 0.578+-0.023, ece 0.117+-0.017, cwece 0.137+-0.009
- far-silent   calib+reliab (n~135): acc 0.532+-0.030, macroAUC 0.722+-0.043, brier 0.587+-0.024, ece 0.120+-0.018, cwece 0.143+-0.011

## fusion paired stats (calibrated - abstract)
- delta macro-AUC: +0.0082 [+0.0068,+0.0097] t p=2.6e-10, W p=1.9e-06
- delta ECE: -0.0106 [-0.0167,-0.0045] t p=1.7e-03, W p=2.7e-03
- delta ECE (reliab-calib): -0.0020 [-0.0039,+0.0000] t p=5.2e-02, W p=5.8e-02
- DeLong class 0: median p=1.4e-01, sig 5/20
- DeLong class 1: median p=2.7e-03, sig 17/20
- DeLong class 2: median p=1.1e-02, sig 15/20
- DeLong class 3: median p=2.2e-02, sig 12/20
- McNemar: sig 7/20; boot dAUC CI excl 0 in 18/20; dECE in 5/20

## ablation (N=80) vs full
- Proposed (full): acc 0.565, AUC 0.810, ECE 0.042
- -ICI (full table): dECE +0.0076 [+0.0026,+0.0125] (p=4.7e-03); dAUC -0.0040 [-0.0060,-0.0019] (p=6.9e-04)
- -constraints: dECE +0.0002 [-0.0002,+0.0006] (p=3.0e-01); dAUC -0.0001 [-0.0001,+0.0000] (p=2.3e-01)
- -prior (MLE): dECE +0.0442 [+0.0321,+0.0562] (p=3.1e-07); dAUC -0.0274 [-0.0312,-0.0237] (p=3.4e-12)
- -reliability: dECE +0.0027 [-0.0008,+0.0062] (p=1.2e-01); dAUC -0.0008 [-0.0014,-0.0003] (p=2.7e-03)

## calibration-set size
- calib_n=0 (abstract): acc 0.565+-0.007, macroF1 0.481+-0.008, macroAUC 0.808+-0.003, brier 0.541+-0.006, ece 0.043+-0.008
- calib_n=30 (estimated): acc 0.570+-0.008, macroF1 0.484+-0.007, macroAUC 0.810+-0.003, brier 0.537+-0.005, ece 0.038+-0.005
- calib_n=100 (estimated): acc 0.573+-0.008, macroF1 0.483+-0.008, macroAUC 0.815+-0.003, brier 0.531+-0.005, ece 0.033+-0.006
- calib_n=300 (estimated): acc 0.576+-0.008, macroF1 0.487+-0.007, macroAUC 0.816+-0.002, brier 0.529+-0.005, ece 0.032+-0.005
- calib_n=1000 (estimated): acc 0.576+-0.007, macroF1 0.487+-0.007, macroAUC 0.817+-0.002, brier 0.528+-0.004, ece 0.033+-0.005
- calib_n=-1 (oracle-marginal): acc 0.576+-0.007, macroF1 0.488+-0.008, macroAUC 0.817+-0.002, brier 0.528+-0.004, ece 0.032+-0.005

## abstract-diagonal sweep (calibrated-fusion reference above)
- diag=0.70: AUC 0.808+-0.003, ECE 0.038+-0.008
- diag=0.80: AUC 0.808+-0.003, ECE 0.040+-0.009
- diag=0.90: AUC 0.808+-0.003, ECE 0.042+-0.009
- diag=0.94: AUC 0.808+-0.003, ECE 0.043+-0.008
- diag=0.99: AUC 0.808+-0.003, ECE 0.043+-0.009
