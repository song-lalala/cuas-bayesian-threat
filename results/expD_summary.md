# Exp-D: double discounting of sensor uncertainty (S=20, test n=2500, N=200, calibration n=300)

## A1 population level: clean-matrix share of the deployed likelihood (standard world)
- Rr: V0 0.870/0.915/0.930; V1 0.880/0.970/1.000; V1-cap0.98 0.880/0.970/0.980; V3 0.949/0.949/0.949; truth 0.880/0.970/1.000; reference V0 0.870/0.915/0.930, V3 0.949
- Re: V0 0.594/0.767/0.804; V1 0.550/0.920/1.000; V1-cap0.98 0.550/0.920/0.980; V3 0.821/0.821/0.821; truth 0.550/0.920/1.000; reference V0 0.594/0.767/0.804, V3 0.821
- Ra: V0 0.618/0.737/0.775; V1 0.500/0.880/1.000; V1-cap0.98 0.500/0.880/0.980; V3 0.790/0.790/0.790; truth 0.500/0.880/1.000; reference V0 0.618/0.737/0.775, V3 0.790
- max residual of the share fit: 2.5e-16

## A1b population level: KL(truth || deployed) in nats, far/near/imminent
- standard Rr: V0 0.0000/0.0015/0.0025; V1 0.0000/0.0000/0.0000; V1-cap0.98 0.0000/0.0000/0.0002; V3 0.0023/0.0002/0.0014
- standard Re: V0 0.0009/0.0134/0.0237; V1 0.0000/0.0000/0.0000; V1-cap0.98 0.0000/0.0000/0.0003; V3 0.0382/0.0059/0.0203
- standard Ra: V0 0.0040/0.0071/0.0196; V1 0.0000/0.0000/0.0000; V1-cap0.98 0.0000/0.0000/0.0002; V3 0.0266/0.0029/0.0172
- structural Rr: V0 0.0052/0.0031/0.0096; V1 0.0053/0.0006/0.0043; V1-cap0.98 0.0053/0.0006/0.0049; V3 0.0112/0.0012/0.0080
- structural Re: V0 0.0860/0.0239/0.0594; V1 0.1068/0.0126/inf; V1-cap0.98 0.1068/0.0126/0.0740; V3 0.0944/0.0159/0.0573
- structural Ra: V0 0.0256/0.0122/0.0380; V1 0.0255/0.0034/0.0286; V1-cap0.98 0.0255/0.0034/0.0229; V3 0.0578/0.0061/0.0355

## A1c large-sample check (actual estimators, estimated n=100000): share far/near/imminent, standard world
- Rr: V0 0.857/0.907/0.921; V1 0.864/0.965/0.992; V3 0.940/0.940/0.940
- Re: V0 0.595/0.771/0.809; V1 0.551/0.925/1.006; V3 0.825/0.825/0.825
- Ra: V0 0.630/0.746/0.780; V1 0.517/0.885/0.995; V3 0.796/0.796/0.796

## A5 standard world, n=300: full test set (mean +- 95% CI half-width)
- V0: acc 0.5770+-0.0081, macroF1 0.4868+-0.0077, macroAUC 0.8169+-0.0025, brier 0.5280+-0.0044, ece 0.0301+-0.0054, cwece 0.0313+-0.0032, far@pd0.9 0.4670+-0.0109
- V1: acc 0.5760+-0.0083, macroF1 0.4884+-0.0074, macroAUC 0.8166+-0.0026, brier 0.5283+-0.0046, ece 0.0339+-0.0055, cwece 0.0314+-0.0033, far@pd0.9 0.4642+-0.0112
- V1-cap0.98: acc 0.5763+-0.0082, macroF1 0.4891+-0.0075, macroAUC 0.8168+-0.0026, brier 0.5281+-0.0045, ece 0.0336+-0.0051, cwece 0.0314+-0.0032, far@pd0.9 0.4638+-0.0116
- V2: acc 0.5746+-0.0078, macroF1 0.4868+-0.0069, macroAUC 0.8157+-0.0026, brier 0.5292+-0.0045, ece 0.0337+-0.0056, cwece 0.0318+-0.0033, far@pd0.9 0.4683+-0.0126
- V3: acc 0.5758+-0.0078, macroF1 0.4870+-0.0073, macroAUC 0.8161+-0.0024, brier 0.5290+-0.0045, ece 0.0321+-0.0052, cwece 0.0313+-0.0032, far@pd0.9 0.4657+-0.0120
- sensor-oracle-marginal: acc 0.5762+-0.0074, macroF1 0.4875+-0.0076, macroAUC 0.8168+-0.0023, brier 0.5281+-0.0044, ece 0.0324+-0.0048, cwece 0.0316+-0.0030, far@pd0.9 0.4615+-0.0114
- sensor-oracle-conditional: acc 0.5769+-0.0073, macroF1 0.4887+-0.0071, macroAUC 0.8179+-0.0024, brier 0.5269+-0.0044, ece 0.0318+-0.0049, cwece 0.0319+-0.0029, far@pd0.9 0.4620+-0.0112
- CPT-oracle: acc 0.5931+-0.0052, macroF1 0.4872+-0.0059, macroAUC 0.8237+-0.0023, brier 0.5160+-0.0036, ece 0.0170+-0.0024, cwece 0.0186+-0.0009, far@pd0.9 0.4554+-0.0119
- Bayes-oracle: acc 0.5939+-0.0053, macroF1 0.4906+-0.0062, macroAUC 0.8247+-0.0024, brier 0.5147+-0.0036, ece 0.0177+-0.0018, cwece 0.0185+-0.0006, far@pd0.9 0.4539+-0.0122

### standard: brier by subset (mean over replications)
| variant | all | far | near | imminent | silent-drone | far-silent | near-silent | imminent-silent |
|---|---|---|---|---|---|---|---|---|
| V0 | 0.5280 | 0.5363 | 0.5367 | 0.5108 | 0.5321 | 0.5873 | 0.5323 | 0.4721 |
| V1 | 0.5283 | 0.5365 | 0.5369 | 0.5115 | 0.5266 | 0.5906 | 0.5243 | 0.4598 |
| V1-cap0.98 | 0.5281 | 0.5364 | 0.5367 | 0.5110 | 0.5265 | 0.5908 | 0.5242 | 0.4592 |
| V2 | 0.5292 | 0.5388 | 0.5371 | 0.5114 | 0.5294 | 0.5945 | 0.5258 | 0.4624 |
| V3 | 0.5290 | 0.5392 | 0.5367 | 0.5108 | 0.5272 | 0.5782 | 0.5290 | 0.4704 |
| sensor-oracle-marginal | 0.5281 | 0.5380 | 0.5357 | 0.5106 | 0.5234 | 0.5734 | 0.5252 | 0.4674 |
| sensor-oracle-conditional | 0.5269 | 0.5349 | 0.5356 | 0.5100 | 0.5222 | 0.5848 | 0.5213 | 0.4553 |
| CPT-oracle | 0.5160 | 0.5250 | 0.5245 | 0.4983 | 0.5042 | 0.5513 | 0.5056 | 0.4519 |
| Bayes-oracle | 0.5147 | 0.5223 | 0.5243 | 0.4973 | 0.5031 | 0.5636 | 0.5014 | 0.4397 |

### standard: ece by subset (mean over replications)
| variant | all | far | near | imminent | silent-drone | far-silent | near-silent | imminent-silent |
|---|---|---|---|---|---|---|---|---|
| V0 | 0.0301 | 0.0454 | 0.0449 | 0.0429 | 0.0626 | 0.1203 | 0.0903 | 0.0980 |
| V1 | 0.0339 | 0.0445 | 0.0469 | 0.0470 | 0.0663 | 0.1192 | 0.0962 | 0.1002 |
| V1-cap0.98 | 0.0336 | 0.0445 | 0.0469 | 0.0462 | 0.0657 | 0.1195 | 0.0990 | 0.0983 |
| V2 | 0.0337 | 0.0479 | 0.0449 | 0.0448 | 0.0688 | 0.1148 | 0.1092 | 0.0980 |
| V3 | 0.0321 | 0.0510 | 0.0457 | 0.0422 | 0.0649 | 0.1170 | 0.0955 | 0.0989 |
| sensor-oracle-marginal | 0.0324 | 0.0497 | 0.0459 | 0.0449 | 0.0678 | 0.1156 | 0.0929 | 0.1084 |
| sensor-oracle-conditional | 0.0318 | 0.0449 | 0.0462 | 0.0471 | 0.0625 | 0.1143 | 0.0941 | 0.1045 |
| CPT-oracle | 0.0170 | 0.0341 | 0.0309 | 0.0308 | 0.0591 | 0.1056 | 0.0852 | 0.0877 |
| Bayes-oracle | 0.0177 | 0.0323 | 0.0325 | 0.0300 | 0.0539 | 0.0822 | 0.0894 | 0.0952 |

### standard: acc by subset (mean over replications)
| variant | all | far | near | imminent | silent-drone | far-silent | near-silent | imminent-silent |
|---|---|---|---|---|---|---|---|---|
| V0 | 0.5770 | 0.5703 | 0.5712 | 0.5894 | 0.5880 | 0.5316 | 0.5997 | 0.6364 |
| V1 | 0.5760 | 0.5694 | 0.5713 | 0.5872 | 0.5892 | 0.5200 | 0.6087 | 0.6434 |
| V1-cap0.98 | 0.5763 | 0.5696 | 0.5711 | 0.5880 | 0.5889 | 0.5196 | 0.6094 | 0.6421 |
| V2 | 0.5746 | 0.5667 | 0.5705 | 0.5867 | 0.5849 | 0.5151 | 0.6043 | 0.6394 |
| V3 | 0.5758 | 0.5674 | 0.5713 | 0.5888 | 0.5931 | 0.5449 | 0.6006 | 0.6367 |
| sensor-oracle-marginal | 0.5762 | 0.5683 | 0.5732 | 0.5870 | 0.5996 | 0.5509 | 0.6078 | 0.6433 |
| sensor-oracle-conditional | 0.5769 | 0.5703 | 0.5729 | 0.5875 | 0.6009 | 0.5424 | 0.6115 | 0.6523 |
| CPT-oracle | 0.5931 | 0.5873 | 0.5863 | 0.6061 | 0.6208 | 0.5754 | 0.6354 | 0.6548 |
| Bayes-oracle | 0.5939 | 0.5890 | 0.5850 | 0.6081 | 0.6259 | 0.5713 | 0.6427 | 0.6678 |

### standard: paired deltas vs V0 (t p / Wilcoxon p)
- V1 | all: dbrier +0.0004 (p 2.2e-01/3.1e-01); dece +0.0038 (p 1.2e-02/1.7e-02); dacc -0.0010 (p 1.2e-01/1.5e-01); dmacroAUC -0.0003 (p 1.5e-01/7.0e-02)
- V1 | far: dbrier +0.0002 (p 5.1e-01/7.6e-01); dece -0.0010 (p 6.1e-01/7.6e-01); dacc -0.0009 (p 4.9e-01/1.0e+00); dmacroAUC -0.0001 (p 7.5e-01/7.0e-01)
- V1 | near: dbrier +0.0002 (p 6.5e-01/1.0e+00); dece +0.0020 (p 2.8e-01/2.8e-01); dacc +0.0000 (p 9.7e-01/8.4e-01); dmacroAUC -0.0002 (p 5.5e-01/4.3e-01)
- V1 | imminent: dbrier +0.0007 (p 3.0e-01/4.7e-01); dece +0.0042 (p 1.1e-01/2.5e-01); dacc -0.0022 (p 1.1e-01/8.5e-02); dmacroAUC -0.0002 (p 6.3e-01/8.1e-01)
- V1 | silent-drone: dbrier -0.0055 (p 1.6e-04/2.6e-04); dece +0.0037 (p 3.8e-01/3.7e-01); dacc +0.0012 (p 6.7e-01/8.4e-01); dmacroAUC +0.0016 (p 1.7e-01/1.7e-01)
- V1 | far-silent: dbrier +0.0032 (p 9.8e-03/1.5e-02); dece -0.0011 (p 8.9e-01/6.7e-01); dacc -0.0116 (p 4.5e-02/7.2e-02); dmacroAUC +0.0018 (p 1.1e-01/1.9e-01)
- V1-cap0.98 | all: dbrier +0.0001 (p 6.3e-01/9.9e-01); dece +0.0035 (p 7.0e-03/1.1e-02); dacc -0.0007 (p 1.8e-01/2.8e-01); dmacroAUC -0.0001 (p 6.5e-01/8.4e-01)
- V1-cap0.98 | far: dbrier +0.0001 (p 5.9e-01/7.3e-01); dece -0.0009 (p 6.5e-01/6.7e-01); dacc -0.0006 (p 6.5e-01/8.2e-01); dmacroAUC -0.0001 (p 8.3e-01/8.1e-01)
- V1-cap0.98 | near: dbrier -0.0000 (p 9.4e-01/9.0e-01); dece +0.0020 (p 1.9e-01/2.3e-01); dacc -0.0001 (p 8.9e-01/6.5e-01); dmacroAUC +0.0001 (p 8.0e-01/9.9e-01)
- V1-cap0.98 | imminent: dbrier +0.0002 (p 6.8e-01/7.6e-01); dece +0.0033 (p 9.1e-02/2.0e-01); dacc -0.0014 (p 2.5e-01/2.8e-01); dmacroAUC +0.0002 (p 5.6e-01/6.7e-01)
- V1-cap0.98 | silent-drone: dbrier -0.0056 (p 4.2e-05/1.3e-04); dece +0.0031 (p 4.2e-01/5.0e-01); dacc +0.0009 (p 7.4e-01/8.4e-01); dmacroAUC +0.0020 (p 1.1e-01/1.0e-01)
- V1-cap0.98 | far-silent: dbrier +0.0035 (p 4.1e-03/6.4e-03); dece -0.0008 (p 9.2e-01/6.5e-01); dacc -0.0120 (p 4.0e-02/7.2e-02); dmacroAUC +0.0022 (p 1.0e-01/1.5e-01)
- V2 | all: dbrier +0.0012 (p 2.2e-04/3.2e-04); dece +0.0035 (p 1.9e-02/2.4e-02); dacc -0.0024 (p 1.2e-03/4.3e-03); dmacroAUC -0.0012 (p 1.3e-06/1.9e-06)
- V2 | far: dbrier +0.0025 (p 6.7e-04/2.6e-04); dece +0.0025 (p 2.3e-01/1.9e-01); dacc -0.0036 (p 1.0e-02/9.6e-03); dmacroAUC -0.0031 (p 9.2e-03/2.3e-03)
- V2 | near: dbrier +0.0004 (p 4.0e-01/4.1e-01); dece +0.0001 (p 9.7e-01/9.6e-01); dacc -0.0008 (p 5.9e-01/2.0e-01); dmacroAUC -0.0007 (p 7.1e-02/8.3e-02)
- V2 | imminent: dbrier +0.0007 (p 2.0e-01/3.3e-01); dece +0.0020 (p 3.1e-01/3.3e-01); dacc -0.0027 (p 4.2e-02/4.9e-02); dmacroAUC +0.0000 (p 9.6e-01/8.4e-01)
- V2 | silent-drone: dbrier -0.0027 (p 4.2e-02/5.8e-02); dece +0.0062 (p 7.4e-02/8.3e-02); dacc -0.0032 (p 3.3e-01/2.7e-01); dmacroAUC +0.0017 (p 4.0e-02/4.9e-02)
- V2 | far-silent: dbrier +0.0072 (p 1.3e-01/2.0e-01); dece -0.0055 (p 5.1e-01/9.6e-01); dacc -0.0165 (p 2.8e-02/4.8e-02); dmacroAUC +0.0072 (p 2.2e-02/2.1e-02)
- V3 | all: dbrier +0.0010 (p 8.5e-04/2.0e-03); dece +0.0020 (p 5.2e-02/5.8e-02); dacc -0.0012 (p 7.9e-02/5.8e-02); dmacroAUC -0.0008 (p 5.4e-04/4.8e-04)
- V3 | far: dbrier +0.0029 (p 3.0e-04/5.9e-04); dece +0.0056 (p 7.6e-03/1.5e-02); dacc -0.0029 (p 8.6e-02/8.9e-02); dmacroAUC -0.0015 (p 6.1e-02/5.8e-02)
- V3 | near: dbrier -0.0000 (p 8.4e-01/7.6e-01); dece +0.0009 (p 5.2e-01/4.3e-01); dacc +0.0000 (p 9.8e-01/9.1e-01); dmacroAUC +0.0000 (p 9.3e-01/5.5e-01)
- V3 | imminent: dbrier +0.0000 (p 7.5e-01/8.1e-01); dece -0.0007 (p 6.0e-01/3.1e-01); dacc -0.0006 (p 3.7e-01/6.1e-01); dmacroAUC -0.0000 (p 6.5e-01/7.8e-01)
- V3 | silent-drone: dbrier -0.0049 (p 2.8e-04/4.8e-04); dece +0.0023 (p 5.3e-01/4.1e-01); dacc +0.0051 (p 2.3e-02/2.2e-02); dmacroAUC -0.0013 (p 2.1e-01/3.3e-01)
- V3 | far-silent: dbrier -0.0091 (p 1.1e-03/2.0e-03); dece -0.0033 (p 6.9e-01/6.2e-01); dacc +0.0133 (p 1.7e-02/2.8e-02); dmacroAUC -0.0044 (p 1.3e-01/2.4e-01)
- sensor-oracle-marginal | all: dbrier +0.0001 (p 6.1e-01/8.1e-01); dece +0.0023 (p 1.1e-01/1.8e-01); dacc -0.0008 (p 4.0e-01/3.7e-01); dmacroAUC -0.0001 (p 7.8e-01/7.3e-01)
- sensor-oracle-marginal | far: dbrier +0.0017 (p 1.5e-02/1.7e-02); dece +0.0043 (p 5.7e-02/7.0e-02); dacc -0.0020 (p 2.6e-01/1.8e-01); dmacroAUC -0.0003 (p 6.6e-01/8.1e-01)
- sensor-oracle-marginal | near: dbrier -0.0010 (p 1.2e-02/6.4e-03); dece +0.0010 (p 5.7e-01/4.1e-01); dacc +0.0019 (p 1.6e-01/1.8e-01); dmacroAUC +0.0009 (p 4.8e-02/4.8e-02)
- sensor-oracle-marginal | imminent: dbrier -0.0002 (p 5.8e-01/6.5e-01); dece +0.0021 (p 2.7e-01/2.8e-01); dacc -0.0024 (p 1.3e-01/1.2e-01); dmacroAUC +0.0005 (p 2.6e-01/2.0e-01)
- sensor-oracle-marginal | silent-drone: dbrier -0.0087 (p 1.2e-03/1.7e-03); dece +0.0053 (p 3.1e-01/4.1e-01); dacc +0.0116 (p 9.8e-03/2.5e-02); dmacroAUC -0.0017 (p 4.2e-01/9.0e-01)
- sensor-oracle-marginal | far-silent: dbrier -0.0140 (p 8.2e-04/1.2e-03); dece -0.0048 (p 6.3e-01/6.7e-01); dacc +0.0193 (p 1.4e-02/2.3e-02); dmacroAUC -0.0070 (p 1.4e-01/1.5e-01)
- sensor-oracle-conditional | all: dbrier -0.0011 (p 3.0e-03/8.5e-04); dece +0.0017 (p 2.9e-01/4.7e-01); dacc -0.0001 (p 9.5e-01/7.9e-01); dmacroAUC +0.0010 (p 2.1e-03/8.5e-04)
- sensor-oracle-conditional | far: dbrier -0.0014 (p 6.1e-03/8.3e-03); dece -0.0005 (p 8.8e-01/7.3e-01); dacc +0.0001 (p 9.7e-01/6.0e-01); dmacroAUC +0.0019 (p 3.7e-03/1.2e-03)
- sensor-oracle-conditional | near: dbrier -0.0011 (p 1.3e-02/1.4e-02); dece +0.0013 (p 5.2e-01/5.5e-01); dacc +0.0016 (p 2.0e-01/3.1e-01); dmacroAUC +0.0008 (p 7.1e-02/8.3e-02)
- sensor-oracle-conditional | imminent: dbrier -0.0008 (p 9.1e-02/1.1e-01); dece +0.0042 (p 4.7e-02/8.3e-02); dacc -0.0019 (p 2.2e-01/3.3e-01); dmacroAUC +0.0016 (p 1.8e-02/1.2e-02)
- sensor-oracle-conditional | silent-drone: dbrier -0.0099 (p 1.9e-04/3.9e-04); dece -0.0001 (p 9.9e-01/7.3e-01); dacc +0.0128 (p 2.2e-03/1.1e-03); dmacroAUC +0.0012 (p 5.0e-01/5.4e-01)
- sensor-oracle-conditional | far-silent: dbrier -0.0026 (p 3.5e-01/2.3e-01); dece -0.0060 (p 4.5e-01/6.2e-01); dacc +0.0108 (p 6.0e-02/4.7e-02); dmacroAUC +0.0007 (p 8.7e-01/1.0e+00)
- CPT-oracle | all: dbrier -0.0120 (p 4.2e-08/1.9e-06); dece -0.0131 (p 2.7e-04/8.2e-05); dacc +0.0161 (p 2.5e-05/9.5e-06); dmacroAUC +0.0067 (p 9.8e-09/1.9e-06)
- CPT-oracle | far: dbrier -0.0113 (p 8.3e-04/1.9e-06); dece -0.0113 (p 1.7e-02/1.7e-02); dacc +0.0170 (p 2.1e-02/4.9e-03); dmacroAUC +0.0066 (p 1.0e-05/4.8e-05)
- CPT-oracle | near: dbrier -0.0122 (p 1.1e-07/1.9e-06); dece -0.0139 (p 7.2e-03/1.1e-02); dacc +0.0151 (p 1.6e-04/6.2e-04); dmacroAUC +0.0067 (p 5.7e-06/1.9e-06)
- CPT-oracle | imminent: dbrier -0.0125 (p 1.8e-06/3.8e-06); dece -0.0121 (p 6.3e-03/5.6e-03); dacc +0.0167 (p 2.0e-04/3.9e-04); dmacroAUC +0.0070 (p 2.2e-04/4.8e-04)
- CPT-oracle | silent-drone: dbrier -0.0280 (p 2.7e-04/2.6e-04); dece -0.0035 (p 4.6e-01/5.7e-01); dacc +0.0328 (p 5.1e-06/1.9e-05); dmacroAUC +0.0086 (p 1.2e-02/1.3e-02)
- CPT-oracle | far-silent: dbrier -0.0360 (p 5.1e-04/3.2e-04); dece -0.0148 (p 1.3e-01/1.8e-01); dacc +0.0438 (p 7.1e-05/3.3e-04); dmacroAUC +0.0045 (p 5.1e-01/4.5e-01)
- Bayes-oracle | all: dbrier -0.0133 (p 6.9e-09/1.9e-06); dece -0.0125 (p 2.4e-04/6.3e-05); dacc +0.0169 (p 2.1e-05/2.0e-04); dmacroAUC +0.0078 (p 1.1e-09/1.9e-06)
- Bayes-oracle | far: dbrier -0.0140 (p 6.0e-05/1.9e-06); dece -0.0131 (p 2.6e-02/3.3e-02); dacc +0.0188 (p 1.1e-02/1.2e-03); dmacroAUC +0.0092 (p 3.6e-08/1.9e-06)
- Bayes-oracle | near: dbrier -0.0124 (p 8.7e-08/1.9e-06); dece -0.0123 (p 1.5e-02/1.7e-02); dacc +0.0138 (p 2.4e-04/1.7e-04); dmacroAUC +0.0068 (p 5.2e-06/1.9e-06)
- Bayes-oracle | imminent: dbrier -0.0135 (p 1.1e-06/1.9e-06); dece -0.0129 (p 1.9e-03/2.0e-03); dacc +0.0187 (p 6.6e-05/1.7e-04); dmacroAUC +0.0079 (p 1.5e-04/3.9e-04)
- Bayes-oracle | silent-drone: dbrier -0.0290 (p 1.3e-04/8.2e-05); dece -0.0087 (p 1.2e-01/1.3e-01); dacc +0.0379 (p 4.1e-06/1.3e-04); dmacroAUC +0.0109 (p 3.6e-03/2.3e-03)
- Bayes-oracle | far-silent: dbrier -0.0237 (p 8.2e-03/1.7e-02); dece -0.0381 (p 1.0e-03/1.4e-03); dacc +0.0397 (p 1.3e-04/3.4e-04); dmacroAUC +0.0131 (p 7.5e-02/3.3e-02)

## A5 structural world, n=300: full test set (mean +- 95% CI half-width)
- V0: acc 0.5728+-0.0081, macroF1 0.4843+-0.0074, macroAUC 0.8143+-0.0027, brier 0.5317+-0.0046, ece 0.0289+-0.0066, cwece 0.0315+-0.0033, far@pd0.9 0.4701+-0.0102
- V1: acc 0.5712+-0.0086, macroF1 0.4869+-0.0067, macroAUC 0.8132+-0.0026, brier 0.5333+-0.0045, ece 0.0333+-0.0068, cwece 0.0321+-0.0034, far@pd0.9 0.4708+-0.0108
- V1-cap0.98: acc 0.5716+-0.0086, macroF1 0.4873+-0.0069, macroAUC 0.8136+-0.0026, brier 0.5327+-0.0045, ece 0.0325+-0.0066, cwece 0.0319+-0.0035, far@pd0.9 0.4702+-0.0107
- V2: acc 0.5751+-0.0076, macroF1 0.4861+-0.0064, macroAUC 0.8147+-0.0025, brier 0.5309+-0.0044, ece 0.0319+-0.0057, cwece 0.0323+-0.0033, far@pd0.9 0.4689+-0.0115
- V3: acc 0.5717+-0.0080, macroF1 0.4861+-0.0074, macroAUC 0.8137+-0.0026, brier 0.5323+-0.0046, ece 0.0319+-0.0064, cwece 0.0317+-0.0033, far@pd0.9 0.4720+-0.0112
- sensor-oracle-marginal: acc 0.5728+-0.0081, macroF1 0.4866+-0.0079, macroAUC 0.8144+-0.0024, brier 0.5315+-0.0045, ece 0.0327+-0.0059, cwece 0.0321+-0.0031, far@pd0.9 0.4692+-0.0114
- sensor-oracle-conditional: acc 0.5768+-0.0078, macroF1 0.4892+-0.0074, macroAUC 0.8167+-0.0024, brier 0.5286+-0.0044, ece 0.0326+-0.0054, cwece 0.0323+-0.0031, far@pd0.9 0.4578+-0.0099
- CPT-oracle: acc 0.5895+-0.0051, macroF1 0.4866+-0.0052, macroAUC 0.8209+-0.0023, brier 0.5197+-0.0036, ece 0.0174+-0.0022, cwece 0.0188+-0.0009, far@pd0.9 0.4643+-0.0116
- Bayes-oracle: acc 0.5935+-0.0055, macroF1 0.4897+-0.0053, macroAUC 0.8234+-0.0024, brier 0.5164+-0.0036, ece 0.0179+-0.0020, cwece 0.0186+-0.0008, far@pd0.9 0.4555+-0.0116

### structural: brier by subset (mean over replications)
| variant | all | far | near | imminent | silent-drone | far-silent | near-silent | imminent-silent |
|---|---|---|---|---|---|---|---|---|
| V0 | 0.5317 | 0.5438 | 0.5389 | 0.5119 | 0.5474 | 0.6306 | 0.5362 | 0.4691 |
| V1 | 0.5333 | 0.5443 | 0.5394 | 0.5157 | 0.5349 | 0.6358 | 0.5199 | 0.4416 |
| V1-cap0.98 | 0.5327 | 0.5443 | 0.5393 | 0.5141 | 0.5349 | 0.6362 | 0.5199 | 0.4413 |
| V2 | 0.5309 | 0.5415 | 0.5393 | 0.5114 | 0.5281 | 0.5851 | 0.5319 | 0.4624 |
| V3 | 0.5323 | 0.5458 | 0.5386 | 0.5120 | 0.5354 | 0.6042 | 0.5292 | 0.4676 |
| sensor-oracle-marginal | 0.5315 | 0.5449 | 0.5377 | 0.5116 | 0.5334 | 0.6021 | 0.5276 | 0.4653 |
| sensor-oracle-conditional | 0.5286 | 0.5380 | 0.5375 | 0.5100 | 0.5228 | 0.5814 | 0.5265 | 0.4553 |
| CPT-oracle | 0.5197 | 0.5322 | 0.5268 | 0.4996 | 0.5143 | 0.5807 | 0.5076 | 0.4502 |
| Bayes-oracle | 0.5164 | 0.5254 | 0.5263 | 0.4973 | 0.5034 | 0.5600 | 0.5064 | 0.4397 |

### structural: ece by subset (mean over replications)
| variant | all | far | near | imminent | silent-drone | far-silent | near-silent | imminent-silent |
|---|---|---|---|---|---|---|---|---|
| V0 | 0.0289 | 0.0491 | 0.0403 | 0.0412 | 0.0692 | 0.1290 | 0.0900 | 0.0989 |
| V1 | 0.0333 | 0.0503 | 0.0465 | 0.0463 | 0.0724 | 0.1288 | 0.0960 | 0.0893 |
| V1-cap0.98 | 0.0325 | 0.0495 | 0.0470 | 0.0453 | 0.0722 | 0.1320 | 0.0983 | 0.0892 |
| V2 | 0.0319 | 0.0457 | 0.0437 | 0.0448 | 0.0651 | 0.1227 | 0.1051 | 0.0980 |
| V3 | 0.0319 | 0.0559 | 0.0431 | 0.0410 | 0.0666 | 0.1355 | 0.0894 | 0.0940 |
| sensor-oracle-marginal | 0.0327 | 0.0583 | 0.0449 | 0.0444 | 0.0694 | 0.1436 | 0.0866 | 0.1030 |
| sensor-oracle-conditional | 0.0326 | 0.0472 | 0.0465 | 0.0471 | 0.0641 | 0.1071 | 0.0883 | 0.1045 |
| CPT-oracle | 0.0174 | 0.0401 | 0.0326 | 0.0303 | 0.0546 | 0.1226 | 0.0821 | 0.0828 |
| Bayes-oracle | 0.0179 | 0.0358 | 0.0308 | 0.0300 | 0.0561 | 0.0963 | 0.0880 | 0.0952 |

### structural: acc by subset (mean over replications)
| variant | all | far | near | imminent | silent-drone | far-silent | near-silent | imminent-silent |
|---|---|---|---|---|---|---|---|---|
| V0 | 0.5728 | 0.5623 | 0.5692 | 0.5869 | 0.5679 | 0.4772 | 0.5961 | 0.6360 |
| V1 | 0.5712 | 0.5621 | 0.5677 | 0.5838 | 0.5796 | 0.4881 | 0.6063 | 0.6498 |
| V1-cap0.98 | 0.5716 | 0.5617 | 0.5683 | 0.5849 | 0.5787 | 0.4855 | 0.6064 | 0.6499 |
| V2 | 0.5751 | 0.5693 | 0.5690 | 0.5867 | 0.5932 | 0.5447 | 0.5995 | 0.6394 |
| V3 | 0.5717 | 0.5597 | 0.5696 | 0.5860 | 0.5773 | 0.5018 | 0.5980 | 0.6358 |
| sensor-oracle-marginal | 0.5728 | 0.5592 | 0.5725 | 0.5871 | 0.5763 | 0.4934 | 0.6020 | 0.6381 |
| sensor-oracle-conditional | 0.5768 | 0.5720 | 0.5706 | 0.5875 | 0.5990 | 0.5483 | 0.5997 | 0.6523 |
| CPT-oracle | 0.5895 | 0.5781 | 0.5832 | 0.6077 | 0.5901 | 0.4988 | 0.6234 | 0.6535 |
| Bayes-oracle | 0.5935 | 0.5896 | 0.5832 | 0.6081 | 0.6145 | 0.5507 | 0.6289 | 0.6678 |

### structural: paired deltas vs V0 (t p / Wilcoxon p)
- V1 | all: dbrier +0.0016 (p 1.7e-03/1.4e-03); dece +0.0044 (p 5.4e-03/4.9e-03); dacc -0.0016 (p 4.1e-02/5.3e-02); dmacroAUC -0.0011 (p 1.1e-02/1.4e-02)
- V1 | far: dbrier +0.0006 (p 4.0e-01/4.1e-01); dece +0.0012 (p 5.6e-01/7.6e-01); dacc -0.0002 (p 9.0e-01/4.2e-01); dmacroAUC -0.0008 (p 4.0e-01/5.5e-01)
- V1 | near: dbrier +0.0006 (p 4.2e-01/7.6e-01); dece +0.0061 (p 5.2e-03/7.3e-03); dacc -0.0015 (p 1.7e-01/3.0e-01); dmacroAUC -0.0007 (p 2.5e-01/2.9e-01)
- V1 | imminent: dbrier +0.0038 (p 8.7e-04/8.5e-04); dece +0.0051 (p 1.3e-01/7.6e-02); dacc -0.0030 (p 9.5e-02/9.1e-02); dmacroAUC -0.0014 (p 6.1e-02/7.6e-02)
- V1 | silent-drone: dbrier -0.0124 (p 5.1e-08/3.8e-06); dece +0.0032 (p 3.6e-01/3.5e-01); dacc +0.0117 (p 1.8e-03/1.9e-03); dmacroAUC +0.0066 (p 4.1e-03/6.1e-04)
- V1 | far-silent: dbrier +0.0052 (p 1.0e-02/1.5e-02); dece -0.0002 (p 9.7e-01/8.4e-01); dacc +0.0109 (p 1.4e-01/2.2e-01); dmacroAUC +0.0079 (p 1.2e-01/1.7e-01)
- V1-cap0.98 | all: dbrier +0.0011 (p 1.0e-02/1.4e-02); dece +0.0036 (p 1.9e-02/2.7e-02); dacc -0.0012 (p 1.5e-01/1.5e-01); dmacroAUC -0.0007 (p 4.6e-02/1.1e-01)
- V1-cap0.98 | far: dbrier +0.0005 (p 4.2e-01/3.9e-01); dece +0.0004 (p 8.4e-01/9.3e-01); dacc -0.0007 (p 7.4e-01/3.5e-01); dmacroAUC -0.0007 (p 4.4e-01/5.7e-01)
- V1-cap0.98 | near: dbrier +0.0004 (p 5.5e-01/8.7e-01); dece +0.0066 (p 1.2e-03/1.2e-03); dacc -0.0009 (p 4.6e-01/7.2e-01); dmacroAUC -0.0005 (p 3.6e-01/5.7e-01)
- V1-cap0.98 | imminent: dbrier +0.0022 (p 5.6e-03/9.4e-03); dece +0.0041 (p 1.8e-01/4.8e-02); dacc -0.0020 (p 2.0e-01/2.1e-01); dmacroAUC -0.0004 (p 4.6e-01/3.5e-01)
- V1-cap0.98 | silent-drone: dbrier -0.0124 (p 1.2e-08/1.9e-06); dece +0.0031 (p 3.8e-01/3.5e-01); dacc +0.0109 (p 3.0e-03/3.1e-03); dmacroAUC +0.0067 (p 3.8e-03/6.1e-04)
- V1-cap0.98 | far-silent: dbrier +0.0055 (p 5.6e-03/7.3e-03); dece +0.0030 (p 6.0e-01/3.5e-01); dacc +0.0083 (p 2.3e-01/3.5e-01); dmacroAUC +0.0078 (p 1.2e-01/1.7e-01)
- V2 | all: dbrier -0.0008 (p 5.3e-02/7.0e-02); dece +0.0029 (p 7.0e-02/7.6e-02); dacc +0.0023 (p 2.2e-02/2.9e-02); dmacroAUC +0.0005 (p 9.5e-02/1.2e-01)
- V2 | far: dbrier -0.0023 (p 3.2e-02/4.4e-02); dece -0.0034 (p 2.3e-01/4.1e-01); dacc +0.0070 (p 5.9e-03/5.6e-03); dmacroAUC +0.0031 (p 3.6e-02/1.7e-02)
- V2 | near: dbrier +0.0005 (p 3.0e-01/3.3e-01); dece +0.0033 (p 7.7e-02/1.3e-01); dacc -0.0002 (p 8.6e-01/6.5e-01); dmacroAUC -0.0005 (p 1.5e-01/1.4e-01)
- V2 | imminent: dbrier -0.0005 (p 4.8e-01/2.8e-01); dece +0.0036 (p 1.5e-01/1.2e-01); dacc -0.0002 (p 8.9e-01/9.7e-01); dmacroAUC +0.0011 (p 5.1e-02/5.8e-02)
- V2 | silent-drone: dbrier -0.0193 (p 1.9e-11/1.9e-06); dece -0.0040 (p 3.5e-01/4.7e-01); dacc +0.0254 (p 2.7e-07/1.3e-04); dmacroAUC +0.0094 (p 2.3e-04/1.2e-04)
- V2 | far-silent: dbrier -0.0456 (p 1.7e-10/1.9e-06); dece -0.0063 (p 5.2e-01/8.1e-01); dacc +0.0675 (p 1.9e-08/1.9e-06); dmacroAUC +0.0290 (p 6.4e-04/2.4e-04)
- V3 | all: dbrier +0.0007 (p 2.7e-02/3.0e-02); dece +0.0029 (p 1.0e-02/1.2e-02); dacc -0.0011 (p 1.7e-01/1.5e-01); dmacroAUC -0.0005 (p 2.4e-02/5.8e-02)
- V3 | far: dbrier +0.0021 (p 1.1e-02/1.5e-02); dece +0.0068 (p 1.5e-03/2.0e-03); dacc -0.0026 (p 2.1e-01/2.5e-01); dmacroAUC -0.0007 (p 4.7e-01/7.0e-01)
- V3 | near: dbrier -0.0002 (p 4.1e-01/2.8e-01); dece +0.0027 (p 1.2e-01/1.8e-01); dacc +0.0004 (p 7.1e-01/3.6e-01); dmacroAUC -0.0001 (p 8.7e-01/9.3e-01)
- V3 | imminent: dbrier +0.0001 (p 2.1e-01/1.8e-01); dece -0.0002 (p 8.4e-01/9.9e-01); dacc -0.0008 (p 2.3e-01/2.1e-01); dmacroAUC -0.0001 (p 1.0e-01/9.0e-02)
- V3 | silent-drone: dbrier -0.0120 (p 4.6e-10/1.9e-06); dece -0.0026 (p 5.6e-01/5.2e-01); dacc +0.0094 (p 3.6e-03/7.5e-03); dmacroAUC -0.0004 (p 8.1e-01/1.0e+00)
- V3 | far-silent: dbrier -0.0264 (p 4.2e-09/1.9e-06); dece +0.0065 (p 4.4e-01/7.8e-01); dacc +0.0246 (p 4.5e-03/7.9e-03); dmacroAUC -0.0017 (p 7.3e-01/9.5e-01)
- sensor-oracle-marginal | all: dbrier -0.0001 (p 7.5e-01/5.2e-01); dece +0.0037 (p 1.6e-02/2.7e-02); dacc +0.0001 (p 9.4e-01/8.4e-01); dmacroAUC +0.0001 (p 6.6e-01/2.6e-01)
- sensor-oracle-marginal | far: dbrier +0.0011 (p 1.4e-01/1.9e-01); dece +0.0092 (p 6.0e-03/9.4e-03); dacc -0.0031 (p 1.3e-01/1.5e-01); dmacroAUC +0.0011 (p 1.9e-01/2.5e-01)
- sensor-oracle-marginal | near: dbrier -0.0012 (p 1.7e-02/9.4e-03); dece +0.0045 (p 6.8e-02/6.4e-02); dacc +0.0032 (p 1.3e-02/1.5e-02); dmacroAUC +0.0011 (p 2.7e-02/3.0e-02)
- sensor-oracle-marginal | imminent: dbrier -0.0003 (p 4.6e-01/2.9e-01); dece +0.0031 (p 1.0e-01/4.0e-02); dacc +0.0003 (p 8.7e-01/8.7e-01); dmacroAUC +0.0001 (p 7.9e-01/4.3e-01)
- sensor-oracle-marginal | silent-drone: dbrier -0.0140 (p 9.0e-06/4.8e-05); dece +0.0002 (p 9.6e-01/8.7e-01); dacc +0.0084 (p 3.2e-02/4.0e-02); dmacroAUC -0.0008 (p 6.7e-01/7.6e-01)
- sensor-oracle-marginal | far-silent: dbrier -0.0286 (p 2.4e-07/3.8e-06); dece +0.0146 (p 1.3e-01/1.1e-01); dacc +0.0162 (p 3.4e-02/5.3e-02); dmacroAUC -0.0043 (p 3.1e-01/2.2e-01)
- sensor-oracle-conditional | all: dbrier -0.0031 (p 2.4e-08/3.8e-06); dece +0.0036 (p 3.9e-02/8.3e-02); dacc +0.0040 (p 8.0e-04/2.9e-03); dmacroAUC +0.0024 (p 2.0e-08/3.8e-06)
- sensor-oracle-conditional | far: dbrier -0.0058 (p 4.3e-07/1.9e-06); dece -0.0019 (p 4.8e-01/3.7e-01); dacc +0.0097 (p 7.3e-05/1.7e-04); dmacroAUC +0.0076 (p 4.6e-07/9.5e-06)
- sensor-oracle-conditional | near: dbrier -0.0014 (p 9.4e-03/7.3e-03); dece +0.0062 (p 1.2e-02/1.2e-02); dacc +0.0014 (p 1.9e-01/3.0e-01); dmacroAUC +0.0014 (p 2.0e-02/3.0e-02)
- sensor-oracle-conditional | imminent: dbrier -0.0019 (p 3.0e-03/3.7e-03); dece +0.0059 (p 1.4e-02/7.3e-03); dacc +0.0006 (p 7.1e-01/8.4e-01); dmacroAUC +0.0026 (p 1.8e-03/8.5e-04)
- sensor-oracle-conditional | silent-drone: dbrier -0.0246 (p 4.0e-09/3.8e-06); dece -0.0051 (p 3.4e-01/4.7e-01); dacc +0.0311 (p 1.0e-08/1.3e-04); dmacroAUC +0.0090 (p 9.8e-04/5.2e-03)
- sensor-oracle-conditional | far-silent: dbrier -0.0493 (p 9.6e-11/1.9e-06); dece -0.0219 (p 7.9e-02/1.4e-01); dacc +0.0711 (p 1.8e-08/1.9e-06); dmacroAUC +0.0198 (p 8.3e-03/1.0e-02)
- CPT-oracle | all: dbrier -0.0120 (p 2.9e-08/1.9e-06); dece -0.0115 (p 1.5e-03/8.5e-04); dacc +0.0167 (p 4.5e-06/8.8e-05); dmacroAUC +0.0066 (p 3.3e-08/1.9e-06)
- CPT-oracle | far: dbrier -0.0116 (p 4.6e-04/1.9e-06); dece -0.0090 (p 1.3e-01/4.1e-01); dacc +0.0157 (p 2.4e-02/8.9e-03); dmacroAUC +0.0080 (p 3.9e-05/1.0e-04)
- CPT-oracle | near: dbrier -0.0120 (p 5.4e-07/1.9e-06); dece -0.0077 (p 7.1e-02/9.7e-02); dacc +0.0140 (p 2.2e-05/2.2e-04); dmacroAUC +0.0070 (p 4.2e-05/1.9e-06)
- CPT-oracle | imminent: dbrier -0.0123 (p 1.1e-06/1.9e-06); dece -0.0109 (p 2.7e-03/3.7e-03); dacc +0.0208 (p 1.1e-05/2.7e-05); dmacroAUC +0.0068 (p 2.8e-04/3.2e-04)
- CPT-oracle | silent-drone: dbrier -0.0331 (p 8.3e-05/1.3e-04); dece -0.0145 (p 1.7e-02/1.7e-02); dacc +0.0222 (p 1.2e-03/1.5e-03); dmacroAUC +0.0082 (p 1.6e-02/1.3e-02)
- CPT-oracle | far-silent: dbrier -0.0499 (p 2.9e-05/3.6e-05); dece -0.0064 (p 6.3e-01/9.0e-01); dacc +0.0216 (p 5.2e-02/6.1e-02); dmacroAUC +0.0055 (p 4.4e-01/5.4e-01)
- Bayes-oracle | all: dbrier -0.0152 (p 1.1e-09/1.9e-06); dece -0.0111 (p 1.9e-03/4.8e-04); dacc +0.0208 (p 8.2e-07/1.0e-04); dmacroAUC +0.0092 (p 7.7e-10/1.9e-06)
- Bayes-oracle | far: dbrier -0.0184 (p 3.7e-06/1.9e-06); dece -0.0133 (p 4.2e-02/4.4e-02); dacc +0.0273 (p 5.4e-04/1.9e-05); dmacroAUC +0.0146 (p 7.0e-08/1.9e-06)
- Bayes-oracle | near: dbrier -0.0126 (p 2.4e-07/1.9e-06); dece -0.0095 (p 1.6e-02/1.9e-02); dacc +0.0139 (p 4.5e-05/6.3e-05); dmacroAUC +0.0071 (p 5.0e-05/5.7e-06)
- Bayes-oracle | imminent: dbrier -0.0147 (p 1.1e-07/1.9e-06); dece -0.0112 (p 1.7e-03/1.4e-03); dacc +0.0212 (p 3.3e-05/1.3e-04); dmacroAUC +0.0089 (p 2.9e-05/6.3e-05)
- Bayes-oracle | silent-drone: dbrier -0.0440 (p 2.1e-06/3.8e-06); dece -0.0131 (p 5.1e-02/1.6e-01); dacc +0.0466 (p 5.9e-07/1.3e-04); dmacroAUC +0.0181 (p 6.2e-05/3.7e-04)
- Bayes-oracle | far-silent: dbrier -0.0706 (p 4.0e-07/1.9e-06); dece -0.0327 (p 2.0e-02/1.1e-02); dacc +0.0735 (p 1.0e-05/2.0e-04); dmacroAUC +0.0257 (p 1.4e-02/2.1e-02)

## A5 calibration-size sweep (full test set; ECE near / imminent)
- standard n=30 V0: AUC 0.8105+-0.0033, Brier 0.5359+-0.0049, ECE 0.0355+-0.0045; ECE near 0.0461, imminent 0.0483
- standard n=30 V1: AUC 0.8089+-0.0034, Brier 0.5381+-0.0049, ECE 0.0380+-0.0059; ECE near 0.0516, imminent 0.0482
- standard n=30 V2: AUC 0.8047+-0.0040, Brier 0.5427+-0.0045, ECE 0.0336+-0.0059; ECE near 0.0471, imminent 0.0489
- standard n=30 V3: AUC 0.8102+-0.0032, Brier 0.5365+-0.0051, ECE 0.0375+-0.0053; ECE near 0.0473, imminent 0.0489
    V1-V0: dBrier +0.0022 (p 2.0e-03), dECE +0.0024 (p 1.8e-01)
    V2-V0: dBrier +0.0068 (p 7.8e-07), dECE -0.0019 (p 4.0e-01)
    V3-V0: dBrier +0.0006 (p 1.7e-01), dECE +0.0020 (p 2.1e-01)
- standard n=100 V0: AUC 0.8154+-0.0027, Brier 0.5300+-0.0046, ECE 0.0297+-0.0050; ECE near 0.0423, imminent 0.0449
- standard n=100 V1: AUC 0.8144+-0.0027, Brier 0.5313+-0.0048, ECE 0.0340+-0.0066; ECE near 0.0455, imminent 0.0485
- standard n=100 V2: AUC 0.8126+-0.0028, Brier 0.5333+-0.0046, ECE 0.0338+-0.0063; ECE near 0.0456, imminent 0.0489
- standard n=100 V3: AUC 0.8151+-0.0026, Brier 0.5306+-0.0047, ECE 0.0333+-0.0061; ECE near 0.0460, imminent 0.0437
    V1-V0: dBrier +0.0013 (p 6.8e-03), dECE +0.0043 (p 1.6e-02)
    V2-V0: dBrier +0.0032 (p 2.4e-06), dECE +0.0041 (p 4.2e-02)
    V3-V0: dBrier +0.0005 (p 7.3e-02), dECE +0.0036 (p 7.3e-03)
- standard n=300 V0: AUC 0.8169+-0.0025, Brier 0.5280+-0.0044, ECE 0.0301+-0.0054; ECE near 0.0449, imminent 0.0429
- standard n=300 V1: AUC 0.8166+-0.0026, Brier 0.5283+-0.0046, ECE 0.0339+-0.0055; ECE near 0.0469, imminent 0.0470
- standard n=300 V2: AUC 0.8157+-0.0026, Brier 0.5292+-0.0045, ECE 0.0337+-0.0056; ECE near 0.0449, imminent 0.0448
- standard n=300 V3: AUC 0.8161+-0.0024, Brier 0.5290+-0.0045, ECE 0.0321+-0.0052; ECE near 0.0457, imminent 0.0422
    V1-V0: dBrier +0.0004 (p 2.2e-01), dECE +0.0038 (p 1.2e-02)
    V2-V0: dBrier +0.0012 (p 2.2e-04), dECE +0.0035 (p 1.9e-02)
    V3-V0: dBrier +0.0010 (p 8.5e-04), dECE +0.0020 (p 5.2e-02)
- standard n=1000 V0: AUC 0.8173+-0.0024, Brier 0.5276+-0.0043, ECE 0.0302+-0.0049; ECE near 0.0452, imminent 0.0437
- standard n=1000 V1: AUC 0.8174+-0.0025, Brier 0.5273+-0.0044, ECE 0.0328+-0.0054; ECE near 0.0461, imminent 0.0476
- standard n=1000 V2: AUC 0.8172+-0.0025, Brier 0.5276+-0.0044, ECE 0.0325+-0.0048; ECE near 0.0460, imminent 0.0470
- standard n=1000 V3: AUC 0.8165+-0.0024, Brier 0.5285+-0.0043, ECE 0.0333+-0.0052; ECE near 0.0474, imminent 0.0443
    V1-V0: dBrier -0.0002 (p 1.3e-01), dECE +0.0026 (p 1.3e-02)
    V2-V0: dBrier +0.0000 (p 7.5e-01), dECE +0.0023 (p 1.4e-02)
    V3-V0: dBrier +0.0009 (p 2.0e-04), dECE +0.0031 (p 4.7e-03)
- structural n=30 V0: AUC 0.8085+-0.0033, Brier 0.5389+-0.0050, ECE 0.0338+-0.0053; ECE near 0.0444, imminent 0.0468
- structural n=30 V1: AUC 0.8069+-0.0034, Brier 0.5409+-0.0050, ECE 0.0357+-0.0063; ECE near 0.0506, imminent 0.0493
- structural n=30 V2: AUC 0.8032+-0.0040, Brier 0.5445+-0.0048, ECE 0.0344+-0.0059; ECE near 0.0492, imminent 0.0489
- structural n=30 V3: AUC 0.8080+-0.0032, Brier 0.5397+-0.0051, ECE 0.0389+-0.0061; ECE near 0.0452, imminent 0.0492
    V1-V0: dBrier +0.0020 (p 9.3e-04), dECE +0.0019 (p 2.2e-01)
    V2-V0: dBrier +0.0055 (p 4.3e-05), dECE +0.0007 (p 7.7e-01)
    V3-V0: dBrier +0.0008 (p 1.1e-01), dECE +0.0051 (p 6.0e-04)
- structural n=100 V0: AUC 0.8128+-0.0029, Brier 0.5336+-0.0048, ECE 0.0297+-0.0058; ECE near 0.0453, imminent 0.0451
- structural n=100 V1: AUC 0.8115+-0.0030, Brier 0.5353+-0.0050, ECE 0.0346+-0.0076; ECE near 0.0470, imminent 0.0494
- structural n=100 V2: AUC 0.8109+-0.0030, Brier 0.5357+-0.0048, ECE 0.0339+-0.0060; ECE near 0.0449, imminent 0.0489
- structural n=100 V3: AUC 0.8125+-0.0029, Brier 0.5339+-0.0049, ECE 0.0316+-0.0063; ECE near 0.0433, imminent 0.0431
    V1-V0: dBrier +0.0018 (p 1.9e-03), dECE +0.0049 (p 6.1e-03)
    V2-V0: dBrier +0.0021 (p 8.2e-03), dECE +0.0042 (p 5.1e-03)
    V3-V0: dBrier +0.0004 (p 2.6e-01), dECE +0.0019 (p 7.2e-02)
- structural n=300 V0: AUC 0.8143+-0.0027, Brier 0.5317+-0.0046, ECE 0.0289+-0.0066; ECE near 0.0403, imminent 0.0412
- structural n=300 V1: AUC 0.8132+-0.0026, Brier 0.5333+-0.0045, ECE 0.0333+-0.0068; ECE near 0.0465, imminent 0.0463
- structural n=300 V2: AUC 0.8147+-0.0025, Brier 0.5309+-0.0044, ECE 0.0319+-0.0057; ECE near 0.0437, imminent 0.0448
- structural n=300 V3: AUC 0.8137+-0.0026, Brier 0.5323+-0.0046, ECE 0.0319+-0.0064; ECE near 0.0431, imminent 0.0410
    V1-V0: dBrier +0.0016 (p 1.7e-03), dECE +0.0044 (p 5.4e-03)
    V2-V0: dBrier -0.0008 (p 5.3e-02), dECE +0.0029 (p 7.0e-02)
    V3-V0: dBrier +0.0007 (p 2.7e-02), dECE +0.0029 (p 1.0e-02)
- structural n=1000 V0: AUC 0.8148+-0.0026, Brier 0.5309+-0.0046, ECE 0.0304+-0.0060; ECE near 0.0442, imminent 0.0431
- structural n=1000 V1: AUC 0.8143+-0.0026, Brier 0.5318+-0.0045, ECE 0.0329+-0.0059; ECE near 0.0466, imminent 0.0470
- structural n=1000 V2: AUC 0.8162+-0.0025, Brier 0.5292+-0.0044, ECE 0.0307+-0.0055; ECE near 0.0458, imminent 0.0470
- structural n=1000 V3: AUC 0.8143+-0.0025, Brier 0.5316+-0.0045, ECE 0.0336+-0.0062; ECE near 0.0451, imminent 0.0434
    V1-V0: dBrier +0.0009 (p 1.4e-02), dECE +0.0025 (p 4.2e-02)
    V2-V0: dBrier -0.0018 (p 4.7e-05), dECE +0.0003 (p 7.9e-01)
    V3-V0: dBrier +0.0006 (p 2.4e-02), dECE +0.0032 (p 6.4e-03)

## reliability weights (n=300, mean over replications) and projection activity
- standard Rr: alpha_V0 0.884/0.931/0.955; kappa 0.822/0.909/0.953 (min 0.539/0.687/0.760); kbar 0.894
- standard Re: alpha_V0 0.721/0.898/0.974; kappa 0.552/0.858/0.990 (min 0.289/0.722/0.822); kbar 0.795
- standard Ra: alpha_V0 0.757/0.920/0.975; kappa 0.448/0.853/0.986 (min 0.273/0.671/0.774); kbar 0.756
- standard Rf: alpha_V0 0.962/0.954/0.960; kappa 0.972/0.959/0.969 (min 0.860/0.886/0.878); kbar 0.966
- standard n=30: projected rows 187/300 (62.3%); by sensor Rr 32/60, Re 33/60, Ra 17/60, Rf 105/120
- standard n=100: projected rows 119/300 (39.7%); by sensor Rr 11/60, Re 17/60, Ra 3/60, Rf 88/120
- standard n=300: projected rows 53/300 (17.7%); by sensor Rr 4/60, Re 4/60, Ra 0/60, Rf 45/120
- standard n=1000: projected rows 8/300 (2.7%); by sensor Rr 0/60, Re 0/60, Ra 0/60, Rf 8/120
- structural Rr: alpha_V0 0.861/0.925/0.962; kappa 0.779/0.898/0.967 (min 0.449/0.663/0.777); kbar 0.880
- structural Re: alpha_V0 0.607/0.878/0.976; kappa 0.353/0.824/0.994 (min 0.142/0.681/0.876); kbar 0.718
- structural Ra: alpha_V0 0.660/0.899/0.977; kappa 0.203/0.802/0.991 (min 0.075/0.545/0.830); kbar 0.658
- structural Rf: alpha_V0 0.962/0.954/0.960; kappa 0.972/0.959/0.969 (min 0.860/0.886/0.878); kbar 0.966
- structural n=30: projected rows 201/300 (67.0%); by sensor Rr 33/60, Re 43/60, Ra 20/60, Rf 105/120
- structural n=100: projected rows 140/300 (46.7%); by sensor Rr 6/60, Re 36/60, Ra 10/60, Rf 88/120
- structural n=300: projected rows 84/300 (28.0%); by sensor Rr 1/60, Re 29/60, Ra 9/60, Rf 45/120
- structural n=1000: projected rows 41/300 (13.7%); by sensor Rr 0/60, Re 29/60, Ra 4/60, Rf 8/120
- RF exact chance accuracy (mean) 0.3949 vs 1/K = 0.3333

## A6 range decomposition of the radio-silent-drone delta (n=300; contribution = n_d/n_silent x subset delta)
- standard V0-V3 brier: total +0.00487 (p 2.8e-04); far +0.00325 (p 1.1e-03); near +0.00110 (p 3.4e-03); imminent +0.00053 (p 8.9e-03)
- standard V0-V3 acc: total -0.00510 (p 2.3e-02); far -0.00470 (p 1.5e-02); near -0.00031 (p 7.8e-01); imminent -0.00009 (p 8.7e-01)
- standard V1-V3 brier: total -0.00063 (p 5.2e-01); far +0.00438 (p 9.1e-04); near -0.00150 (p 3.1e-02); imminent -0.00351 (p 4.9e-04)
- standard V1-V3 acc: total -0.00389 (p 2.3e-01); far -0.00870 (p 2.7e-03); near +0.00258 (p 1.1e-01); imminent +0.00223 (p 6.1e-02)
- structural V0-V3 brier: total +0.01197 (p 4.6e-10); far +0.00918 (p 4.6e-09); near +0.00230 (p 1.5e-05); imminent +0.00048 (p 1.5e-03)
- structural V0-V3 acc: total -0.00941 (p 3.6e-03); far -0.00876 (p 4.1e-03); near -0.00068 (p 4.5e-01); imminent +0.00003 (p 9.7e-01)
- structural V1-V3 brier: total -0.00047 (p 7.4e-01); far +0.01098 (p 2.5e-07); near -0.00303 (p 1.9e-04); imminent -0.00842 (p 5.6e-07)
- structural V1-V3 acc: total +0.00231 (p 5.3e-01); far -0.00506 (p 1.1e-01); near +0.00268 (p 3.6e-02); imminent +0.00468 (p 5.8e-03)

## V1 cap: V1-cap0.98 - V1 (same C_ref; only the deployed weight is capped at 0.98 like V0)
- standard all brier: 0.5283 -> 0.5281 (d -0.0002, t p 7.5e-04)
- standard all ece: 0.0339 -> 0.0336 (d -0.0003, t p 6.0e-01)
- standard all acc: 0.5760 -> 0.5763 (d +0.0003, t p 2.0e-01)
- standard far brier: 0.5365 -> 0.5364 (d -0.0001, t p 2.7e-01)
- standard far ece: 0.0445 -> 0.0445 (d +0.0001, t p 8.4e-01)
- standard far acc: 0.5694 -> 0.5696 (d +0.0003, t p 4.9e-01)
- standard near brier: 0.5369 -> 0.5367 (d -0.0002, t p 4.7e-02)
- standard near ece: 0.0469 -> 0.0469 (d -0.0000, t p 9.7e-01)
- standard near acc: 0.5713 -> 0.5711 (d -0.0002, t p 4.1e-01)
- standard imminent brier: 0.5115 -> 0.5110 (d -0.0005, t p 7.7e-03)
- standard imminent ece: 0.0470 -> 0.0462 (d -0.0008, t p 4.0e-01)
- standard imminent acc: 0.5872 -> 0.5880 (d +0.0008, t p 9.5e-02)
- standard silent-drone brier: 0.5266 -> 0.5265 (d -0.0001, t p 5.3e-01)
- standard silent-drone ece: 0.0663 -> 0.0657 (d -0.0006, t p 7.4e-01)
- standard silent-drone acc: 0.5892 -> 0.5889 (d -0.0003, t p 4.6e-01)
- structural all brier: 0.5333 -> 0.5327 (d -0.0006, t p 2.0e-05)
- structural all ece: 0.0333 -> 0.0325 (d -0.0008, t p 1.7e-01)
- structural all acc: 0.5712 -> 0.5716 (d +0.0004, t p 2.2e-01)
- structural far brier: 0.5443 -> 0.5443 (d -0.0000, t p 3.3e-01)
- structural far ece: 0.0503 -> 0.0495 (d -0.0008, t p 1.4e-01)
- structural far acc: 0.5621 -> 0.5617 (d -0.0004, t p 1.8e-01)
- structural near brier: 0.5394 -> 0.5393 (d -0.0002, t p 7.3e-03)
- structural near ece: 0.0465 -> 0.0470 (d +0.0005, t p 4.2e-01)
- structural near acc: 0.5677 -> 0.5683 (d +0.0006, t p 1.2e-01)
- structural imminent brier: 0.5157 -> 0.5141 (d -0.0016, t p 5.3e-05)
- structural imminent ece: 0.0463 -> 0.0453 (d -0.0010, t p 4.2e-01)
- structural imminent acc: 0.5838 -> 0.5849 (d +0.0010, t p 2.4e-01)
- structural silent-drone brier: 0.5349 -> 0.5349 (d -0.0000, t p 9.9e-01)
- structural silent-drone ece: 0.0724 -> 0.0722 (d -0.0002, t p 9.0e-01)
- structural silent-drone acc: 0.5796 -> 0.5787 (d -0.0009, t p 2.9e-01)

## subset-selection check: exact vs range-marginal sensor tables (positive d = the exact model scores worse)
- standard sensor-oracle-conditional - sensor-oracle-marginal | all brier: 0.5281 -> 0.5269 (d -0.0012, t p 9.5e-07)
- standard sensor-oracle-conditional - sensor-oracle-marginal | far brier: 0.5380 -> 0.5349 (d -0.0031, t p 9.9e-05)
- standard sensor-oracle-conditional - sensor-oracle-marginal | silent-drone brier: 0.5234 -> 0.5222 (d -0.0012, t p 1.3e-01)
- standard sensor-oracle-conditional - sensor-oracle-marginal | far-silent brier: 0.5734 -> 0.5848 (d +0.0114, t p 3.5e-05)
- standard Bayes-oracle - CPT-oracle | all brier: 0.5160 -> 0.5147 (d -0.0013, t p 5.5e-08)
- standard Bayes-oracle - CPT-oracle | far brier: 0.5250 -> 0.5223 (d -0.0028, t p 1.2e-05)
- standard Bayes-oracle - CPT-oracle | silent-drone brier: 0.5042 -> 0.5031 (d -0.0011, t p 1.3e-01)
- standard Bayes-oracle - CPT-oracle | far-silent brier: 0.5513 -> 0.5636 (d +0.0123, t p 1.3e-07)
- structural sensor-oracle-conditional - sensor-oracle-marginal | all brier: 0.5315 -> 0.5286 (d -0.0030, t p 1.8e-08)
- structural sensor-oracle-conditional - sensor-oracle-marginal | far brier: 0.5449 -> 0.5380 (d -0.0069, t p 2.8e-07)
- structural sensor-oracle-conditional - sensor-oracle-marginal | silent-drone brier: 0.5334 -> 0.5228 (d -0.0106, t p 7.9e-09)
- structural sensor-oracle-conditional - sensor-oracle-marginal | far-silent brier: 0.6021 -> 0.5814 (d -0.0207, t p 5.0e-07)
- structural Bayes-oracle - CPT-oracle | all brier: 0.5197 -> 0.5164 (d -0.0033, t p 1.7e-09)
- structural Bayes-oracle - CPT-oracle | far brier: 0.5322 -> 0.5254 (d -0.0067, t p 2.0e-08)
- structural Bayes-oracle - CPT-oracle | silent-drone brier: 0.5143 -> 0.5034 (d -0.0109, t p 7.6e-10)
- structural Bayes-oracle - CPT-oracle | far-silent brier: 0.5807 -> 0.5600 (d -0.0207, t p 1.2e-07)

## diagnostic (added after the main run; not part of the rule): sensor layers on the TRUE internal CPTs
- consistency: 7680 learned-internal values equal the main run exactly
- signed calibration gap = mean top-label confidence - accuracy (positive = over-confident), full test set:
    standard learned-internal: V0 +0.0108+-0.0110 (conf 0.5878); V1 +0.0170+-0.0110 (conf 0.5930); V3 +0.0173+-0.0105 (conf 0.5931); sensor-oracle-conditional +0.0162+-0.0101 (conf 0.5931)
    standard true-internal: V0 -0.0033+-0.0052 (conf 0.5893); V1 +0.0021+-0.0053 (conf 0.5936); V3 +0.0021+-0.0053 (conf 0.5938); sensor-oracle-conditional -0.0003+-0.0048 (conf 0.5936)
    standard learned-internal V1 - V0 conf_gap | all: +0.0108 -> +0.0170 (d +0.0062, t p 8.2e-07)
    standard learned-internal V1 - V0 conf_gap | near: +0.0085 -> +0.0152 (d +0.0067, t p 3.7e-06)
    standard learned-internal V1 - V0 conf_gap | imminent: +0.0098 -> +0.0219 (d +0.0121, t p 3.6e-06)
    standard learned-internal V0 - exact-sensor conf_gap: +0.0162 -> +0.0108 (d -0.0054 [-0.0079,-0.0029], t p 2.5e-04, W p 3.9e-04)
    standard true-internal V1 - V0 conf_gap | all: -0.0033 -> +0.0021 (d +0.0054, t p 3.5e-07)
    standard true-internal V1 - V0 conf_gap | near: -0.0092 -> -0.0005 (d +0.0087, t p 2.8e-07)
    standard true-internal V1 - V0 conf_gap | imminent: -0.0034 -> +0.0057 (d +0.0092, t p 5.1e-05)
    standard true-internal V0 - exact-sensor conf_gap: -0.0003 -> -0.0033 (d -0.0030 [-0.0052,-0.0008], t p 1.1e-02, W p 1.5e-02)
    structural learned-internal: V0 +0.0110+-0.0113 (conf 0.5838); V1 +0.0175+-0.0114 (conf 0.5887); V3 +0.0187+-0.0108 (conf 0.5904); sensor-oracle-conditional +0.0163+-0.0102 (conf 0.5930)
    structural true-internal: V0 -0.0027+-0.0053 (conf 0.5858); V1 +0.0021+-0.0050 (conf 0.5900); V3 +0.0034+-0.0055 (conf 0.5913); sensor-oracle-conditional +0.0003+-0.0050 (conf 0.5938)
    structural learned-internal V1 - V0 conf_gap | all: +0.0110 -> +0.0175 (d +0.0065, t p 5.3e-07)
    structural learned-internal V1 - V0 conf_gap | near: +0.0058 -> +0.0144 (d +0.0086, t p 4.0e-06)
    structural learned-internal V1 - V0 conf_gap | imminent: +0.0082 -> +0.0232 (d +0.0150, t p 1.9e-06)
    structural learned-internal V0 - exact-sensor conf_gap: +0.0163 -> +0.0110 (d -0.0052 [-0.0077,-0.0028], t p 2.4e-04, W p 8.5e-04)
    structural true-internal V1 - V0 conf_gap | all: -0.0027 -> +0.0021 (d +0.0048, t p 6.8e-04)
    structural true-internal V1 - V0 conf_gap | near: -0.0088 -> -0.0015 (d +0.0073, t p 1.4e-04)
    structural true-internal V1 - V0 conf_gap | imminent: -0.0060 -> +0.0061 (d +0.0121, t p 3.0e-05)
    structural true-internal V0 - exact-sensor conf_gap: +0.0003 -> -0.0027 (d -0.0030 [-0.0059,-0.0001], t p 4.0e-02, W p 4.4e-02)
- standard V0|true-internal: Brier 0.5164, ECE 0.0192, cwECE 0.0185, acc 0.5926; ECE near 0.0316, imminent 0.0313
- standard V1|true-internal: Brier 0.5166, ECE 0.0186, cwECE 0.0190, acc 0.5915; ECE near 0.0309, imminent 0.0305
- standard V3|true-internal: Brier 0.5172, ECE 0.0197, cwECE 0.0186, acc 0.5917; ECE near 0.0347, imminent 0.0305
- standard CPT-oracle: Brier 0.5160, ECE 0.0170, cwECE 0.0186, acc 0.5931; ECE near 0.0309, imminent 0.0308
- standard Bayes-oracle: Brier 0.5147, ECE 0.0177, cwECE 0.0185, acc 0.5939; ECE near 0.0325, imminent 0.0300
    V1|true-internal - V0|true-internal | all brier: d +0.0001 (t p 6.4e-01, W p 7.8e-01)
    V1|true-internal - V0|true-internal | all ece: d -0.0006 (t p 6.8e-01, W p 7.8e-01)
    V1|true-internal - V0|true-internal | near brier: d -0.0002 (t p 5.7e-01, W p 3.9e-01)
    V1|true-internal - V0|true-internal | near ece: d -0.0007 (t p 7.2e-01, W p 6.5e-01)
    V1|true-internal - V0|true-internal | imminent brier: d +0.0003 (t p 6.2e-01, W p 9.9e-01)
    V1|true-internal - V0|true-internal | imminent ece: d -0.0008 (t p 7.3e-01, W p 7.0e-01)
    V3|true-internal - V0|true-internal | all brier: d +0.0008 (t p 8.4e-04, W p 8.5e-04)
    V3|true-internal - V0|true-internal | all ece: d +0.0005 (t p 7.0e-01, W p 6.0e-01)
    V3|true-internal - V0|true-internal | near brier: d -0.0003 (t p 9.4e-02, W p 5.3e-02)
    V3|true-internal - V0|true-internal | near ece: d +0.0031 (t p 4.0e-02, W p 5.3e-02)
    V3|true-internal - V0|true-internal | imminent brier: d -0.0001 (t p 4.9e-01, W p 3.7e-01)
    V3|true-internal - V0|true-internal | imminent ece: d -0.0008 (t p 3.9e-01, W p 5.7e-01)
    V0|true-internal - Bayes-oracle | all brier: d +0.0017 (t p 1.7e-07, W p 5.7e-06)
    V0|true-internal - Bayes-oracle | all ece: d +0.0016 (t p 2.1e-01, W p 2.0e-01)
    V0|true-internal - Bayes-oracle | near brier: d +0.0018 (t p 1.2e-05, W p 4.8e-05)
    V0|true-internal - Bayes-oracle | near ece: d -0.0009 (t p 6.3e-01, W p 6.0e-01)
    V0|true-internal - Bayes-oracle | imminent brier: d +0.0017 (t p 5.1e-05, W p 6.3e-05)
    V0|true-internal - Bayes-oracle | imminent ece: d +0.0013 (t p 3.9e-01, W p 4.3e-01)
    V1|true-internal - Bayes-oracle | all brier: d +0.0019 (t p 6.0e-06, W p 1.3e-05)
    V1|true-internal - Bayes-oracle | all ece: d +0.0010 (t p 2.7e-01, W p 2.6e-01)
    V1|true-internal - Bayes-oracle | near brier: d +0.0016 (t p 1.1e-03, W p 8.5e-04)
    V1|true-internal - Bayes-oracle | near ece: d -0.0016 (t p 4.1e-01, W p 3.9e-01)
    V1|true-internal - Bayes-oracle | imminent brier: d +0.0020 (t p 3.6e-03, W p 1.0e-03)
    V1|true-internal - Bayes-oracle | imminent ece: d +0.0005 (t p 8.1e-01, W p 9.9e-01)
- structural V0|true-internal: Brier 0.5204, ECE 0.0170, cwECE 0.0193, acc 0.5885; ECE near 0.0328, imminent 0.0312
- structural V1|true-internal: Brier 0.5218, ECE 0.0187, cwECE 0.0200, acc 0.5879; ECE near 0.0302, imminent 0.0318
- structural V3|true-internal: Brier 0.5208, ECE 0.0180, cwECE 0.0190, acc 0.5878; ECE near 0.0300, imminent 0.0298
- structural CPT-oracle: Brier 0.5197, ECE 0.0174, cwECE 0.0188, acc 0.5895; ECE near 0.0326, imminent 0.0303
- structural Bayes-oracle: Brier 0.5164, ECE 0.0179, cwECE 0.0186, acc 0.5935; ECE near 0.0308, imminent 0.0300
    V1|true-internal - V0|true-internal | all brier: d +0.0015 (t p 1.4e-02, W p 2.7e-02)
    V1|true-internal - V0|true-internal | all ece: d +0.0017 (t p 1.6e-01, W p 2.2e-01)
    V1|true-internal - V0|true-internal | near brier: d +0.0002 (t p 8.0e-01, W p 4.5e-01)
    V1|true-internal - V0|true-internal | near ece: d -0.0026 (t p 2.5e-01, W p 9.0e-02)
    V1|true-internal - V0|true-internal | imminent brier: d +0.0034 (t p 1.1e-02, W p 8.3e-03)
    V1|true-internal - V0|true-internal | imminent ece: d +0.0006 (t p 7.5e-01, W p 7.0e-01)
    V3|true-internal - V0|true-internal | all brier: d +0.0004 (t p 1.4e-01, W p 1.5e-01)
    V3|true-internal - V0|true-internal | all ece: d +0.0010 (t p 3.5e-01, W p 2.8e-01)
    V3|true-internal - V0|true-internal | near brier: d -0.0005 (t p 5.7e-02, W p 2.7e-02)
    V3|true-internal - V0|true-internal | near ece: d -0.0029 (t p 1.3e-01, W p 8.3e-02)
    V3|true-internal - V0|true-internal | imminent brier: d +0.0000 (t p 9.1e-01, W p 1.0e+00)
    V3|true-internal - V0|true-internal | imminent ece: d -0.0014 (t p 2.2e-01, W p 1.4e-01)
    V0|true-internal - Bayes-oracle | all brier: d +0.0040 (t p 3.4e-12, W p 1.9e-06)
    V0|true-internal - Bayes-oracle | all ece: d -0.0009 (t p 5.3e-01, W p 5.5e-01)
    V0|true-internal - Bayes-oracle | near brier: d +0.0023 (t p 3.2e-05, W p 1.0e-04)
    V0|true-internal - Bayes-oracle | near ece: d +0.0020 (t p 3.4e-01, W p 1.5e-01)
    V0|true-internal - Bayes-oracle | imminent brier: d +0.0031 (t p 9.2e-08, W p 1.9e-06)
    V0|true-internal - Bayes-oracle | imminent ece: d +0.0012 (t p 5.1e-01, W p 6.0e-01)
    V1|true-internal - Bayes-oracle | all brier: d +0.0054 (t p 3.6e-09, W p 1.9e-06)
    V1|true-internal - Bayes-oracle | all ece: d +0.0008 (t p 4.8e-01, W p 5.7e-01)
    V1|true-internal - Bayes-oracle | near brier: d +0.0024 (t p 2.2e-03, W p 5.9e-04)
    V1|true-internal - Bayes-oracle | near ece: d -0.0006 (t p 7.4e-01, W p 9.9e-01)
    V1|true-internal - Bayes-oracle | imminent brier: d +0.0065 (t p 3.2e-06, W p 1.9e-06)
    V1|true-internal - Bayes-oracle | imminent ece: d +0.0018 (t p 3.7e-01, W p 3.3e-01)

## A7 decision (rule fixed before the run, standard world, n=300)
- Brier (all): dV1-V0 = +0.0004 [-0.0002,+0.0010], t p = 2.21e-01, W p = 3.12e-01 -> not significantly worse
- ECE (all): dV1-V0 = +0.0038 [+0.0009,+0.0066], t p = 1.18e-02, W p = 1.72e-02 -> significantly WORSE
- ECE near: 0.0449 -> 0.0469 (d +0.0020, t p = 2.82e-01); ECE imminent: 0.0429 -> 0.0470 (d +0.0042, t p = 1.09e-01) -> NOT reduced in both
- VERDICT: KEEP V0 (adoption conditions not all met); analysis only
