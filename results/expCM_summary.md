# Exp-CM: the comparators' free values and the attribution of the gains

Added after the main results. Values chosen for a comparator were chosen after the fact. p: paired t over 20 replications, unadjusted.

## Reproduction of the main study's rows
- largest |Brier difference| against the main study's raw files: {'Proposed scarcity': 0.0, 'B1-MLE scarcity': 0.0, 'uninformative MAP': 0.0, 'expert MAP (full table)': 0.0, 'expert cMAP (full table)': 0.0}
- Proposed through the pipeline against elicited, constrained+ICI, ESS 2 in the grid: 0.00e+00

## Means by arm (macro-AUC / Brier / ECE)
- MLE: N=10 0.661/0.696/0.111; N=20 0.697/0.673/0.139; N=50 0.761/0.602/0.107; N=80 0.783/0.577/0.086; N=100 0.792/0.562/0.074; N=200 0.810/0.536/0.044; N=400 0.816/0.526/0.032
- uniform full, ESS 2 (B3): N=10 0.671/0.712/0.143; N=20 0.715/0.677/0.152; N=50 0.769/0.613/0.125; N=80 0.785/0.582/0.102; N=100 0.793/0.569/0.088; N=200 0.808/0.541/0.056; N=400 0.814/0.528/0.038
- uniform full, ESS 0.5: N=10 0.669/0.680/0.098; N=20 0.707/0.640/0.079; N=50 0.768/0.581/0.044; N=80 0.787/0.561/0.040; N=100 0.795/0.552/0.037; N=200 0.810/0.534/0.032; N=400 0.816/0.526/0.027
- uniform constrained, ESS 0.2: N=10 0.761/0.632/0.060; N=20 0.773/0.609/0.077; N=50 0.790/0.575/0.066; N=80 0.797/0.561/0.060; N=100 0.802/0.552/0.051; N=200 0.812/0.534/0.034; N=400 0.817/0.526/0.029
- uniform constrained+ICI, ESS 0.2: N=10 0.763/0.632/0.078; N=20 0.781/0.590/0.059; N=50 0.803/0.554/0.041; N=80 0.808/0.541/0.035; N=100 0.811/0.537/0.037; N=200 0.817/0.528/0.030; N=400 0.820/0.522/0.024
- uniform constrained+ICI, ESS 2: N=10 0.763/0.710/0.124; N=20 0.771/0.678/0.142; N=50 0.793/0.621/0.150; N=80 0.802/0.591/0.130; N=100 0.806/0.578/0.123; N=200 0.815/0.548/0.086; N=400 0.819/0.530/0.055
- elicited full, ESS 2: N=10 0.788/0.593/0.088; N=20 0.793/0.580/0.079; N=50 0.801/0.557/0.059; N=80 0.805/0.547/0.051; N=100 0.807/0.543/0.047; N=200 0.815/0.531/0.036; N=400 0.818/0.525/0.030
- elicited constrained, ESS 2: N=10 0.790/0.592/0.087; N=20 0.794/0.579/0.079; N=50 0.802/0.557/0.059; N=80 0.806/0.547/0.050; N=100 0.808/0.542/0.047; N=200 0.815/0.531/0.036; N=400 0.818/0.524/0.029
- elicited constrained+ICI, ESS 2 (Proposed): N=10 0.793/0.586/0.077; N=20 0.797/0.572/0.066; N=50 0.806/0.551/0.051; N=80 0.810/0.541/0.042; N=100 0.811/0.538/0.040; N=200 0.817/0.528/0.030; N=400 0.820/0.522/0.024

## Proposed - the same structure with a weak uniform prior (ESS 0.2)
- N=10: acc +0.0643 (p=0.0022); macroAUC +0.0298 (p=0.00033); brier -0.0458 (p=0.016); ece -0.0009 (p=0.96); far@pd0.9 -0.0519 (p=0.0022)
- N=20: acc +0.0248 (p=0.054); macroAUC +0.0162 (p=0.0049); brier -0.0179 (p=0.14); ece +0.0071 (p=0.58); far@pd0.9 -0.0362 (p=0.0089)
- N=50: acc +0.0051 (p=0.25); macroAUC +0.0035 (p=0.041); brier -0.0030 (p=0.51); ece +0.0098 (p=0.083); far@pd0.9 +0.0001 (p=0.99)
- N=80: acc +0.0015 (p=0.65); macroAUC +0.0018 (p=0.075); brier -0.0001 (p=0.97); ece +0.0067 (p=0.078); far@pd0.9 +0.0020 (p=0.67)
- N=100: acc -0.0002 (p=0.95); macroAUC +0.0003 (p=0.72); brier +0.0016 (p=0.47); ece +0.0025 (p=0.43); far@pd0.9 +0.0036 (p=0.34)
- N=200: acc -0.0001 (p=0.96); macroAUC +0.0001 (p=0.87); brier +0.0002 (p=0.85); ece +0.0005 (p=0.85); far@pd0.9 +0.0006 (p=0.8)
- N=400: acc -0.0006 (p=0.52); macroAUC +0.0002 (p=0.3); brier -0.0002 (p=0.66); ece -0.0001 (p=0.95); far@pd0.9 +0.0000 (p=0.99)

## That structure with ESS 0.2 - MLE at a larger N
- N=10 against N=50: acc -0.0633 (p=0.00045); macroAUC +0.0019 (p=0.71); brier +0.0300 (p=0.022); ece -0.0291 (p=0.047); far@pd0.9 +0.0267 (p=0.1)
- N=20 against N=50: acc -0.0108 (p=0.26); macroAUC +0.0198 (p=0.0016); brier -0.0122 (p=0.093); ece -0.0481 (p=0.00014); far@pd0.9 +0.0073 (p=0.66)
- N=50 against N=100: acc +0.0004 (p=0.92); macroAUC +0.0104 (p=0.00072); brier -0.0084 (p=0.016); ece -0.0328 (p=4.1e-06); far@pd0.9 -0.0124 (p=0.12)
- N=50 against N=200: acc -0.0164 (p=8.2e-05); macroAUC -0.0076 (p=0.00056); brier +0.0174 (p=2.4e-05); ece -0.0025 (p=0.51); far@pd0.9 +0.0098 (p=0.16)
- N=80 against N=200: acc -0.0074 (p=0.015); macroAUC -0.0020 (p=0.18); brier +0.0050 (p=0.0057); ece -0.0086 (p=0.0098); far@pd0.9 +0.0042 (p=0.44)

## Walks, each step a paired difference
- N=10 the order of VII-B: uniform - MLE: macroAUC +0.0095 (p=0.028); brier +0.0154 (p=0.025); ece +0.0321 (p=0.017) | elicited - uniform: macroAUC +0.1173 (p=8.2e-09); brier -0.1187 (p=3.6e-08); ece -0.0545 (p=0.00065) | constraints: macroAUC +0.0014 (p=0.087); brier -0.0009 (p=0.074); ece -0.0010 (p=0.2) | ICI: macroAUC +0.0028 (p=0.026); brier -0.0058 (p=0.0012); ece -0.0107 (p=0.019)
- N=10 structure first, ESS 2: constraints - uniform full: macroAUC +0.0958 (p=8.8e-08); brier -0.0071 (p=1.4e-07); ece +0.0148 (p=0.028) | ICI: macroAUC -0.0043 (p=0.35); brier +0.0051 (p=1.2e-05); ece -0.0337 (p=0.05) | elicited: macroAUC +0.0301 (p=0.00042); brier -0.1234 (p=1.9e-08); ece -0.0473 (p=0.012)
- N=10 structure first, ESS 0.2: structure - MLE: macroAUC +0.1013 (p=4.2e-10); brier -0.0643 (p=1.3e-08); ece -0.0331 (p=0.012) | elicited at ESS 2: macroAUC +0.0298 (p=0.00033); brier -0.0458 (p=0.016); ece -0.0009 (p=0.96)
- N=50 the order of VII-B: uniform - MLE: macroAUC +0.0081 (p=0.0036); brier +0.0114 (p=0.036); ece +0.0187 (p=0.19) | elicited - uniform: macroAUC +0.0325 (p=1.3e-08); brier -0.0563 (p=3.1e-10); ece -0.0664 (p=9.6e-06) | constraints: macroAUC +0.0008 (p=0.1); brier -0.0005 (p=0.036); ece -0.0004 (p=0.14) | ICI: macroAUC +0.0039 (p=0.0028); brier -0.0061 (p=0.0015); ece -0.0074 (p=0.054)
- N=50 structure first, ESS 2: constraints - uniform full: macroAUC +0.0250 (p=6.2e-11); brier -0.0083 (p=3.2e-11); ece +0.0023 (p=0.25) | ICI: macroAUC -0.0010 (p=0.77); brier +0.0158 (p=4.4e-10); ece +0.0226 (p=0.0067) | elicited: macroAUC +0.0132 (p=8.2e-07); brier -0.0704 (p=1.3e-11); ece -0.0991 (p=2.3e-07)
- N=50 structure first, ESS 0.2: structure - MLE: macroAUC +0.0419 (p=1.7e-09); brier -0.0485 (p=1.2e-09); ece -0.0652 (p=2e-07) | elicited at ESS 2: macroAUC +0.0035 (p=0.041); brier -0.0030 (p=0.51); ece +0.0098 (p=0.083)
- N=80 the order of VII-B: uniform - MLE: macroAUC +0.0027 (p=0.13); brier +0.0049 (p=0.074); ece +0.0155 (p=0.15) | elicited - uniform: macroAUC +0.0199 (p=1.1e-08); brier -0.0348 (p=4.4e-10); ece -0.0511 (p=6.3e-06) | constraints: macroAUC +0.0009 (p=0.18); brier -0.0005 (p=0.074); ece -0.0009 (p=0.013) | ICI: macroAUC +0.0040 (p=0.00069); brier -0.0055 (p=0.00094); ece -0.0076 (p=0.0047)
- N=80 structure first, ESS 2: constraints - uniform full: macroAUC +0.0154 (p=2.4e-11); brier -0.0060 (p=1.2e-08); ece +0.0003 (p=0.85) | ICI: macroAUC +0.0011 (p=0.57); brier +0.0148 (p=2.9e-10); ece +0.0278 (p=0.0039) | elicited: macroAUC +0.0083 (p=4.6e-07); brier -0.0495 (p=1e-12); ece -0.0877 (p=9.3e-08)
- N=80 structure first, ESS 0.2: structure - MLE: macroAUC +0.0256 (p=1.7e-10); brier -0.0357 (p=2.7e-10); ece -0.0509 (p=1.9e-08) | elicited at ESS 2: macroAUC +0.0018 (p=0.075); brier -0.0001 (p=0.97); ece +0.0067 (p=0.078)

## Proposed - full tables with a uniform mean, by ESS, and BDeu (ECE, Brier, macro-AUC)
- uniform N=10: 0.01: -0.0319 (p=0.031), -0.1083 (p=3.5e-06), +0.1308 (p=7.9e-10) | 0.05: -0.0254 (p=0.064), -0.1024 (p=5.5e-06), +0.1300 (p=9.6e-10) | 0.1: -0.0204 (p=0.11), -0.0973 (p=8.3e-06), +0.1290 (p=1.1e-09) | 0.2: -0.0142 (p=0.24), -0.0921 (p=1.1e-05), +0.1275 (p=1.6e-09) | 0.5: -0.0218 (p=0.07), -0.0936 (p=4.2e-06), +0.1240 (p=2.7e-09) | 1: -0.0469 (p=0.00099), -0.1066 (p=3.2e-07), +0.1219 (p=4.6e-09) | 2: -0.0662 (p=1.8e-05), -0.1254 (p=1.4e-08), +0.1216 (p=7.3e-09)
- uniform N=20: 0.01: -0.0695 (p=2.3e-05), -0.0989 (p=2.9e-08), +0.0992 (p=3.1e-12) | 0.05: -0.0583 (p=0.00011), -0.0909 (p=1.1e-07), +0.0978 (p=3.1e-12) | 0.1: -0.0455 (p=0.0012), -0.0833 (p=4.2e-07), +0.0965 (p=3.1e-12) | 0.2: -0.0261 (p=0.037), -0.0738 (p=2.8e-06), +0.0944 (p=4e-12) | 0.5: -0.0133 (p=0.28), -0.0680 (p=9.8e-06), +0.0899 (p=9.7e-12) | 1: -0.0466 (p=0.0014), -0.0794 (p=9e-07), +0.0851 (p=6e-11) | 2: -0.0861 (p=1.1e-06), -0.1048 (p=5.4e-09), +0.0821 (p=3.2e-10)
- uniform N=50: 0.01: -0.0530 (p=3.4e-06), -0.0502 (p=2e-06), +0.0449 (p=5.4e-10) | 0.05: -0.0431 (p=3.6e-05), -0.0456 (p=4.6e-06), +0.0438 (p=5.6e-10) | 0.1: -0.0321 (p=0.00054), -0.0411 (p=1.1e-05), +0.0425 (p=7.1e-10) | 0.2: -0.0143 (p=0.061), -0.0350 (p=3.9e-05), +0.0405 (p=8.7e-10) | 0.5: +0.0072 (p=0.29), -0.0302 (p=7.1e-05), +0.0378 (p=1.7e-09) | 1: -0.0179 (p=0.051), -0.0381 (p=1.7e-06), +0.0365 (p=5.1e-09) | 2: -0.0742 (p=1.8e-07), -0.0629 (p=4.4e-10), +0.0372 (p=9.4e-09)
- uniform N=80: 0.01: -0.0431 (p=3.8e-07), -0.0350 (p=3.1e-07), +0.0272 (p=4.3e-12) | 0.05: -0.0361 (p=3.8e-06), -0.0321 (p=7.4e-07), +0.0266 (p=3.7e-12) | 0.1: -0.0287 (p=5.2e-05), -0.0292 (p=2e-06), +0.0258 (p=3.5e-12) | 0.2: -0.0167 (p=0.0024), -0.0248 (p=9.3e-06), +0.0249 (p=5.3e-12) | 0.5: +0.0017 (p=0.71), -0.0199 (p=5.6e-05), +0.0233 (p=2e-11) | 1: -0.0098 (p=0.14), -0.0233 (p=4e-06), +0.0233 (p=3e-11) | 2: -0.0597 (p=1.7e-07), -0.0407 (p=4.1e-10), +0.0248 (p=6.4e-11)
- uniform N=100: 0.01: -0.0335 (p=7.1e-09), -0.0231 (p=1.5e-05), +0.0187 (p=2e-08) | 0.05: -0.0285 (p=5.5e-08), -0.0213 (p=3.2e-05), +0.0184 (p=2.3e-08) | 0.1: -0.0224 (p=1.1e-06), -0.0194 (p=7.3e-05), +0.0179 (p=2.4e-08) | 0.2: -0.0128 (p=0.00054), -0.0166 (p=0.00025), +0.0174 (p=2.1e-08) | 0.5: +0.0030 (p=0.4), -0.0134 (p=0.0009), +0.0167 (p=1.2e-08) | 1: -0.0053 (p=0.38), -0.0163 (p=5.3e-05), +0.0171 (p=5.5e-09) | 2: -0.0483 (p=2.5e-06), -0.0309 (p=1.6e-09), +0.0187 (p=2.1e-09)
- uniform N=200: 0.01: -0.0128 (p=8.7e-07), -0.0081 (p=1.5e-05), +0.0064 (p=3.2e-06) | 0.05: -0.0114 (p=1.9e-05), -0.0077 (p=2.6e-05), +0.0064 (p=3.3e-06) | 0.1: -0.0086 (p=0.0002), -0.0073 (p=4.5e-05), +0.0063 (p=3.4e-06) | 0.2: -0.0046 (p=0.026), -0.0067 (p=0.00011), +0.0064 (p=3.2e-06) | 0.5: -0.0017 (p=0.43), -0.0059 (p=0.00036), +0.0067 (p=1.6e-06) | 1: -0.0022 (p=0.51), -0.0070 (p=6.7e-05), +0.0076 (p=3.6e-07) | 2: -0.0261 (p=3e-05), -0.0135 (p=1.4e-08), +0.0092 (p=4.8e-08)
- uniform N=400: 0.01: -0.0074 (p=0.00065), -0.0042 (p=1.4e-05), +0.0036 (p=2.6e-06) | 0.05: -0.0068 (p=0.0021), -0.0041 (p=1.8e-05), +0.0035 (p=3.3e-06) | 0.1: -0.0062 (p=0.0056), -0.0040 (p=2.5e-05), +0.0035 (p=3.6e-06) | 0.2: -0.0048 (p=0.017), -0.0038 (p=4e-05), +0.0035 (p=3.8e-06) | 0.5: -0.0026 (p=0.13), -0.0036 (p=9.1e-05), +0.0038 (p=2.4e-06) | 1: -0.0023 (p=0.29), -0.0039 (p=5.9e-05), +0.0043 (p=9.1e-07) | 2: -0.0134 (p=0.00016), -0.0060 (p=5.9e-07), +0.0056 (p=2.2e-07)
- BDeu N=10: 1: -0.0223 (p=0.1), -0.0984 (p=7.5e-06), +0.1290 (p=7.8e-10) | 2: -0.0183 (p=0.15), -0.0934 (p=1e-05), +0.1279 (p=8.2e-10) | 5: -0.0242 (p=0.059), -0.0927 (p=5.4e-06), +0.1266 (p=5.7e-10) | 10: -0.0366 (p=0.006), -0.1018 (p=7.4e-07), +0.1267 (p=3.9e-10) | 20: -0.0462 (p=0.00042), -0.1177 (p=4.6e-08), +0.1281 (p=4.9e-10)
- BDeu N=20: 1: -0.0487 (p=0.00062), -0.0873 (p=2.2e-07), +0.0975 (p=4.7e-12) | 2: -0.0326 (p=0.014), -0.0788 (p=1.1e-06), +0.0965 (p=4.9e-12) | 5: -0.0122 (p=0.29), -0.0698 (p=7.3e-06), +0.0943 (p=7.2e-12) | 10: -0.0225 (p=0.083), -0.0742 (p=2.7e-06), +0.0932 (p=8.7e-12) | 20: -0.0528 (p=0.00031), -0.0931 (p=4.8e-08), +0.0935 (p=2e-11)
- BDeu N=50: 1: -0.0401 (p=7.8e-05), -0.0447 (p=6e-06), +0.0436 (p=3.7e-10) | 2: -0.0272 (p=0.0021), -0.0397 (p=1.7e-05), +0.0424 (p=4.6e-10) | 5: +0.0001 (p=0.99), -0.0323 (p=7.4e-05), +0.0405 (p=4.3e-10) | 10: -0.0031 (p=0.66), -0.0328 (p=2.6e-05), +0.0395 (p=8.2e-10) | 20: -0.0465 (p=5e-05), -0.0482 (p=5e-08), +0.0404 (p=8.2e-10)
- BDeu N=80: 1: -0.0338 (p=9.7e-06), -0.0317 (p=8.9e-07), +0.0266 (p=2.6e-12) | 2: -0.0256 (p=0.00016), -0.0284 (p=2.7e-06), +0.0259 (p=2.6e-12) | 5: -0.0075 (p=0.095), -0.0226 (p=2.6e-05), +0.0248 (p=4e-12) | 10: +0.0046 (p=0.34), -0.0206 (p=4e-05), +0.0242 (p=9.9e-12) | 20: -0.0269 (p=0.00099), -0.0288 (p=1.9e-07), +0.0251 (p=1.3e-11)
- BDeu N=100: 1: -0.0271 (p=1.4e-07), -0.0210 (p=3.7e-05), +0.0183 (p=2.2e-08) | 2: -0.0201 (p=8.2e-06), -0.0189 (p=9.4e-05), +0.0179 (p=2.3e-08) | 5: -0.0044 (p=0.16), -0.0151 (p=0.00055), +0.0172 (p=1.6e-08) | 10: +0.0021 (p=0.63), -0.0139 (p=0.00066), +0.0169 (p=5.5e-09) | 20: -0.0202 (p=0.007), -0.0204 (p=3.1e-06), +0.0178 (p=9.9e-10)
- BDeu N=200: 1: -0.0108 (p=3.2e-05), -0.0077 (p=3e-05), +0.0065 (p=3e-06) | 2: -0.0081 (p=0.00023), -0.0072 (p=5.8e-05), +0.0065 (p=3.1e-06) | 5: -0.0025 (p=0.25), -0.0064 (p=0.00023), +0.0065 (p=2.8e-06) | 10: +0.0001 (p=0.97), -0.0061 (p=0.00039), +0.0070 (p=1.5e-06) | 20: -0.0071 (p=0.11), -0.0085 (p=1.6e-05), +0.0080 (p=3.8e-07)
- BDeu N=400: 1: -0.0068 (p=0.0015), -0.0041 (p=1.9e-05), +0.0035 (p=3.1e-06) | 2: -0.0058 (p=0.0088), -0.0040 (p=2.6e-05), +0.0035 (p=3.3e-06) | 5: -0.0033 (p=0.058), -0.0037 (p=5.2e-05), +0.0036 (p=3e-06) | 10: -0.0005 (p=0.75), -0.0037 (p=7.7e-05), +0.0038 (p=1.4e-06) | 20: -0.0051 (p=0.054), -0.0043 (p=1.8e-05), +0.0045 (p=3.4e-07)

## B0 with its taus multiplied by f
- f=1.0: alone acc 0.450, macroAUC 0.784, brier 0.658, ece 0.098, far@pd0.9 0.523 | as prior N=10 acc 0.515, macroAUC 0.787, brier 0.621, ece 0.123, far@pd0.9 0.510 | N=50 acc 0.555, macroAUC 0.799, brier 0.566, ece 0.079, far@pd0.9 0.485
- f=0.8: alone acc 0.477, macroAUC 0.790, brier 0.635, ece 0.088, far@pd0.9 0.516 | as prior N=10 acc 0.530, macroAUC 0.794, brier 0.601, ece 0.105, far@pd0.9 0.505 | N=50 acc 0.559, macroAUC 0.803, brier 0.556, ece 0.060, far@pd0.9 0.479
- f=0.6: alone acc 0.503, macroAUC 0.798, brier 0.606, ece 0.056, far@pd0.9 0.500 | as prior N=10 acc 0.544, macroAUC 0.800, brier 0.578, ece 0.070, far@pd0.9 0.493 | N=50 acc 0.563, macroAUC 0.807, brier 0.547, ece 0.035, far@pd0.9 0.477
- f=0.5: alone acc 0.524, macroAUC 0.802, brier 0.590, ece 0.032, far@pd0.9 0.477 | as prior N=10 acc 0.547, macroAUC 0.803, brier 0.569, ece 0.046, far@pd0.9 0.487 | N=50 acc 0.564, macroAUC 0.808, brier 0.544, ece 0.031, far@pd0.9 0.474
- f=0.4: alone acc 0.525, macroAUC 0.803, brier 0.585, ece 0.045, far@pd0.9 0.473 | as prior N=10 acc 0.546, macroAUC 0.804, brier 0.569, ece 0.054, far@pd0.9 0.483 | N=50 acc 0.563, macroAUC 0.808, brier 0.545, ece 0.038, far@pd0.9 0.474
- f=0.3: alone acc 0.512, macroAUC 0.795, brier 0.632, ece 0.147, far@pd0.9 0.487 | as prior N=10 acc 0.534, macroAUC 0.799, brier 0.599, ece 0.112, far@pd0.9 0.489 | N=50 acc 0.559, macroAUC 0.805, brier 0.553, ece 0.060, far@pd0.9 0.480
- Proposed N=10 - B0 f=1.0 alone: acc +0.0839 (p=2.1e-07); macroAUC +0.0085 (p=0.068); brier -0.0722 (p=1.7e-05); ece -0.0213 (p=0.1); far@pd0.9 -0.0170 (p=0.11)
- Proposed N=20 - B0 f=1.0 alone: acc +0.0969 (p=1.3e-10); macroAUC +0.0127 (p=0.0011); brier -0.0865 (p=1.5e-08); ece -0.0324 (p=0.0037); far@pd0.9 -0.0207 (p=0.02)
- Proposed N=50 - B0 f=1.0 alone: acc +0.1097 (p=3.1e-15); macroAUC +0.0221 (p=1.1e-09); brier -0.1079 (p=4.3e-15); ece -0.0467 (p=1.2e-06); far@pd0.9 -0.0411 (p=6.3e-07)
- Proposed N=80 - B0 f=1.0 alone: acc +0.1150 (p=4.1e-15); macroAUC +0.0261 (p=2.1e-13); brier -0.1174 (p=1.1e-19); ece -0.0560 (p=3.7e-11); far@pd0.9 -0.0447 (p=4.9e-08)
- Proposed N=100 - B0 f=1.0 alone: acc +0.1175 (p=4.2e-16); macroAUC +0.0272 (p=1.2e-14); brier -0.1201 (p=1e-21); ece -0.0585 (p=4.2e-12); far@pd0.9 -0.0456 (p=3.2e-09)
- Proposed N=200 - B0 f=1.0 alone: acc +0.1272 (p=1.6e-16); macroAUC +0.0328 (p=1e-17); brier -0.1305 (p=1.9e-23); ece -0.0679 (p=6.7e-15); far@pd0.9 -0.0562 (p=3.3e-11)
- Proposed N=400 - B0 f=1.0 alone: acc +0.1326 (p=6.3e-18); macroAUC +0.0359 (p=2.7e-18); brier -0.1363 (p=1.5e-24); ece -0.0737 (p=2.7e-17); far@pd0.9 -0.0591 (p=1.2e-11)
- Proposed N=10 - B0 f=0.5 alone: acc +0.0097 (p=0.37); macroAUC -0.0090 (p=0.052); brier -0.0041 (p=0.75); ece +0.0451 (p=0.0017); far@pd0.9 +0.0290 (p=0.0092)
- Proposed N=20 - B0 f=0.5 alone: acc +0.0226 (p=0.0064); macroAUC -0.0048 (p=0.16); brier -0.0184 (p=0.063); ece +0.0340 (p=0.0023); far@pd0.9 +0.0253 (p=0.0052)
- Proposed N=50 - B0 f=0.5 alone: acc +0.0354 (p=3.8e-07); macroAUC +0.0046 (p=0.041); brier -0.0398 (p=1.3e-07); ece +0.0196 (p=0.0076); far@pd0.9 +0.0049 (p=0.37)
- Proposed N=80 - B0 f=0.5 alone: acc +0.0407 (p=5.9e-08); macroAUC +0.0085 (p=4.4e-05); brier -0.0493 (p=2.3e-12); ece +0.0104 (p=0.023); far@pd0.9 +0.0013 (p=0.78)
- Proposed N=100 - B0 f=0.5 alone: acc +0.0432 (p=4.4e-09); macroAUC +0.0097 (p=1.5e-06); brier -0.0520 (p=1.7e-14); ece +0.0079 (p=0.049); far@pd0.9 +0.0005 (p=0.9)
- Proposed N=200 - B0 f=0.5 alone: acc +0.0530 (p=8.3e-11); macroAUC +0.0153 (p=5e-12); brier -0.0624 (p=8.8e-18); ece -0.0015 (p=0.66); far@pd0.9 -0.0102 (p=0.0024)
- Proposed N=400 - B0 f=0.5 alone: acc +0.0583 (p=7.7e-13); macroAUC +0.0184 (p=9.5e-15); brier -0.0682 (p=5.6e-20); ece -0.0073 (p=0.022); far@pd0.9 -0.0130 (p=0.00019)
- Proposed N=10 - B0 f=0.5 as prior at N=10: acc -0.0130 (p=0.35); macroAUC -0.0101 (p=0.058); brier +0.0178 (p=0.26); ece +0.0307 (p=0.034); far@pd0.9 +0.0191 (p=0.044)
- Proposed N=50 - B0 f=0.5 as prior at N=50: acc -0.0044 (p=0.36); macroAUC -0.0016 (p=0.44); brier +0.0068 (p=0.19); ece +0.0207 (p=0.0074); far@pd0.9 +0.0084 (p=0.082)
- B0 f=0.5 as prior N=10 - MLE N=50: acc +0.0141 (p=0.097); macroAUC +0.0419 (p=2.5e-09); brier -0.0335 (p=0.00051); ece -0.0607 (p=8.9e-06); far@pd0.9 -0.0443 (p=0.00049)
- f=1.0 as prior against expB: largest |Brier difference| 0.00e+00

## B5 by L2 strength C
- C=1 against expB: largest |Brier difference| 0.00e+00
- N=10 C=0.001: means ECE 0.143, Brier 0.732, macro-AUC 0.589; Proposed - B5: macroAUC +0.2038 (p=2.7e-16); brier -0.1461 (p=1.4e-06); ece -0.0667 (p=0.023)
- N=10 C=0.003: means ECE 0.144, Brier 0.732, macro-AUC 0.589; Proposed - B5: macroAUC +0.2038 (p=2.7e-16); brier -0.1457 (p=1.5e-06); ece -0.0668 (p=0.022)
- N=10 C=0.01: means ECE 0.142, Brier 0.731, macro-AUC 0.589; Proposed - B5: macroAUC +0.2036 (p=2.5e-16); brier -0.1443 (p=1.7e-06); ece -0.0656 (p=0.025)
- N=10 C=0.03: means ECE 0.140, Brier 0.728, macro-AUC 0.589; Proposed - B5: macroAUC +0.2033 (p=2.3e-16); brier -0.1414 (p=2.4e-06); ece -0.0630 (p=0.029)
- N=10 C=0.1: means ECE 0.135, Brier 0.724, macro-AUC 0.590; Proposed - B5: macroAUC +0.2026 (p=1.6e-16); brier -0.1378 (p=3.9e-06); ece -0.0584 (p=0.031)
- N=10 C=0.3: means ECE 0.150, Brier 0.734, macro-AUC 0.590; Proposed - B5: macroAUC +0.2028 (p=6.4e-17); brier -0.1480 (p=1.8e-06); ece -0.0737 (p=0.0048)
- N=10 C=1: means ECE 0.224, Brier 0.784, macro-AUC 0.587; Proposed - B5: macroAUC +0.2053 (p=1.1e-17); brier -0.1975 (p=2.7e-08); ece -0.1471 (p=5.2e-07)
- N=10 C=3: means ECE 0.305, Brier 0.853, macro-AUC 0.584; Proposed - B5: macroAUC +0.2085 (p=3.2e-18); brier -0.2663 (p=1.5e-10); ece -0.2282 (p=1.1e-10)
- N=10 C=10: means ECE 0.376, Brier 0.923, macro-AUC 0.581; Proposed - B5: macroAUC +0.2114 (p=1.4e-18); brier -0.3366 (p=1.7e-12); ece -0.2988 (p=3.8e-13)
- N=50 C=0.001: means ECE 0.057, Brier 0.667, macro-AUC 0.674; Proposed - B5: macroAUC +0.1322 (p=1.9e-14); brier -0.1167 (p=4.4e-15); ece -0.0057 (p=0.55)
- N=50 C=0.003: means ECE 0.058, Brier 0.665, macro-AUC 0.674; Proposed - B5: macroAUC +0.1322 (p=1.6e-14); brier -0.1149 (p=6e-15); ece -0.0063 (p=0.5)
- N=50 C=0.01: means ECE 0.058, Brier 0.660, macro-AUC 0.674; Proposed - B5: macroAUC +0.1319 (p=1e-14); brier -0.1097 (p=1.6e-14); ece -0.0064 (p=0.46)
- N=50 C=0.03: means ECE 0.053, Brier 0.651, macro-AUC 0.675; Proposed - B5: macroAUC +0.1314 (p=3.2e-15); brier -0.1003 (p=1.3e-13); ece -0.0019 (p=0.82)
- N=50 C=0.1: means ECE 0.057, Brier 0.642, macro-AUC 0.676; Proposed - B5: macroAUC +0.1302 (p=6.3e-16); brier -0.0914 (p=2.5e-12); ece -0.0057 (p=0.52)
- N=50 C=0.3: means ECE 0.088, Brier 0.651, macro-AUC 0.677; Proposed - B5: macroAUC +0.1289 (p=4.8e-16); brier -0.1005 (p=1e-11); ece -0.0372 (p=0.0006)
- N=50 C=1: means ECE 0.161, Brier 0.691, macro-AUC 0.678; Proposed - B5: macroAUC +0.1286 (p=1.8e-15); brier -0.1407 (p=5.1e-12); ece -0.1100 (p=6.1e-09)
- N=50 C=3: means ECE 0.243, Brier 0.753, macro-AUC 0.676; Proposed - B5: macroAUC +0.1302 (p=7.9e-15); brier -0.2022 (p=3e-13); ece -0.1922 (p=2.2e-12)
- N=50 C=10: means ECE 0.328, Brier 0.835, macro-AUC 0.672; Proposed - B5: macroAUC +0.1342 (p=3.1e-14); brier -0.2843 (p=9.1e-15); ece -0.2765 (p=6.6e-15)
- N=200 C=0.001: means ECE 0.054, Brier 0.656, macro-AUC 0.736; Proposed - B5: macroAUC +0.0806 (p=8e-16); brier -0.1278 (p=2.1e-24); ece -0.0236 (p=0.011)
- N=200 C=0.003: means ECE 0.071, Brier 0.650, macro-AUC 0.737; Proposed - B5: macroAUC +0.0799 (p=5.3e-16); brier -0.1216 (p=7.7e-24); ece -0.0406 (p=5.2e-05)
- N=200 C=0.01: means ECE 0.066, Brier 0.635, macro-AUC 0.738; Proposed - B5: macroAUC +0.0785 (p=2.2e-16); brier -0.1073 (p=2e-22); ece -0.0363 (p=2.3e-05)
- N=200 C=0.03: means ECE 0.052, Brier 0.617, macro-AUC 0.741; Proposed - B5: macroAUC +0.0759 (p=1.2e-16); brier -0.0891 (p=9.7e-21); ece -0.0214 (p=0.0026)
- N=200 C=0.1: means ECE 0.053, Brier 0.601, macro-AUC 0.747; Proposed - B5: macroAUC +0.0701 (p=3.3e-16); brier -0.0726 (p=1.9e-18); ece -0.0227 (p=5.2e-06)
- N=200 C=0.3: means ECE 0.062, Brier 0.594, macro-AUC 0.754; Proposed - B5: macroAUC +0.0627 (p=2.4e-15); brier -0.0656 (p=1e-15); ece -0.0315 (p=1.4e-08)
- N=200 C=1: means ECE 0.080, Brier 0.596, macro-AUC 0.760; Proposed - B5: macroAUC +0.0567 (p=1.5e-14); brier -0.0677 (p=6.5e-14); ece -0.0499 (p=3.1e-08)
- N=200 C=3: means ECE 0.103, Brier 0.604, macro-AUC 0.762; Proposed - B5: macroAUC +0.0547 (p=3.7e-14); brier -0.0760 (p=1.6e-13); ece -0.0728 (p=2.2e-09)
- N=200 C=10: means ECE 0.124, Brier 0.613, macro-AUC 0.762; Proposed - B5: macroAUC +0.0547 (p=6.1e-14); brier -0.0855 (p=2.1e-13); ece -0.0935 (p=1.3e-10)
- N=400 C=0.001: means ECE 0.055, Brier 0.649, macro-AUC 0.754; Proposed - B5: macroAUC +0.0657 (p=2.5e-17); brier -0.1269 (p=1.3e-24); ece -0.0310 (p=0.0014)
- N=400 C=0.003: means ECE 0.077, Brier 0.638, macro-AUC 0.756; Proposed - B5: macroAUC +0.0643 (p=3.6e-17); brier -0.1163 (p=1.7e-23); ece -0.0530 (p=1.6e-07)
- N=400 C=0.01: means ECE 0.070, Brier 0.618, macro-AUC 0.759; Proposed - B5: macroAUC +0.0613 (p=9.4e-17); brier -0.0961 (p=1.6e-21); ece -0.0459 (p=5.4e-07)
- N=400 C=0.03: means ECE 0.060, Brier 0.597, macro-AUC 0.763; Proposed - B5: macroAUC +0.0566 (p=2.5e-16); brier -0.0752 (p=1.5e-19); ece -0.0356 (p=7.3e-07)
- N=400 C=0.1: means ECE 0.046, Brier 0.579, macro-AUC 0.771; Proposed - B5: macroAUC +0.0487 (p=9.8e-16); brier -0.0564 (p=3.3e-17); ece -0.0222 (p=2.1e-06)
- N=400 C=0.3: means ECE 0.045, Brier 0.568, macro-AUC 0.779; Proposed - B5: macroAUC +0.0412 (p=8e-15); brier -0.0462 (p=2.4e-15); ece -0.0210 (p=6.4e-06)
- N=400 C=1: means ECE 0.051, Brier 0.566, macro-AUC 0.783; Proposed - B5: macroAUC +0.0368 (p=3.3e-14); brier -0.0439 (p=1.7e-14); ece -0.0265 (p=7.6e-08)
- N=400 C=3: means ECE 0.058, Brier 0.568, macro-AUC 0.784; Proposed - B5: macroAUC +0.0357 (p=6.1e-14); brier -0.0458 (p=2.3e-14); ece -0.0333 (p=1.7e-07)
- N=400 C=10: means ECE 0.064, Brier 0.570, macro-AUC 0.784; Proposed - B5: macroAUC +0.0358 (p=1.2e-13); brier -0.0476 (p=2.3e-14); ece -0.0399 (p=1.5e-08)

## One number per sensor
- matrix rows against expS_calibsize_raw: largest |Brier difference| 0.00e+00
- n=30: mean numbers radar 0.649, EO/IR 0.657, acoustic 0.715, RF 0.781; one number - matrix: acc +0.0036 (p=0.081); macroAUC +0.0040 (p=0.00018); brier -0.0050 (p=0.00037); ece -0.0097 (p=0.0005); far@pd0.9 -0.0121 (p=0.033)
- n=100: mean numbers radar 0.669, EO/IR 0.696, acoustic 0.752, RF 0.880; one number - matrix: acc +0.0027 (p=0.079); macroAUC +0.0002 (p=0.61); brier -0.0008 (p=0.25); ece -0.0049 (p=0.02); far@pd0.9 -0.0029 (p=0.23)
- n=300: mean numbers radar 0.687, EO/IR 0.705, acoustic 0.756, RF 0.913; one number - matrix: acc +0.0002 (p=0.81); macroAUC -0.0001 (p=0.68); brier -0.0001 (p=0.78); ece -0.0029 (p=0.054); far@pd0.9 +0.0006 (p=0.81)
- n=1000: mean numbers radar 0.700, EO/IR 0.712, acoustic 0.758, RF 0.925; one number - matrix: acc -0.0002 (p=0.83); macroAUC -0.0004 (p=0.031); brier +0.0002 (p=0.58); ece -0.0039 (p=0.0036); far@pd0.9 +0.0032 (p=0.089)
- one number n=30 - matrix n=300: acc -0.0023 (p=0.13); macroAUC -0.0019 (p=0.0019); brier +0.0026 (p=0.00084); ece -0.0043 (p=0.022); far@pd0.9 +0.0087 (p=0.0047)
- one number n=100 - matrix n=300: acc -0.0002 (p=0.86); macroAUC -0.0008 (p=0.032); brier +0.0008 (p=0.14); ece -0.0037 (p=0.039); far@pd0.9 +0.0003 (p=0.92)

## CNN errors under the scenarios' mix
- P0: pool share of model confusions 0.784, pool drone/background 0.040; scenarios: error 0.047, share of model confusions 0.636, drone/background 0.017
- P1: pool share of model confusions 0.832, pool drone/background 0.028; scenarios: error 0.047, share of model confusions 0.617, drone/background 0.018
- P1c: pool share of model confusions 0.828, pool drone/background 0.029; scenarios: error 0.050, share of model confusions 0.583, drone/background 0.021
- P2: pool share of model confusions 0.767, pool drone/background 0.043; scenarios: error 0.074, share of model confusions 0.447, drone/background 0.043
- P3: pool share of model confusions 0.800, pool drone/background 0.053; scenarios: error 0.082, share of model confusions 0.448, drone/background 0.049
- worst fold: scenario share of model confusions 0.272, largest scenario drone/background 0.086, largest pool drone/background 0.114

## The world, by class and emission
- standard, clutter, emitting: P(T=high) 0.005, P(T>=medium) 0.262, P(N=top) 0.002
- standard, clutter, silent: P(T=high) 0.021, P(T>=medium) 0.298, P(N=top) 0.065
- standard, bird, emitting: P(T=high) 0.010, P(T>=medium) 0.274, P(N=top) 0.023
- standard, bird, silent: P(T=high) 0.188, P(T>=medium) 0.698, P(N=top) 0.772
- standard, drone, emitting: P(T=high) 0.130, P(T>=medium) 0.560, P(N=top) 0.528
- standard, drone, silent: P(T=high) 0.241, P(T>=medium) 0.823, P(N=top) 0.994
- standard, all, all: P(T=high) 0.091, P(T>=medium) 0.465, P(N=top) 0.360
- emit-drone-only, clutter, emitting: P(T=high) 0.005, P(T>=medium) 0.262, P(N=top) 0.002
- emit-drone-only, clutter, silent: P(T=high) 0.005, P(T>=medium) 0.262, P(N=top) 0.002
- emit-drone-only, bird, emitting: P(T=high) 0.010, P(T>=medium) 0.274, P(N=top) 0.023
- emit-drone-only, bird, silent: P(T=high) 0.010, P(T>=medium) 0.274, P(N=top) 0.023
- emit-drone-only, drone, emitting: P(T=high) 0.130, P(T>=medium) 0.560, P(N=top) 0.528
- emit-drone-only, drone, silent: P(T=high) 0.241, P(T>=medium) 0.823, P(N=top) 0.994
- emit-drone-only, all, all: P(T=high) 0.068, P(T>=medium) 0.411, P(N=top) 0.264

## Emission acting on drones only (emit-drone-only)
- N=50: Proposed Brier 0.5093, ECE 0.0450; graph 0.0016; excess 0.0378 (standard 0.0346); best alternative cMAP-full -0.0058 (p=0.00017); verdict holds
- N=200: Proposed Brier 0.4842, ECE 0.0346; graph 0.0016; excess 0.0127 (standard 0.0120); best alternative cMAP-full -0.0024 (p=0.0016); verdict holds
- Proposed N=10 - MLE N=50: acc -0.0086 (p=0.46); macroAUC +0.0321 (p=3.1e-05); brier -0.0132 (p=0.39); ece -0.0472 (p=0.0038); far@pd0.9 -0.0135 (p=0.43)
- standard, RF channel, all: calib+rel - no RF+rel: acc +0.0005 (p=0.65); macroAUC +0.0030 (p=2.6e-09); brier -0.0049 (p=9.7e-13); ece -0.0013 (p=0.4) | calibrated - no RF: acc -0.0004 (p=0.74); macroAUC +0.0027 (p=8.5e-08); brier -0.0046 (p=8.6e-12); ece -0.0006 (p=0.66)
- standard, RF channel, radio-silent drones: calib+rel - no RF+rel: acc +0.0030 (p=0.58); macroAUC -0.0070 (p=0.063); brier -0.0242 (p=6.8e-06); ece -0.0189 (p=0.025) | calibrated - no RF: acc +0.0014 (p=0.78); macroAUC -0.0101 (p=0.0082); brier -0.0265 (p=3.4e-06); ece -0.0235 (p=0.018)
- standard, RF channel, radio-silent non-drones: calib+rel - no RF+rel: acc -0.0002 (p=0.81); macroAUC -0.0004 (p=0.061); brier +0.0020 (p=0.00043); ece +0.0028 (p=0.072) | calibrated - no RF: acc -0.0013 (p=0.19); macroAUC -0.0012 (p=0.00037); brier +0.0026 (p=2.2e-05); ece +0.0041 (p=0.015)
- emit-drone-only, RF channel, all: calib+rel - no RF+rel: acc +0.0003 (p=0.76); macroAUC +0.0034 (p=3e-11); brier -0.0055 (p=2.1e-14); ece +0.0032 (p=0.033) | calibrated - no RF: acc -0.0006 (p=0.62); macroAUC +0.0031 (p=1.6e-10); brier -0.0051 (p=5.8e-14); ece +0.0044 (p=0.015)
- emit-drone-only, RF channel, radio-silent drones: calib+rel - no RF+rel: acc -0.0039 (p=0.44); macroAUC -0.0058 (p=0.1); brier -0.0160 (p=0.00069); ece +0.0024 (p=0.78) | calibrated - no RF: acc +0.0006 (p=0.92); macroAUC -0.0080 (p=0.027); brier -0.0195 (p=0.00012); ece -0.0048 (p=0.52)
- emit-drone-only, RF channel, radio-silent non-drones: calib+rel - no RF+rel: acc -0.0020 (p=0.21); macroAUC -0.0001 (p=0.9); brier +0.0004 (p=0.52); ece +0.0002 (p=0.93) | calibrated - no RF: acc -0.0032 (p=0.016); macroAUC -0.0012 (p=0.023); brier +0.0025 (p=0.00072); ece +0.0010 (p=0.56)

## S1: N=10 against MLE at N=50, and the ESS against the prior's direction
- Proposed and MLE rows against expW_raw: largest |Brier difference| 0.00e+00
- standard: Proposed N=10 - MLE N=50: acc +0.0010 (p=0.93); macroAUC +0.0318 (p=3.1e-06); brier -0.0158 (p=0.31); ece -0.0300 (p=0.047); far@pd0.9 -0.0252 (p=0.05)
  - standard, N=50, constrained full table, elicited, ESS 2 - MLE: acc +0.0231 (p=1.6e-05); brier -0.0454 (p=1.2e-06); ece -0.0480 (p=7.2e-07)
  - standard, N=50, constrained full table, elicited, ESS 0.2 - MLE: acc +0.0140 (p=1.7e-05); brier -0.0244 (p=3.4e-09); ece -0.0085 (p=0.041)
  - standard, N=50, constrained full table, uniform, ESS 2 - MLE: acc +0.0066 (p=0.0037); brier +0.0031 (p=0.55); ece +0.0211 (p=0.12)
  - standard, N=50, constrained full table, uniform, ESS 0.2 - MLE: acc +0.0076 (p=0.00037); brier -0.0268 (p=5.3e-12); ece -0.0407 (p=1.7e-11)
  - standard, N=200, constrained full table, elicited, ESS 2 - MLE: acc +0.0035 (p=0.051); brier -0.0054 (p=5.9e-05); ece -0.0083 (p=4.3e-05)
  - standard, N=200, constrained full table, elicited, ESS 0.2 - MLE: acc +0.0020 (p=0.00094); brier -0.0021 (p=1.2e-06); ece -0.0028 (p=0.0018)
  - standard, N=200, constrained full table, uniform, ESS 2 - MLE: acc +0.0015 (p=0.16); brier +0.0034 (p=0.00044); ece +0.0123 (p=0.038)
  - standard, N=200, constrained full table, uniform, ESS 0.2 - MLE: acc +0.0007 (p=0.077); brier -0.0019 (p=1.6e-08); ece -0.0095 (p=2.5e-11)
- nonadd-0.5: Proposed N=10 - MLE N=50: acc -0.0521 (p=0.003); macroAUC -0.0024 (p=0.72); brier +0.0518 (p=0.0088); ece -0.0008 (p=0.96); far@pd0.9 +0.0013 (p=0.92)
  - nonadd-0.5, N=50, constrained full table, elicited, ESS 2 - MLE: acc +0.0167 (p=0.014); brier -0.0222 (p=0.00078); ece -0.0465 (p=1.6e-06)
  - nonadd-0.5, N=50, constrained full table, elicited, ESS 0.2 - MLE: acc +0.0228 (p=4.1e-07); brier -0.0258 (p=3.7e-07); ece -0.0177 (p=0.00012)
  - nonadd-0.5, N=50, constrained full table, uniform, ESS 2 - MLE: acc +0.0113 (p=0.00011); brier +0.0109 (p=0.058); ece +0.0437 (p=0.0057)
  - nonadd-0.5, N=50, constrained full table, uniform, ESS 0.2 - MLE: acc +0.0133 (p=3e-05); brier -0.0294 (p=1.2e-09); ece -0.0397 (p=3.9e-09)
  - nonadd-0.5, N=200, constrained full table, elicited, ESS 2 - MLE: acc -0.0041 (p=0.025); brier +0.0016 (p=0.18); ece -0.0076 (p=0.013)
  - nonadd-0.5, N=200, constrained full table, elicited, ESS 0.2 - MLE: acc -0.0004 (p=0.53); brier -0.0012 (p=0.00051); ece -0.0027 (p=0.047)
  - nonadd-0.5, N=200, constrained full table, uniform, ESS 2 - MLE: acc -0.0010 (p=0.32); brier +0.0068 (p=1.9e-05); ece +0.0312 (p=2.9e-05)
  - nonadd-0.5, N=200, constrained full table, uniform, ESS 0.2 - MLE: acc +0.0000 (p=0.96); brier -0.0022 (p=2.8e-06); ece -0.0062 (p=0.00019)
- nonadd-1.0: Proposed N=10 - MLE N=50: acc -0.2904 (p=1.5e-13); macroAUC -0.0867 (p=6e-11); brier +0.2836 (p=3e-11); ece +0.1223 (p=9e-08); far@pd0.9 -0.0059 (p=0.56)
  - nonadd-1.0, N=50, constrained full table, elicited, ESS 2 - MLE: acc -0.0338 (p=2.6e-05); brier +0.0467 (p=3.3e-08); ece +0.0188 (p=0.0058)
  - nonadd-1.0, N=50, constrained full table, elicited, ESS 0.2 - MLE: acc +0.0094 (p=0.0015); brier -0.0181 (p=2.6e-05); ece -0.0174 (p=0.0018)
  - nonadd-1.0, N=50, constrained full table, uniform, ESS 2 - MLE: acc +0.0112 (p=0.035); brier +0.0350 (p=3.1e-06); ece +0.0932 (p=1.7e-08)
  - nonadd-1.0, N=50, constrained full table, uniform, ESS 0.2 - MLE: acc +0.0090 (p=0.0045); brier -0.0258 (p=2.1e-07); ece -0.0238 (p=0.00021)
  - nonadd-1.0, N=200, constrained full table, elicited, ESS 2 - MLE: acc -0.0091 (p=0.00035); brier +0.0121 (p=1.4e-07); ece +0.0137 (p=0.00016)
  - nonadd-1.0, N=200, constrained full table, elicited, ESS 0.2 - MLE: acc -0.0003 (p=0.8); brier -0.0015 (p=0.0023); ece -0.0073 (p=0.00018)
  - nonadd-1.0, N=200, constrained full table, uniform, ESS 2 - MLE: acc +0.0019 (p=0.2); brier +0.0085 (p=1.3e-07); ece +0.0326 (p=1.9e-06)
  - nonadd-1.0, N=200, constrained full table, uniform, ESS 0.2 - MLE: acc +0.0014 (p=0.27); brier -0.0029 (p=8e-06); ece -0.0073 (p=0.00097)
