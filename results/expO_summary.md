# Exp-O: EO/IR channel on real thermal-infrared clips, mixed world with two real channels

## G1 (boxes)
```
manifest: 1391 files match their SHA-256
box CSV ir_boxes_v1.0.0.csv: SHA-256 14c663b0abb09824abbd0af76abcb13ee22d22217b0e9caf8fc6fb6d7fc81b0b, 119475 box rows, 365 label files, labels ['AIRPLANE', 'BIRD', 'DRONE', 'HELICOPTER']
boxes by clip class (rows) and label (columns):
label       AIRPLANE   BIRD  DRONE  HELICOPTER
ccls                                          
AIRPLANE       23243      0      0           0
BIRD               0  28527      0           0
DRONE              0    146  50606           0
HELICOPTER         0      0      0       16953
boxes whose label differs from the clip class (ignored): 146; DRONE_096 76 (bin MEDIUM, session 119), DRONE_101 21 (bin CLOSE, session 119), DRONE_104 49 (bin DISTANT, session 119)
rows with several boxes of the clip class (largest used): 4452
decoded frames, all clips: 114095; last boxed row = decoded frame count in 362/365 clips; others (last boxed row - frames): DRONE_067 -2, HELICOPTER_006 -1, HELICOPTER_034 -1
label time = (row - 1)/30 + video stream start time in 365/365 clips; stream start 0 s in 349 clips and 1/30 s in 16; frames are matched by row (frame = row - 1); round(time_s * 30) would be one frame late in those 16
G1 criterion: sampled windows with a box of the clip class 3650/3650 = 1.0000 (>= 0.95: pass); the sampled frame is the first frame of its window in 3650/3650
box sizes of the sampled frames (pixels; side = max(w, h)), by class and bin:
                      n  w_median  w<5  w>=15  side_median  side_p90  side_p95  side_max  side>64
cls        bin                                                                                   
AIRPLANE   CLOSE     90    30.008    0  0.956       30.008    48.200    53.401   134.129        2
           MEDIUM   250    17.514    0  0.688       17.514    24.558    27.328    42.000        0
           DISTANT  400    14.000    0  0.450       14.000    24.959    28.918    52.914        0
BIRD       CLOSE    100    19.707    0  0.870       20.000    25.121    26.183    32.824        0
           MEDIUM   230    15.000    0  0.543       15.033    20.045    22.000    27.000        0
           DISTANT  460    13.490    0  0.237       13.599    16.000    16.700    21.000        0
DRONE      CLOSE    240    29.323    0  1.000       29.323    42.374    45.062   104.625        2
           MEDIUM   940    18.048    0  0.874       18.109    24.000    25.245    30.074        0
           DISTANT  390    14.087    0  0.423       14.087    18.303    20.858    26.474        0
HELICOPTER CLOSE    150    40.500    0  1.000       40.500    74.530    83.135    89.326       33
           MEDIUM   200    24.000    0  0.965       24.000    29.758    32.001    36.000        0
           DISTANT  200    15.214    0  0.535       15.257    22.156    24.618    27.212        0
c3: sampled frames with side > 64 (resized to 64x64): 37/3650 = 0.0101; AIRPLANE/CLOSE 2 (0.022 of the cell), DRONE/CLOSE 2 (0.008 of the cell), HELICOPTER/CLOSE 33 (0.220 of the cell)
share of sampled frames left out of G2 for a mask of side M (box side > M - 4):
                     M=16   M=24   M=32   M=40   M=48
cls        bin                                       
AIRPLANE   CLOSE    0.967  0.956  0.689  0.322  0.144
           MEDIUM   0.936  0.248  0.040  0.028  0.000
           DISTANT  0.718  0.220  0.052  0.028  0.025
BIRD       CLOSE    1.000  0.430  0.050  0.000  0.000
           MEDIUM   0.900  0.100  0.000  0.000  0.000
           DISTANT  0.826  0.004  0.000  0.000  0.000
DRONE      CLOSE    1.000  0.929  0.575  0.179  0.062
           MEDIUM   0.984  0.300  0.021  0.000  0.000
           DISTANT  0.700  0.064  0.000  0.000  0.000
HELICOPTER CLOSE    1.000  1.000  0.973  0.627  0.453
           MEDIUM   0.995  0.825  0.120  0.000  0.000
           DISTANT  0.805  0.195  0.000  0.000  0.000
ALL                 0.888  0.325  0.117  0.050  0.029
per-label-file provenance (manifest SHA-256 of the label file, frames, boxes): expO_g1_label_files.csv (365 rows)
```

## G2 (confound gate, mask M = 40; both checks must meet both conditions)
- 1-NN masked (LOSO) (n 3466): accuracy 0.438 vs limit 0.541 (majority share 0.441) -> pass; balanced accuracy 0.409 vs limit 0.35 -> FAIL; overall FAIL
  recall AIRPLANE 0.450, BIRD 0.376, DRONE 0.505, HELICOPTER 0.305; 4x4 confusion (rows true AIRPLANE, BIRD, DRONE, HELICOPTER): 312 179 181 21 200 297 240 53 221 423 771 112 69 134 114 139; no-information accuracy (sum of squared shares) 0.303
- CNN masked (4-fold sessions) (n 3466): accuracy 0.467 vs limit 0.541 (majority share 0.441) -> pass; balanced accuracy 0.396 vs limit 0.35 -> FAIL; overall FAIL
  recall AIRPLANE 0.491, BIRD 0.220, DRONE 0.659, HELICOPTER 0.215; 4x4 confusion (rows true AIRPLANE, BIRD, DRONE, HELICOPTER): 340 59 182 112 190 174 325 101 244 123 1006 154 46 65 247 98
- G2 overall: FAIL
- per fold, 1-NN masked (LOSO): fold 0 n 709 acc 0.460 (majority 0.282) balanced 0.477; fold 1 n 1334 acc 0.429 (majority 0.622) balanced 0.386; fold 2 n 688 acc 0.459 (majority 0.366) balanced 0.412; fold 3 n 735 acc 0.415 (majority 0.362) balanced 0.377
- per fold, CNN masked (4-fold sessions): fold 0 n 709 acc 0.271 (majority 0.282) balanced 0.277; fold 1 n 1334 acc 0.587 (majority 0.622) balanced 0.476; fold 2 n 688 acc 0.491 (majority 0.366) balanced 0.458; fold 3 n 735 acc 0.415 (majority 0.362) balanced 0.370
- patches left out of G2 (box side > M - 4), share by class and bin: AIRPLANE/CLOSE 0.32 (29/90); AIRPLANE/MEDIUM 0.03 (7/250); AIRPLANE/DISTANT 0.03 (11/400); BIRD/CLOSE 0.00 (0/100); BIRD/MEDIUM 0.00 (0/230); BIRD/DISTANT 0.00 (0/460); DRONE/CLOSE 0.18 (43/240); DRONE/MEDIUM 0.00 (0/940); DRONE/DISTANT 0.00 (0/390); HELICOPTER/CLOSE 0.63 (94/150); HELICOPTER/MEDIUM 0.00 (0/200); HELICOPTER/DISTANT 0.00 (0/200)

## G3 (test sessions per class and bin)
| class | bin | clips | patches | sessions | per fold | thin |
|---|---|---|---|---|---|---|
| AIRPLANE | CLOSE | 9 | 90 | 8 | 2/2/2/2 |  |
| AIRPLANE | MEDIUM | 25 | 250 | 17 | 4/4/5/4 |  |
| AIRPLANE | DISTANT | 40 | 400 | 25 | 6/6/6/7 |  |
| BIRD | CLOSE | 10 | 100 | 9 | 2/2/2/3 |  |
| BIRD | MEDIUM | 23 | 230 | 22 | 6/6/5/5 |  |
| BIRD | DISTANT | 46 | 460 | 34 | 8/8/9/9 |  |
| DRONE | CLOSE | 24 | 240 | 7 | 3/2/1/1 |  |
| DRONE | MEDIUM | 94 | 940 | 11 | 3/2/3/3 |  |
| DRONE | DISTANT | 39 | 390 | 4 | 1/1/1/1 |  |
| HELICOPTER | CLOSE | 15 | 150 | 8 | 2/1/2/3 |  |
| HELICOPTER | MEDIUM | 20 | 200 | 9 | 3/3/2/1 |  |
| HELICOPTER | DISTANT | 20 | 200 | 8 | 1/2/2/3 |  |
- test fold 0: 4 classes, 73 clips, 35 sessions, 730 patches; clips (sessions) AIRPLANE 20 (11), BIRD 20 (16), DRONE 18 (4), HELICOPTER 15 (4)
- test fold 1: 4 classes, 138 clips, 32 sessions, 1380 patches; clips (sessions) AIRPLANE 19 (10), BIRD 20 (16), DRONE 85 (2), HELICOPTER 14 (4)
- test fold 2: 4 classes, 75 clips, 36 sessions, 750 patches; clips (sessions) AIRPLANE 18 (12), BIRD 20 (15), DRONE 27 (3), HELICOPTER 10 (6)
- test fold 3: 4 classes, 79 clips, 36 sessions, 790 patches; clips (sessions) AIRPLANE 17 (12), BIRD 19 (16), DRONE 27 (3), HELICOPTER 16 (5)

## G2 session bootstrap (descriptive, not part of the decision)
- 1-NN masked (LOSO): balanced accuracy 0.409, 95% interval [0.347, 0.483], share of resamples <= 0.35 0.033, <= 0.25 0.000 (B 2000, seed 0, 135 sessions)
- CNN masked (4-fold sessions): balanced accuracy 0.396, 95% interval [0.338, 0.449], share of resamples <= 0.35 0.063, <= 0.25 0.000 (B 2000, seed 0, 135 sessions)

## [N] node level after the G2 failure (a descriptive check only; [R], [E], [H] not run)
- CNN unmasked, all patches (n 3650): accuracy 0.622, balanced accuracy 0.546, macro-F1 0.546, 3-way report accuracy 0.655
  recall AIRPLANE 0.570, BIRD 0.543, DRONE 0.815, HELICOPTER 0.255; 4x4 confusion (rows true AIRPLANE, BIRD, DRONE, HELICOPTER): 422 162 119 37 140 429 127 94 104 133 1279 54 82 168 160 140; 3x3 report confusion (rows true drone, bird, other): 1279 133 158 127 429 234 279 330 681
  per bin: CLOSE n 580 acc 0.695 balanced 0.622 drone recall 0.912; MEDIUM n 1620 acc 0.722 balanced 0.570 drone recall 0.891; DISTANT n 1450 acc 0.481 balanced 0.425 drone recall 0.569
  balanced accuracy per fold: fold 0 0.468, fold 1 0.611, fold 2 0.684, fold 3 0.433
- CNN unmasked, G2 patches (n 3466): accuracy 0.615, balanced accuracy 0.518, macro-F1 0.514, 3-way report accuracy 0.643
  recall AIRPLANE 0.561, BIRD 0.543, DRONE 0.815, HELICOPTER 0.154; 4x4 confusion (rows true AIRPLANE, BIRD, DRONE, HELICOPTER): 389 161 114 29 140 429 127 94 104 133 1245 45 66 168 152 70; 3x3 report confusion (rows true drone, bird, other): 1245 133 149 127 429 234 266 329 554
  per bin: CLOSE n 414 acc 0.684 balanced 0.563 drone recall 0.939; MEDIUM n 1613 acc 0.720 balanced 0.569 drone recall 0.891; DISTANT n 1439 acc 0.478 balanced 0.422 drone recall 0.569
  balanced accuracy per fold: fold 0 0.448, fold 1 0.600, fold 2 0.656, fold 3 0.386
- 1-NN unmasked, G2 patches (n 3466): accuracy 0.540, balanced accuracy 0.434, macro-F1 0.426, 3-way report accuracy 0.569
  recall AIRPLANE 0.359, BIRD 0.523, DRONE 0.767, HELICOPTER 0.088; 4x4 confusion (rows true AIRPLANE, BIRD, DRONE, HELICOPTER): 249 239 171 34 131 413 201 45 151 139 1171 66 65 155 196 40; 3x3 report confusion (rows true drone, bird, other): 1171 139 217 201 413 176 367 394 388
  per bin: CLOSE n 414 acc 0.739 balanced 0.634 drone recall 0.944; MEDIUM n 1613 acc 0.613 balanced 0.420 drone recall 0.848; DISTANT n 1439 acc 0.402 balanced 0.346 drone recall 0.482
  balanced accuracy per fold: fold 0 0.427, fold 1 0.464, fold 2 0.459, fold 3 0.402
- masked against unmasked on the same G2 patches: 1-NN: accuracy 0.438 -> 0.540, balanced accuracy 0.409 -> 0.434; CNN: accuracy 0.467 -> 0.615, balanced accuracy 0.396 -> 0.518
- session bootstrap, 1-NN unmasked, G2 patches: +0.434, 95% interval [0.400, 0.481] (B 2000, seed 0, 135 sessions)
- session bootstrap, CNN unmasked, G2 patches: +0.518, 95% interval [0.460, 0.584] (B 2000, seed 0, 135 sessions)
- session bootstrap, CNN unmasked, all patches: +0.546, 95% interval [0.482, 0.612] (B 2000, seed 0, 139 sessions)
- session bootstrap, 1-NN unmasked minus masked: +0.025, 95% interval [-0.042, 0.090] (B 2000, seed 0, 135 sessions), share of resamples <= 0: 0.255
- session bootstrap, CNN unmasked minus masked: +0.122, 95% interval [0.077, 0.178] (B 2000, seed 0, 135 sessions), share of resamples <= 0: 0.000
