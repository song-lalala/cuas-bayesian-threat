# Exp-G v2: real-sample end-to-end RF study (reps=20)

- CNN (recording-level split): val_acc 0.8279, test_acc 0.8143
- abstract: acc 0.547+-0.007, macroAUC 0.798+-0.003, ece 0.063+-0.009, cwece 0.045+-0.005
- calib-hard: acc 0.578+-0.008, macroAUC 0.818+-0.002, ece 0.029+-0.005, cwece 0.031+-0.003
- calib-soft: acc 0.579+-0.007, macroAUC 0.819+-0.002, ece 0.030+-0.005, cwece 0.031+-0.003
- measured vs abstract dmacroAUC: +0.0202 (t p=8.7e-15)
- measured vs abstract dece: -0.0334 (t p=2.5e-08)
- virtual soft vs hard dmacroAUC: +0.0003 (t p=9.5e-02)
- virtual soft vs hard dece: +0.0001 (t p=9.3e-01)
