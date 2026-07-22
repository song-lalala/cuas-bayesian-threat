# cuas-bayesian-threat

Statistically grounded, sensor-calibrated hierarchical Bayesian network for
counter-UAS (C-UAS) threat assessment of low-slow-small (LSS) drones, with a
runnable synthetic digital-twin study.

This repository accompanies a study on making the conditional probability
tables (CPTs) of a hierarchical C-UAS threat BN *statistically grounded* under
data scarcity (independence-of-causal-influence reduction, equivalent-sample-size
Dirichlet priors, monotonicity-constrained MAP, and MAP-EM), and on calibrating
its measurement layer from *measured sensor confusion matrices* (Dirichlet
smoothing, context-dependent reliability discounting, virtual/likelihood
evidence). All experiments here run in pure Python — no AirSim/GPU is required:
a ground-truth generative BN with monotone CPTs stands in for the digital twin,
and sensor reports are drawn through literature-informed confusion matrices.
Real AirSim logs or public-dataset confusion matrices can replace
`dataio.generate_dataset` and the matrices in `model_spec.py` without touching
the rest of the pipeline.

## Files
- `cuas_bn.py`   — discrete factor algebra + exact inference (variable elimination), virtual (likelihood) evidence. Self-test: `python cuas_bn.py`.
- `model_spec.py`— network structure, ground-truth / expert-prior / heuristic CPTs (monotone ordinal-response), sensor confusion matrices.
- `dataio.py`    — scenario sampler (digital-twin proxy) + sensor evidence layer (confusion-matrix likelihood, reliability discounting by range, virtual evidence, missing sensors, abstract-sensor baseline).
- `grounding.py` — MLE, MAP (ESS Dirichlet), constrained-MAP (SLSQP monotonicity), MAP-EM (latent H, N, C).
- `metrics.py`   — accuracy, macro-F1, macro-AUC, Brier, ECE, FAR@Pd; per-scenario BN inference with reliability caching by range.
- `experiments.py` — Exp1 (scarcity sweep), Exp2 (sensors), Exp3 (ablation); writes `results/`.
- `exp_constraints.py` — weak-prior study: monotonicity-violation rate MAP vs cMAP.
- `replot_fig3.py` — regenerate `results/exp1_scarcity.png` with publication styling.
- `VALIDATION_OPTIONS.md` — single-world validation design and deferred AirSim / real-dataset options.

## Run
```
pip install -r requirements.txt
python experiments.py        # ~5-6 min; writes results/RESULTS.md, CSVs, PNGs
python exp_constraints.py    # ~1-2 min; writes results/exp_constraints.csv
```

## Coherence (single world)
One generative BN emits labels, kinematics, and ALL sensor reports; the sensor
confusion matrices are **estimated** from a labelled calibration split of that
same world (`estimate_confusion`, n=300), not read off the true CPTs — the same
procedure whether the world is synthetic, AirSim, or real. See
`VALIDATION_OPTIONS.md` for the deferred AirSim / real-dataset designs.

## Key results (means over seeds, 2500-scenario test; oracle acc 0.60 / AUC 0.83 / ECE 0.008)
- **Scarcity:** prior-grounded CPTs reach asymptotic accuracy+calibration with ~5-10x fewer scenarios than MLE; at N=50 ECE 0.046 (proposed) vs 0.081 (MLE); heuristic/expert-only never adapt (`results/exp1_scarcity.png`).
- **Fusion:** calibrated 4-sensor fusion 0.590 acc / 0.815 AUC vs best single 0.558 / 0.794; calibrated vs abstract ECE 0.030 vs 0.057; reliability lowers far-range radio-silent ECE 0.149 → 0.100.
- **Calibration size:** confusion matrices estimated from just n=30 labelled scenarios already beat abstract (AUC 0.810 vs 0.799, ECE 0.031 vs 0.057); n≈300 reaches oracle (0.815 vs 0.816) (`results/exp2b_calibsize.csv`).
- **Constraints:** under a weak prior, cMAP cuts threat-CPT monotonicity violations 5-8x (5.9% → 1.9% at N=50) at no accuracy cost.
- **Ablation (N=80):** removing the prior (MLE) is the largest regression (ECE 0.042 → 0.066); reliability / virtual-evidence effects are small and calibration-side.

## Caveats
Synthetic ground truth; absolute accuracies reflect the chosen aleatoric-noise
regime (adjacent threat-level overlap is realistic). The comparative
conclusions (grounding ≫ MLE under scarcity; calibrated ≫ abstract sensors)
are the transferable findings. Confusion matrices are literature-informed, not
yet fit to public RF/EO-IR datasets — that substitution is a planned next step.

## Citation
```bibtex
@misc{song_cuas_bayesian_threat,
  author = {Song, Moogeun},
  title  = {cuas-bayesian-threat: statistically grounded, sensor-calibrated
            Bayesian network for counter-UAS threat assessment},
  year   = {2026},
  howpublished = {\url{https://github.com/song-lalala/cuas-bayesian-threat}}
}
```

## License
Released under the MIT License. See `LICENSE`.
