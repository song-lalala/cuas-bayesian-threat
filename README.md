# cuas-bayesian-threat

Statistically grounded, sensor-calibrated hierarchical Bayesian network for
counter-UAS (C-UAS) threat assessment of low-slow-small (LSS) drones, with a
runnable synthetic study on a fully specified generative world.

This repository accompanies a study on making the conditional probability
tables (CPTs) of a hierarchical C-UAS threat BN *statistically grounded* under
data scarcity (an ordinal independence-of-causal-influence threat node,
equivalent-sample-size Dirichlet priors held fixed at alpha_0 = 2 (no
validation labels in the main pipeline -- label-budget fairness), a convex
monotonicity-constrained estimator with status-logged solves, and MAP-EM for
the latent-layer regime),
and on calibrating its measurement layer from *estimated sensor confusion
matrices* plus reliability weights fitted by logistic regression. The
evaluation world (v2) is a generative BN with range-dependent sensor
degradation, fully specified in `model_spec.py` and in Table 3 of the
manuscript; every experiment runs one protocol of **20 independent
replications with 95% CIs**. All experiments except the DroneRF ones run in
pure Python (no AirSim/GPU).

## Files
- `cuas_bn.py`   — discrete factor algebra + exact inference (variable elimination), virtual (likelihood) evidence. Self-test: `python cuas_bn.py`.
- `model_spec.py`— world v2: structure, ground-truth / synthetic-expert-prior / heuristic-surrogate CPTs (monotone ordinal-response), range-degraded sensor tables (`SENSOR_DEGRADE`).
- `dataio.py`    — scenario sampler, range-marginal & range-conditional confusion estimation, **logistic-regression reliability fit** (`fit_reliability`), abstract-sensor baseline (diagonal parameterized), inference-BN assembly.
- `grounding.py` — MLE, MAP (ESS Dirichlet), **cvxpy constrained estimator** (`cmap_cpt`, interior-point, solver tally in `CMAP_STATS`), **ordinal cumulative-logit ICI** (`ordinal_ici_cpt`, the adopted C1 form), Noisy-MAX / weighted-sum reference fitters, MAP-EM (`em_internal`, constrained final M-step, estimated sensors in the E-step).
- `pipeline.py`  — shared grounding pipeline: method registry (B0/B1/B2/MAP/cMAP-full/Proposed) with the ESS **held fixed at alpha_0 = 2** for all reported runs (equal label budgets); `select_ess` remains available as the Sec. VII-E(iii) option study.
- `metrics.py`   — accuracy, macro-F1, macro-AUC, Brier, top-label ECE (15 bins), classwise ECE, FAR@Pd, reliability-diagram data.
- `exp_significance.py` — **main study**: one 20-replication loop produces Tables 4–6, Fig. 3, subset analyses, abstract-diagonal sweep, DeLong/McNemar/bootstrap and across-replication tests (Sec. VII-B/D).
- `exp_constraints.py` — weak-prior monotonicity-violation study (estimated sensors, fitted reliability).
- `exp_prior_ess.py` — adversarial (reversed-monotonicity) prior stress test, full ESS range, regime maps at ESS 8 and 50, ordinal-ICI conservative-failure check (Sec. VII-E; Figs. 5–6).
- `exp_correlated.py` — correlated failures via a shared nuisance root with observed / noisy-observed / latent variants; reliability miscalibration ±50% and far-heavy context-shift transfer; P(e) monitoring (Sec. VII-F; Fig. 7).
- `exp_rf_e2e.py` — real-sample end-to-end DroneRF study: segment-level 60/15/25 split (H/L band pairs kept together), validation-checkpointed CNN, held-out spectrograms passed through the pipeline at test time, hard vs. scaled-likelihood-softmax variants (Secs. VII-C/G; needs the DroneRF data).
- `exp_mapem.py` — latent-layer MAP-EM study (C2-iii executed) (Sec. VII-B).
- `exp_baselines.py` — standard baselines (uninformative-mean MAP, Laplace, L2 logistic regression) and the realistic-expert condition (B0 table as the prior mean), same 20 replications (Sec. VII-B).
- `exp_noise_level.py` — sensitivity of the headline comparison to the world's aleatoric sharpness tau_T in {0.25, 0.40, 0.60} (Sec. VII-B).
- `exp_latency.py` — per-track inference cost, deployment assembly, and supervised-vs-MAP-EM grounding cost (Sec. IV-D).
- `exp_reliability_classes.py` — binned reliability decomposition for all four threat levels (Sec. VII-D, Table 8 lower block).
- `exp_sensitivity.py` — one-way linear-fractional sensitivity (tornado) + credal bounds for the worked-track evidence (Sec. VII-H).
- `rf_confusion.py` — legacy segment-level RF calibration (kept for comparison with common practice; the manuscript reports the segment-level numbers from `exp_rf_e2e.py --train`).
- `VALIDATION_OPTIONS.md` — single-world validation design and deferred AirSim / real-dataset options.

## Run
```
pip install -r requirements.txt        # or requirements.lock for exact versions
python exp_significance.py     # ~40-60 min; Tables 4-6, Fig. 3, Sec. VII-B/D
python exp_constraints.py      # ~2 min
python exp_prior_ess.py        # ~1-2 h;  Sec. VII-E, Figs. 5-6
python exp_correlated.py       # ~20 min; Sec. VII-F, Fig. 7
python exp_mapem.py            # ~15 min; latent-layer study
python exp_baselines.py        # ~4 min;  standard + realistic-expert baselines
python exp_noise_level.py      # ~5 min;  tau_T sweep
python exp_latency.py          # ~15 s;   inference-cost measurement
python exp_reliability_classes.py  # ~30 s;   per-class reliability
python exp_sensitivity.py      # ~1 min;  Sec. VII-H tornado
# DroneRF-dependent (see "DroneRF data" below):
python exp_rf_e2e.py --train /path/droneRF_four.npz   # CNN, segment-level split
python exp_rf_e2e.py                                   # Sec. VII-G study
```

## DroneRF data (required only for `exp_rf_e2e.py` / `rf_confusion.py`)
The DroneRF recordings are **not** redistributed with this repository.
Reproducing Secs. VII-C and VII-G requires a one-time separate download of the
public DroneRF dataset (Al-Sa'd et al., *Data in Brief* 26:104313, 2019;
https://data.mendeley.com/datasets/f4c2b4n755), processed into
`droneRF_four.npz` with arrays `X` (N,128,128) float32 log-spectrograms,
`y` (N,) int64, `class_names` (4,), and `meta` (N,2) carrying the source CSV
file per segment (used for the segment-level split). **Every other
experiment in the paper runs self-contained from this repository.**

## Reproducibility tolerance
All reported numbers are means over independent replications of Monte-Carlo
simulations. Re-runs on different machines / BLAS builds / library versions
typically match to within **±0.001–0.004**, not to the third decimal. Seeds
are fixed in the scripts and exact library versions are pinned in
`requirements.lock`. Console note: runs print numpy warnings (`overflow
encountered in exp`, `divide by zero encountered in log`) in the thousands.
Both are benign: the first is the ordinal-response logistic saturating in
`_oi_table`, whose row is immediately clipped to `1e-9` and renormalized, and
the second is the convex solver evaluating a log at a boundary point. The
tally that matters is in `results/expS_solver.txt`. Solver note: every
constrained solve behind the reported tables is status-logged; the run
summaries record the tally (all optimal in the reported runs).

### Checking a re-run against the paper
`check_manuscript_numbers.py` compares what the scripts produced against
what is quoted in the paper. Run it from this directory:

```
python check_manuscript_numbers.py
```

Without the manuscript sources it verifies `results/` alone — that the
files exist, that the headline figures are internally consistent, and that
the four-step ablation path sums to its total — and says so. If you also
have the LaTeX sources, point it at them to check every quoted number,
table and figure reference as well:

```
CUAS_PAPER_DIR=/path/to/paper/ieeeaccess python check_manuscript_numbers.py
```

## Evaluation protocol status (tracked here per the manuscript)
| Protocol item | Status |
|---|---|
| Scarcity, fusion (+hard-condition subsets), calibration-set size, abstract-diagonal sweep, ablation incl. (−ICI), majority baseline — 20 replications, CIs, DeLong/McNemar/bootstrap | done (`exp_significance.py`) — Secs. VII-B/D |
| Constraint-violation study (weak prior) | done (`exp_constraints.py`) |
| Latent-layer MAP-EM (C2-iii) | done (`exp_mapem.py`) — Sec. VII-B |
| Adversarial prior quality; full ESS range; regime maps (ESS 8 & 50) | done (`exp_prior_ess.py`) — Sec. VII-E |
| Correlated failures (observed/noisy/latent W); reliability miscalibration & context-shift transfer; P(e) monitoring | done (`exp_correlated.py`) — Sec. VII-F |
| Real-sample end-to-end DroneRF study (segment-level splits) | done (`exp_rf_e2e.py`) — Secs. VII-C/G |
| One-way sensitivity (tornado) + credal bounds | done (`exp_sensitivity.py`) — Sec. VII-H |
| Standard baselines (uninformative MAP, Laplace, discriminative) and realistic-expert prior | done (`exp_baselines.py`) — Sec. VII-B |
| World noise-level (tau_T) sensitivity | done (`exp_noise_level.py`) — Sec. VII-B |
| Dedicated dropout/spoofing and SNR sweeps; embedded-latency profiling | deferred to a dedicated follow-up study (extended evaluation) |

## Key results (means ± 95% CI over 20 independent replications; 2500-scenario test draws; oracle acc 0.593 / AUC 0.824 / ECE 0.017; majority 0.434)
- **Scarcity (equal label budgets):** the proposed grounding reaches unconstrained MLE's performance with ~4–5× fewer scenarios (N=10: 0.534/0.793/0.077 vs MLE N=50: 0.533/0.761/0.107); resolvable even at N=400 (ΔECE −0.008, p=7.9e-4); beats the prior-only B2 at every N.
- **Fusion:** calibrated fusion beats the abstract model on every metric (AUC +0.008, p=2.6e-10; ECE −0.011; FAR −0.005, p=0.03; oracle FAR 0.455 = noise ceiling), with the big effects where the abstract model is structurally wrong: radio-silent drones acc 0.503→0.593, far-range radio-silent 0.438→0.545 (ECE 0.156→0.117). Diagonal sweep 0.70–0.99: insensitive to the 0.94 choice.
- **Reliability (fitted):** condition-dependent in both directions — improves the far-range subset (ECE 0.051→0.045, p=0.008) but worsens radio-silent Brier (+0.005, p<1e-3; mechanism reported); ±50% miscalibration moves ECE ≤0.004; far-heavy deployment keeps the benefit (0.031 vs 0.036).
- **Constraints & ICI:** constraints are inert under correct priors, rescue calibration under strongly reversed priors at small N (ECE 0.072→0.031), and can hurt at λ=0.5 or reversed-prior large-N — mapped at both ESS values. The 9-parameter ordinal-ICI threat node beats the 54-parameter table on both axes (−ICI ablation: ΔECE +0.008, ΔAUC −0.004, both significant, in favor of ICI) and fails conservatively under a reversed prior (ECE 0.019 at the cost of discrimination).
- **Correlated failures:** naive conditional-independence fusion degrades to adverse-subset ECE 0.070 at induced error correlation ρ=0.26; the nuisance-parent fusion restores 0.039 observed / 0.043 noisily observed / 0.064 latent (all significant; +15 parameters).
- **MAP-EM (latent H,N,C):** prior-only 0.512/0.780/0.103 → EM 0.564/0.804/0.046 at N=200; fully observed 0.577/0.817/0.030.
- **DroneRF end-to-end (real samples through the pipeline):** segment-level split (H/L band pairs kept together) gives CNN test acc 0.814 (a naive random split gives 0.850); note this is within-recording generalization, since each flight mode is one recording cut into segments; the measured channel improves the threat posterior over the abstract RF model by +0.020 macro-AUC (p=8.7e-15) / −0.033 ECE; softmax virtual evidence ≈ hard reports for this confident classifier.
- **Ablation path (N=80, one change per step):** smoothing +0.015 ECE (p=0.15, n.s.) -> elicited prior mean -0.051 (p=6.3e-6) -> monotonicity constraints -0.001 (inert under a good prior) -> ordinal ICI node -0.008 (p=4.7e-3); the steps sum to the -0.044 total. The `expert MAP`/`expert cMAP` variants in `exp_baselines.py` exist so that each delta belongs to exactly one change.
- **Standard baselines:** smoothing alone does not explain the gain — uninformative-mean MAP at the same ESS reaches only 0.442/0.671/0.143 at N=10 (proposed: 0.534/0.793/0.077; dAUC +0.122, p<1e-7) and stays behind at N=400; L2 logistic regression trails at every N (0.552/0.783/0.051 at N=400). The coarse B0 table used as the prior mean already beats unsmoothed MLE at every N.
- **World noise:** the conclusion holds for tau_T in {0.25, 0.40, 0.60} (oracle accuracy 0.629/0.593/0.507) and the margin grows with noise (N=50 dECE -0.042/-0.055/-0.088).
- **Cost:** a track update costs 0.106 ms on one desktop CPU core (p95 0.126 ms, ~9,500 updates/s); the latent-layer MAP-EM fit costs 4.3x the supervised grounding offline and nothing at inference.
- **Per-class reliability:** every threat level is over-confident in the (.50,.75] bin (negligible -0.056, low -0.022, medium -0.016, high -0.050); the largest deviation anywhere is the high level's +0.069 in the top bin, against a CPT-oracle maximum of 0.029. Level names come from `model_spec.STATES["T"]`, never hardcoded.
- **Sensitivity:** the largest single-cell swing of the worked-track P(T=high) is 0.14; the top-5 one-at-a-time envelope [0.565, 0.700] (baseline 0.633) leaves the decision invariant.

## Caveats
Synthetic ground truth; absolute accuracies reflect the chosen aleatoric-noise
regime. The world, the synthetic prior, and the proposed threat
parameterization share the monotone ordinal-response family: conclusions
certify the grounding machinery when that domain assumption holds (Sec. VII-E
quantifies what breaks when the prior contradicts it); structural
misspecification is future work. Of the four modalities, RF is executed on
real data (segment-level DroneRF, Secs. VII-C/G); EO/IR, acoustic, and radar
remain specified but not executed on public datasets.

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
