# cuas-bayesian-threat

Statistically grounded, sensor-calibrated hierarchical Bayesian network for
counter-UAS (C-UAS) threat assessment of low-slow-small (LSS) drones, with a
runnable synthetic study on a fully specified generative world and real-data
studies of the radio-frequency (RF) channel.

This repository accompanies a study on making the conditional probability
tables (CPTs) of a hierarchical C-UAS threat BN *statistically grounded* under
data scarcity (an ordinal independence-of-causal-influence threat node,
equivalent-sample-size Dirichlet priors held fixed at alpha_0 = 2 (no
validation labels in the main pipeline -- label-budget fairness), a convex
monotonicity-constrained estimator, and MAP-EM for
the latent-layer regime),
and on calibrating its measurement layer from *estimated sensor confusion
matrices* plus reliability weights fitted by logistic regression. The
evaluation world (v2) is a generative BN with range-dependent sensor
degradation, fully specified in `model_spec.py` and in Table 3 of the paper;
the main experiments run one protocol of **20 independent replications with
95% CIs**. All experiments except the RF and infrared ones run in pure Python
(no GPU needed).

## Files

Core:
- `cuas_bn.py`   — discrete factor algebra + exact inference (variable elimination), virtual (likelihood) evidence. Self-test: `python cuas_bn.py`.
- `model_spec.py`— world v2: structure, ground-truth / synthetic-expert-prior / heuristic-surrogate CPTs (monotone ordinal-response), range-degraded sensor tables (`SENSOR_DEGRADE`).
- `dataio.py`    — scenario sampler, range-marginal & range-conditional confusion estimation, **logistic-regression reliability fit** (`fit_reliability`), abstract-sensor baseline (diagonal parameterized), inference-BN assembly.
- `grounding.py` — MLE, MAP (ESS Dirichlet), **cvxpy constrained estimator** (`cmap_cpt`, interior-point, solver tally in `CMAP_STATS`), **ordinal cumulative-logit ICI** (`ordinal_ici_cpt`), Noisy-MAX / weighted-sum reference fitters, MAP-EM (`em_internal`, constrained final M-step, estimated sensors in the E-step).
- `pipeline.py`  — shared grounding pipeline: method registry (B0/B1/B2/MAP/cMAP-full/Proposed) with the ESS **held fixed at alpha_0 = 2** per column in the main study (every BN method sees the same labeled scenarios and calibration split; other studies state their ESS); `select_ess` remains available as the Sec. VII-E(iii) option study.
- `metrics.py`   — accuracy, macro-F1, macro-AUC, Brier, top-label ECE (15 bins), classwise ECE, FAR@Pd, reliability-diagram data.
- `param_counts.py` — free-parameter counts of the network and of the threat node.

Synthetic studies:
- `exp_significance.py` — **main study**: one 20-replication loop produces Tables 4, 6 and 7, the data of Fig. 3, subset analyses, abstract-diagonal sweep, DeLong/McNemar/bootstrap and across-replication tests (Secs. VII-B, VII-D). `exp_calibsize_ci.py` re-evaluates the calibration-size cells of Table 7; `analyze_subsets.py` is a post-analysis of the raw files.
- `exp_baselines.py` — standard baselines (uninformative-mean MAP, Laplace, L2 logistic regression) and the realistic-expert condition (B0 table as the prior mean), same 20 replications (Table 5, Sec. VII-B).
- `exp_base_rule.py` — the scoring rule of the base framework as a baseline (B6), published, recalibrated and fitted, and the abstract-diagonal sweep extended below 0.70 (Secs. VI-C, VII-B). `exp_base_rule_worlds.py` runs the same comparison in the S1 worlds and at large N; it was added after the first results were seen and was not fixed in advance.
- `exp_noise_level.py` — sensitivity of the headline comparison to the world's aleatoric sharpness tau_T in {0.25, 0.40, 0.60} (Sec. VII-B, Supplementary Material Sec. S1).
- `exp_mapem.py` — latent-layer MAP-EM study (C2-iii) (Sec. VII-B).
- `exp_latent_identifiability.py` — what the unlabeled case can identify (Jacobian rank, recodings, their reach) and the quality of the latent posteriors with and without labels (Secs. IV-D, VII-B).
- `exp_constraints.py` — noisy-prior monotonicity-violation study (8 seeds, ESS 8, N = 20 and 50, one fixed test and calibration set; estimated sensors, fitted reliability; Sec. VII-B).
- `exp_reliability_variants.py`, `reliability_cc.py`, `world_options.py` — the double discount of a range-dependent weight on a range-marginal matrix: deployed layer, chance-corrected layer, range-conditional matrix, no discount, exact tables, in the standard and in a structural-degradation world (Table 8, Secs. V-B, VII-B).
- `exp_reliability_classes.py` — binned reliability decomposition for all four threat levels (Table 9, Sec. VII-D).
- `exp_replication_intervals.py` — intervals that resample the replications for the binned reliability of P(T=high), and the pipeline against the CPT oracle on the same resampled replications (Secs. VII-D, VII-H).
- `exp_ece_bins.py` — ECE against the bin count (Sec. VI-D).
- `exp_estimator_consistency.py`, `world_nonmonotone.py` — what the constrained estimator is and what it converges to, MAP-EM behavior, and the projection level of the ICI family (Proposition 1, Secs. IV-A, IV-C, IV-D, Appendix A).
- `exp_prior_ess.py` — adversarial (reversed-monotonicity) prior stress test, full ESS range, regime maps at ESS 8 and 50, ordinal-ICI failure mode under a reversed prior (Sec. VII-E; Supplementary Material Sec. S2, Figs. S1, S2). `exp_ess_sensitivity.py` repeats the headline comparison at other ESS values (Sec. VII-B). `exp_prior_deployed_ess.py` runs the reversed prior at the deployed alpha_0 = 2; it was added after the main results (Sec. VII-E).
- `exp_correlated.py` — correlated failures via a shared nuisance root with observed / noisy-observed / latent variants; reliability miscalibration ±50% and far-heavy context-shift transfer; P(e) monitoring (Sec. VII-F; Fig. 5). `exp_jam_scaling.py` extends the nuisance parent to all four reports with |W| = 2 to 4 (Sec. VII-F).
- `exp_misspecification.py`, `world_misspec.py`, `exp_additivity.py` — structural misspecification and test-time shift, and the additivity and monotonicity tests on the training data with the adaptive rule (Table 10, Sec. VII-G).
- `exp_latency.py` — per-track inference cost, deployment assembly, and supervised-vs-MAP-EM grounding cost (Secs. V-C, VII-B).
- `exp_sensitivity.py` — one-way linear-fractional sensitivity (tornado) + a one-at-a-time envelope for the worked-track evidence (Sec. VII-I).
- `exp_worked_track_seeds.py` — the worked track over seeds 0-19, which redraw the training data and the prior, with and without the RF report, and the table cells that actually move its posterior (Secs. VII-A, VII-I).
- `exp_comparators.py` — the comparators' free values, added after the main results: the prior mean crossed with the structure at several ESS and BDeu totals, B0's flatness, B5's L2 strength, one correct-report probability per sensor, the CNN's errors under the scenarios' mix, the world by class and emission, emission acting on drones only, N=10 under S1, and the ESS against the prior's direction in S1 (Secs. VI-A, VI-C, VII-B, VII-G, VII-H, Supplementary Material Sec. S5).
- `exp_reason_checks.py` — controls behind sentences that give a reason, added after the main results: the rule of the base framework on RF silence, how concentrated the softmax is, the binary-context weight of DroneDetect, the additivity test by training size, the fitted weights over calibration splits and scaled, the RF channel's upper bound, the softmax split on what the scenarios draw, the root marginals, the adverse subset under correlated failures, the no-RF and hand-set rows of Table 6, the hand-set table at diagonal 0.75 and the sweep of its diagonal (Supplementary Material Sec. S4, Table S2), a network with a uniform prior mean against the rule and the rule's T>=medium reading, the worked track with the data and the prior redrawn apart, the additivity test's false alarms at N=10 and 20, and the reversed-prior grid of Table S1 (Secs. VI-A, VII-A, VII-B, VII-E to VII-H, Supplementary Material Secs. S3, S4).

Real-data studies (data not included, see below):
- `prepare_dronerf_four.py` — builds `droneRF_four.npz` from the DroneRF recordings (Secs. VI-E, VII-C).
- `exp_rf_e2e.py` — real-sample end-to-end DroneRF study: segment-level 60/15/25 split (H/L band pairs kept together), validation-checkpointed CNN, held-out spectrograms passed through the pipeline at test time, hard vs. scaled-likelihood-softmax variants (Secs. VII-C, VII-H).
- `exp_rf_independent.py` — the same channel under stricter splits (segment group, time block, held-out flight mode), on DroneDetect under real interference, and high-threat reliability with real RF outputs (Table 11, Sec. VII-H). Split functions are copied from github.com/song-lalala/drone-rf-id; `--check-t3` compares them with that repository's originals. `exp_rf_type_id.py` reads its saved outputs for AR-versus-Bebop identification.
- `exp_rf_control.py` — the mixed-world study against a control with no RF channel (Sec. VII-H). It needs no recordings: no spectrogram enters the control, and the real-RF variants are read from `results/expI_e2e_raw.csv`. It also gives the contrasts with each CNN counted once.
- `exp_eo_independent.py`, `export_boxes.m` — the gate on a public thermal-infrared set for the EO/IR node (Key results, EO/IR gate; the paper does not use the set). **The main mixed-world run of this script was not performed:** a rule fixed before the results were seen required that the background alone not identify the class, and it did (see Key results), so `--stage main` stops. `results/expO_g2_run1/` keeps the first run of the gate, which started at 10:47 on 2026-11-06 under the accuracy limit alone and ended at 10:59. The balanced-accuracy limit was added at 10:49, while that run was still preparing its data and before any of its output had been read. Under the accuracy limit alone the gate would have passed (`pass: True` in that folder's log); the second run, under both limits (10:59 to 11:10), failed and is the one reported.
- `rf_confusion.py` — an earlier RF calibration with the same CNN as `exp_rf_e2e.py` on a naive random 70/30 split over segments and bands, with the checkpoint also chosen on its test part: the 84.9% that the paper reports only for comparison with common practice. The paper's other RF numbers come from `exp_rf_e2e.py` and `exp_rf_independent.py`. Running `rf_confusion.py` overwrites `results/rf_confusion.png`, the file of Fig. 4, with its own matrix; `replot_rf_confusion.py` restores that file.
- `results/rf_cm_splitA.csv`, `results/rf_cm_splitB.csv`, `results/expG_splits_run.log` — the test set of that naive split divided into two halves of 681 spectrograms each, as row-normalized matrices with add-0.5 smoothing. The code that made them is not in this repository; `check_manuscript_numbers.py` reads them to recount the 84.9%.
- `results/expG_train.log` and `results/expG_run.log` keep what the script printed at the time: "session-level" and "recording-level" there mean this segment-level split. `results/expN_run.log` says "five-fold check" for what `results/expN_summary.md` calls Proposed(N=10) - MLE(N=50). `results/RESULTS.md` holds the first results of July 2026, not the values of the paper.

Plots (no experiment runs):
- `replot_scarcity_strip.py` (Fig. 3), `replot_rf_confusion.py` (Fig. 4), `replot_print_size.py` (Figs. 5 and S3), `replot_figs.py` (Supplementary Material Figs. S1 and S2). `replot_fig3.py` is an earlier version of Fig. 3.

- `check_manuscript_numbers.py` — consistency check, see below.

## Tables and figures

| In the paper | Script | Result file in `results/` |
|---|---|---|
| Table 3 | `model_spec.py`, `param_counts.py` | the specification is the code |
| Tables 4, 6 | `exp_significance.py` | `expS_table_stats.csv`, `expS_scarcity_raw.csv`, `expS_fusion_raw.csv` |
| Table 5 | `exp_baselines.py` | `expB_baselines_raw.csv` |
| Table 6, the no-RF and hand-set rows | `exp_reason_checks.py` | `expRS_fusion.csv` |
| Table 7 | `exp_significance.py`; `exp_calibsize_ci.py` repeats these cells alone | `expS_table_stats.csv`, `expS_calibsize_raw.csv` |
| Supplementary Material Sec. S4 (the hand-set table at 0.75) and Table S2 | `exp_reason_checks.py` | `expRS_handset.csv` |
| Table 8 | `exp_reliability_variants.py` | `expD_runs_raw.csv` |
| Table 9 | `analyze_subsets.py` (top block), `exp_reliability_classes.py` (lower block) | `expS_highclass_reliability.csv`, `expR_reliability_classes.csv` |
| Table 10 | `exp_misspecification.py` | `expW_raw.csv`, `expW_judgment.csv` |
| Table 11 | `exp_rf_independent.py` | `expI_node_dronerf.csv`, `expI_e2e_raw.csv`, `expI_dd_node.csv`, `expI_dd_reliability.csv` |
| Fig. 3 | `replot_scarcity_strip.py` | `exp1_scarcity_strip4.png` from `expS_scarcity_raw.csv` |
| Fig. 4 | `replot_rf_confusion.py` | `rf_confusion.png` from `expG_cnn.npz` |
| Fig. 5 | `replot_print_size.py` | `fig_expF_corr_print.png` from `expF_corr_raw.csv` |
| Figs. S1, S2 | `replot_figs.py` | `fig_expE_regime.png` from `expE_prior_raw.csv`, `fig_expE_ess.png` from `expE_ess_raw.csv` |
| Fig. S3 | `replot_print_size.py` | `fig_expT_tornado_print.png` from `expT_summary.md` |
| Sec. IV-A, IV-C, IV-D, Appendix A | `exp_estimator_consistency.py`, `world_nonmonotone.py` | `expC_summary.md` |
| Sec. IV-D (what the data identify), Sec. VII-B (latent posteriors) | `exp_latent_identifiability.py` | `expEM_summary.md`, `expEM_rank.csv`, `expEM_reach.csv`, `expEM_recoding.csv`, `expEM_latent_raw.csv` |
| Sec. VI-D (ECE and the bin count) | `exp_ece_bins.py` | `expK_ece_bins.csv` |
| Sec. VI-C, VII-B (rule of the base framework) | `exp_base_rule.py`, `exp_base_rule_worlds.py` | `expB6_summary.md`, `expB6w_summary.md` |
| Sec. VII-A, VII-I (seeds, cells) | `exp_worked_track_seeds.py`, `exp_sensitivity.py` | `expTS_summary.md`, `expTS_seeds.csv`, `expTS_cells.csv`, `expTS_rf_report.csv`, `expT_summary.md` |
| Sec. VII-B (per-range, oracle and calibration-size breakdowns) | `exp_reliability_variants.py` | `expD_summary.md` |
| Sec. VII-D, VII-H (replication intervals) | `exp_replication_intervals.py` | `expBS_summary.md`, `expBS_intervals.csv`, `expBS_paired.csv` |
| Sec. VII-F | `exp_correlated.py`, `exp_jam_scaling.py` | `expF_summary.md`, `expJ_summary.md` |
| Sec. VII-G | `exp_misspecification.py`, `exp_additivity.py` | `expW_summary.md`, `expW_add_summary.md` |
| Sec. VII-H | `exp_rf_e2e.py`, `exp_rf_independent.py` | `expG_summary.md`, `expI_summary.md` |
| Sec. VII-H (control with no RF channel) | `exp_rf_control.py` | `expRC_summary.md`, `expRC_contrasts.csv`, `expRC_per_cnn.csv` |
| Sec. VII-E (reversed prior at alpha_0 = 2) | `exp_prior_deployed_ess.py` | `expED_summary.md`, `expED_raw.csv` |
| Secs. VI-A, VII-A, VII-B, VII-E to VII-H (controls behind sentences that give a reason), Supplementary Material Secs. S3, S4 and Tables S1, S2 | `exp_reason_checks.py` | `expRS_summary.md`, `expRS_b6_silent.csv`, `expRS_b6_control.csv`, `expRS_softmax.csv`, `expRS_transfer.csv`, `expRS_flags.csv`, `expRS_weights.csv`, `expRS_rf_bound.csv`, `expRS_split.csv`, `expRS_rel_scale.csv`, `expRS_roots.csv`, `expRS_adverse.csv`, `expRS_fusion.csv`, `expRS_rule_b3.csv`, `expRS_track_split.csv`, `expRS_flags_small.csv`, `expRS_lambda.csv`, `expRS_handset.csv` |
| Secs. VI-A, VII-B, VII-G, VII-H (the comparators' free values, emission acting on drones only), one number per sensor (Supplementary Material Sec. S4), Supplementary Material Sec. S5 and Tables S3, S4 | `exp_comparators.py` | `expCM_summary.md`, `expCM_factorial.csv`, `expCM_b0.csv`, `expCM_b5.csv`, `expCM_scalar.csv`, `expCM_node_errors.csv`, `expCM_world.csv`, `expCM_sens.csv`, `expCM_s1.csv` |
| Supplementary Material Sec. S2 | `exp_prior_ess.py` | `expE_summary.md`, `expE_prior_raw.csv` |
| Sec. VII-B (constraints under a noisy prior) | `exp_constraints.py` | `exp_constraints.csv`, `exp_constraints_raw.csv` |

Tables 1 and 2 (variables, notation) and Figs. 1 and 2 (schematics) are not results.
The `table` column of `expS_table_stats.csv` keeps earlier keys: `T3` is Table 4, `T4` is Table 6, `T5cs` is Table 7, and `T5abl` is the ablation path of Sec. VII-B.

## Run
These print their options or test themselves without running an experiment:
```
pip install -r requirements.txt        # or requirements.lock for exact versions
python cuas_bn.py
python exp_base_rule.py --selftest
python exp_jam_scaling.py --selftest
python replot_scarcity_strip.py --selftest
python replot_print_size.py --selftest
python replot_rf_confusion.py --selftest
python exp_significance.py --help      # every exp_*.py that takes options has --help
python check_manuscript_numbers.py
```
Full runs:
```
python exp_significance.py     # ~40-60 min; Tables 4, 6, 7, data of Fig. 3, Secs. VII-B, VII-D
python analyze_subsets.py      # after exp_significance.py; top block of Table 9
python exp_calibsize_ci.py     # the calibration-size cells of Table 7
python exp_ece_bins.py         # Sec. VI-D
python exp_constraints.py      # ~2 min
python exp_prior_ess.py        # ~1-2 h;  Sec. VII-E, Figs. S1, S2
python exp_ess_sensitivity.py  # Sec. VII-B
python exp_prior_deployed_ess.py       # ~4 min; Sec. VII-E
python exp_correlated.py       # ~20 min; Sec. VII-F, Fig. 5
python exp_mapem.py            # ~15 min; latent-layer study
python exp_latent_identifiability.py   # Secs. IV-D, VII-B
python exp_baselines.py        # ~4 min;  standard + realistic-expert baselines
python exp_noise_level.py      # ~5 min;  tau_T sweep
python exp_latency.py          # ~15 s;   inference-cost measurement
python exp_reliability_classes.py  # ~30 s;   per-class reliability
python exp_replication_intervals.py    # Secs. VII-D, VII-H
python exp_sensitivity.py      # ~1 min;  Sec. VII-I tornado
python exp_worked_track_seeds.py       # Secs. VII-A, VII-I
python exp_estimator_consistency.py
python exp_reliability_variants.py
python exp_misspecification.py
python exp_additivity.py
python exp_base_rule.py
python exp_base_rule_worlds.py --stage flags
python exp_base_rule_worlds.py --stage main
python exp_jam_scaling.py --stage main
python exp_jam_scaling.py --stage timing
python exp_reason_checks.py    # after the runs it reads; its softmax part reads the CNN outputs of the real-data runs
python exp_comparators.py --workers 20    # after the runs whose rows it reproduces; its CNN-error part reads expI_node_dronerf.csv
python replot_scarcity_strip.py; python replot_rf_confusion.py; python replot_print_size.py; python replot_figs.py
```
Runs that need the external data below:
```
python prepare_dronerf_four.py --raw /path/DroneRF --out /path/droneRF_four.npz --order data_derived/droneRF_four_order.csv
python exp_rf_e2e.py --train /path/droneRF_four.npz    # CNN, segment-level split
python exp_rf_e2e.py                                   # Secs. VII-C, VII-H
python exp_rf_independent.py --cache /path/cache --build-dd-zip /path/DroneDetect_V2.zip
python exp_rf_independent.py --cache /path/cache --dronerf /path/droneRF_four.npz --workers 7
python exp_rf_type_id.py
python exp_rf_control.py                               # no recordings needed; Sec. VII-H
python exp_eo_independent.py --data /path/data --boxes BOXES.csv --manifest data_derived/manifest_sha256_v1.0.0.csv --stage g1
python exp_eo_independent.py --data /path/data --boxes BOXES.csv --manifest data_derived/manifest_sha256_v1.0.0.csv --stage g2
```

## External data
None of the recordings below are redistributed with this repository; download
them from their sources and follow the terms stated there.

| Dataset | Source | Needed by |
|---|---|---|
| DroneRF | Allahham et al., *Data in Brief* 26:104313, 2019; https://data.mendeley.com/datasets/f4c2b4n755 (CC BY 4.0) | `prepare_dronerf_four.py`, which makes the input of `exp_rf_e2e.py --train`, `exp_rf_independent.py`, `rf_confusion.py` |
| DroneDetect (V2) | Swinney and Woods, IEEE DataPort, 2021, doi:10.21227/5jjj-1m32 (CC BY 4.0) | `exp_rf_independent.py` |
| Thermal-infrared clips | Svanström et al., *Data in Brief* 39:107521, 2021; github.com/DroneDetectionThesis/Drone-detection-dataset, tag v1.0.0 (CC0-1.0) | `exp_eo_independent.py` |

- DroneRF data: `prepare_dronerf_four.py` processes the recordings into `droneRF_four.npz` with arrays `X` (N,128,128) float32 log-spectrograms, `y` (N,) int64, `class_names` (4,), and `meta` (N,2) carrying the source CSV file and chunk of each spectrogram (used for the segment-level split). Each file gives up to 10 chunks of 1,000,000 samples; each chunk goes through `scipy.signal.stft` with a 512-sample Hann window every 2048 samples (one-sided), 20 log10(|Z| + 1e-8), a bilinear resize to 128 x 128, and zero mean and unit standard deviation per spectrogram. `data_derived/droneRF_four_order.csv` holds the row order of the file used here, which followed parallel completion and not the file names; the split indices in `results/` refer to those rows. Rebuilt with the versions of `requirements.lock`, 4,539 of the 4,540 spectrograms equal that file bit for bit, and the other differs in one cell by one float32 step of its dB value (2e-7 after normalization; `results/expCM_dronerf_rebuild.txt`).
- The infrared boxes come from the MATLAB label files of that dataset through `export_boxes.m`. So that MATLAB is not needed, `data_derived/ir_boxes_v1.0.0.csv.gz` holds its output (gunzip it to get `BOXES.csv`); `data_derived/manifest_sha256_v1.0.0.csv` lists the SHA-256 of the dataset's files, checked at the start of every stage.
- The `.npz` files in `results/` hold CNN outputs, labels and split indices computed from these datasets, not recordings.
- Every other experiment runs self-contained from this repository.

## Reproducibility tolerance
All reported numbers are means over independent replications of Monte-Carlo
simulations. One independent re-run on another machine, of an earlier
version of the code, differed from the reported values by about 0.001 to 0.004;
we have no other measurement across machines. Seeds
are fixed in the scripts and exact library versions are pinned in
`requirements.lock`. Console note: runs print numpy warnings (`overflow
encountered in exp`, `divide by zero encountered in log`) in the thousands.
Both are benign: the first is the ordinal-response logistic saturating in
`_oi_table`, whose row is immediately clipped to `1e-9` and renormalized, and
the second is the convex solver evaluating a log at a boundary point. The
tally that matters is in `results/expS_solver.txt`. Solver note: a constrained
solve counts as solved if its status is optimal or optimal-inaccurate; otherwise it
falls back to the closed form. The runs keep only the tally (main study: 1,100
solved, no fallback), not each status.

### Checking a re-run against the paper
`check_manuscript_numbers.py` compares what the scripts produced against
what is quoted in the paper. Run it from this directory:

```
python check_manuscript_numbers.py
```

Without the paper's LaTeX sources it verifies `results/` alone (that the
files exist, and that the results agree with each other and with the conditions
the paper states) and says so. A clone gives every file the time of the checkout,
so it compares modification times only with the sources present; after a re-run,
`CUAS_GATE_FRESHNESS=1` asks for that comparison (results newer than the code). If you also have the sources, point it at them to check every quoted
number, table and sentence as well:

```
CUAS_PAPER_DIR=/path/to/latex/sources python check_manuscript_numbers.py
```

## Evaluation protocol status
| Protocol item | Status |
|---|---|
| Scarcity, fusion (+hard-condition subsets), calibration-set size, abstract-diagonal sweep, ablation incl. (−ICI), majority baseline — 20 replications, CIs, DeLong/McNemar/bootstrap | done (`exp_significance.py`) — Secs. VII-B, VII-D |
| Constraint-violation study (noisy prior) | done (`exp_constraints.py`) — Sec. VII-B |
| Latent-layer MAP-EM (C2-iii) | done (`exp_mapem.py`) — Sec. VII-B |
| What the unlabeled case identifies; quality of the latent posteriors | done (`exp_latent_identifiability.py`) — Secs. IV-D, VII-B |
| What the constrained estimator is and what it converges to | done (`exp_estimator_consistency.py`) — Sec. IV-C, Appendix A |
| Standard baselines and realistic-expert prior | done (`exp_baselines.py`) — Sec. VII-B |
| The rule of the base framework as a baseline | done (`exp_base_rule.py`, `exp_base_rule_worlds.py`) — Secs. VI-C, VII-B |
| World noise-level (tau_T) sensitivity | done (`exp_noise_level.py`) — Sec. VII-B, Supplementary Material Sec. S1 |
| Reliability-layer variants and the double discount | done (`exp_reliability_variants.py`) — Sec. VII-B, Table 8 |
| Adversarial prior quality; full ESS range; regime maps (ESS 8 & 50) | done (`exp_prior_ess.py`) — Sec. VII-E |
| Correlated failures (observed/noisy/latent W); reliability miscalibration & context-shift transfer; P(e) monitoring; nuisance parent over all four reports | done (`exp_correlated.py`, `exp_jam_scaling.py`) — Sec. VII-F |
| Structural misspecification and test-time shift; additivity and monotonicity tests | done (`exp_misspecification.py`, `exp_additivity.py`) — Sec. VII-G, Table 10 |
| Real RF data from the sensor node to the threat posterior (segment, time-block, held-out flight-mode splits); DroneDetect under interference | done (`exp_rf_e2e.py`, `exp_rf_independent.py`) — Secs. VII-C, VII-H, Table 11 |
| Control with no RF channel | done (`exp_rf_control.py`) — Sec. VII-H |
| EO/IR node on a public infrared set | gate failed, main run not performed (`exp_eo_independent.py`) — Key results, EO/IR gate |
| One-way sensitivity (tornado) + one-at-a-time envelope | done (`exp_sensitivity.py`) — Sec. VII-I |
| Worked track over seeds that redraw the training data and the prior, with and without the RF report; cells that move its posterior | done (`exp_worked_track_seeds.py`) — Secs. VII-A, VII-I |
| The comparators' free values; one number per sensor; emission acting on drones only | done (`exp_comparators.py`), added after the main results — Secs. VII-B, VII-G, VII-H, Supplementary Material Secs. S4, S5 and Table S5 |
| Reversed prior at the deployed ESS; controls behind sentences that give a reason | done (`exp_prior_deployed_ess.py`, `exp_reason_checks.py`), added after the main results — Secs. VII-B, VII-E to VII-H, Supplementary Material Table S5 |
| Replication-level intervals for the reliability bins | done (`exp_replication_intervals.py`) — Secs. VII-D, VII-H |

## Key results (means ± 95% CI over 20 independent replications; 2500-scenario test draws; oracle acc 0.593 / AUC 0.824 / ECE 0.017; majority 0.434)
These hold in the evaluation world of Sec. VI-A, given labeled latent states and a prior near the truth unless stated otherwise.
- **Scarcity (equal label budgets):** N=10 is not significantly worse than unconstrained MLE at N=50 on any of six metrics and better on macro-AUC (0.793 vs. 0.761, p<1e-3) and, at the margin, on ECE (0.077 vs. 0.107, p=0.05, Wilcoxon 0.053) and FAR (p=0.05, Wilcoxon 0.070); accuracy differs by +0.001 [-0.022, +0.025], Brier by -0.016 [-0.047, +0.016]. The same tests do not separate N=10 from MLE at N=100 on five of the six metrics, and the prior alone (B2) is not significantly worse than MLE at N=50 on accuracy, Brier, ECE or FAR, so this comparison mainly measures the prior; Supplementary Material Sec. S5 replaces it by a weak uniform one. At N=50 the proposed model beats MLE at N=100 on macro-AUC (+0.014) and ECE (−0.023), but MLE at N=200 overtakes it. The residual advantage at equal N persists to N=400 (ΔECE −0.008, p=7.9e-4), and the operationally meaningful gains are confined to N ≲ 50. The proposed model beats the prior-only B2 at every N.
- **Standard baselines:** uninformative-mean MAP at the same ESS, a heavy setting for a uniform mean, reaches ECE 0.143 at N=10, worse than unsmoothed MLE (0.111), and the proposed model beats it by +0.122 macro-AUC and −0.125 Brier (both p<1e-7). A lighter uniform prior closes the ECE gap (alpha_0=0.5: p>=0.06 at every N), but no ESS from 0.01 to 2 and no BDeu total from 1 to 20 closes the macro-AUC or Brier gap (p<1e-3; Table S3). L2 logistic regression trails at every size; at C<=0.1 its ECE gap at N=50 closes (p>=0.46), and no C from 0.001 to 10 closes the macro-AUC or Brier gap. The coarse B0 table used as the prior mean beats unsmoothed MLE at every N on accuracy, macro-AUC and Brier, but at N=10 it trails MLE at N=50 on Brier (+0.019, p=0.02). B0 is flat (tau 0.95/0.9/0.95/1.0 for H/N/C/T against the world's 0.28 and 0.40); Supplementary Material Sec. S5 and Table S4 vary its flatness.
- **Ablation path (N=80, one change per step, one order, alpha_0=2):** uniform mean +0.015 ECE (p=0.15, n.s.) -> elicited prior mean -0.051 (p=6.3e-6) -> monotonicity constraints -0.001 -> ordinal ICI node -0.008 (p=4.7e-3); the steps sum, to rounding, to the -0.044 total. The prior mean crossed with the structure, with a weak uniform prior chosen after the main results, is in Supplementary Material Sec. S5 and Table S3. The `expert MAP`/`expert cMAP` variants in `exp_baselines.py` exist so that each delta belongs to exactly one change.
- **The rule of the base framework (T=high):** the published rule ranks at AUC 0.752, and its output is not a probability; rescaling brings the Brier score from 0.405 to 0.079 (0.082 for a constant forecast at the base rate 0.091). The proposed model (AUC 0.876–0.890) is ahead of the published, rescaled and fitted forms at every N on both AUC and Brier (p≤0.007). On radio-silent drones the published rule's mean output is 0.853 against 0.175 for the CPT oracle and 0.178 for the proposed model at N=200; its hand rule on silence explains only part of that (0.537 with that input off, 0.240 observed). At N=10,000 the fitted form still trails (0.846 against 0.893) but nearly catches up (0.862 against 0.865) in the S1 strong world. Supplementary Material Sec. S4 sets the rule under a hand-set prior, against a network with a uniform prior mean (B3 does not beat the published rule on AUC at N≤20), in its P(T>=medium) reading (the rule's scores above 0.75 are right 67% of the time) and in the S1 worlds.
- **World noise:** rebuilt at tau_T in {0.25, 0.40, 0.60} (oracle accuracy 0.629/0.593/0.507), the grounded model beats unsmoothed MLE on macro-AUC and Brier at every training size tested (all p≤0.003), with an ECE margin that grows with noise (-0.042/-0.055/-0.088 at N=50). Three gaps at tau_T=0.25 are not significant. The proposed model at N=10 keeps its macro-AUC lead over MLE at N=50 in all three worlds, but not its ECE lead at tau_T=0.25; each world's expert prior reads that world's tau_T (Supplementary Material Sec. S1).
- **Worked track:** the RF "none" barely lowers the drone posterior (0.98 to 0.96) and raises P(silent) from 0.45 to 0.83, so it raises P(T=high) from 0.52 to 0.63 (true world 0.60 to 0.76); the abstract sensor model ignores emission, so its P(T=high) barely moves with or without that report (0.525, 0.526).
- **Fusion:** calibrated four-modality fusion beats the best single sensor on accuracy, F1, AUC and Brier, though not on ECE. Its RF channel, read through the latent emission state, adds information over the same fusion without RF. With the fitted discount on both sides, the deployed RF channel adds +0.003 macro-AUC and −0.005 Brier overall, with accuracy and ECE unresolved (p=0.65, 0.40), and on radio-silent drones −0.024 Brier (p<1e-5) with macro-AUC −0.007 (p=0.06; −0.010, p=0.008, without the discount). The abstract sensor model, our stand-in (the base framework has no report-level sensor model), ignores emission and is worse than no RF channel on radio-silent drones: its accuracy there is 0.503, and with no RF channel the radio-silent accuracy is 0.592 (calibrated +0.001, p=0.78). Calibrated fusion beats it on accuracy, AUC, Brier, ECE and FAR (macro-F1 +0.006, p=0.07; AUC 0.816 vs. 0.808; ECE 0.032 vs. 0.043; FAR 0.466 vs. 0.470, p=0.03; oracle FAR 0.455), while on non-drones the abstract model leads (calibrated accuracy −0.008, macro-AUC −0.005, Brier +0.007, p<1e-3); both subsets condition on the latent class. Against a hand-set emission-aware table, estimated matrices without the discount gain overall (+0.005 macro-AUC, −0.008 Brier, −0.016 ECE, all p<1e-5) but not on radio-silent drones (accuracy −0.020, macro-AUC −0.008, Brier +0.037, all p≤0.014). That table's diagonal, 0.94, lies far above the sensors' estimated accuracy (mean correct-report probability 0.69, 0.71, 0.76 for radar, EO/IR, acoustic). Swept from 0.55 to 0.99 (Table S2), it ties the estimated matrices at 0.70 to 0.80 on accuracy, macro-AUC and Brier (p≥0.18; at 0.75 its ECE is lower by 0.003, p=0.008) and loses on Brier elsewhere (p≤0.008), so estimating supplies the level without choosing it. From 0.75 up the table leads on radio-silent drones (accuracy, Brier, p≤0.007) but trails on non-drones (Brier +0.002 at 0.75, +0.019 at 0.94, p≤0.03), and among all scenarios with an RF report of none it ties at 0.75 and 0.80 and trails above them (Brier, p≤0.004), so that lead comes from conditioning on the latent class. The RF row of that table, the table at diagonal 0.75 (the best of the sweep, chosen after the main results), one correct-report probability per sensor and the emission-aware RF row at one level are in Supplementary Material Secs. S4 and S5. Over the abstract model's own diagonal sweep 0.55–0.99 the AUC and Brier gaps of calibrated fusion hold at every value (p<1e-9), while the ECE gap is not significant at 0.60, 0.65 and 0.70.
- **Reliability (fitted):** little change on average (ECE p=0.05); it helps on the far-range subset (Brier p<1e-3, ECE p=0.008) and hurts radio-silent drones (Brier p<1e-3). On the true threat-layer CPTs the deployed layer is less confident than the exact sensor tables (p=0.011), a double discount; the chance-corrected layer, which removes it, fails the adoption rule for the standard world (Brier unchanged, p=0.22, ECE +0.0038, p=0.012, near- and imminent-range ECE does not fall), and the structural world meets none of its conditions (Brier +0.0016, p=0.002, ECE +0.0044, p=0.005), so the fitted weight of Eq. (10) stays as an empirical discount: it takes the place of P(reliable | context) in the network, but its fitted value is not an estimate of that probability. Its small average effect (Brier −0.0010, ECE −0.0020) comes with a radio-silent cost (+0.0049); on the true CPTs the ECE part vanishes (p=0.70), the same offset of the threat layer's over-confidence, and the Brier part stays (−0.0008, p<1e-3). ±50% miscalibration moves ECE by at most 0.004, but halving the weights costs 0.008 Brier and 0.005 macro-AUC (both p<1e-8) and a quarter of the fitted value raises ECE by 0.012 (p=0.006), so ECE alone understates what a mis-set weight costs; far-heavy deployment keeps the benefit (0.031 vs. 0.036). The fitted weights follow the direction of the world's range degradation but not its size, and they vary with the calibration split (radar's far range is its highest in 4 of 20 splits; RF, with no range effect, spans 0.889 to 0.98).
- **Constraints & ICI:** the constraints restore admissibility and are inert under a good prior; they help under a strongly, but not completely, reversed prior (up to +0.063 ECE and +0.030 Brier), hurt under a half-reversed one, and under a fully reversed prior at small N drive the estimate beneath an evidence-free reference. The 9-parameter ordinal-ICI threat node beats the 54-parameter table on both axes at N=80 (ΔECE +0.008, ΔAUC −0.004, both significant, in favor of ICI). Under a reversed prior it collapses toward uninformative predictions (accuracy 0.408, Brier 0.670–0.676), which is a failure, even though it is conservative. At the deployed alpha_0 = 2 the proposed model under the fully reversed prior trails MLE at N=50 by 0.066 accuracy and 0.041 Brier; accuracy recovers by N=200, but Brier and ECE stay worse at N=400 (+0.010, +0.018). With the right direction but half the true effects (lambda=0.25), the model at N=10 keeps a macro-AUC lead over MLE at N=50 (+0.021) but trails by 0.036 accuracy, 0.058 Brier and 0.064 ECE (p≤0.02), though at N=50 it still leads MLE at N=50 on macro-AUC, Brier and ECE (p≤0.025; Table S1).
- **MAP-EM (latent H,N,C):** prior-only 0.512/0.780/0.103 → EM 0.564/0.804/0.046 at N=200; fully observed 0.577/0.817/0.030. Without labels the data fix the tables only up to recodings of the latent states (Jacobian rank 112 of 126 at the true tables, 72 of 72 if the threat table is known). Threat-level metrics survive (0.01 to 0.02 in accuracy and ECE), the latent layer does not: at N=200 the top state of H is right 80% of the time against 87% with labels (79% for the prior alone, 88% with the true tables).
- **Correlated failures:** naive conditional-independence fusion degrades to adverse-subset ECE 0.070 at induced error correlation ρ=0.26 (mean confidence stays near 0.60 while accuracy falls from 0.58 to 0.54; kinematic and context evidence alone holds 0.53 at every severity); the nuisance parent restores 0.039 observed / 0.043 noisily observed / 0.064 never observed (all significant; +15 parameters). Over all four reports the cost is 27(|W|−1) parameters and the mean update stays under 0.14 ms; at |W|=4 the gain depends mainly on knowing the nuisance state at run time (0.031 with W observed, 0.007 marginalized). This jamming model is an assumption of the synthetic world.
- **Misspecification and shift:** the model breaks when its own assumptions fail (a non-additive threat response, S1; a wrong monotonicity constraint, S2), and the result at N=10 does not survive S1: against MLE at N=50 the proposed model loses 0.052 accuracy and 0.052 Brier in weak S1 and 0.290 and 0.284 in strong S1 (p<=0.009). With enough data both failures show in the training data: a likelihood-ratio test flags the ICI node in every S1 replication at N=200 (at most 7 of 20 at N≤20 and, on a weak violation, at N=50) and in two of 100 standard-world replications, and the H constraint under S2 in 17 and 20 of 20 at N=50 and 200 but never in the standard world; the S2 test was measured from N=50 only. At N=50 the switch on a weak violation is worse than the full table on Brier (+0.0055, p=0.007) and than the fixed model on ECE (+0.0102, p=0.024). In strong S1 at N=200 unsmoothed MLE beats the constrained full table at an ESS of 2 per column whatever its mean (elicited +0.012, uniform +0.008 Brier; Supplementary Material Sec. S5 lowers the ESS). Under S3 (missing edges) no estimator within the graph recovers the loss. Under S4 the model degrades only when silent targets dominate (N=200) or the context weight doubles. With emission acting on drones only, the proposed model holds at both sizes (Supplementary Material Sec. S5).
- **Real RF data:** a segment-level split gives CNN test accuracy 81.4% (a naive random split gives 84.9%); this is within-recording generalization. Through the pipeline the measured channel beats the abstract RF model by +0.020 macro-AUC (p=8.7e-15) and −0.033 ECE, but that model is worse than no RF channel at all (−0.016 macro-AUC, +0.031 ECE). Over a control with no RF channel the measured channel adds +0.0037 to +0.0045 macro-AUC and −0.0059 to −0.0071 Brier across the splits (p<1e-7 over replications, p<=0.014 with each CNN counted once), with accuracy and ECE unresolved over replications (p>0.10). Background, recorded once, is never held out by recording, and each split tests it on 5 to 10 segments (100 to 200 spectrograms). Under the scenarios' mix (81% background) drone-model confusions are 45 to 64% of the CNN errors per split type (77 to 83% of the test pool) and drone/background errors 1.7 to 4.9% of scenarios (8.6% in the hardest fold). Soft evidence adds little because only the softmax's mass on background moves P(T) and the hard report already takes most of what that split can give: an error-free split adds +0.0055 macro-AUC and -0.0083 Brier over the control, only +0.0009 and -0.0014 over the hard report. On what the scenarios draw, the top probability averages 0.86 to 0.93 across runs and the split falls below 0.9 in 9% to 32%. On DroneDetect, discounting by the interference condition improves the RF likelihood under Wi-Fi and under both sources. Threat labels in this mixed world are synthetic.
- **EO/IR gate:** on the public infrared set, with the target masked, the background alone identified the class across recording sessions (balanced accuracy 0.41, session-bootstrap interval [0.35, 0.48], and 0.40, [0.34, 0.45], against 0.25 for no information and a limit of 0.35 that both intervals reach), so the set was not used to calibrate the node.
- **Cost:** a track update costs 0.106 ms on one desktop CPU core (95th percentile 0.126 ms, about 9,500 updates/s); the latent-layer MAP-EM fit costs 4.3x the supervised grounding offline and nothing at inference.
- **Per-class reliability:** every threat level is over-confident in the (.50,.75] bin (negligible -0.056, low -0.022, medium -0.016, high -0.050); the largest deviation anywhere is the high level's +0.069 in the top bin. Level names come from `model_spec.STATES["T"]`, never hardcoded. Intervals that resample the replications keep the two decision-relevant deviations of the high level away from zero and contain zero for every oracle bin; in the mixed world they leave two of 25 bins outside for the pipeline and two of 25 for the CPT oracle. Paired with the oracle on the same replications, the signed deviation differs in (.50,.75] under all five splits (+0.046 to +0.057) and in (.25,.50] under four, but in the top bin under none; its size differs in none of the 25 bins.
- **Sensitivity:** the largest single-cell swing of the worked-track P(T=high) is 0.14; the top-5 one-at-a-time envelope [0.565, 0.700] (baseline 0.633) is not a joint credal propagation. Of the 183 table cells 62 move that posterior at all (25 by more than 0.001). Over seeds 0-19, which redraw the training data and the prior, it ranges from 0.633 to 0.832 (mean 0.730; the data alone 0.632 to 0.753, the prior alone 0.561 to 0.746), always above 0.5; seed 0, used here, is the lowest.

## Caveats
The synthetic evaluation runs in one fully specified generative BN, the
evaluation world of Sec. VI-A; absolute accuracies reflect its chosen
aleatoric-noise regime. The world and the model share monotone, nearly additive
dependence, so the comparative conclusions hold when the domain's
monotone-additive structure holds: Sec. VII-G maps where they break, and
Sec. VII-E what a wrong prior costs. Of the four modalities, only the RF
likelihood is grounded in real measurements (DroneRF, DroneDetect; Secs. VII-C,
VII-H); DroneRF's background and Phantom classes, recorded once each, cannot be
held out by recording. The EO/IR, acoustic, and radar likelihoods are estimated in the evaluation world. Supervised training rows carry the threat
level, the three latent states, and the true class and emission state, which the
table of the non-cooperative node uses; the marginals of class and emission come from the generative
specification, and counting them on the calibration split changes no metric by
more than 0.002. Rules
described as fixed in advance were written into the scripts or our plan before
the runs; they were not registered externally, and the repository's history
does not date them.

## Citation
```bibtex
@misc{song_cuas_bayesian_threat,
  author = {Song, Moogeun},
  title  = {cuas-bayesian-threat: statistically grounded, sensor-calibrated
            Bayesian network for counter-UAS threat assessment},
  year   = {2026}
}
```

## License
Released under the MIT License. See `LICENSE`.
