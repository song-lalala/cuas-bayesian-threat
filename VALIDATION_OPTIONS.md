# Validation-world designs — chosen + deferred options

The experimental validation must use a **single coherent "world"**: whatever
generates the ground-truth labels (class, intent, threat) must also generate the
sensor observations, so that the sensor confusion matrices are *emergent from
the same reality* rather than injected from a different source. Mixing sources
(e.g. synthetic-BN labels + AirSim sensor stats, or per-modality public
datasets) creates a two-worlds mismatch and is the incoherence we are avoiding.

Principle in all designs: **one world → labelled data (labels + kinematics +
ALL sensor outputs) → confusion matrices ESTIMATED from a labelled calibration
subset → used as BN likelihoods.** Identical procedure whether the world is
synthetic, AirSim, or real.

---

## Chosen: Option 1 — Unified synthetic world + estimated confusion matrices
One generative BN emits labels, kinematics, and all four sensor reports. The
sensor confusion matrices are **estimated from a labelled calibration split of
that same world** (Dirichlet-smoothed counts), exactly as one would from AirSim
or real logs — not read off the true CPTs. Runnable now; fully coherent;
honest. AirSim/real data are framed only as a future higher-fidelity step, not
mixed in. Implemented here (`estimate_confusion`, calibration set, sensor-
calibration-size sub-experiment).

---

## Deferred — to try later

### Option 2 — AirSim single-world + physics signal models  (most realistic)
AirSim (Unreal) is the single world:
- **From AirSim directly:** scripted scenarios give true class/intent labels;
  trajectories give kinematics (v, range-rate, altitude, bearing, CPA); camera
  gives **EO/IR imagery** → run a real detector (e.g. YOLO) → EO/IR reports and
  its empirical confusion matrix; depth/LiDAR give a 3D branch.
- **Add-on signal models driven by the SAME AirSim state** (so still one world):
  - radar **micro-Doppler** from rotor kinematics / body RCS + range;
  - **RF** from an emission/comms model (incl. radio-silent = no emission);
  - **acoustic** from motor-RPM harmonics + range/wind propagation.
- Confusion matrices estimated from the AirSim-generated labelled set.
Effort: needs Unreal + GPU; AirSim scenario scripts + 3 signal models.
Payoff: highest-fidelity, single coherent physical world, camera-realistic EO/IR.
Entry points to reuse: keep `cuas_bn`, `grounding`, `metrics`; replace
`dataio.generate_dataset` with an AirSim loader and `model_spec` sensor CPTs
with estimated ones.

### Option 3 — Partial AirSim + public datasets  (honest hybrid, per-modality)
- EO/IR + kinematics from AirSim; **RF from DroneRF/DroneDetect**, **radar from a
  micro-Doppler dataset**, **acoustic from a drone-audio dataset**.
- Each sensor node's confusion matrix estimated from its own dataset.
- **Caveat to state explicitly:** modalities come from different
  sources/worlds, so cross-modal correlations and a shared physical scene are
  not captured; treat as a per-node calibration study, not a joint-world test.
Effort: moderate (download + train per-modality classifiers, extract matrices).
Payoff: uses *real* sensor data; weaker joint-world coherence.

### Option 4 (bonus, cheap) — latent-variable MAP-EM result
Already coded (`grounding.em_internal`): run the scarcity study with H, N, C
*unobserved* (only inputs, sensors, T labelled) to demonstrate the MAP-EM
sim-to-CPT machinery as a separate figure.
