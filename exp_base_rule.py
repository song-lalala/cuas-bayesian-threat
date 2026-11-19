"""
Exp-B6: the actual scoring rule of the base framework [5] as a baseline,
and the abstract-diagonal sweep extended
below 0.70. Core files are not modified; results go to results/expB6_*.

Source of the rule
  M. Song et al., J. Korea Academia-Industrial cooperation Society 27(3),
  228-241, 2026, doi:10.5762/KAIS.2026.27.3.228 (the manuscript's [5]):
  Eq. (1) S_base = sum_i w_i s_i over the input nodes of a latent variable;
  Analyzing scores 0; under RF Silent the Protocol node is missing and leaves
  the sum (no renormalization is stated). Eq. (2) S_total = S_base + expert
  adjustments of its Table 3. Eq. (3) P(Z) = 1 / (1 + exp(-k (S_total - c))), k = 6,
  c = 0.5. Eq. (4) Noisy-OR, P(T) = 1 - prod_Z (1 - P(Z)). Eq. (5) two thresholds
  tau_low and tau_high; [5] gives no numeric values for them.
  Table 2 of [5], weights: Hostile Proximity .48, Approach .28, Acceleration .15,
  Altitude .09; Non-cooperative RF Status .40, RF Protocol .2, Trajectory .2,
  RCS .2; Context Route .5, Swarm .30, Environment .20.
  Table 3 of [5], rules: A Altitude=Masking and Approach=Fast_Appr +0.5; B
  Approach=Non_Closing and Proximity!=Critical -2.0; 1 RF=Silent and RCS=Stable
  +0.5; 2 Protocol=Standard -1.0; 3 RCS=Fluctuating -1.0; alpha Env=Adverse and
  Route=Violation +0.5; beta Swarm=Swarm +0.5; gamma Env=Favorable and
  Route=On_Route and Swarm=Single -1.0.
  Table 6 of [5], worked example: 0.470 -> 0.455, 0.600 -> 0.646, 1.200 -> 0.985, final
  0.997; reproduced by selftest(), which also reproduces the cold-start value
  0.047 of its Sec. 3.3.2.

FIXED BEFORE THE RUN (in this docstring, before any comparison is run; not registered externally)

[M] Map from the world of this study to the inputs of [5]. Source of a state
    score: T6 = Table 6 of [5]; A = assumed here (Table 2 of [5] has no state
    scores; the assumption takes the endpoints of the 0-1 range in threat order
    and 0.5 for a middle state).
    Hostile Maneuver
      Proximity (.48): d = imminent -> Critical 1.0 (A); near -> Near 0.5 (T6);
        far -> Far 0.0 (A).
      Approach Rate (.28): rdot = closing and v = fast -> Fast_Appr 1.0 (A);
        rdot = closing and v in {hover, slow} -> Slow_Appr 0.5 (T6); rdot =
        steady -> Slow_Appr 0.5 (T6), because the 0-15 m/s range of Slow_Appr
        includes 0; rdot = opening -> Non_Closing 0.0 (A). The speed v enters
        only here, since [5] has no speed input and calls approach rate the
        better indicator.
      Acceleration (.15), Altitude (.09): no counterpart, see below.
    Non-Cooperative Track
      RF Status (.40): Rf = none -> Silent 1.0 (A); consumer, mil -> Detected
        0.0 (T6).
      RF Protocol (.2): Rf = mil -> Non_Std 1.0 (T6); consumer -> Standard 0.0
        (A); none -> missing, leaves the sum as in Eq. (1).
      Trajectory (.2): no counterpart, see below.
      RCS (.2): Rr = drone -> Stable 1.0 (T6); Rr = bird, clutter -> Fluctuating
        0.0 (A), because rule 3 of [5] reads irregular RCS as a non-UAV sign.
    Context-based Threat
      Route (.5): z = nofly -> Violation 1.0 (T6); permitted -> On_Route 0.0 (A).
      Swarm (.30), Environment (.20): no counterpart, see below.
    Inputs without a counterpart (main map M2, fixed before any
    comparison, replacing the renormalization of the first draft): the world has
    no such variable, so they are held at their benign state with the original
    weights and no renormalization: Acceleration Stable 0.0 (T6), Altitude Legal
    0.0 (A), Trajectory Direct 0.0 (A), Swarm Single 0.0 (T6), Environment
    Favorable 0.0 (A). This follows [5], whose Sec. 3.3.2 sets the contribution of
    a node without data to 0.0 and whose Eq. (1) drops the missing Protocol
    without renormalizing. The weighted sums stop at Hostile 0.76, Non-cooperative
    0.8 (0.6 under Silent) and Context 0.5.
    Rules that can fire: B (rdot = opening and d != imminent), 1, 2, 3, gamma (z =
    permitted). Rules that never fire here: A (Altitude), alpha (Environment), beta
    (Swarm).
    Parents: B6-pub follows the inputs of [5], so Context has no d although the
    world's C has parents (d, z), and EO/IR and acoustic reports are unused
    because [5] has no input node for them (its RCS node mentions EO/IR fusion;
    here it reads the radar report only). B6-fit follows the world's parents and
    takes all evidence, because it is the best member of the family.
    selftest() checks the assumed scores against the qualitative levels of Table
    5 of [5] (S1, S3, S4, S5; S2 is the worked example).
    Descriptive alternatives of the map (rule constants unchanged):
      M1 inputs without a counterpart removed and the remaining weights
         renormalized per latent variable (Hostile .632, .368; Non-cooperative .5,
         .25, .25; Context 1.0), the map of the first draft; rule gamma cannot fire,
         and Route alone gives P(C) = 0.953 under nofly, a property [5] does not
         have;
      M3 M2 with Approach Rate from rdot alone (closing Fast_Appr, steady
         Slow_Appr, opening Non_Closing);
      M4 M2 with rdot = steady -> Non_Closing, so rule B fires there unless d =
         imminent.
[V] Variants.
      B6-pub    the constants of [5] with map M2, no training. Uses less
                evidence than Proposed (no Re, no Ra).
      B6-recal  B6-pub recalibrated on the calibration split (300 scenarios of
                the replication): logistic regression of T = high on the logit
                of B6-pub (clipped at 1e-6; C = 1e6). It removes the question of
                which level the P(T) of [5] means, and leaves the ranking. It
                uses T labels of the calibration split, which Proposed does not.
      B6-fit    the same functional form (a sigmoid per latent variable, then
                Noisy-OR) with the same evidence as Proposed, weights and
                constants fitted by maximum likelihood of T = high on the same
                N training scenarios:
                  logit P(H) = b + v (2) + rdot (2) + Fast_Appr (rdot = closing
                               and v = fast) + d (2) + rule B;
                  logit P(N) = b + Rf (2) + Rr (2) + Re (2) + Ra (1) + rule 1;
                  logit P(C) = b + z (1) + d (2);
                22 parameters, so the form contains B6-pub, one-hot against the first state of model_spec (v
                hover, rdot opening, d far, z permitted, Rf consumer; Rr, Re, Ra
                drone), signs free. L-BFGS-B with bounds +-20, 5 starts (slopes 0 with the
                intercepts at the base rate; 4 starts adding N(0, 0.5^2) from
                default_rng(61000 + 1000 r + N)), best likelihood kept. Without a
                positive or a negative in the training set the output is the
                constant training rate clipped at 1e-12 and counted. Unlike
                Proposed, whose training also uses the labels of H, N, C, its
                likelihood sees only T = high, because this form has no latent
                labels.
    Descriptive additions, to keep the comparison from being a straw man:
      B6-fit-r  B6-fit with a N(0, 2.5^2) prior on the slopes (intercepts free),
                since Proposed has a prior and pure maximum likelihood has none
                at small N.
      B6-fit-5  B6-fit restricted to the inputs of [5] (no Re, Ra; no d in C; 17
                parameters), so that constants against estimation can be read on
                equal evidence.
      B6-fit-p  B6-fit with a N(mu, 2.5^2) prior on the slopes, mu read from the
                constants of [5] under M2 through logit P(Z) = k (sum w s + sum a
                - c): Hostile rdot=steady .84, rdot=closing .84, Fast_Appr .84,
                d=near 1.44, d=imminent 2.88, rule B -12; Non-cooperative Rf=mil
                7.2, Rf=none 8.4, Rr=bird -7.2, Rr=clutter -7.2, rule 1 3.0;
                Context z=nofly 9.0; mean 0 for the terms [5] lacks (v, Re, Ra, d
                in C). Intercepts free, with one more start at these values and the
                intercepts of [5] (-3, -7.8, -9); selftest() checks that this point
                reproduces B6-pub on every evidence combination. It is fitted also
                without a positive in training, since the prior keeps a ranking.
      B0-as-prior MAP  exp_baselines.py path: the hand-set table of [5]'s kind as
                prior mean at the same ESS, full tables. These are not the values
                of [5] but a table of its kind. Its metrics must reproduce
                expB_baselines_raw.csv within 1e-9 at the shared N.
      Proposed-EM  exp_mapem.py path: T, inputs and reports observed, H, N, C, cls,
                emit latent, the same prior as Proposed, 12 iterations. Its metrics
                must reproduce expM_raw.csv within 1e-9 at the shared N.
      CPT-oracle  true internal CPTs with the true range-marginal sensor tables,
                the upper reference used in Table 9 of this study.
[O] Target and metrics. The output of [5] is one probability, so the comparison
    is binary. Primary target T = high, compared with P(T = high) of Proposed:
    AUC; Brier score mean (p - y)^2; ECE as defined in this study (metrics.ece:
    top-label, 15 equal-width bins) on the two-column output (1 - p, p); ece_pos,
    the same bins on p itself (descriptive); the decomposition of Table 9 of this study (bins
    (0,.10], (.10,.25], (.25,.50], (.50,.75], (.75,1]; 95% Clopper-Pearson);
    alarm precision and recall at 0.5 and 0.25 ([5] has no numeric thresholds).
    Secondary target T >= medium: AUC only (Proposed scores P(medium) + P(high));
    descriptive. Subsets, descriptive: radio-silent drones (cls = drone, emit =
    silent), radio-silent non-drones (Rf = none, cls != drone), far range.
[R] Same conditions as the main experiment (exp_significance.py): world
    ms.ground_truth_bn(); replications r = 0..19; test 2,500 scenarios seed
    9000 + r; calibration 300 scenarios seed 5000 + r; training seed r; N in
    {10, 20, 50, 100, 200, 400}. Proposed is the deployed pipeline as tabulated
    (pl.ground_internal("Proposed"), estimated confusion, fitted reliability); its
    macro-AUC, ECE, Brier score and accuracy must reproduce expS_scarcity_raw.csv
    within 1e-9. Differences are Proposed minus variant per replication; paired
    t-test over the 20 replications, as in the manuscript; Holm-adjusted p over
    the family of Proposed-against-B6-pub, -recal, -fit and -fit-r on AUC, Brier
    and ECE at all N is reported beside the raw p, and over the family of the two
    comparisons of [C] below, separately.
[C] Reading. No verdict word: the sign and significance of every difference are
    reported. If B6-fit or B6-fit-r is ahead of Proposed, or not significantly
    behind it (raw p >= 0.05), on AUC or Brier at some N, this is stated and the
    contribution claim is narrowed to the remaining N. The differences read as:
    scale, B6-recal against B6-pub; hand-set constants against estimation on equal
    evidence, B6-fit-5 against B6-pub; added evidence, B6-fit against B6-fit-5;
    functional form, prior and latent labels together, Proposed against B6-fit and
    B6-fit-r. The prior of Proposed is near the truth in this experiment and that of
    B6-fit-r has mean 0, so at small N these differences reflect knowledge more than
    form. Two comparisons hold one of them fixed: same knowledge, B0-as-prior MAP
    against B6-fit-p; same data, Proposed-EM against B6-fit and B6-fit-r. The rule
    above applies to both: at an N where the BN side is not ahead, no gain of the
    functional form is claimed. Variants are not added after the results are seen,
    and analyses made after the run are marked as outside this registration.
[A] Abstract-diagonal sweep (separate). The manuscript sweeps the diagonal of the
    abstract sensor model over 0.70-0.99 (exp_significance.py absweep), while
    measured values fall below 0.70 (DroneDetect under interference 0.667, 0.567).
    Same procedure (Proposed at N = 200, abstract sensor tables, no discount, all
    20 replications) with diagonals 0.55, 0.60, 0.65 added; the five old values
    are recomputed and must reproduce expS_absweep_raw.csv within 1e-9. Reported
    as found on macro-AUC, ECE, Brier score and accuracy, with the paired
    difference from the calibrated pipeline. Results in a new file.
Determinism: one worker per replication, 1 BLAS thread per worker; replication 0
is rerun in the parent at the end and must match the pool's rows exactly.

Run:  python exp_base_rule.py --selftest        (no result files)
      python exp_base_rule.py --workers 20      (the study)
      python exp_base_rule.py --summary-only
"""
from __future__ import annotations
import os
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import sys, time, argparse, math
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
RES = os.path.join(HERE, "results")
P = "expB6"
N_GRID = (10, 20, 50, 100, 200, 400)
REPS = 20
TEST_N = 2500
CALIB_N = 300
ABS_N = 200
DIAG_NEW = (0.55, 0.60, 0.65)
DIAG_OLD = (0.70, 0.80, 0.90, 0.94, 0.99)
TBINS = [0.0, 0.10, 0.25, 0.50, 0.75, 1.0]
PROB_CLIP = 1e-6
FIT_CLIP = 1e-12
FIT_BOUND = 20.0
FIT_STARTS = 5
PRIOR_SD = 2.5

# ------------------------------------------------------------- the rule of [5]
K5, C5 = 6.0, 0.5
WEIGHTS = {
    "H": {"Proximity": .48, "Approach": .28, "Acceleration": .15, "Altitude": .09},
    "N": {"RF_Status": .40, "RF_Protocol": .2, "Trajectory": .2, "RCS": .2},
    "C": {"Route": .5, "Swarm": .30, "Environment": .20},
}
# state scores; T6 = Table 6 of [5], A = assumed (see [M]); states marked "unused"
# never occur under the maps of this study
SCORES = {
    "Proximity": {"Critical": 1.0, "Near": 0.5, "Far": 0.0},                   # A, T6, A
    "Approach": {"Fast_Appr": 1.0, "Slow_Appr": 0.5, "Non_Closing": 0.0, "Analyzing": 0.0},  # A, T6, A, Eq.(1)
    "Acceleration": {"High": 1.0, "Stable": 0.0, "Analyzing": 0.0},            # A, T6, Eq.(1)
    "Altitude": {"Masking": 1.0, "Anomaly": 0.5, "Legal": 0.0, "Analyzing": 0.0},  # T6, A unused, A, cold start
    "RF_Status": {"Silent": 1.0, "Detected": 0.0, "Analyzing": 0.0},           # A, T6, cold start
    "RF_Protocol": {"Non_Std": 1.0, "Standard": 0.0, "Analyzing": 0.0},        # T6, A, cold start
    "Trajectory": {"Loitering": 1.0, "Direct": 0.0, "Analyzing": 0.0},         # T6, A, Eq.(1)
    "RCS": {"Stable": 1.0, "Fluctuating": 0.0, "Analyzing": 0.0},              # T6, A, Eq.(1)
    "Route": {"Violation": 1.0, "Warning": 0.5, "On_Route": 0.0, "Analyzing": 0.0},  # T6, A unused, A, cold start
    "Swarm": {"Swarm": 1.0, "Cluster": 0.5, "Single": 0.0, "Analyzing": 0.0},  # A, A unused, T6, cold start
    "Environment": {"Adverse": 1.0, "Favorable": 0.0, "Analyzing": 0.0},       # T6, A, cold start
}


def _rules(s):
    """Table 3 of [5]. A rule fires only if every input it reads is present."""
    g = s.get
    return {
        "H": [0.5 * (g("Altitude") == "Masking" and g("Approach") == "Fast_Appr"),
              -2.0 * (g("Approach") == "Non_Closing" and g("Proximity") not in ("Critical", None))],
        "N": [0.5 * (g("RF_Status") == "Silent" and g("RCS") == "Stable"),
              -1.0 * (g("RF_Protocol") == "Standard"),
              -1.0 * (g("RCS") == "Fluctuating")],
        "C": [0.5 * (g("Environment") == "Adverse" and g("Route") == "Violation"),
              0.5 * (g("Swarm") == "Swarm"),
              -1.0 * (g("Environment") == "Favorable" and g("Route") == "On_Route" and g("Swarm") == "Single")],
    }


def _sig(x):
    return 1.0 / (1.0 + math.exp(-x))


def rule5(states):
    """Eq. (1)-(4) of [5]. states: input -> state; an input that is not a key has
    no counterpart and is removed with the remaining weights renormalized; the
    state "NaN" (RF Protocol under Silent) leaves the sum without renormalizing.
    Returns (P(T), {Z: (S_base, S_total, P(Z))})."""
    rules = _rules(states)
    out, q = {}, 1.0
    for z, w in WEIGHTS.items():
        have = {i: wi for i, wi in w.items() if i in states}
        tot = sum(have.values())
        sb = sum((wi / tot) * SCORES[i][states[i]] for i, wi in have.items() if states[i] != "NaN")
        st = sb + float(sum(rules[z]))
        pz = _sig(K5 * (st - C5))
        out[z] = (sb, st, pz)
        q *= 1.0 - pz
    return 1.0 - q, out


def map_states(d, rdot, v, z, rf, rr, m="M2"):
    """[M]: world evidence (integer states of model_spec) -> inputs of [5]."""
    s = {"Proximity": ("Far", "Near", "Critical")[d]}
    if m == "M3":
        s["Approach"] = ("Non_Closing", "Slow_Appr", "Fast_Appr")[rdot]
    elif m == "M4":
        s["Approach"] = ("Non_Closing", "Non_Closing", "Fast_Appr" if v == 2 else "Slow_Appr")[rdot]
    else:
        s["Approach"] = ("Non_Closing", "Slow_Appr", "Fast_Appr" if v == 2 else "Slow_Appr")[rdot]
    s["RF_Status"] = "Silent" if rf == 2 else "Detected"
    s["RF_Protocol"] = ("Standard", "Non_Std", "NaN")[rf]           # Rf: consumer, mil, none
    s["RCS"] = "Stable" if rr == 0 else "Fluctuating"               # Rr: drone, bird, clutter
    s["Route"] = "Violation" if z == 1 else "On_Route"
    if m != "M1":
        s.update(Acceleration="Stable", Altitude="Legal", Trajectory="Direct", Swarm="Single",
                 Environment="Favorable")
    return s


def b6_pub(scns, m="M2"):
    cache, out = {}, np.empty(len(scns))
    for i, x in enumerate(scns):
        key = (x["d"], x["rdot"], x["v"], x["z"], x["Rf"], x["Rr"])
        if key not in cache:
            cache[key] = rule5(map_states(*key, m=m))[0]
        out[i] = cache[key]
    return out


def selftest():
    """The worked example of [5] (Table 6), the cold-start value, the map, and
    the gradient and recovery of the fitted form. Prints and asserts; no files."""
    ex = {"Proximity": "Near", "Approach": "Slow_Appr", "Acceleration": "Stable", "Altitude": "Masking",
          "RF_Status": "Detected", "RF_Protocol": "Non_Std", "Trajectory": "Loitering", "RCS": "Stable",
          "Route": "Violation", "Swarm": "Single", "Environment": "Adverse"}
    pt, z = rule5(ex)
    print(f"Table 6 of [5]: Hostile S {z['H'][1]:.3f} -> {z['H'][2]:.3f} (paper 0.470 -> 0.455); "
          f"Non-coop S {z['N'][1]:.3f} -> {z['N'][2]:.3f} (0.600 -> 0.646); "
          f"Context S_base {z['C'][0]:.3f}, S {z['C'][1]:.3f} -> {z['C'][2]:.3f} (0.700, 1.200 -> 0.985); "
          f"final {pt:.3f} (0.997)")
    got = (z["H"][1], z["H"][2], z["N"][1], z["N"][2], z["C"][0], z["C"][1], z["C"][2], pt)
    want = (0.470, 0.455, 0.600, 0.646, 0.700, 1.200, 0.985, 0.997)
    assert all(round(a, 3) == b for a, b in zip(got, want)), got
    cold = {i: "Analyzing" for w in WEIGHTS.values() for i in w}
    cold["Proximity"] = "Far"            # Table 2 of [5] has no Analyzing state for Proximity; it scores 0 there
    pc, zc = rule5(cold)
    print(f"cold start of Sec. 3.3.2: P(Z) {zc['H'][2]:.3f} (paper about 0.047)")
    assert round(zc["H"][2], 3) == 0.047 and round(zc["N"][2], 3) == 0.047 and round(zc["C"][2], 3) == 0.047
    # the weights of Table 2 of [5] sum to 1 per latent variable
    assert all(abs(sum(w.values()) - 1.0) < 1e-12 for w in WEIGHTS.values())
    # the assumed state scores against the qualitative Table 5 of [5], with the
    # inputs of its Table 4 (Environment and the RF entries marked Prob. there are
    # set as noted; Table 5 gives no level for the entries marked Prob.)
    com = {"Acceleration": "Stable", "Altitude": "Legal", "Trajectory": "Direct", "RCS": "Stable"}
    sc = {"S1 Transit": dict(com, Proximity="Far", Approach="Non_Closing", RF_Status="Detected", RF_Protocol="Standard",
                             Route="On_Route", Swarm="Single", Environment="Favorable"),
          "S3 Attack": dict(com, Proximity="Critical", Approach="Fast_Appr", Acceleration="High", RF_Status="Detected",
                            RF_Protocol="Non_Std", Route="Violation", Swarm="Single", Environment="Favorable"),
          "S4 Swarm": dict(com, Proximity="Near", Approach="Slow_Appr", RF_Status="Detected", RF_Protocol="Non_Std",
                           Route="Violation", Swarm="Swarm", Environment="Favorable"),
          "S5 Auto": dict(com, Proximity="Near", Approach="Slow_Appr", RF_Status="Silent", RF_Protocol="NaN",
                          Route="Violation", Swarm="Single", Environment="Adverse")}
    lv = {k: rule5(v)[1] for k, v in sc.items()}
    for k, v in lv.items():
        print(f"Table 5 of [5], {k}: P(Hostile) {v['H'][2]:.3f}, P(NonCoop) {v['N'][2]:.3f}, P(Context) {v['C'][2]:.3f}")
    assert all(lv["S1 Transit"][z][2] < 0.1 for z in "HNC")                      # Low, Low, Low
    assert lv["S3 Attack"]["H"][2] > 0.8 and 0.2 < lv["S3 Attack"]["C"][2] < 0.8  # High, Prob., Mid
    assert 0.2 < lv["S4 Swarm"]["H"][2] < 0.8 and lv["S4 Swarm"]["C"][2] > 0.8    # Mid, Prob., High
    assert 0.2 < lv["S5 Auto"]["H"][2] < 0.8 and lv["S5 Auto"]["N"][2] > 0.8 and lv["S5 Auto"]["C"][2] > 0.8  # Mid, High, High
    # the map: the main map M2, the alternatives and the rules, by hand
    a = rule5(map_states(2, 2, 2, 1, 2, 0))[1]      # imminent, closing fast, nofly, Rf none, Rr drone
    assert abs(a["H"][1] - 0.76) < 1e-12 and abs(a["N"][1] - (0.40 + 0.2 + 0.5)) < 1e-12 and abs(a["C"][1] - 0.5) < 1e-12
    b = rule5(map_states(1, 0, 0, 0, 0, 1))[1]      # near, opening, permitted, Rf consumer, Rr bird
    assert abs(b["H"][1] - (0.24 - 2.0)) < 1e-12    # rule B
    assert abs(b["N"][1] - (0.0 - 1.0 - 1.0)) < 1e-12 and abs(b["C"][1] - (-1.0)) < 1e-12    # rules 2, 3; rule gamma
    c = rule5(map_states(1, 1, 0, 1, 1, 0))[1]      # near, steady, nofly, Rf mil, Rr drone
    assert abs(c["H"][1] - 0.38) < 1e-12 and abs(c["N"][1] - 0.4) < 1e-12 and abs(c["C"][1] - 0.5) < 1e-12
    e = rule5(map_states(2, 2, 2, 1, 2, 0, m="M1"))[1]   # the renormalized map
    assert abs(e["H"][1] - 1.0) < 1e-12 and abs(e["N"][1] - (0.5 + 0.25 + 0.5)) < 1e-12 and abs(e["C"][1] - 1.0) < 1e-12
    assert round(e["C"][2], 3) == 0.953
    assert map_states(0, 2, 0, 0, 0, 0, m="M3")["Approach"] == "Fast_Appr"
    assert map_states(0, 2, 0, 0, 0, 0)["Approach"] == "Slow_Appr"
    f4 = rule5(map_states(1, 1, 0, 1, 1, 0, m="M4"))[1]  # steady is Non_Closing: rule B fires at d = near
    assert abs(f4["H"][1] - (0.24 - 2.0)) < 1e-12
    f5 = rule5(map_states(2, 1, 0, 1, 1, 0, m="M4"))[1]  # not at d = imminent
    assert abs(f5["H"][1] - 0.48) < 1e-12
    print("map checks by hand: 17 conditions passed (M2, M1, M3, M4, rules B, 1, 2, 3, gamma)")
    # the prior mean of B6-fit-p with the intercepts of [5] is B6-pub, on every
    # evidence combination
    from itertools import product as _pr
    grid = [dict(d=d_, rdot=r_, v=v_, z=z_, Rf=f_, Rr=q_, Re=e_, Ra=a_, T=0)
            for d_, r_, v_, z_, f_, q_, e_, a_ in _pr(range(3), range(3), range(3), range(2), range(3), range(3), range(3), range(2))]
    gap = float(np.max(np.abs(form_predict(prior_vector("full", True), design(grid, "full")) - b6_pub(grid))))
    print(f"prior mean of B6-fit-p against B6-pub on {len(grid)} combinations: max |difference| {gap:.1e}")
    assert gap < 1e-12
    # gradient of the fitted form against central differences, and recovery of a
    # known member of the family (no world data, no Proposed)
    rng = np.random.default_rng(0)
    XH, XN, XC = (np.c_[np.ones(400), rng.integers(0, 2, (400, k))].astype(float) for k in (3, 4, 2))
    th = rng.normal(0, 0.5, 3 + 1 + 4 + 1 + 2 + 1)
    y = rng.integers(0, 2, 400).astype(float)
    f = lambda t: nll_grad(t, (XH, XN, XC), y, None)
    v, gr = f(th)
    num = np.array([(f(th + 1e-6 * e_)[0] - f(th - 1e-6 * e_)[0]) / 2e-6 for e_ in np.eye(len(th))])
    dg = float(np.max(np.abs(num - gr)))
    print(f"analytic against numeric gradient of the Noisy-OR likelihood: max |difference| {dg:.1e}")
    assert dg < 1e-3
    true = np.array([-2.5, 2.0, -1.0, 0.5, -3.0, 2.5, 1.0, -1.0, 0.5, -3.0, 2.0, 1.0])
    n = 40000
    XH, XN, XC = (np.c_[np.ones(n), rng.integers(0, 2, (n, k))].astype(float) for k in (3, 4, 2))
    py = form_predict(true, (XH, XN, XC))
    y = (rng.random(n) < py).astype(float)
    th_hat, val, conv = fit_form((XH, XN, XC), y, None, np.random.default_rng(1))
    pd_ = np.abs(form_predict(th_hat, (XH, XN, XC)) - py).mean()
    print(f"recovery on 40,000 draws of a known member: mean |P error| {pd_:.4f}, converged {conv}")
    assert pd_ < 0.01
    print("selftest passed")


# ------------------------------------------------------------ the fitted form
FEATS = {
    "full": {"H": ["b", "v=slow", "v=fast", "rdot=steady", "rdot=closing", "Fast_Appr", "d=near", "d=imminent", "rule B"],
             "N": ["b", "Rf=mil", "Rf=none", "Rr=bird", "Rr=clutter", "Re=bird", "Re=other", "Ra=none", "rule 1"],
             "C": ["b", "z=nofly", "d=near", "d=imminent"]},
    "five": {"H": ["b", "v=slow", "v=fast", "rdot=steady", "rdot=closing", "Fast_Appr", "d=near", "d=imminent", "rule B"],
             "N": ["b", "Rf=mil", "Rf=none", "Rr=bird", "Rr=clutter", "rule 1"],
             "C": ["b", "z=nofly"]},
}


# B6-fit-p: slope means read from the constants of [5] under M2, logit P(Z) =
# k (sum w s + sum a - c), against the reference state of each input; terms [5]
# lacks have mean 0. The intercepts are used only as a start.
PRIOR5 = {
    "H:b": K5 * (0.0 - C5), "H:rdot=steady": K5 * 0.28 * 0.5, "H:rdot=closing": K5 * 0.28 * 0.5,
    "H:Fast_Appr": K5 * 0.28 * 0.5, "H:d=near": K5 * 0.48 * 0.5, "H:d=imminent": K5 * 0.48, "H:rule B": K5 * -2.0,
    "N:b": K5 * (0.2 - 1.0 - C5), "N:Rf=mil": K5 * (0.2 + 1.0), "N:Rf=none": K5 * (0.40 + 1.0),
    "N:Rr=bird": K5 * (-1.0 - 0.2), "N:Rr=clutter": K5 * (-1.0 - 0.2), "N:rule 1": K5 * 0.5,
    "C:b": K5 * (-1.0 - C5), "C:z=nofly": K5 * (0.5 + 1.0),
}


def prior_vector(kind, with_intercepts):
    v = [PRIOR5.get(f"{z}:{nm}", 0.0) if (nm != "b" or with_intercepts) else 0.0
         for z in "HNC" for nm in FEATS[kind][z]]
    return np.array(v, float)


def _feat(x, name):
    if name == "b":
        return 1.0
    if name == "Fast_Appr":
        return float(x["rdot"] == 2 and x["v"] == 2)
    if name == "rule B":
        return float(x["rdot"] == 0 and x["d"] != 2)
    if name == "rule 1":
        return float(x["Rf"] == 2 and x["Rr"] == 0)
    var, st = name.split("=")
    import model_spec as ms
    return float(x[var] == ms.STATES[var].index(st))


def design(scns, kind):
    cache, rows = {}, {z: [] for z in "HNC"}
    for x in scns:
        key = (x["v"], x["rdot"], x["d"], x["z"], x["Rf"], x["Rr"], x["Re"], x["Ra"])
        if key not in cache:
            cache[key] = {z: [_feat(x, nm) for nm in FEATS[kind][z]] for z in "HNC"}
        for z in "HNC":
            rows[z].append(cache[key][z])
    return tuple(np.array(rows[z], float) for z in "HNC")


def _split(th, Xs):
    out, i = [], 0
    for X in Xs:
        out.append(th[i:i + X.shape[1]])
        i += X.shape[1]
    return out


def form_predict(th, Xs):
    logq = np.zeros(Xs[0].shape[0])
    for t, X in zip(_split(th, Xs), Xs):
        logq -= np.logaddexp(0.0, X @ t)        # log(1 - sigmoid(z)) = -log(1 + exp(z))
    return 1.0 - np.exp(logq)


def nll_grad(th, Xs, y, prior_sd, prior_mu=None):
    """Negative log likelihood of y under Noisy-OR of per-latent sigmoids, its
    gradient, and an optional N(prior_mu, prior_sd^2) prior on the slopes (not on
    the intercept, the first column of each X; prior_mu None means mean 0)."""
    logq = np.zeros(len(y))
    ps = []
    for t, X in zip(_split(th, Xs), Xs):
        z = X @ t
        logq -= np.logaddexp(0.0, z)
        ps.append(np.exp(z - np.logaddexp(0.0, z)))
    q = np.exp(logq)
    p = np.clip(1.0 - q, 1e-12, 1.0)
    val = -float(np.sum(y * np.log(p) + (1.0 - y) * logq))
    w = y * q / p - (1.0 - y)
    grads = [-(X.T @ (w * pz)) for X, pz in zip(Xs, ps)]
    if prior_sd is not None:
        mus = _split(prior_mu, Xs) if prior_mu is not None else [np.zeros(X.shape[1]) for X in Xs]
        for t, mu, gr in zip(_split(th, Xs), mus, grads):
            val += float(np.sum((t[1:] - mu[1:]) ** 2)) / (2.0 * prior_sd ** 2)
            gr[1:] += (t[1:] - mu[1:]) / prior_sd ** 2
    return val, np.concatenate(grads)


def fit_form(Xs, y, prior_sd, rng, prior_mu=None, extra_start=None):
    from scipy.optimize import minimize
    n_par = sum(X.shape[1] for X in Xs)
    ybar = float(np.clip(y.mean(), 1e-4, 1 - 1e-4))
    b0 = float(np.log(1.0 - (1.0 - ybar) ** (1.0 / 3.0)) - np.log((1.0 - ybar) ** (1.0 / 3.0)))
    base = np.zeros(n_par)
    i = 0
    for X in Xs:
        base[i] = b0
        i += X.shape[1]
    best, best_val, best_conv = None, np.inf, False
    starts = [base + (rng.normal(0.0, 0.5, n_par) if s else 0.0) for s in range(FIT_STARTS)]
    if extra_start is not None:
        starts.append(extra_start)
    for th0 in starts:
        r = minimize(nll_grad, np.clip(th0, -FIT_BOUND, FIT_BOUND), args=(Xs, y, prior_sd, prior_mu), jac=True,
                     method="L-BFGS-B", bounds=[(-FIT_BOUND, FIT_BOUND)] * n_par,
                     options={"maxiter": 3000, "ftol": 1e-10, "gtol": 1e-6})
        if r.fun < best_val:
            best, best_val, best_conv = r.x.copy(), float(r.fun), bool(r.success)
    return best, best_val, best_conv


def fit_predict(train, test, kind, prior_sd, seed_rng, use_prior5=False):
    """Returns (P(T = high) on test, parameter rows, degenerate flag). A training
    set without a positive or a negative gives the constant training rate, except
    under the prior of [5] (B6-fit-p), which still ranks."""
    y = np.array([float(x["T"] == 3) for x in train])
    deg = bool(y.min() == y.max())
    if deg and not use_prior5:
        return np.full(len(test), float(np.clip(y.mean(), FIT_CLIP, 1 - FIT_CLIP))), [], True
    Xs = design(train, kind)
    th, val, conv = fit_form(Xs, y, prior_sd, seed_rng,
                             prior_mu=prior_vector(kind, False) if use_prior5 else None,
                             extra_start=prior_vector(kind, True) if use_prior5 else None)
    names = [f"{z}:{nm}" for z in "HNC" for nm in FEATS[kind][z]]
    assert len(names) == len(th)
    rows = [{"param": nm, "value": float(v), "nll": val, "converged": conv} for nm, v in zip(names, th)]
    return np.clip(form_predict(th, design(test, kind)), FIT_CLIP, 1 - FIT_CLIP), rows, deg


def recalibrate(p_cal, y_cal, p_test):
    from sklearn.linear_model import LogisticRegression
    lg = lambda p: np.log(np.clip(p, PROB_CLIP, 1 - PROB_CLIP) / (1 - np.clip(p, PROB_CLIP, 1 - PROB_CLIP)))
    m = LogisticRegression(C=1e6, max_iter=2000).fit(lg(p_cal)[:, None], y_cal)
    return m.predict_proba(lg(p_test)[:, None])[:, 1], float(m.intercept_[0]), float(m.coef_[0, 0])


# ------------------------------------------------------------------ metrics
def binary_metrics(p, y):
    """p: P(T = high); y: T in 0..3."""
    import metrics as mt
    from sklearn.metrics import roc_auc_score
    hit = (y == 3).astype(int)
    row = {"auc": float(roc_auc_score(hit, p)), "brier_bin": float(np.mean((p - hit) ** 2)),
           "ece_bin": float(mt.ece(np.c_[1 - p, p], hit))}
    bins = np.linspace(0, 1, mt.N_BINS + 1)
    e = 0.0
    for b in range(mt.N_BINS):
        m = (p > bins[b]) & (p <= bins[b + 1])
        if m.any():
            e += m.mean() * abs(hit[m].mean() - p[m].mean())
    row["ece_pos"] = float(e)
    for thr in (0.5, 0.25):
        rule = p >= thr
        row[f"tp_{thr}"], row[f"fp_{thr}"] = int((rule & (hit == 1)).sum()), int((rule & (hit == 0)).sum())
        row[f"fn_{thr}"] = int((~rule & (hit == 1)).sum())
    return row


def bin_rows(p, y):
    hit = y == 3
    rows = []
    for lo, hi in zip(TBINS[:-1], TBINS[1:]):
        m = (p > lo) & (p <= hi)
        rows.append({"bin_lo": lo, "bin_hi": hi, "n": int(m.sum()), "sum_pred": float(p[m].sum()),
                     "hits": int(hit[m].sum())})
    return rows


def subset_rows(p, y, masks):
    from sklearn.metrics import roc_auc_score
    hit = (y == 3).astype(int)
    rows = []
    for name, m in masks.items():
        if m.sum() == 0:
            continue
        auc = float(roc_auc_score(hit[m], p[m])) if 0 < hit[m].sum() < m.sum() else np.nan
        rows.append({"subset": name, "n": int(m.sum()), "pos": int(hit[m].sum()), "auc": auc,
                     "brier_bin": float(np.mean((p[m] - hit[m]) ** 2)), "mean_p": float(p[m].mean()),
                     "share_ge_0.5": float(np.mean(p[m] >= 0.5))})
    return rows


# --------------------------------------------------------------- replication
def replicate(r, reps_n_grid=N_GRID):
    import model_spec as ms, dataio, metrics as mt, pipeline as pl, grounding as g
    import exp_mapem as em
    from sklearn.metrics import roc_auc_score
    t0 = time.time()
    gt = ms.ground_truth_bn()
    test = dataio.generate_dataset(gt, TEST_N, seed=9000 + r)
    calib = dataio.generate_dataset(gt, CALIB_N, seed=5000 + r)
    est = dataio.estimate_confusion(calib)
    alpha, _ = dataio.fit_reliability(calib)
    y = np.array([x["T"] for x in test])
    ge2 = (y >= 2).astype(int)
    masks = {"silent-drone": np.array([x["cls"] == 2 and x["emit"] == 1 for x in test]),
             "silent-report-nondrone": np.array([x["Rf"] == 2 and x["cls"] != 2 for x in test]),
             "far-range": np.array([x["d"] == 0 for x in test])}
    rows, bins, subs, params, abs_rows = [], [], [], [], []

    def add(method, N, p, score_ge2=None, extra=None):
        rows.append({"rep": r, "N": N, "method": method, **binary_metrics(p, y),
                     "auc_ge_medium": float(roc_auc_score(ge2, p if score_ge2 is None else score_ge2)),
                     **(extra or {})})
        bins.extend({"rep": r, "N": N, "method": method, **b} for b in bin_rows(p, y))
        subs.extend({"rep": r, "N": N, "method": method, **s} for s in subset_rows(p, y, masks))

    # no training: N = 0
    Po, _ = mt.posteriors({k: gt.cpts[k] for k in ms.INTERNAL}, test, ms.SENSORS)
    add("CPT-oracle", 0, Po[:, 3], Po[:, 2] + Po[:, 3])
    pub = b6_pub(test)
    add("B6-pub", 0, pub)
    add("B6-pub-M1", 0, b6_pub(test, "M1"))
    add("B6-pub-M3", 0, b6_pub(test, "M3"))
    add("B6-pub-M4", 0, b6_pub(test, "M4"))
    yc = np.array([float(x["T"] == 3) for x in calib])
    prc, a_, b_ = recalibrate(b6_pub(calib), yc, pub)
    add("B6-recal", 0, prc, extra={"recal_a": a_, "recal_b": b_})

    b0 = ms.heuristic_cpts()
    prior = ms.expert_prior_cpts(np.random.default_rng(1000 + r))
    for N in reps_n_grid:
        data = dataio.generate_dataset(gt, N, seed=r)
        cpts, _ = pl.ground_internal("Proposed", data, r, gt=gt)
        partial = [{k: row[k] for k in em.KEEP} for row in data]
        bn_methods = (("Proposed", cpts),
                      ("B0-as-prior MAP", {n: g.map_cpt(n, data, b0[n], pl.ESS_FIXED) for n in ms.INTERNAL}),
                      ("Proposed-EM", g.em_internal(partial, prior, pl.ESS_FIXED, n_iter=em.EM_ITERS, seed=r,
                                                    constrained=True, sensor_cpts=est)))
        for name, cp in bn_methods:
            met, (Pm, yy) = mt.evaluate(cp, test, sensor_cpts=est, alpha=alpha)
            assert np.array_equal(yy, y)
            add(name, N, Pm[:, 3], Pm[:, 2] + Pm[:, 3], extra={k: met[k] for k in REPRO_KEYS})
        for name, kind, sd, up in (("B6-fit", "full", None, False), ("B6-fit-r", "full", PRIOR_SD, False),
                                   ("B6-fit-p", "full", PRIOR_SD, True), ("B6-fit-5", "five", None, False)):
            pf, pr, deg = fit_predict(data, test, kind, sd, np.random.default_rng(61000 + 1000 * r + N), up)
            add(name, N, pf, extra={"degenerate": deg})
            params.extend({"rep": r, "N": N, "method": name, **q} for q in pr)
        if N == ABS_N:
            for dgv in DIAG_NEW + DIAG_OLD:
                m_, _ = mt.evaluate(cpts, test, sensor_cpts=dataio.abstract_sensor_cpts_diag(dgv), alpha=None)
                abs_rows.append({"rep": r, "diag": dgv, **m_})
            m_, _ = mt.evaluate(cpts, test, sensor_cpts=est, alpha=None)
            abs_rows.append({"rep": r, "diag": -1.0, **m_})    # calibrated, no discount (reference)
    return r, rows, bins, subs, params, abs_rows, time.time() - t0


REPRO_KEYS = ("acc", "macroF1", "macroAUC", "brier", "ece", "cwece", "far@pd0.9")


def _save(out, res):
    pd.DataFrame(out["rows"]).to_csv(os.path.join(res, f"{P}_raw.csv"), index=False)
    pd.DataFrame(out["bins"]).to_csv(os.path.join(res, f"{P}_bins.csv"), index=False)
    pd.DataFrame(out["subs"]).to_csv(os.path.join(res, f"{P}_subsets.csv"), index=False)
    pd.DataFrame(out["params"]).to_csv(os.path.join(res, f"{P}_fit_params.csv"), index=False)
    pd.DataFrame(out["abs"]).to_csv(os.path.join(res, f"{P}_absweep_raw.csv"), index=False)


def check_reproduction(rows, abs_rows):
    """[R], [A]: Proposed and the old diagonals against the files of expS."""
    d = pd.DataFrame(rows)
    out = []
    for method, fname, other in (("Proposed", "expS_scarcity_raw.csv", "Proposed"),
                                 ("B0-as-prior MAP", "expB_baselines_raw.csv", "B0-as-prior MAP"),
                                 ("Proposed-EM", "expM_raw.csv", "MAP-EM (latent)")):
        ref = pd.read_csv(os.path.join(RES, fname), float_precision="round_trip")
        ref = ref[ref.method == other].set_index(["rep", "N"])
        z = d[(d.method == method)].to_dict("records")
        dev = [abs(float(ref.loc[(x["rep"], x["N"]), k]) - float(x[k])) for x in z
               if (x["rep"], x["N"]) in ref.index for k in REPRO_KEYS]
        assert len(dev) > 0, fname
        out.append((f"{method} against {fname}", max(dev), len(dev)))
    ab = pd.read_csv(os.path.join(RES, "expS_absweep_raw.csv"), float_precision="round_trip").set_index(["rep", "diag"])
    a = pd.DataFrame(abs_rows)
    a = a[a["diag"].isin(DIAG_OLD)]
    deva = [abs(float(ab.loc[(x.rep, x.diag), k]) - float(getattr(x, k))) for x in a.itertuples()
            for k in ("macroAUC", "ece", "brier", "acc")]
    assert len(deva) > 0
    out.append(("old diagonals against expS_absweep_raw.csv", max(deva), len(deva)))
    return out


# -------------------------------------------------------------------- summary
def _holm(ps):
    ps = np.asarray(ps, float)
    order = np.argsort(ps)
    adj, cur = np.empty(len(ps)), 0.0
    for i, j in enumerate(order):
        cur = max(cur, (len(ps) - i) * ps[j])
        adj[j] = min(cur, 1.0)
    return adj


def _cp(k, n):
    from scipy import stats as st
    return (float(st.beta.ppf(0.025, k, n - k + 1)) if k > 0 else 0.0,
            float(st.beta.ppf(0.975, k + 1, n - k)) if k < n else 1.0)


def summarize(res):
    d = pd.read_csv(os.path.join(res, f"{P}_raw.csv"), float_precision="round_trip")
    bn = pd.read_csv(os.path.join(res, f"{P}_bins.csv"))
    sb = pd.read_csv(os.path.join(res, f"{P}_subsets.csv"))
    ab = pd.read_csv(os.path.join(res, f"{P}_absweep_raw.csv"), float_precision="round_trip")
    L = ["# Exp-B6: the scoring rule of [5] against the deployed pipeline (target T = high)", ""]
    get = lambda m, N, k: d[(d.method == m) & (d.N == N)].sort_values("rep")[k].values
    FIX = ("B6-pub", "B6-pub-M1", "B6-pub-M3", "B6-pub-M4", "B6-recal", "CPT-oracle")
    FIT = ("B6-fit", "B6-fit-r", "B6-fit-p", "B6-fit-5")
    BNS = ("Proposed", "Proposed-EM", "B0-as-prior MAP")
    KS = (("auc", "AUC"), ("brier_bin", "Brier"), ("ece_bin", "ECE"), ("ece_pos", "ECE on p"), ("auc_ge_medium", "AUC T>=medium"))
    L += ["## Mean over replications", "### No training (independent of N)",
          "| method | " + " | ".join(n for _, n in KS) + " |", "|---|" + "---|" * len(KS)]
    for m in FIX:
        L.append(f"| {m} | " + " | ".join(f"{get(m, 0, k).mean():.4f}" for k, _ in KS) + " |")
    L += ["", "### By training size", "| N | method | " + " | ".join(n for _, n in KS) + " | degenerate fits |",
          "|---|---|" + "---|" * (len(KS) + 1)]
    for N in N_GRID:
        for m in BNS + FIT:
            z = d[(d.method == m) & (d.N == N)]
            dg = int((z.degenerate == True).sum()) if m in FIT else ""
            L.append(f"| {N} | {m} | " + " | ".join(f"{get(m, N, k).mean():.4f}" for k, _ in KS) + f" | {dg} |")
    # paired differences
    rows = []
    for N in N_GRID:
        for m in ("B6-pub", "B6-recal", "B6-fit", "B6-fit-r"):
            nb = 0 if m in FIX else N
            for k, nm in KS[:3]:
                a, b = get("Proposed", N, k), get(m, nb, k)
                diff = a - b
                from scipy import stats as st
                pv = float(st.ttest_1samp(diff, 0.0).pvalue) if np.ptp(diff) > 0 else np.nan
                rows.append({"N": N, "against": m, "metric": nm, "delta": float(diff.mean()), "p": pv})
    t = pd.DataFrame(rows)
    t["p_holm"] = _holm(t.p.fillna(1.0).values)
    t.to_csv(os.path.join(res, f"{P}_paired.csv"), index=False)
    L += ["", "## Proposed minus variant (paired over 20 replications; positive favors Proposed on AUC, negative on Brier and ECE)",
          "| N | against | dAUC (p, Holm) | dBrier (p, Holm) | dECE (p, Holm) |", "|---|---|---|---|---|"]
    for N in N_GRID:
        for m in ("B6-pub", "B6-recal", "B6-fit", "B6-fit-r"):
            z = t[(t.N == N) & (t.against == m)].set_index("metric")
            L.append(f"| {N} | {m} | " + " | ".join(f"{z.loc[k].delta:+.4f} ({z.loc[k].p:.1e}, {z.loc[k].p_holm:.1e})"
                                                    for k in ("AUC", "Brier", "ECE")) + " |")
    # the two comparisons that hold knowledge or data fixed
    rows2 = []
    for N in N_GRID:
        for a_, b_ in (("B0-as-prior MAP", "B6-fit-p"), ("Proposed-EM", "B6-fit"), ("Proposed-EM", "B6-fit-r")):
            for k, nm in KS[:3]:
                diff = get(a_, N, k) - get(b_, N, k)
                pv = float(st.ttest_1samp(diff, 0.0).pvalue) if np.ptp(diff) > 0 else np.nan
                rows2.append({"N": N, "bn": a_, "against": b_, "metric": nm, "delta": float(diff.mean()), "p": pv})
    t2 = pd.DataFrame(rows2)
    t2["p_holm"] = _holm(t2.p.fillna(1.0).values)
    t2.to_csv(os.path.join(res, f"{P}_paired_same_level.csv"), index=False)
    L += ["", "## Same knowledge and same data (BN side minus rule form; p paired, Holm over this family)",
          "| N | comparison | dAUC (p, Holm) | dBrier (p, Holm) | dECE (p, Holm) |", "|---|---|---|---|---|"]
    for N in N_GRID:
        for a_, b_ in (("B0-as-prior MAP", "B6-fit-p"), ("Proposed-EM", "B6-fit"), ("Proposed-EM", "B6-fit-r")):
            z = t2[(t2.N == N) & (t2.bn == a_) & (t2.against == b_)].set_index("metric")
            L.append(f"| {N} | {a_} - {b_} | " + " | ".join(f"{z.loc[k].delta:+.4f} ({z.loc[k].p:.1e}, {z.loc[k].p_holm:.1e})"
                                                          for k in ("AUC", "Brier", "ECE")) + " |")
    # the decomposition of [C]
    L += ["", "## Reading of [C] (differences of means; p paired over replications)",
          "| N | scale: recal - pub Brier | constants: fit-5 - pub AUC | evidence: fit - fit-5 AUC | form+prior+labels: Proposed - fit AUC | Proposed - fit-r AUC |",
          "|---|---|---|---|---|---|"]
    for N in N_GRID:
        c1 = _paired2(get("B6-recal", 0, "brier_bin"), get("B6-pub", 0, "brier_bin"))
        c2 = _paired2(get("B6-fit-5", N, "auc"), get("B6-pub", 0, "auc"))
        c3 = _paired2(get("B6-fit", N, "auc"), get("B6-fit-5", N, "auc"))
        c4 = _paired2(get("Proposed", N, "auc"), get("B6-fit", N, "auc"))
        c5 = _paired2(get("Proposed", N, "auc"), get("B6-fit-r", N, "auc"))
        L.append(f"| {N} | {c1} | {c2} | {c3} | {c4} | {c5} |")
    # [C] statement
    near = t[(t.against.isin(["B6-fit", "B6-fit-r"])) & (t.metric.isin(["AUC", "Brier"]))]
    nsb = near[(near.p >= 0.05) | near.p.isna() | ((near.metric == "AUC") & (near["delta"] <= 0)) | ((near.metric == "Brier") & (near["delta"] >= 0))]
    L += ["", "## [C] N where B6-fit or B6-fit-r is ahead of Proposed or not significantly behind it (raw p >= 0.05)"]
    L += [f"- N {int(x.N)}, {x.against}, {x.metric}: difference {x.delta:+.4f}, p {x.p:.2g}" for x in nsb.itertuples()] or ["- none"]
    n2 = t2[t2.metric.isin(["AUC", "Brier"])]
    n2 = n2[(n2.p >= 0.05) | n2.p.isna() | ((n2.metric == "AUC") & (n2.delta <= 0)) | ((n2.metric == "Brier") & (n2.delta >= 0))]
    L += ["", "## [C] N where the BN side is not ahead in the same-knowledge or same-data comparison (no gain of the form is claimed there)"]
    L += [f"- N {int(x.N)}, {x.bn} - {x.against}, {x.metric}: difference {x.delta:+.4f}, p {x.p:.2g}" for x in n2.itertuples()] or ["- none"]
    # Table 9 form
    L += ["", "## Table 9 form, pooled over replications"]
    for m, N in (("Proposed", 200), ("Proposed-EM", 200), ("B0-as-prior MAP", 200), ("B6-pub", 0), ("B6-recal", 0),
                 ("B6-fit", 200), ("B6-fit-r", 200), ("B6-fit-p", 200), ("CPT-oracle", 0)):
        L += ["", f"### {m}" + (f" (N = {N})" if N else ""), "| bin | n | predicted | observed [95% CP] | outside |", "|---|---|---|---|---|"]
        z = bn[(bn.method == m) & (bn.N == N)]
        for lo in TBINS[:-1]:
            q = z[z.bin_lo == lo]
            n, k, sp = int(q.n.sum()), int(q.hits.sum()), float(q.sum_pred.sum())
            if n == 0:
                continue
            l_, h_ = _cp(k, n)
            L.append(f"| ({lo:.2f}, {q.bin_hi.iloc[0]:.2f}] | {n} | {sp / n:.3f} | {k / n:.3f} [{l_:.3f}, {h_:.3f}] | {'*' if not (l_ <= sp / n <= h_) else ''} |")
    # alarms
    L += ["", "## Alarms, pooled over replications (precision, recall)", "| method | at 0.5 | at 0.25 |", "|---|---|---|"]
    for m, N in (("Proposed", 200), ("Proposed-EM", 200), ("B0-as-prior MAP", 200), ("B6-pub", 0), ("B6-pub-M1", 0),
                 ("B6-pub-M3", 0), ("B6-pub-M4", 0), ("B6-recal", 0), ("B6-fit", 200), ("B6-fit-r", 200), ("B6-fit-p", 200),
                 ("CPT-oracle", 0)):
        z = d[(d.method == m) & (d.N == N)]
        cell = []
        for thr in (0.5, 0.25):
            tp, fp, fn = z[f"tp_{thr}"].sum(), z[f"fp_{thr}"].sum(), z[f"fn_{thr}"].sum()
            cell.append(f"{tp / max(tp + fp, 1):.3f}, {tp / max(tp + fn, 1):.3f}")
        L.append(f"| {m}{f' (N = {N})' if N else ''} | {cell[0]} | {cell[1]} |")
    # subsets
    L += ["", "## Subsets, mean over replications (descriptive)", "| subset | method | n | AUC | Brier | mean p | share p >= 0.5 |", "|---|---|---|---|---|---|---|"]
    for s_ in sb.subset.unique():
        for m, N in (("Proposed", 200), ("Proposed-EM", 200), ("B0-as-prior MAP", 200), ("B6-pub", 0), ("B6-recal", 0),
                     ("B6-fit", 200), ("B6-fit-r", 200), ("B6-fit-p", 200), ("CPT-oracle", 0)):
            z = sb[(sb.subset == s_) & (sb.method == m) & (sb.N == N)]
            L.append(f"| {s_} | {m} | {z.n.mean():.0f} | {z.auc.mean():.3f} | {z.brier_bin.mean():.3f} | {z.mean_p.mean():.3f} | {z['share_ge_0.5'].mean():.3f} |")
    # the sweep
    L += ["", "## [A] Abstract-diagonal sweep (Proposed at N = 200, no discount)", "| diagonal | macro-AUC | ECE | Brier | accuracy | dAUC, dECE against calibrated (p) |", "|---|---|---|---|---|---|"]
    ref = ab[ab["diag"] == -1.0].sort_values("rep")
    for dgv in sorted(set(DIAG_NEW + DIAG_OLD)):
        z = ab[ab["diag"] == dgv].sort_values("rep")
        L.append(f"| {dgv:.2f} | {z.macroAUC.mean():.4f} | {z.ece.mean():.4f} | {z.brier.mean():.4f} | {z.acc.mean():.4f} | "
                 f"{_paired2(z.macroAUC.values, ref.macroAUC.values)}, {_paired2(z.ece.values, ref.ece.values)} |")
    L.append(f"| calibrated | {ref.macroAUC.mean():.4f} | {ref.ece.mean():.4f} | {ref.brier.mean():.4f} | {ref.acc.mean():.4f} | |")
    chk = os.path.join(res, f"{P}_checks.txt")
    if os.path.exists(chk):
        L = L[:1] + ["", "## Checks", "```", open(chk, encoding="utf-8").read().strip(), "```"] + L[1:]
    txt = "\n".join(L)
    open(os.path.join(res, f"{P}_summary.md"), "w", encoding="utf-8").write(txt + "\n")
    print(txt)


def _paired2(a, b):
    from scipy import stats as st
    d_ = np.asarray(a, float) - np.asarray(b, float)
    pv = float(st.ttest_1samp(d_, 0.0).pvalue) if np.ptp(d_) > 0 else float("nan")
    return f"{d_.mean():+.4f} ({pv:.1e})"


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--workers", type=int, default=20)
    ap.add_argument("--reps", type=int, default=REPS)
    ap.add_argument("--res", default=RES, help="output folder (default results/)")
    ap.add_argument("--no-summary", action="store_true", help="code-path check: write raw files only, print no comparison")
    ap.add_argument("--summary-only", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        selftest()
        return
    if a.summary_only:
        summarize(a.res)
        return
    selftest()
    from multiprocessing import Pool
    t0 = time.time()
    results = {}
    with Pool(min(a.workers, a.reps)) as pool:
        for out in pool.imap_unordered(replicate, range(a.reps)):
            results[out[0]] = out
            print(f"[expB6] rep {out[0]} done ({out[6]:.0f}s; {time.time() - t0:.0f}s total)", flush=True)
    merged = {"rows": [], "bins": [], "subs": [], "params": [], "abs": []}
    for r in range(a.reps):
        _, rows, bins, subs, params, abs_rows, _ = results[r]
        merged["rows"] += rows; merged["bins"] += bins; merged["subs"] += subs
        merged["params"] += params; merged["abs"] += abs_rows
    os.makedirs(a.res, exist_ok=True)
    _save(merged, a.res)
    checks = check_reproduction(merged["rows"], merged["abs"])
    # determinism: replication 0 again, in this process
    again = replicate(0)
    same = (pd.DataFrame(again[1]).equals(pd.DataFrame(results[0][1]))
            and pd.DataFrame(again[5]).equals(pd.DataFrame(results[0][5])))
    txt = ("".join(f"{nm}: max |difference| {v:.2e} over {n} values\n" for nm, v, n in checks)
           + f"replication 0 rerun in the parent process equals the pool's rows: {same}\n"
           f"threads per worker: OMP {os.environ.get('OMP_NUM_THREADS')}, MKL {os.environ.get('MKL_NUM_THREADS')}, "
           f"OpenBLAS {os.environ.get('OPENBLAS_NUM_THREADS')}; workers {min(a.workers, a.reps)}; "
           f"elapsed {time.time() - t0:.0f}s")
    open(os.path.join(a.res, f"{P}_checks.txt"), "w", encoding="utf-8").write(txt + "\n")
    print(txt, flush=True)
    assert all(v <= 1e-9 for _, v, _ in checks) and same, "reproduction or determinism check failed"
    if not a.no_summary:
        summarize(a.res)


if __name__ == "__main__":
    main()
