"""
Consistency gate: verify that the numbers quoted in main.tex and in the
supplementary material match the results/ files (single source of truth).
Exits nonzero on any mismatch. Run after any experiment rerun.

Without the manuscript sources it verifies results/ alone. With the
environment variable CUAS_GATE_DRAFT=1 it also prints the number of checks of
the first blocks.
"""
from __future__ import annotations
import re, sys, os, glob
sys.stdout.reconfigure(encoding="utf-8")   # failure messages may hold characters outside cp949
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
DRAFT = os.environ.get("CUAS_GATE_DRAFT") == "1"

# Where the manuscript sources live. The default is a sibling checkout of
# the paper directory; set CUAS_PAPER_DIR to point somewhere else. The
# manuscript-side checks are skipped (not failed) when it is absent, so a
# reader who cloned only this repository can still verify results/.
PAPER = os.environ.get(
    "CUAS_PAPER_DIR",
    os.path.join(os.path.dirname(HERE), "paper", "ieeeaccess"))
TEX = os.path.join(PAPER, "main.tex")
HAVE_PAPER = os.path.exists(TEX)
if not HAVE_PAPER:
    print(f"note: no manuscript at {PAPER} -- checking results only. "
          "Set CUAS_PAPER_DIR to run the manuscript checks.")
tex = open(TEX, encoding="utf-8").read().replace("\n", " ") if HAVE_PAPER \
    else ""
# normalize LaTeX digit-group spacing so "12{,}500" matches "12,500"
tex = tex.replace("{,}", ",")

fails = []
_supp_later = []


def need_supp_later(label, snippet):
    """A sentence that moved to the supplement; checked where the supplement has been read."""
    _supp_later.append((label, snippet))


# ---- freshness gate: every results file quoted below must be newer than the
# core pipeline code that produces it (stale-results guard).
CORE = ["model_spec.py", "dataio.py", "grounding.py", "metrics.py",
        "pipeline.py", "cuas_bn.py"]
core_mtime = max(os.path.getmtime(os.path.join(HERE, f)) for f in CORE)
RESULT_FILES = [
    ("expS_table_stats.csv", "exp_significance.py"),
    ("expS_fusion_raw.csv", "exp_significance.py"),
    ("expS_scarcity_raw.csv", "exp_significance.py"),
    ("expE_prior_raw.csv", "exp_prior_ess.py"),
    ("expF_corr_raw.csv", "exp_correlated.py"),
    ("expF_rel_raw.csv", "exp_correlated.py"),
    ("expG_rf_e2e_raw.csv", "exp_rf_e2e.py"),
    ("expG_cnn.npz", "exp_rf_e2e.py"),
    ("expM_raw.csv", "exp_mapem.py"),
    ("expT_summary.md", "exp_sensitivity.py"),
    ("exp1_scarcity_strip4.png", "replot_scarcity_strip.py"),
    ("fig_expF_corr_print.png", "replot_print_size.py"),
    ("fig_expT_tornado_print.png", "replot_print_size.py"),
    ("exp_constraints.csv", "exp_constraints.py"),
    ("expA_ess_sensitivity.csv", "exp_ess_sensitivity.py"),
    ("expA_baselines.csv", "exp_ess_sensitivity.py"),
    ("expB_baselines_raw.csv", "exp_baselines.py"),
    ("expN_noise_raw.csv", "exp_noise_level.py"),
    ("expL_latency.csv", "exp_latency.py"),
    ("expR_reliability_classes.csv", "exp_reliability_classes.py"),
    ("expK_ece_bins.csv", "exp_ece_bins.py"),
    # the double-discounting study depends on two modules besides its
    # own script, so all three are listed
    ("expD_runs_raw.csv", ("exp_reliability_variants.py", "reliability_cc.py",
                           "world_options.py")),
    ("expD_population.csv", ("exp_reliability_variants.py",
                             "reliability_cc.py", "world_options.py")),
    ("expD_trueCPT_raw.csv", ("exp_reliability_variants.py",
                              "reliability_cc.py", "world_options.py")),
    # the estimator study and its non-monotone world
    ("expC_consistency_raw.csv", ("exp_estimator_consistency.py",
                                  "world_nonmonotone.py")),
    ("expC_pseudotrue.csv", ("exp_estimator_consistency.py",
                             "world_nonmonotone.py")),
    ("expC_mapem_raw.csv", ("exp_estimator_consistency.py",
                            "world_nonmonotone.py")),
    ("expC_mapem_trace.csv", ("exp_estimator_consistency.py",
                              "world_nonmonotone.py")),
    ("expC_ici_raw.csv", ("exp_estimator_consistency.py",
                          "world_nonmonotone.py")),
    ("expC_ici_floor.csv", ("exp_estimator_consistency.py",
                            "world_nonmonotone.py")),
    # the misspecification and shift study, its worlds, and the
    # additivity diagnostic with the adaptive rule (which reads the former)
    ("expW_raw.csv", ("exp_misspecification.py", "world_misspec.py",
                      "world_nonmonotone.py")),
    ("expW_judgment.csv", ("exp_misspecification.py", "world_misspec.py",
                           "world_nonmonotone.py")),
    ("expW_summary.md", ("exp_misspecification.py", "world_misspec.py",
                         "world_nonmonotone.py")),
    ("expW_diag_raw.csv", ("exp_misspecification.py", "world_misspec.py",
                           "world_nonmonotone.py")),
    ("expW_add_diag_raw.csv", ("exp_additivity.py", "exp_misspecification.py",
                               "world_misspec.py", "world_nonmonotone.py")),
    ("expW_add_adaptive_raw.csv", ("exp_additivity.py",
                                   "exp_misspecification.py",
                                   "world_misspec.py", "world_nonmonotone.py")),
    # the RF study on independent recordings and DroneDetect: its own
    # script plus the two modules it imports (CNN class, mixed-world run)
] + [(f, ("exp_rf_independent.py", "exp_rf_e2e.py", "rf_confusion.py"))
     for f in ("expI_cnn_dronerf.npz", "expI_cnn_dronedetect.npz",
               "expI_node_dronerf.csv", "expI_e2e_raw.csv",
               "expI_highthreat_bins.csv", "expI_alarm.csv",
               "expI_dd_node.csv", "expI_dd_reliability.csv",
               "expI_summary.md")
    # the EO/IR gate. Files of the node stage and the
    # bootstrap depend on the script; the G2 files are older than its last
    # edit (the node stage), so they depend on the CNN modules here, and the
    # expO block below guards the G2 code path of the script by a hash.
] + [(f, ("exp_eo_independent.py", "exp_rf_independent.py", "rf_confusion.py"))
     for f in ("expO_g2_bootstrap.csv", "expO_node.csv", "expO_node_predictions.csv",
               "expO_node_bootstrap.csv", "expO_node_cnn.npz")
] + [(f, ("exp_rf_independent.py", "rf_confusion.py"))
     for f in ("expO_g2.csv", "expO_g2_predictions.csv", "expO_g2_cnn.npz", "expO_g3_folds.csv")
    # the rule of the base framework and the runs added after its first
    # results; the type-identification figure from the saved CNN outputs
] + [(f, ("exp_base_rule.py",))
     for f in ("expB6_raw.csv", "expB6_bins.csv", "expB6_absweep_raw.csv", "expB6_summary.md")
] + [(f, ("exp_base_rule_worlds.py", "exp_base_rule.py", "exp_additivity.py",
          "exp_misspecification.py", "world_misspec.py"))
     for f in ("expB6w_flags_raw.csv", "expB6w_raw.csv", "expB6w_bins.csv", "expB6w_big_raw.csv",
               "expB6w_big_bins.csv", "expB6w_world_facts.csv", "expB6w_paired.csv",
               "expB6w_summary.md", "expB6w_checks.txt")
] + [("expI_type_id.csv", ("exp_rf_type_id.py", os.path.join("results", "expI_cnn_dronerf.npz")))
    # the nuisance parent over all four reports
] + [(f, ("exp_jam_scaling.py", "exp_correlated.py"))
     for f in ("expJ_raw.csv", "expJ_params.csv", "expJ_factors.csv", "expJ_calib_counts.csv",
               "expJ_latency.csv", "expJ_paired.csv", "expJ_checks.txt", "expJ_summary.md")
    # the control with no RF channel, what the unlabeled case identifies,
    # intervals over replications, the worked track over training seeds
] + [(f, ("exp_rf_control.py", "exp_rf_e2e.py"))
     for f in ("expRC_raw.csv", "expRC_contrasts.csv", "expRC_per_cnn.csv", "expRC_summary.md")
] + [(f, ("exp_latent_identifiability.py",))
     for f in ("expEM_rank.csv", "expEM_reach.csv", "expEM_recoding.csv", "expEM_latent_raw.csv", "expEM_summary.md")
] + [(f, ("exp_replication_intervals.py",))
     for f in ("expBS_synthetic_bins.csv", "expBS_intervals.csv", "expBS_paired.csv", "expBS_summary.md")
] + [(f, ("exp_worked_track_seeds.py", "exp_sensitivity.py"))
     for f in ("expTS_seeds.csv", "expTS_cells.csv", "expTS_rf_report.csv", "expTS_summary.md")
] + [(f, "exp_reason_checks.py")
     for f in ("expRS_b6_silent.csv", "expRS_b6_control.csv", "expRS_softmax.csv", "expRS_transfer.csv", "expRS_flags.csv", "expRS_weights.csv", "expRS_rf_bound.csv", "expRS_split.csv", "expRS_rel_scale.csv", "expRS_roots.csv", "expRS_adverse.csv", "expRS_fusion.csv", "expRS_rule_b3.csv", "expRS_track_split.csv", "expRS_flags_small.csv", "expRS_lambda.csv", "expRS_handset.csv", "expRS_summary.md")
] + [(f, ("exp_prior_deployed_ess.py", "exp_prior_ess.py"))
     for f in ("expED_raw.csv", "expED_summary.md")
] + [(f, ("exp_comparators.py", "world_misspec.py"))
     for f in ("expCM_factorial.csv", "expCM_b0.csv", "expCM_b5.csv", "expCM_scalar.csv", "expCM_node_errors.csv", "expCM_world.csv", "expCM_sens.csv", "expCM_s1.csv", "expCM_summary.md")]

# ---- banned-phrase gate: wording retired in earlier revisions must not
# reappear in the manuscript or the README.
# Wording that MISDESCRIBES the protocol, as regexes so that variants
# ("capture-session-level", "ESS selection integrated", ...) cannot slip
# through. The honest limitation "a recording-level holdout is impossible"
# is deliberately allowed -- only claims that OUR split/study is
# recording-level are banned.
BANNED = {
    r"recording-level\s+(split|study|dronerf|test|train|held-out|numbers)":
        "the DroneRF split is by segment, not by recording",
    r"with\s+recording-level": "the split unit is the segment",
    r"capture[-\s]session": "DroneRF's 227 units are segments, not sessions",
    r"sessions?\s+never\s+seen": "the held-out unit is the segment",
    r"ESS\s+is\s+select": "the ESS is fixed; nothing is selected per run",
    r"ESS\s+selection\s+(integrated|per\s+training|redrawn)":
        "the ESS is fixed for all reported runs",
    r"(fixes|fixed|held)[^.]{0,40}\ba\s+priori\b":
        "the ESS was chosen from the plateau of its sweep, so say so",
    r"credal\s+intervals?\s+propagated":
        "the analysis is a one-at-a-time envelope",
    r"measured\s+multi-sensor\s+confusion":
        "synthetic-world matrices are estimated, not measured",
    r"(now\s+contains|tracks\s+the\s+frozen|are\s+included\s+in)"
    r"[^.]{0,40}released\s+repository":
        "this text does not describe the state of the public repository",
    r"54-entry": "the threat table has 72 entries and 54 free parameters",
    r"\blabelled\b": "the manuscript uses US spelling (labeled)",
    r"fast enough for (online|real-time)":
        "the per-update inference cost is measured -- quote it",
}



DOCS = [os.path.join(HERE, "README.md")] + ([TEX] if HAVE_PAPER else [])
for doc in DOCS:
    if not os.path.exists(doc):
        fails.append(f"banned-phrase: {doc} missing")
        continue
    body = open(doc, encoding="utf-8").read()
    for pattern, why in BANNED.items():
        m = re.search(pattern, body, re.I)
        if m:
            fails.append(f"banned wording \"{m.group(0)}\" in "
                         f"{os.path.basename(doc)} -- {why}")



# ---- (a) control-character gate: LaTeX written through a non-raw Python
# string silently loses its backslash ("\\texttt" -> TAB + "exttt"), which
# builds without error and prints garbage. Tabs are legal in .py sources and
# in latexdiff's own preamble, so only the manuscript and the README are swept.
CTRL = {"\a": "BEL", "\b": "BS", "\t": "TAB", "\v": "VT", "\f": "FF"}
CTRL_DOCS = [d for d in DOCS if not d.endswith(".py")]
for doc in CTRL_DOCS:
    if not os.path.exists(doc):
        continue
    body = open(doc, encoding="utf-8", newline="").read()
    for ch, name in CTRL.items():
        if ch in body:
            where = body[:body.index(ch)].count("\n") + 1
            fails.append(f"control char: {os.path.basename(doc)} line {where} "
                         f"contains a raw {name} -- a LaTeX command was eaten "
                         "by a non-raw Python string")
    if re.search(r"\r(?!\n)", body):
        fails.append(f"control char: {os.path.basename(doc)} has a bare CR")

# A clone gives every file the time of the checkout, so modification times mean
# nothing there. They are compared where the manuscript sources are present, or
# when CUAS_GATE_FRESHNESS=1 asks for it after a re-run.
FRESH = HAVE_PAPER or os.environ.get("CUAS_GATE_FRESHNESS") == "1"
if not FRESH:
    print("note: modification times are not compared. Set CUAS_GATE_FRESHNESS=1 after a re-run.")
for rf, script in RESULT_FILES:
    path = os.path.join(RES, rf)
    if not os.path.exists(path):
        fails.append(f"freshness: {rf} missing")
        continue
    if not FRESH:
        continue
    scripts = script if isinstance(script, tuple) else (script,)
    need_after = max([core_mtime] + [os.path.getmtime(os.path.join(HERE, s))
                                     for s in scripts])
    if os.path.getmtime(path) < need_after:
        fails.append(f"freshness: {rf} is OLDER than the code that makes it "
                     f"-- rerun {scripts[0]}")


def need(label, value, decimals=3, signed=False):
    """Assert that `value`, rounded, appears in the tex. The compact table
    style drops the leading zero ($.442$), so accept that form too.
    Without a manuscript to check against, this is a no-op."""
    if not HAVE_PAPER:
        return
    s = f"{value:+.{decimals}f}" if signed else f"{value:.{decimals}f}"
    forms = {s, s.lstrip("+")}
    forms |= {f.replace("0.", ".", 1) for f in list(forms)
              if f.startswith(("0.", "+0.", "-0."))}
    forms |= {f.replace("+0.", "+.").replace("-0.", "-.") for f in list(forms)}
    if not any(f in tex for f in forms):
        fails.append(f"{label}: expected {s} in main.tex")


# ---- main study -----------------------------------------------------------
ts = pd.read_csv(os.path.join(RES, "expS_table_stats.csv"))


def cell(table, name):
    return ts[(ts.table == table) & (ts.cell == name)].iloc[0]

for N in (10, 50, 200, 400):
    r = cell("T3", f"N{N}-Proposed")
    need(f"T3 Proposed N{N} acc", r["acc"])
    need(f"T3 Proposed N{N} ece", r["ece"])
    need(f"T3 Proposed N{N} auc", r["auc"])
    r = cell("T3", f"N{N}-B1-MLE")
    need(f"T3 MLE N{N} acc", r["acc"])
for cfg in ["all (abstract model)", "all (calibrated)", "all (calib+reliab)"]:
    r = cell("T4", cfg)
    need(f"T4 {cfg} acc", r["acc"])
    need(f"T4 {cfg} ece", r["ece"])
    need(f"T4 {cfg} auc", r["macroAUC"])

fus = pd.read_csv(os.path.join(RES, "expS_fusion_raw.csv"))
ps = fus[fus.config == "_paired_stats"]
need("fusion dAUC mean", ps["dAUC"].mean(), signed=True)
need("fusion dECE mean", ps["dECE"].mean(), signed=True)
for sub, cfgname, met, dec in [
        ("silent-drone", "abstract", "acc", 3),
        ("silent-drone", "calibrated", "acc", 3),
        ("far-silent", "abstract", "ece", 3),
        ("far-silent", "calibrated", "ece", 3)]:
    d = fus[fus.config == f"_subset_{sub}_{cfgname}"]
    need(f"subset {sub} {cfgname} {met}", d[met].mean(), dec)

# ---- E2 / E3 / E4 / EM ----------------------------------------------------
pr = pd.read_csv(os.path.join(RES, "expE_prior_raw.csv"))
x = pr[(pr.lam == 1.0) & (pr.ess == 50.0) & (pr.N == 50)]
need("E2 lam1 ess50 N50 MAP ece", x[x.method == "MAP"]["ece"].mean())
need("E2 lam1 ess50 N50 cMAP ece", x[x.method == "cMAP"]["ece"].mean())
x8 = pr[(pr.lam == 1.0) & (pr.ess == 8.0) & (pr.N == 400)]
need_supp_later("E2 lam1 ess8 N400 cMAP acc", f"(${x8[x8.method == 'cMAP']['acc'].mean():.3f}$ vs.\\ MLE")
need_supp_later("E2 lam1 ess8 N400 cMAP ece", f"(ECE ${x8[x8.method == 'cMAP']['ece'].mean():.3f}$; MAP")

fc = pd.read_csv(os.path.join(RES, "expF_corr_raw.csv"))
s1 = fc[fc.severity == 1.0]
need("E3 naive adv ece s1", s1[s1.fusion == "naive-CI"]["ece_adv"].mean())
need("E3 W-obs adv ece s1", s1[s1.fusion == "W-observed"]["ece_adv"].mean())
need("E3 W-latent adv ece s1", s1[s1.fusion == "W-latent"]["ece_adv"].mean())

gg = pd.read_csv(os.path.join(RES, "expG_rf_e2e_raw.csv"))
for name in ["abstract", "calib-hard"]:
    d = gg[gg.rf_likelihood == name]
    need(f"E4 {name} auc", d["macroAUC"].mean())
    need(f"E4 {name} ece", d["ece"].mean())

em = pd.read_csv(os.path.join(RES, "expM_raw.csv"))
d = em[(em.N == 200) & (em.method == "MAP-EM (latent)")]
need("EM N200 acc", d["acc"].mean())
need("EM N200 ece", d["ece"].mean())

# ---- CNN ------------------------------------------------------------------
cc = np.load(os.path.join(RES, "expG_cnn.npz"), allow_pickle=True)
need("CNN recording test acc (as %)", float(cc["test_acc"]) * 100, 1)

# ---- standard baselines and the noise sweep -------------------------------
bl = pd.read_csv(os.path.join(RES, "expB_baselines_raw.csv"))
for N, meth, keys in [(10, "uninformative MAP", ("acc", "macroAUC", "ece")),
                      (10, "logistic regression", ("acc", "macroAUC")),
                      (400, "logistic regression", ("acc", "macroAUC", "ece")),
                      (50, "B0-as-prior MAP", ("acc", "macroAUC", "ece"))]:
    d = bl[(bl.N == N) & (bl.method == meth)]
    for k in keys:
        need(f"expB {meth} N{N} {k}", d[k].mean())

# The Sec. VII-B ablation path: every step must change exactly one thing, so
# the gate walks the same chain the manuscript does. Quoting a bundled delta
# as if it belonged to the prior alone is the error this catches.
_PATH = ["B1-MLE", "uninformative MAP", "expert MAP (full table)",
         "expert cMAP (full table)", "Proposed"]
_STEPS = ["smoothing", "prior mean", "constraints", "ICI"]
_w = {m: bl[(bl.N == 80) & (bl.method == m)].set_index("rep") for m in _PATH}
for _c, _a, _lab in zip(_PATH, _PATH[1:], _STEPS):
    for k in ("ece", "macroAUC"):
        need(f"expB N80 {_lab} {k}", (_w[_a][k] - _w[_c][k]).mean(),
             signed=True)
for k in ("ece", "macroAUC"):
    _sum = sum((_w[_a][k] - _w[_c][k]).mean()
               for _c, _a in zip(_PATH, _PATH[1:]))
    _tot = (_w["Proposed"][k] - _w["B1-MLE"][k]).mean()
    if abs(_sum - _tot) > 5e-4:
        fails.append(f"expB N80 {k}: the steps sum to {_sum:+.4f} but the "
                     f"total is {_tot:+.4f} -- the path is not a chain")

nz = pd.read_csv(os.path.join(RES, "expN_noise_raw.csv"))
for tau in (0.25, 0.6):
    d = nz[(nz.tau == tau) & (nz.method == "CPT-oracle")]
    need(f"expN oracle acc tau={tau}", d["acc"].mean())

# ---- measured cost and the per-class decomposition ------------------------
lat = pd.read_csv(os.path.join(RES, "expL_latency.csv"))
nq = int(lat.n_queries.sum())
if HAVE_PAPER and f"{nq//1000},{nq%1000:03d}" not in tex:
    fails.append(f"expL: expected the timed-query count {nq:,} in main.tex")
need("expL mean query ms", lat.q_mean_ms.mean())
need("expL p95 query ms", lat.q_p95_ms.mean())
need("expL max query ms", lat.q_max_ms.max())
need("expL assemble ms", lat.assemble_s.mean() * 1e3, 1)
need_supp_later("expL MAP-EM s", f"${lat.mapem_s.mean():.2f}$ s against")
need_supp_later("expL grounding s", f"${lat.ground_s.mean():.2f}$ s at")
need("expL EM/supervised ratio", lat.mapem_s.mean() / lat.ground_s.mean(), 1)

import model_spec as _ms
rel = pd.read_csv(os.path.join(RES, "expR_reliability_classes.csv"))
rel["dev"] = rel.observed - rel.predicted
for cls in _ms.STATES["T"]:
    d = rel[(rel["class"] == cls) & (rel.bin == "(0.50,0.75]")]
    if d.empty:
        fails.append(f"expR: no (.50,.75] row for threat level {cls!r}")
        continue
    need(f"expR {cls} dev (.50,.75]", d["dev"].iloc[0], signed=True)
need("expR high top-bin dev",
     rel[(rel["class"] == "high") & (rel.bin == "(0.75,1.00]")]["dev"].iloc[0],
     signed=True)
oracle_max = (rel.oracle_observed - rel.oracle_predicted).abs().max()
need("expR max oracle deviation", oracle_max)

# ---- the ECE bin-count sweep quoted in Sec. VI-D -----------------
kb = pd.read_csv(os.path.join(RES, "expK_ece_bins.csv"))
_prop = kb[kb.model == "Proposed"].groupby("bins")["ece"].mean()
_orac = kb[kb.model == "CPT-oracle"].groupby("bins")["ece"].mean()
need("expK proposed ECE min", _prop.min())
need("expK proposed ECE max", _prop.max())
need("expK oracle ECE min", _orac.min())
need("expK oracle ECE max", _orac.max())
_gap = []
for _b in sorted(kb.bins.unique()):
    _p = kb[(kb.bins == _b) & (kb.model == "Proposed")]\
        .set_index("rep")["ece"]
    _o = kb[(kb.bins == _b) & (kb.model == "CPT-oracle")]\
        .set_index("rep")["ece"]
    _gap.append((_p - _o).mean())
need("expK excess over oracle, min", min(_gap))
need("expK excess over oracle, max", max(_gap))

# The threat-level names are defined once, in the model spec; a table that
# invents its own (an earlier draft said "guarded") silently mislabels rows.
for _lvl in _ms.STATES["T"]:
    if _lvl not in set(rel["class"]):
        fails.append(f"labels: results file omits threat level {_lvl!r}")
# Presence anywhere in the paper proves nothing -- Table 1 names every level
# already. Check the row labels of the per-class block itself, in order.
_block = re.search(r"observed \$-\$ predicted, all four classes"
                   r"(.{0,600}?)\\bottomrule", tex, re.S)
if not HAVE_PAPER:
    pass
elif not _block:
    fails.append("labels: the per-class block of Table 8 was not found")
else:
    _rows = re.findall(r"(?:\\\\|\}\})\s*([A-Za-z]+)\s*&", _block.group(1))
    if _rows[:4] != list(_ms.STATES["T"]):
        fails.append(f"labels: Table 8 rows are {_rows[:4]}, but the model "
                     f"spec defines {list(_ms.STATES['T'])}")
for _bad in set(rel["class"]) - set(_ms.STATES["T"]):
    fails.append(f"labels: results file invents threat level {_bad!r}")

# ---- the reliability-layer study (Secs. V-B, VII-B, Table 8) --------
# Every number the manuscript quotes from exp_reliability_variants.py is
# recomputed from its raw replication files, and every claim the text makes
# in words (direction, significance, break-even) is checked against them.
from scipy import stats as _st


def _rt(name):
    # exact parse: pandas' default float converter is not correctly rounded
    return pd.read_csv(os.path.join(RES, name), float_precision="round_trip")


_dr, _dt, _dp = (_rt("expD_runs_raw.csv"), _rt("expD_trueCPT_raw.csv"),
                 _rt("expD_population.csv"))
_n_expd = 0


def _sel(df, world, variant, subset, n=300):
    q = df[(df.world == world) & (df.variant == variant)
           & (df.subset == subset) & (df.calib_n == n)]
    if q.rep.nunique() != 20:
        fails.append(f"expD: {world}/{variant}/{subset}/n={n} has "
                     f"{q.rep.nunique()} replications, not 20")
    return q.set_index("rep")


def _paired(df, world, a, b, subset, k, n=300):
    d = (_sel(df, world, a, subset, n)[k]
         - _sel(df, world, b, subset, n)[k]).dropna()
    return float(d.mean()), float(_st.ttest_1samp(d.values, 0.0).pvalue)


def _ptex(p, dec):
    if p < 1e-4:
        return "p{<}10^{-4}"
    if p < 1e-3:
        return "p{<}10^{-3}"
    return f"p{{=}}{p:.{dec}f}"


def need_tex(label, snippet):
    global _n_expd
    _n_expd += 1
    if HAVE_PAPER and snippet not in tex:
        fails.append(f"{label}: expected {snippet!r} in main.tex")


def need_true(label, cond):
    global _n_expd
    _n_expd += 1
    if not cond:
        fails.append(f"{label}: the manuscript's wording no longer matches "
                     "the results")


# Sec. V-B: the EO/IR population weights, and the exactness claim
_pe = _dp[(_dp.world == "standard") & (_dp.source == "population")]
_D = ("far", "near", "imminent")
_sh = _pe[_pe.sensor == "Re"].set_index(["variant", "d"])["share"]
need_tex("expD EO/IR deployed weight",
         "$" + "/".join(f"{_sh[('V0', d)]:.3f}" for d in _D) + "$")
need_tex("expD EO/IR world reliability",
         "$" + "/".join(f"{_sh[('truth', d)]:.2f}" for d in _D) + "$")
need_tex("expD EO/IR no-discount weight",
         f"${_sh[('V3', 'far')]:.3f}$ with no discount")
_v1 = _pe[_pe.variant == "V1"].set_index(["sensor", "d"])["share"]
_tr = _pe[_pe.variant == "truth"].set_index(["sensor", "d"])["share"]
need_true("expD chance-corrected layer is exact at population level",
          len(_v1) == 9 and float((_v1 - _tr).abs().max()) < 1e-9)

# Table 8: five rows x (standard, structural) x (Brier, ECE, silent Brier)
_T8 = [("Deployed (V0)", "V0"), ("Chance-corrected (V1)", "V1"),
       ("Range-conditional (V2)", "V2"), ("No discount (V3)", "V3"),
       ("Exact tables (oracle)", "sensor-oracle-conditional")]
_blk = re.search(r"\\label\{tab:relvariants\}(.*?)\\bottomrule", tex, re.S)
if HAVE_PAPER and not _blk:
    fails.append("expD: Table 8 (tab:relvariants) not found in main.tex")
elif HAVE_PAPER:
    for _lab, _v in _T8:
        _want = [f"{_sel(_dr, w, _v, sub)[k].mean():.4f}"
                 for w in ("standard", "structural")
                 for sub, k in (("all", "brier"), ("all", "ece"),
                                ("silent-drone", "brier"))]
        _m = re.search(re.escape(_lab) + r"\s*&(.*?)\\\\", _blk.group(1))
        _got = re.findall(r"\$([0-9.]+)\$", _m.group(1)) if _m else None
        _n_expd += 1
        if _got != _want:
            fails.append(f"expD Table 8 row {_lab!r}: main.tex {_got} != "
                         f"results {_want}")

# Sec. VII-B text
_d, _p = _paired(_dr, "standard", "V0", "V3", "all", "ece")
need_tex("expD average ECE, discount vs none",
         rf"changes little (Table~\ref{{tab:sensors}}; ECE ${_ptex(_p, 2)}$)")
need_true("expD average ECE change not significant", _p >= 0.05)
_d, _p = _paired(_dr, "standard", "V0", "V3", "silent-drone", "brier")
need_tex("expD radio-silent Brier, discount vs none",
         f"hurts radio-silent drones (Brier ${_d:+.4f}$, ${_ptex(_p, 3)}$)")
need_true("expD discount hurts radio-silent drones", _d > 0 and _p < 0.05)
_dB, _pB = _paired(_dr, "standard", "V0", "V3", "far", "brier")
_dE, _pE = _paired(_dr, "standard", "V0", "V3", "far", "ece")
need_tex("expD far-range Brier/ECE",
         f"(Brier ${_ptex(_pB, 3)}$, ECE ${_ptex(_pE, 3)}$)")
need_true("expD far-range improvement", _dB < 0 and _dE < 0)
_d, _p = _paired(_dt, "standard", "V0|true-internal",
                 "sensor-oracle-conditional|true-internal", "all", "conf_gap")
need_supp_later("expD double discount on the true threat layer",
         f"lower by ${abs(_d):.3f}$, ${_ptex(_p, 3)}$")
need_true("expD double discount direction", _d < 0 and _p < 0.05)
_d, _p = _paired(_dr, "standard", "V1", "V0", "all", "brier")
need_supp_later("expD chance-corrected Brier", f"unchanged (${_ptex(_p, 2)}$)")
need_true("expD chance-corrected Brier tie", _p >= 0.05)
_d, _p = _paired(_dr, "standard", "V1", "V0", "all", "ece")
need_supp_later("expD chance-corrected ECE", f"higher ECE (${_ptex(_p, 3)}$)")
need_true("expD chance-corrected ECE worse", _d > 0 and _p < 0.05)
_g = _sel(_dt, "standard", "sensor-oracle-conditional|learned-internal",
          "all")["conf_gap"].mean()
need_supp_later("expD learned threat layer over-confidence",
         f"over-confident by ${_g:.3f}$")
_, _pb = _paired(_dt, "standard", "V1|true-internal", "V0|true-internal",
                 "all", "brier")
_, _pe2 = _paired(_dt, "standard", "V1|true-internal", "V0|true-internal",
                  "all", "ece")
need_supp_later("expD true threat layer tie",
         f"(Brier ${_ptex(_pb, 2)}$, ECE ${_ptex(_pe2, 2)}$)")
need_true("expD true threat layer tie", _pb >= 0.05 and _pe2 >= 0.05)
# exact range decomposition of the radio-silent Brier change (V0 - V3)
_tot, _far = [], []
for _r in range(20):
    _c = {}
    for _dn in ("far", "near", "imminent"):
        _a = _sel(_dr, "standard", "V0", f"{_dn}-silent").loc[_r]
        _b = _sel(_dr, "standard", "V3", f"{_dn}-silent").loc[_r]
        _c[_dn] = _a["n_sub"] * (_a["brier"] - _b["brier"])
    _ns = _sel(_dr, "standard", "V0", "silent-drone").loc[_r, "n_sub"]
    _tot.append(sum(_c.values()) / _ns)
    _far.append(_c["far"] / _ns)
_share_far = float(np.mean(_far) / np.mean(_tot))
need_supp_later("expD radio-silent decomposition",
         "wo-thirds of the radio-silent Brier loss arises at far range and "
         "one-third at near and imminent range")
need_true(f"expD far share of the radio-silent loss ({_share_far:.3f})",
          0.60 <= _share_far <= 0.73)
_d, _p = _paired(_dr, "standard", "sensor-oracle-conditional",
                 "sensor-oracle-marginal", "far-silent", "brier")
need_supp_later("expD selection effect on far-range radio-silent drones",
         f"(${_d:+.3f}$ on far-range radio-silent drones, ${_ptex(_p, 3)}$)")
_d2, _p2 = _paired(_dr, "standard", "sensor-oracle-conditional",
                   "sensor-oracle-marginal", "far", "brier")
need_true("expD exact tables improve the far-range subset",
          _d > 0 and _d2 < 0 and _p2 < 0.05)
# the range-conditional matrix: radio-silent recovery and break-even
_s0 = _sel(_dr, "structural", "V0", "silent-drone")["brier"].mean()
_s2 = _sel(_dr, "structural", "V2", "silent-drone")["brier"].mean()
_so = _sel(_dr, "structural", "sensor-oracle-conditional",
           "silent-drone")["brier"].mean()
_d, _p = _paired(_dr, "structural", "V2", "V0", "silent-drone", "brier")
need_true("expD V2 recovers much of the radio-silent loss",
          _d < 0 and _p < 0.05 and (_s0 - _s2) / (_s0 - _so) >= 0.5)
_better = {}
for _w in ("standard", "structural"):
    for _n in (30, 100, 300, 1000):
        _d, _p = _paired(_dr, _w, "V2", "V0", "all", "brier", _n)
        _better[(_w, _n)] = ("better" if _d < 0 and _p < 0.05 else
                             "worse" if _d > 0 and _p < 0.05 else "tie")
need_supp_later("expD V2 break-even",
         "only in that world and only with $n{=}1000$ labeled scenarios, and "
         "it loses at $n{=}30$ in both worlds")
need_true(f"expD V2 break-even pattern {_better}",
          [k for k, v in _better.items() if v == "better"]
          == [("structural", 1000)]
          and _better[("standard", 30)] == "worse"
          and _better[("structural", 30)] == "worse")
if HAVE_PAPER and _n_expd == 0:
    fails.append("expD: no check ran")
if DRAFT:      # the count is printed in draft mode only: default output is fixed
    print(f"expD checks run: {_n_expd} (Table 8 rows count once each)")

# ---- the estimator study (Sec. IV-A, IV-C, IV-D, Appendix A) ------
# Numbers quoted from exp_estimator_consistency.py, recomputed from its raw
# files; claims made in words (monotone decrease, no label switching,
# "only at N=5,000", consistency) are checked against them.
_n0 = _n_expd
_cc = _rt("expC_consistency_raw.csv")
_cp = _rt("expC_pseudotrue.csv")
_cm = _rt("expC_mapem_raw.csv")
_ct = _rt("expC_mapem_trace.csv")
_ci = _rt("expC_ici_raw.csv")
_cf = _rt("expC_ici_floor.csv")


def _sci(x):
    m, e = f"{x:.1e}".split("e")
    return rf"{m}\times10^{{{int(e)}}}"


def _cmean(world, node, est, N, col):
    q = _cc[(_cc.world == world) & (_cc.node == node) & (_cc.estimator == est)
            & (_cc.N == N)][col]
    if len(q) != 20:
        fails.append(f"expC: {world}/{node}/{est}/N={N} has {len(q)} reps")
    return float(q.mean())


_NS = sorted(_cc.N.unique())
# Sec. IV-C
need_tex("expC threat-table KL N=50 -> 20,000",
         f"falls from ${_cmean('standard', 'T', 'cMAP', 50, 'wkl'):.2f}$ at "
         f"$N{{=}}50$ to ${_cmean('standard', 'T', 'cMAP', 20000, 'wkl'):.3f}$"
         " at $N{=}20,000$")
for _node in ("H", "N", "C", "T"):
    _seq = [_cmean("standard", _node, "cMAP", n, "wkl") for n in _NS]
    need_true(f"expC cMAP KL decreases with N for {_node}",
              all(a > b for a, b in zip(_seq, _seq[1:])))
_slopes = {}
for _node in ("H", "N", "C"):
    _m = [_cmean("standard", _node, "cMAP", n, "maxabs_cmap_cmle") for n in _NS]
    _slopes[_node] = float(np.polyfit(np.log(_NS), np.log(_m), 1)[0])
need_supp_later("expC cMAP-cMLE rate", "(log--log slope $-1.0$ for $H$, $N$, and $C$;")
need_true(f"expC cMAP-cMLE slopes {_slopes}",
          all(f"{s:+.1f}" == "-1.0" for s in _slopes.values()))
_mT = [_cmean("standard", "T", "cMAP", n, "maxabs_cmap_cmle") for n in _NS]
_sT = float(np.polyfit(np.log(_NS), np.log(_mT), 1)[0])
_wT = _ms.ground_truth_bn().query(["H", "N", "C"]).table.reshape(-1)
_nrare = int((_wT < 0.01).sum())
_word = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six"}
need_supp_later("expC rare threat-parent configurations",
         f"the threat table, {_word.get(_nrare, _nrare)} of whose {_wT.size} parent "
         "configurations have probability below $0.01$, approaches this rate "
         "more slowly")
need_true(f"expC threat table approaches the rate more slowly (slope {_sT:+.2f})",
          _sT > max(_slopes.values()) + 0.1)
need_tex("expC non-monotone world: distance to the projection",
         f"(KL ${_cmean('H-v-U', 'H', 'cMAP', 20000, 'wkl_to_pseudotrue'):.3f}$"
         " at $N{=}20,000$)")
_proj = float(_cp[_cp.world == "H-v-U"].kl_truth_to_projection.iloc[0])
need_tex("expC non-monotone world: projection vs truth",
         f"which lies at KL ${_proj:.2f}$ from the truth")
_seq = [_cmean("H-v-U", "H", "MAP", n, "wkl") for n in _NS]
need_true("expC unconstrained posterior mean converges to the truth",
          all(a > b for a, b in zip(_seq, _seq[1:])) and _seq[-1] < 0.01)
# Sec. IV-A
_fl = float(_cf.kl_truth_to_family.iloc[0])
need_tex("expC ICI family floor", f"lies at KL ${_sci(_fl)}$ from the truth")
_ici = _ci.groupby("N")[["wkl_ici", "wkl_full_cmap"]].mean()
need_tex("expC ICI vs full table at N=20,000",
         f"(KL ${_sci(_ici.wkl_ici[20000])}$ against "
         f"${_sci(_ici.wkl_full_cmap[20000])}$)")
need_true("expC ICI closer to the truth than the full table at every N",
          bool((_ici.wkl_ici < _ici.wkl_full_cmap).all()))
# Sec. IV-D and Appendix A
_num = _ct[_ct.iter.astype(str).str.fullmatch(r"\d+")].copy()
_num["iter"] = _num["iter"].astype(int)
_steps = _dec = 0
_rel = {}
for (_r, _N, _v), _g in _num.groupby(["rep", "N", "variant"]):
    _lp = _g.sort_values("iter")["logpost"].values
    _dl = np.diff(_lp)[0 if _v == "final-only" else 1:]
    _steps += len(_dl)
    _dec += int((_dl < -1e-9 * np.abs(_lp).max()).sum())
    if _v == "final-only":
        _rel.setdefault(_N, []).append(abs(_lp[-1] - _lp[12]) / abs(_lp[12]))
need_tex("expC MAP-EM monotone steps",
         f"(0 of ${_steps // 1000},{_steps % 1000:03d}$ iterations)")
need_true(f"expC MAP-EM log posterior never decreased ({_dec} decreases)",
          _dec == 0)
need_supp_later("expC MAP-EM not converged at 12",
         r"rises by about $0.2\%$ by iteration 50")
need_true(f"expC MAP-EM relative change 12->50 near 0.2% "
          f"{ {k: round(float(np.mean(v)), 5) for k, v in _rel.items()} }",
          all(0.0015 <= np.mean(v) <= 0.0025 for v in _rel.values()))
_f = _cm[_cm.variant == "final-only"].set_index(["N", "rep", "iters"])
_e = _cm[_cm.variant == "every-iteration"].set_index(["N", "rep", "iters"])


def _d50(N, k):
    d = (_f.loc[N].xs(50, level="iters")[k]
         - _f.loc[N].xs(12, level="iters")[k]).values
    return float(d.mean()), float(_st.ttest_1samp(d, 0.0).pvalue)


_dE, _pE = _d50(200, "ece")
need_supp_later("expC 50 vs 12 iterations, ECE at N=200",
         f"(ECE ${_dE:+.3f}$, ${_ptex(_pE, 3)}$)")
_dA, _pA = _d50(5000, "macroAUC")
need_supp_later("expC 50 vs 12 iterations, AUC at N=5,000",
         f"(AUC ${_dA:+.4f}$, ${_ptex(_pA, 3)}$)")
_k50 = [float(_f.loc[n].xs(50, level="iters")["wkl_H"].mean()) for n in (200, 1000, 5000)]
_k12 = float(_f.loc[5000].xs(12, level="iters")["wkl_H"].mean())
need_supp_later("expC MAP-EM latent-table error at 50 iterations",
         f"the column-weighted KL of $H$ to the truth falls in this world (${_k50[0]:.2f}$, ${_k50[1]:.2f}$, "
         f"${_k50[2]:.3f}$ at $N{{=}}200$, $1,000$, $5,000$)")
need_supp_later("expC MAP-EM 12-iteration gap",
         f"while 12 iterations stop at ${_k12:.3f}$ at $N{{=}}5,000$")
need_true(f"expC MAP-EM latent error falls with N at 50 iterations {_k50}",
          _k50[0] > _k50[1] > _k50[2] and _k12 > _k50[2])
need_true("expC more iterations help discrimination only at N=5,000",
          _dA > 0 and _pA < 0.05
          and all(not (_d50(n, "macroAUC")[0] > 0 and _d50(n, "macroAUC")[1] < 0.05)
                  for n in (200, 1000)))
_mx = max(abs(float((_e.loc[N].xs(12, level="iters")[k]
                     - _f.loc[N].xs(12, level="iters")[k]).mean()))
          for N in (200, 1000, 5000) for k in ("acc", "macroAUC", "ece"))
need_tex("expC every-iteration vs final-only",
         "changes the threat metrics by at most $0.002$")
need_true(f"expC every-iteration vs final-only at 12 ({_mx:.4f})",
          round(_mx, 3) <= 0.002)
need_tex("expC no label switching", "no latent label switched")
need_true("expC no label switching",
          bool(_cm[["label_ok_H", "label_ok_N", "label_ok_C"]].values.all()))
_drop = []
for (_r, _N, _v), _g in _ct[_ct.variant == "final-only"].groupby(["rep", "N", "variant"]):
    _s = _g.set_index(_g.iter.astype(str))["logpost"]
    _drop.append((_N, float(_s["12"] - _s["12c"])))
_dm = pd.DataFrame(_drop, columns=["N", "d"]).groupby("N").d.mean()
need_tex("expC final constrained step drop",
         f"lowers the log posterior by ${_dm.min():.1f}$--${_dm.max():.1f}$ "
         "nats on average")
need_true("expC final constrained step lowers the log posterior",
          bool((_dm > 0).all()))
if DRAFT:
    print(f"expC checks run: {_n_expd - _n0}")

# ---- misspecification and shift (Sec. VII-G, Table tab:misspec) ----
# Numbers quoted from exp_misspecification.py and exp_additivity.py,
# recomputed from their raw files. The verdict column is recomputed here
# from the rule fixed before the run [J] (independently of the script's own
# judge()), and the claims made in words are checked against the results.
_n0w = _n_expd
_wr = _rt("expW_raw.csv")
_wj = _rt("expW_judgment.csv")
_wd = _rt("expW_diag_raw.csv")
_wa = _rt("expW_add_diag_raw.csv")
_wad = _rt("expW_add_adaptive_raw.csv")
_WALT = ["cMAP-full", "MAP", "B1-MLE", "B5-logistic"]
_WLAB = {"standard": "Standard", "nonadd-0.5": "S1 weak", "nonadd-1.0": "S1 strong",
         "H-v-U": "S2", "edges-0.5": "S3 weak", "edges-1.5": "S3 strong",
         "more-drones": "S4a drones", "more-silent": "S4a silent",
         "sensor-x1.5": "S4b sensors", "C-weight-x2": r"S4c $C{\times}2$",
         "tauT-0.60": r"S4c $\tau_T$"}
_WSH = {"cMAP-full": "cMAP", "MAP": "MAP", "B1-MLE": "MLE",
        "B5-logistic": "logistic", "Proposed-Hfree": "$H$-free"}
_WVW = {"holds": "hold", "degrades": "degr.", "breaks": "break"}


def _wb(world, method, N=-1, k="brier"):
    q = _wr[(_wr.world == world) & (_wr.method == method) & (_wr.N == N)]
    if len(q) != 20:
        fails.append(f"expW: {world}/{method}/N={N} has {len(q)} reps")
    return q.set_index("rep")[k]


def _wpaired(a, b):
    d = (a - b).dropna()
    return float(d.mean()), float(_st.ttest_1samp(d, 0.0).pvalue)


def _wjudge(world, N):
    """Rule [J] of exp_misspecification.py, recomputed."""
    alts = _WALT + (["Proposed-Hfree"] if world == "H-v-U" else [])
    best = min(alts, key=lambda m: _wb(world, m, N).mean())
    dm, p = _wpaired(_wb(world, "Proposed", N), _wb(world, best, N))
    exc = float((_wb(world, "Proposed", N) - _wb(world, "CPT-oracle")).mean())
    exc0 = float((_wb("standard", "Proposed", N)
                  - _wb("standard", "CPT-oracle")).mean())
    if dm > 0 and p < 0.05:
        v = "breaks"
    else:
        v = "holds" if exc <= 2 * exc0 else "degrades"
    return best, dm, p, exc, v


_wv = {}
_wgraph = {}
for _w in _WLAB:
    _co = _wb(_w, "CPT-oracle")
    _wgraph[_w] = float((_co - _wb(_w, "Bayes-oracle")).mean())
    _fam = (f"${float((_wb(_w, 'family-best') - _co).mean()):.3f}$"
            if _w in ("nonadd-0.5", "nonadd-1.0", "H-v-U") else "--")
    _j = {N: _wjudge(_w, N) for N in (50, 200)}
    _wv[_w] = _j
    _star = "^{*}" if _j[200][2] < 0.05 else ""
    need_tex(f"expW table row {_w}",
             f"{_WLAB[_w]} & ${_wgraph[_w]:.3f}$ & {_fam} & ${_j[50][3]:.3f}$/"
             f"${_j[200][3]:.3f}$ & {_WSH[_j[200][0]]} ${_j[200][1]:+.3f}{_star}$ & "
             f"{_WVW[_j[50][4]]}/{_WVW[_j[200][4]]} \\\\")
need_true("expW verdicts agree with expW_judgment.csv",
          all(_wj[(_wj.world == _w) & (_wj.N == N)].verdict.iloc[0] == _wv[_w][N][4]
              for _w in _WLAB for N in (50, 200)))
_wverd = {_w: tuple(_wv[_w][N][4] for N in (50, 200)) for _w in _WLAB}

# scenario definitions quoted in the text (world_misspec.py)
import world_misspec as _wm
_wsd = _wm.world_bn("more-drones").cpts["cls"].table
_wse = _wm.world_bn("more-silent").cpts["emit"].table
need_tex("expW S4a definitions", r"(S4a: $60\%$ drones, $80\%$ radio-silent)")
need_true("expW S4a worlds: 60% drones, 80% silent",
          abs(_wsd[2] - 0.60) < 1e-12 and abs(_wse[1] - 0.80) < 1e-12)
need_tex("expW S1/S3/S4 levels",
         r"of weight $0.5$ (weak) or $1.5$ (strong) are absent from the model (S3)")
need_true("expW world levels (S1 0.5/1.0, S3 0.5/1.5, S4b x1.5, S4c tau 0.60)",
          _wm.WORLDS_S1 == ("nonadd-0.5", "nonadd-1.0")
          and _wm.WORLDS_S3 == ("edges-0.5", "edges-1.5")
          and "sensor-x1.5" in _wm.WORLDS_S4 and "tauT-0.60" in _wm.WORLDS_S4
          and "C-weight-x2" in _wm.WORLDS_S4)

# KL of the S1 threat tables (and the standard one) from the ICI family
_wkl = {}
for _l in open(os.path.join(RES, "expW_summary.md"), encoding="utf-8"):
    _m = re.match(r"- (\S+): KL\(T \|\| ICI\) ([0-9.e+-]+)", _l)
    if _m:
        _wkl[_m.group(1)] = float(_m.group(2))
need_tex("expW S1 KL from the ICI family",
         f"(S1, at KL ${_wkl['nonadd-0.5']:.3f}$ and ${_wkl['nonadd-1.0']:.2f}$ "
         "from the ICI family)")
need_tex("expW Discussion KL of the standard world",
         f"the ICI family approximates to KL ${_sci(_wkl['standard'])}$")

# verdicts stated in words
need_true("expW S1, S2 break at both N",
          all(_wverd[_w] == ("breaks", "breaks")
              for _w in ("nonadd-0.5", "nonadd-1.0", "H-v-U")))
need_true("expW S3: estimator holds, graph loss at least 10x the standard",
          all(_wverd[_w] == ("holds", "holds") and _wv[_w][N][1] < 0
              and _wv[_w][N][2] < 0.05 for _w in ("edges-0.5", "edges-1.5")
              for N in (50, 200))
          and min(_wgraph["edges-0.5"], _wgraph["edges-1.5"])
          >= 10 * _wgraph["standard"])
need_true("expW S4 degrades only for silent targets (N=200) and C weight x2",
          _wverd["more-silent"] == ("holds", "degrades")
          and _wverd["C-weight-x2"] == ("degrades", "degrades")
          and all(_wverd[_w] == ("holds", "holds")
                  for _w in ("more-drones", "sensor-x1.5", "tauT-0.60")))
_we0 = float(_wb("standard", "Proposed", 200, "ece").mean())
_wet = float(_wb("tauT-0.60", "Proposed", 200, "ece").mean())
need_tex("expW tau_T ECE", f"its ECE triples (${_we0:.3f}$ to ${_wet:.3f}$)")
need_true(f"expW tau_T ECE triples ({_wet / _we0:.2f}x)", 2.5 <= _wet / _we0 <= 3.5)
need_true("expW logistic trails Proposed in every cell",
          all(_wb(_w, "B5-logistic", N).mean() > _wb(_w, "Proposed", N).mean()
              for _w in _WLAB for N in (50, 200)))
need_true("expW shift toward silent targets or in the threat response "
          "degrades calibration",
          all(_wpaired(_wb(_w, "Proposed", N, "ece"),
                       _wb("standard", "Proposed", N, "ece"))[0] > 0
              and _wpaired(_wb(_w, "Proposed", N, "ece"),
                           _wb("standard", "Proposed", N, "ece"))[1] < 0.05
              for _w in ("more-silent", "C-weight-x2", "tauT-0.60")
              for N in (50, 200)))

# the two diagnostics and the adaptive rule
_waA = _wa[_wa.test == "A"]
def _wflags(df, world, N, node=None):
    q = df[(df.world == world) & (df.N == N)]
    if node is not None:
        q = q[q.node == node]
    return int(q.flag.sum()), len(q)
_wA50 = [_wflags(_waA, _w, 50) for _w in ("nonadd-0.5", "nonadd-1.0")]
_wA200 = [_wflags(_waA, _w, 200) for _w in ("nonadd-0.5", "nonadd-1.0")]
_wA0 = sum(_wflags(_waA, "standard", N)[0] for N in (50, 200, 1000))
_wA0n = sum(_wflags(_waA, "standard", N)[1] for N in (50, 200, 1000))
_wnum = {1: "one", 2: "two", 3: "three"}
need_tex("expW additivity test detection and false alarms",
         f"flags the ICI node in every S1 replication at $N{{=}}200$ "
         f"(weak and strong: {_wA50[0][0]} and {_wA50[1][0]} of 20 at $N{{=}}50$;")
need_true("expW additivity test: false alarms from N = 50, one of 60", _wA0 == 1 and _wA0n == 60)
need_true("expW additivity test flags every S1 replication at N=200",
          all(f == n == 20 for f, n in _wA200))
_wD = [_wflags(_wd, "H-v-U", N, "H")[0] for N in (50, 200)]
_wD0 = sum(_wflags(_wd, "standard", N, "H")[0] for N in (50, 200, 1000))
need_tex("expW monotonicity test on H",
         f"flags the $H$ constraint under S2 in {_wD[0]} and {_wD[1]} of 20 at "
         "$N{=}50$ and $200$ but never in the standard world")
need_true("expW monotonicity test never flags H in the standard world", _wD0 == 0)


def _wada(world, N):
    return _wad[(_wad.world == world) & (_wad.N == N)].set_index("rep").brier


_wmatch = all(_wpaired(_wada(_w, 200), _wb(_w, alt, 200))[1] >= 0.05
              or _wpaired(_wada(_w, 200), _wb(_w, alt, 200))[0] == 0
              for _w, alt in (("nonadd-0.5", "cMAP-full"),
                              ("nonadd-1.0", "cMAP-full"),
                              ("H-v-U", "Proposed-Hfree")))
_wbetter = all(_wb(_w, alt, 200).mean() < _wb(_w, "Proposed", 200).mean()
               for _w, alt in (("nonadd-0.5", "cMAP-full"),
                               ("nonadd-1.0", "cMAP-full"),
                               ("H-v-U", "Proposed-Hfree")))
need_tex("expW adaptive rule", "then matches the better fixed choice at $N{=}200$")
need_true("expW adaptive rule matches the better fixed choice at N=200",
          _wmatch and _wbetter)
_wml = _wpaired(_wada("nonadd-1.0", 200), _wb("nonadd-1.0", "B1-MLE", 200))
need_tex("expW strong S1: MLE against that full table",
         "though in strong S1 unsmoothed MLE beats that full table at an ESS of $2$ per column whatever its mean")
need_true(f"expW strong S1: MLE beats the adaptive rule at N=200 {_wml}",
          _wml[0] > 0 and _wml[1] < 0.05)
need_true("expW adaptive rule is Proposed itself in the standard world",
          bool((_wad[_wad.world == "standard"].source == "Proposed").all()))
if DRAFT:
    print(f"expW checks run: {_n_expd - _n0w}")

# ---- the RF channel on stricter DroneRF splits, on
# DroneDetect under real interference, and high-threat reliability with real
# RF outputs (Sec. VII-H, Table tab:rfsplits). Every number is recomputed here
# from the result files of exp_rf_independent.py; the soft-evidence sentence
# of Sec. VII-H is checked against expG_rf_e2e_raw.csv.
_n0i = _n_expd
_ind = _rt("expI_node_dronerf.csv")
_ie2 = _rt("expI_e2e_raw.csv")
_ihb = _rt("expI_highthreat_bins.csv")
_ial = _rt("expI_alarm.csv")
_idn = _rt("expI_dd_node.csv")
_irl = _rt("expI_dd_reliability.csv")
_ig0 = _rt("expG_rf_e2e_raw.csv")
_IPR = ("P0", "P1", "P1c", "P2", "P3")
_ISEED = [42, 123, 2024, 7, 99]
_IC = ("CLEAN", "BLUE", "WIFI", "BOTH")


def _iv(proto, variant, k):
    q = _ie2[(_ie2.proto == proto) & (_ie2.rf_likelihood == variant)]
    if len(q) != 20:
        fails.append(f"expI: {proto}/{variant} has {len(q)} reps")
    return q.set_index("rep")[k].sort_index()


def _ipaired(a, b):
    d = (a - b).dropna()
    return float(d.mean()), float(_st.ttest_1samp(d, 0.0).pvalue)


def _iconf(s, k):
    return np.array(list(map(int, s.split()))).reshape(k, k)


def _icp(k, n):
    lo = _st.beta.ppf(0.025, k, n - k + 1) if k > 0 else 0.0
    hi = _st.beta.ppf(0.975, k + 1, n - k) if k < n else 1.0
    return lo, hi


# the manuscript's own split (P0) must reproduce expG bit for bit
need_true("expI P0 reproduces expG_rf_e2e_raw.csv",
          all(np.array_equal(_iv("P0", v, k).values,
                             _ig0[_ig0.rf_likelihood == v].set_index("rep")[k]
                             .sort_index().values)
              for v in ("abstract", "calib-hard", "calib-soft")
              for k in ("acc", "macroAUC", "ece", "brier", "cwece")))
need_true("expI split counts: 1/5/5/5 segment-type splits, 4 folds",
          [int((_ind.proto == p).sum()) for p in _IPR] == [1, 5, 5, 5, 4])

# Table tab:rfsplits, DroneRF block: runs, accuracy, Bebop+AR, (ii) minus (i)
_ILAB = {"P0": r"segment (Sec.~\ref{sec:dronerf})", "P1": "segment group",
         "P1c": r"\quad size-matched", "P2": "time block",
         "P3": "held-out flight mode"}
_ie1 = {}
_icc = pd.read_csv(os.path.join(RES, "expRC_contrasts.csv"))
for _p in _IPR:
    _q = _ind[_ind.proto == _p]
    _ie1[_p] = (_ipaired(_iv(_p, "calib-hard", "macroAUC"), _iv(_p, "abstract", "macroAUC")),
                _ipaired(_iv(_p, "calib-hard", "ece"), _iv(_p, "abstract", "ece")))
    _icq = _icc[(_icc.proto == _p) & (_icc.A == "calib-hard") & (_icc.B == "no-RF")]
    need_true(f"expI table row {_p}: one contrast over no RF", len(_icq) == 1)
    need_tex(f"expI table row {_p}",
             f"{_ILAB[_p]} & {len(_q)} & {_q.acc.mean():.3f} & {_q.acc_bebop_ar.mean():.3f} & "
             f"${_ie1[_p][0][0]:+.3f}$ & ${_ie1[_p][1][0]:+.3f}$ & "
             + f"${_icq.d_macroAUC.iloc[0]:+.4f}$".replace("0.", ".", 1) + " \\\\")

# Table tab:rfsplits, DroneDetect block
_ddn = {c: _idn[_idn.cond == c] for c in _IC}
_ddr = {c: _irl[(_irl.fit == "4-condition") & (_irl.cond == c)].set_index("seed") for c in _IC}
_igain = {c: _ipaired(_ddr[c].ll_own, _ddr[c].ll_none) for c in _IC}
need_tex("expI DroneDetect header", r"DroneDetect & clean & Bluetooth & Wi-Fi & both & & \\")
need_tex("expI DroneDetect chunk accuracy",
         "chunk acc. & " + " & ".join(f"{_ddn[c].acc.mean():.3f}" for c in _IC) + r" & & \\")
need_tex("expI DroneDetect recording accuracy",
         "recording acc. & " + " & ".join(f"{_ddn[c].rec_acc.mean():.3f}" for c in _IC) + r" & & \\")
need_tex("expI DroneDetect alpha (Eq. 10)",
         r"$\alpha_{\mathrm{RF}}$, Eq.~\eqref{eq:reliab} & "
         + " & ".join(f"{_ddr[c].alpha.mean():.2f}" for c in _IC) + r" & & \\")
need_tex("expI DroneDetect log-likelihood gain",
         r"log-lik.\ gain & " + " & ".join(f"${_igain[c][0]:+.3f}$" for c in _IC) + r" & & \\")

# "Stricter splits" paragraph
_x = _ind[_ind.proto == "P1"].set_index("key").acc.loc[_ISEED]
_xc = _ind[_ind.proto == "P1c"].set_index("key").acc.loc[_ISEED]
_y = _ind[_ind.proto == "P2"].set_index("key").acc.loc[_ISEED]
_pn = float(_st.ttest_rel(_x, _y).pvalue)
_pnc = float(_st.ttest_rel(_xc, _y).pvalue)
need_tex("expI time-block loss p",
         f"but not significantly (paired $p{{=}}{_pn:.2f}$), also at matched training size")
need_true(f"expI time-block loss positive, not significant, also size-matched "
          f"(p {_pn:.3f}, {_pnc:.3f})",
          (_x - _y).mean() > 0 and _pn >= 0.05 and (_xc - _y).mean() > 0 and _pnc >= 0.05)
_b2 = _ind[_ind.proto == "P2"].acc_bebop_ar
_b3q = _ind[_ind.proto == "P3"]
_pw = float(_st.ttest_ind(_b2, _b3q.acc_bebop_ar, equal_var=False).pvalue)
need_tex("expI held-out-mode folds and Welch p",
         f"four folds spanning ${_b3q.acc_bebop_ar.min():.2f}$ to ${_b3q.acc_bebop_ar.max():.2f}$ "
         f"(Welch $p{{=}}{_pw:.2f}$)")
need_true("expI held-out-mode loss positive and not significant",
          _b2.mean() > _b3q.acc_bebop_ar.mean() and _pw >= 0.05)
need_tex("expI hardest fold",
         f"the hardest fold being AR in one flight mode (recall ${_b3q.recall_2.min():.2f}$)")
_ipmax = max(max(_ie1[_p][0][1], _ie1[_p][1][1]) for _p in _IPR)
need_supp_later("expI gain p, in the supplement", r"Over replications the gains of the stricter splits have $p<10^{-7}$;")
need_true(f"expI every split reproduces the gain of (ii) over (i) (max p {_ipmax:.1e})",
          all(_ie1[_p][0][0] > 0 and _ie1[_p][1][0] < 0 for _p in _IPR) and _ipmax < 1e-7)
_iacc = _ind.groupby("proto").acc.mean()
_iauc = {_p: _iv(_p, "calib-hard", "macroAUC").mean() for _p in _IPR}
_iorg = {_p: _iv(_p, "CPT-oracle", "macroAUC").mean() - _iauc[_p] for _p in _IPR}
need_tex("expI node and threat spreads",
         f"Node accuracy moves by up to ${_iacc.max() - _iacc.min():.2f}$ across splits, the "
         f"threat-level macro-AUC by less than $0.001$, against a gap of "
         f"${float(np.mean(list(_iorg.values()))):.3f}$ to the CPT oracle")
need_true("expI threat AUC spread under 0.001; oracle gap 0.007 in every split",
          max(_iauc.values()) - min(_iauc.values()) < 0.001
          and all(round(v, 3) == 0.007 for v in _iorg.values()))
_ish, _irates, _iworst = [], [], 0.0
for _p in _IPR:
    _sh_p, _m_p, _f_p = [], [], []
    for _s in _ind[_ind.proto == _p].confusion:
        _C = _iconf(_s, 4)
        _sh_p.append((_C[1:, 1:].sum() - np.trace(_C[1:, 1:])) / (_C.sum() - np.trace(_C)))
        _m_p.append(_C[1:, 0].sum() / _C[1:].sum())
        _f_p.append(_C[0, 1:].sum() / _C[0].sum())
        _iworst = max(_iworst, _m_p[-1], _f_p[-1])
    _ish.append(float(np.mean(_sh_p)))
    _irates += [float(np.mean(_m_p)), float(np.mean(_f_p))]
need_true("expI error composition: the per-split sentence left the body for Sec. S5 (checked there by expCM S5 errors)",
          not HAVE_PAPER or "of the CNN errors per split type" not in tex)
need_true("expI most CNN errors are drone-model confusions", min(_ish) > 0.5)

# "Interference" paragraph (DroneDetect)
_zd = np.load(os.path.join(RES, "expI_cnn_dronedetect.npz"))
_yd = _zd["meta__y"]
_va_acc, _te_acc = [], []
for _s in _ISEED:
    _pr = np.full(len(_yd), -1)
    _pr[_zd[f"DD_{_s}__rest"]] = _zd[f"DD_{_s}__rest_softmax"].argmax(1)
    _va, _te = _zd[f"DD_{_s}__va"], _zd[f"DD_{_s}__te"]
    _va_acc.append(float((_pr[_va] == _yd[_va]).mean()))
    _te_acc.append(float((_pr[_te] == _yd[_te]).mean()))
need_supp_later("expI CLEAN validation vs test accuracy",
         f"(accuracy ${np.mean(_va_acc):.3f}$, against ${np.mean(_te_acc):.3f}$ on clean test recordings)")
need_true("expI CLEAN test accuracy agrees with the table",
          f"{np.mean(_te_acc):.3f}" == f"{_ddn['CLEAN'].acc.mean():.3f}")
need_tex("expI interference gain p",
         f"under Wi-Fi and under both sources ($p{{=}}{_igain['WIFI'][1]:.3f}$ each, "
         "positive for all five seeds)")
need_true("expI discount pays off under WIFI and BOTH in every seed (same rounded p), "
          "not under CLEAN and BLUE",
          all(((_ddr[c].ll_own - _ddr[c].ll_none) > 0).all() and _igain[c][1] < 0.05
              for c in ("WIFI", "BOTH"))
          and f"{_igain['WIFI'][1]:.3f}" == f"{_igain['BOTH'][1]:.3f}"
          and all(_igain[c][1] >= 0.05 for c in ("CLEAN", "BLUE")))
_iao = max(abs(_ddr[c].alpha.mean() - _ddr[c].omega.mean()) for c in _IC)
need_supp_later("expI weights vs chance-corrected ones",
         "the seed-averaged weights lie within $0.05$ of the chance-corrected ones")
need_true(f"expI seed-averaged |alpha - omega| under 0.05 ({_iao:.4f})", _iao < 0.05)
_ORD = {"BLUE": 0, "WIFI": 1, "BOTH": 2}
_itr = []
for _B in ("BLUE", "WIFI", "BOTH"):
    for _c in [x for x in ("BLUE", "WIFI", "BOTH") if x != _B]:
        _q = _irl[(_irl.fit == f"clean+{_B}") & (_irl.cond == _c)].set_index("seed")
        _g, _pg = _ipaired(_q.ll_transfer, _q.ll_none)
        _itr.append((_B, _c, _g, _pg, float(_q.alpha.mean()), float(_ddr[_c].alpha.mean())))
_iok = [t for t in _itr if t[2] > 0 and t[3] < 0.05]
_ino = [t for t in _itr if not (t[2] > 0 and t[3] < 0.05)]
need_supp_later("expI transfer",
         f"gains in the three directions toward a harsher condition "
         f"(${min(t[2] for t in _iok):+.3f}$ to ${max(t[2] for t in _iok):+.3f}$ nats, $p<0.02$)")
need_true("expI transfer: 3 directions toward harsher (p<0.02); the others toward milder, "
          "with a smaller transferred weight",
          len(_iok) == 3 and all(_ORD[t[1]] > _ORD[t[0]] for t in _iok)
          and max(t[3] for t in _iok) < 0.02
          and all(_ORD[t[1]] < _ORD[t[0]] and t[4] < t[5] for t in _ino))
need_true("expI severity order follows the clean-trained accuracy (BLUE > WIFI > BOTH)",
          _ddn["BLUE"].acc.mean() > _ddn["WIFI"].acc.mean() > _ddn["BOTH"].acc.mean())
need_tex("expI DroneDetect enters the reliability only",
         "DroneDetect has no background recordings, so it enters the RF node's reliability only;")
need_true("expI no DroneDetect end-to-end rows", set(_ie2.proto) == set(_IPR))

# "High-threat reliability with real RF outputs" paragraph
_idev, _iout, _ioutdev = {}, {}, []
for _p in _IPR:
    for _v in ("calib-hard", "CPT-oracle"):
        _q = _ihb[(_ihb.proto == _p) & (_ihb.variant == _v)]
        _d, _o = [], 0
        for _lo in sorted(_q.bin_lo.unique()):
            _b = _q[_q.bin_lo == _lo]
            _n, _k, _sp = int(_b.n.sum()), int(_b.hits.sum()), float(_b.sum_pred.sum())
            _l, _h = _icp(_k, _n)
            _isout = not (_l <= _sp / _n <= _h)
            _o += _isout
            _d.append(_k / _n - _sp / _n)
            if _isout and _v == "calib-hard":
                _ioutdev.append(abs(_k / _n - _sp / _n))
        _idev[(_p, _v)], _iout[(_p, _v)] = _d, _o
_imid = [-_idev[(_p, "calib-hard")][3] for _p in _IPR]
_itop = [_idev[(_p, "calib-hard")][4] for _p in _IPR]
need_tex("expI high-threat shape",
         f"over-confident by ${min(_imid):.2f}$ to ${max(_imid):.2f}$ in $(.50,.75]$ and "
         f"under-confident by ${min(_itop):.2f}$ to ${max(_itop):.2f}$ in the top bin, "
         "against $0.050$ and $0.069$ in the synthetic world")
need_tex("expI synthetic-world values quoted from Table tab:highrel",
         "high    & $-0.002$ & $+0.012$ & $-0.013$ & $-0.050$ & $+0.069$")
need_true("expI over-confident in (.50,.75] and under-confident in the top bin, every split",
          min(_imid) > 0 and min(_itop) > 0)
need_supp_later("expI bins outside the interval",
         f"binomial intervals put three or four of the five bins outside, by at most "
         f"${max(_ioutdev):.2f}$, against one of $25$ for the CPT oracle fed the same RF evidence")
need_true("expI 3 or 4 bins outside in every split; oracle 1 of 25",
          all(_iout[(_p, "calib-hard")] in (3, 4) for _p in _IPR)
          and sum(_iout[(_p, "CPT-oracle")] for _p in _IPR) == 1)
_ipc, _irc, _iopc, _iorc = [], [], [], []
for _p in _IPR:
    for _v, _P, _R in (("calib-hard", _ipc, _irc), ("CPT-oracle", _iopc, _iorc)):
        _a = _ial[(_ial.proto == _p) & (_ial.variant == _v)]
        _tp, _fp, _fn = _a.tp.sum(), _a.fp.sum(), _a.fn.sum()
        _P.append(_tp / (_tp + _fp))
        _R.append(_tp / (_tp + _fn))
_a0 = _ial[(_ial.proto == "P0") & (_ial.variant == "calib-hard")]
_iprev = float((_a0.tp.sum() + _a0.fn.sum()) / _a0[["tp", "fp", "fn", "tn"]].sum().sum())
need_tex("expI alarm at 0.5",
         f"reaches precision ${min(_ipc):.2f}$ to ${max(_ipc):.2f}$ and recall "
         f"${float(np.mean(_irc)):.2f}$ at a high-threat prevalence of ${100 * _iprev:.1f}\\%$, and "
         f"the CPT oracle, which uses the true tables, shares the low recall "
         f"(${min(_iorc):.2f}$ to ${max(_iorc):.2f}$)")
need_true("expI alarm recall rounds alike in every split",
          len({f"{r:.2f}" for r in _irc}) == 1)
_i25 = {"calib-hard": ([], []), "CPT-oracle": ([], [])}
for _p in _IPR:
    for _v in _i25:
        _q = _ihb[(_ihb.proto == _p) & (_ihb.variant == _v)]
        _top3 = _q[_q.bin_lo >= 0.25]
        _i25[_v][0].append(_top3.hits.sum() / _top3.n.sum())
        _i25[_v][1].append(_top3.hits.sum() / _q.hits.sum())
_hp, _hr = _i25["calib-hard"]
_op, _orr = _i25["CPT-oracle"]
need_supp_later("expI descriptive threshold 0.25",
         f"gives precision ${min(_hp):.2f}$ to ${max(_hp):.2f}$ and recall "
         f"${float(np.mean(_hr)):.2f}$ (oracle ${float(np.mean(_op)):.2f}$ and "
         f"${min(_orr):.2f}$ to ${max(_orr):.2f}$)")
need_true("expI threshold-0.25 recall and oracle precision round alike in every split",
          len({f"{v:.2f}" for v in _hr}) == 1 and len({f"{v:.2f}" for v in _op}) == 1)

# Sec. VII-H soft-evidence sentence: expG and the
# stricter splits
_gh = _ig0[_ig0.rf_likelihood == "calib-hard"].set_index("rep").sort_index()
_gs = _ig0[_ig0.rf_likelihood == "calib-soft"].set_index("rep").sort_index()
_isd = {k: _ipaired(_gs[k], _gh[k]) for k in ("acc", "macroAUC", "ece", "brier")}
need_tex("expI soft variant (expG)", "Variant (iii) differs from (ii) by less than $0.0012$ on every metric (Supplementary Material, Sec.~S4),")
need_supp_later("expI soft variant (expG), the detail",
         f"Variant (iii) is slightly "
         f"better than (ii) on accuracy and Brier score ($p{{=}}{_isd['acc'][1]:.3f}$ and "
         f"${_isd['brier'][1]:.3f}$) and indistinguishable on macro-AUC and ECE "
         f"($p{{=}}{_isd['macroAUC'][1]:.3f}$ and ${_isd['ece'][1]:.2f}$)")
need_true("expI soft variant: every |difference| under 0.0012; better on accuracy and Brier "
          "(p<0.05); not on macro-AUC and ECE",
          max(abs(v[0]) for v in _isd.values()) < 0.0012
          and _isd["acc"][0] > 0 and _isd["acc"][1] < 0.05
          and _isd["brier"][0] < 0 and _isd["brier"][1] < 0.05
          and _isd["macroAUC"][1] >= 0.05 and _isd["ece"][1] >= 0.05)
_issp = [_ipaired(_iv(_p, "calib-soft", k), _iv(_p, "calib-hard", k))
         for _p in ("P1", "P1c", "P2", "P3") for k in ("acc", "macroAUC", "ece", "brier")]
_issig = [_ipaired(_iv(_p, "calib-soft", k), _iv(_p, "calib-hard", k))[1]
          for _p in ("P1", "P1c", "P2", "P3") for k in ("macroAUC", "brier")]
need_supp_later("expI soft variant under the stricter splits",
         r"its edge grows but stays below $0.0024$, significant on macro-AUC and Brier score "
         r"over replications ($p\le0.008$) but not in every split when each CNN counts once ($p$ from $0.016$ to $0.094$)")
need_true("expI soft edge under 0.0024 in the stricter splits; macro-AUC and Brier p <= 0.008",
          max(abs(d) for d, _ in _issp) < 0.0024 and max(_issig) <= 0.008)

# abstract and conclusion
need_tex("expI abstract",
         "the RF channel adds about 0.004 macro-AUC over no channel under three splits, below its 0.0055 bound with error-free reports.")
need_true("expI the macro-AUC gain over the abstract RF model differs by less than 0.002 across splits",
          max(_ie1[_p][0][0] for _p in _IPR) - min(_ie1[_p][0][0] for _p in _IPR) < 0.002)
need_tex("expI conclusion",
         "on DroneDetect, discounting by the interference condition improves the RF likelihood under "
         "Wi-Fi and combined interference")
if DRAFT:
    print(f"expI checks run: {_n_expd - _n0i}")

# ---- the EO/IR gate on real thermal-infrared clips
# (Sec. VI-B). G2 failed, so the mixed-world study was
# not run. Every number is recomputed here from the saved predictions of
# exp_eo_independent.py, not read from its summary.
_n0o = _n_expd
_og2 = _rt("expO_g2.csv").set_index("check")
_opr = _rt("expO_g2_predictions.csv")
_onp = _rt("expO_node_predictions.csv")
_ond = _rt("expO_node.csv")
_ogb = _rt("expO_g2_bootstrap.csv").set_index("name")
_onb = _rt("expO_node_bootstrap.csv").set_index("name")
_ogf = _rt("expO_g3_folds.csv")
_ONN, _OCN = "1-NN masked (LOSO)", "CNN masked (4-fold sessions)"


def _obal(y, p):
    """Balanced accuracy: mean recall over the four classes."""
    return float(np.mean([np.mean(p[y == k] == k) for k in range(4)]))




def _oci(tab, name):
    return f"[{tab.loc[name].lo:.2f}, {tab.loc[name].hi:.2f}]"


_oy = _opr.y.values
_oan, _oac = float((_opr.pred_nn == _opr.y).mean()), float((_opr.pred_cnn == _opr.y).mean())
_obn, _obc = _obal(_oy, _opr.pred_nn.values), _obal(_oy, _opr.pred_cnn.values)
_omaj = float(np.bincount(_oy, minlength=4).max() / len(_oy))
_og = _onp[_onp.in_g2].reset_index(drop=True)
_oun = float((_og.pred_nn_g2 == _og.y).mean())
_ouc = float((_og.pred_cnn == _og.y).mean())
_obun, _obuc = _obal(_og.y.values, _og.pred_nn_g2.values), _obal(_og.y.values, _og.pred_cnn.values)
_oba = _obal(_onp.y.values, _onp.pred_cnn.values)
need_true("expO G2 table agrees with the saved predictions",
          max(abs(_og2.loc[_ONN].acc - _oan), abs(_og2.loc[_OCN].acc - _oac),
              abs(_og2.loc[_ONN].balanced_acc - _obn), abs(_og2.loc[_OCN].balanced_acc - _obc),
              abs(_og2.loc[_ONN].chance - _omaj)) < 1e-12)
need_true("expO gate: accuracy limit passes both checks, balanced limit 0.35 fails both",
          bool(_og2.pass_acc.all()) and not bool(_og2.pass_balanced.any()) and not bool(_og2["pass"].any())
          and bool((_og2.limit_balanced == 0.35).all())
          and _oan <= _omaj + 0.10 and _oac <= _omaj + 0.10 and _obn > 0.35 and _obc > 0.35)
need_true("expO node run used the patches of G2",
          len(_og) == len(_opr) == 3466 and len(_onp) == 3650
          and all(np.array_equal(_og[c].values, _opr[c].values) for c in ("cls", "num", "frame", "session", "fold", "y")))
_oq = _ond[_ond.scope == "all"].set_index(["model", "patches"])
need_true("expO node table agrees with the saved predictions",
          max(abs(_oq.loc[("CNN unmasked", "G2")].balanced_acc - _obuc), abs(_oq.loc[("CNN unmasked", "G2")].acc - _ouc),
              abs(_oq.loc[("1-NN unmasked", "G2")].balanced_acc - _obun), abs(_oq.loc[("1-NN unmasked", "G2")].acc - _oun),
              abs(_oq.loc[("CNN unmasked", "all")].balanced_acc - _oba)) < 1e-12)
need_true("expO bootstrap estimates and intervals are consistent; no resample at 0.25 or below",
          abs(_ogb.loc[_ONN].estimate - _obn) < 1e-12 and abs(_ogb.loc[_OCN].estimate - _obc) < 1e-12
          and bool((_ogb.lo > 0.25).all()) and bool((_ogb["share_le_0.25"] == 0).all())
          and bool((_ogb.B == 2000).all()) and bool((_ogb.seed == 0).all()))
# the node-level comparison the wording rests on
_odc, _odn = _onb.loc["CNN unmasked minus masked"], _onb.loc["1-NN unmasked minus masked"]
need_true("expO target adds information for the CNN (interval above 0), not shown for the 1-NN; "
          "the masked background gives about half of the CNN's margin over 0.25 (ratio within 0.4-0.6)",
          _odc.lo > 0 and abs(_odc.estimate - (_obuc - _obc)) < 1e-12 and _odn.lo < 0 < _odn.hi
          and 0.4 <= (_obc - 0.25) / (_obuc - 0.25) <= 0.6)
need_true("expO estimated drone sessions: 12", int(_ogf.DRONE_sessions.sum()) == 12)
need_true("expO share of the sample in G2 rounds to 95%", f"{len(_opr) / len(_onp):.0%}" == "95%")

# the G2 files are older than the script, which gained the node stage after
# G2 ran: guard the procedure itself instead (source of the functions and
# constants of the G2 path, identical in the versions that ran G2)
import ast as _ast
import hashlib as _hl
_OSRC = open(os.path.join(HERE, "exp_eo_independent.py"), encoding="utf-8").read()
_OFUN = ("check_manifest", "load_description", "clip_path", "decode", "sessions", "load_boxes",
         "sampled_frames", "patch_of", "build_patches", "dominant_bin", "assign_folds", "masked",
         "_task", "run_folds", "pooled_test", "nn_loso", "prepare")
_OCON = ("CLASSES", "BINS", "SESSION_T", "WIN", "MAX_FRAMES", "PATCH", "MASK_M", "MASK_MARGIN", "NFOLD", "SEED")
_oseg = [_ast.get_source_segment(_OSRC, _n) for _n in _ast.parse(_OSRC).body
         if (isinstance(_n, _ast.FunctionDef) and _n.name in _OFUN)
         or (isinstance(_n, _ast.Assign) and getattr(_n.targets[0], "id", None) in _OCON)]
need_true("expO G2 code path unchanged since G2 ran",
          len(_oseg) == len(_OFUN) + len(_OCON)
          and _hl.sha256("\n".join(_oseg).encode("utf-8")).hexdigest()
          == "5ffbe70ce644f467eef6f6e78197d47303e1ad6c45d2000f3bdd988625375182")

# Sec. VI-B
_obi = pd.read_csv(os.path.join(RES, "expO_g2_bootstrap.csv"))
need_true("expO the infrared bootstrap file has the two classifiers' intervals", len(_obi) == 2 and (_obi.lo <= 0.35).all())
for _sn in ("balanced accuracy", "session-bootstrap interval", "under a rule fixed before the results were seen",
            "across recording sessions reconstructed from frame continuity",
            "could therefore partly measure the background rather than the target",
            "need recordings in which drones and non-drone targets share scenes and recording conditions and range is measured",
            "thermal-infrared"):
    need_true(f"expO the infrared data check left main.tex: {_sn}", not HAVE_PAPER or _sn not in tex)
need_tex("expO no public dataset found",
         "We found no public dataset that records the four modalities on the same scenes with non-drone targets "
         "and threat labels")
need_tex("expO node by node",
         "or non-drone targets \\cite{yu2026dronerfcmm}, so real data can enter only node by node.")
need_true("expO the drones-only check is not described", not HAVE_PAPER or "can at best be examined for drones alone" not in tex)
need_true("expO the withdrawn statements and citations are gone",
          not HAVE_PAPER or not any(t in tex for t in ("huang2023antiuav410", "alemadi2019audio", "distractor",
                                                       "public benchmarks", "public-dataset instantiation")))

if DRAFT:
    print(f"expO checks run: {_n_expd - _n0o}")

# ---- the rule of the base framework (Secs. I, IV-A, VI-C, VII-B), with the S1
# worlds and large N of exp_base_rule_worlds.py and the AR-versus-Bebop figure.
# Every number is recomputed here from
# the replication-level files, not read from the summaries.
_n0b = _n_expd
_b6r = _rt("expB6_raw.csv")
_b6b = _rt("expB6_bins.csv")
_b6s = _rt("expB6_absweep_raw.csv")
_b6w = _rt("expB6w_raw.csv")
_b6wb = _rt("expB6w_bins.csv")
_b6g = _rt("expB6w_big_raw.csv")
_b6f = _rt("expB6w_world_facts.csv").set_index("world")
_b6a = _rt("expB6w_flags_raw.csv")
_tid = _rt("expI_type_id.csv")
_B6N = (10, 20, 50, 100, 200, 400)
_BKEY = {"AUC": "auc", "Brier": "brier_bin", "ECE": "ece_bin"}


def _bv(df, m, N, k, world=None):
    q = df[(df.method == m) & (df.N == N)]
    if world is not None:
        q = q[q.world == world]
    q = q.sort_values("rep")
    if q.rep.nunique() != 20 or len(q) != 20:
        fails.append(f"expB6: {world}/{m}/N={N} has {len(q)} rows, not 20")
    return q[k].values


def _bp(a, b):
    d = a - b
    return float(d.mean()), float(_st.ttest_1samp(d, 0.0).pvalue)


def _bn(N, m):
    """The rule without training is stored at N = 0."""
    return 0 if m in ("B6-pub", "B6-recal") else N


def _w1(net, rule, N, metric, world="nonadd-1.0"):
    k = _BKEY[metric]
    return _bp(_bv(_b6w, net, N, k, world), _bv(_b6w, rule, _bn(N, rule), k, world))


def _ahead(metric, delta):
    return delta > 0 if metric == "AUC" else delta < 0


def _f(x, d=3):
    return f"{x:.{d}f}"


def _bin_obs(df, m, N, lo, world=None):
    q = df[(df.method == m) & (df.N == N) & (df.bin_lo == lo)]
    if world is not None:
        q = q[q.world == world]
    return float(q.hits.sum()) / float(q.n.sum()), int(q.n.sum())


# ---- standard world: the published rule
_a_pub = _bv(_b6r, "B6-pub", 0, "auc")
_b_pub = _bv(_b6r, "B6-pub", 0, "brier_bin")
_b_rec = _bv(_b6r, "B6-recal", 0, "brier_bin")
_o75, _n75 = _bin_obs(_b6b, "B6-pub", 0, 0.75)
need_true("expB6 pooled bin above 0.75 has data", _n75 > 1000)
_bin_all = _b6b[(_b6b.method == "B6-pub") & (_b6b.N == 0)]
_prev = _bin_all.groupby("rep").hits.sum() / _bin_all.groupby("rep").n.sum()
_base = float((_prev * (1 - _prev)).mean())
_rule_first = (f"On $T{{=}}\\text{{high}}$ the published rule (B6-pub) ranks at AUC ${_f(_a_pub.mean())}$, and its output is not a probability "
               f"of that class: scores above $0.75$ are high-threat ${100 * _o75:.0f}\\%$ of the time. "
               f"Rescaling (B6-recal) brings the Brier score from ${_f(_b_pub.mean())}$ to ${_f(_b_rec.mean())}$ (${_f(_base)}$ for a constant forecast at the base rate ${_f(float(_prev.mean()))}$).")
need_tex("expB6 published rule and rescaling", _rule_first)
_fit5 = [_bp(_a_pub, _bv(_b6r, "B6-fit-5", N, "auc")) for N in _B6N]
need_true("expB6 constants rank ahead of the same form up to N = 200, not at 400",
          all(d > 0 and p < 0.002 for d, p in _fit5[:5]) and _fit5[5][0] < 0)
need_tex("expB6 constants against the same form",
         "Its constants rank better than the same form fitted on the same inputs up to $N{=}200$ ($p<0.002$).")
_pa = [_bv(_b6r, "Proposed", N, "auc").mean() for N in _B6N]
_pb = [_bv(_b6r, "Proposed", N, "brier_bin").mean() for N in _B6N]
need_true("expB6 Proposed ranges are the extremes", _f(min(_pa)) == _f(_pa[0]) and max(_pa) == _pa[-1]
          and max(_pb) == _pb[0] and min(_pb) == _pb[-1])
_pall = [_bp(_bv(_b6r, "Proposed", N, _BKEY[m]), _bv(_b6r, r, _bn(N, r), _BKEY[m]))
         for N in _B6N for r in ("B6-pub", "B6-recal", "B6-fit") for m in ("AUC", "Brier")]
_pok = all(_ahead(m, _bp(_bv(_b6r, "Proposed", N, _BKEY[m]), _bv(_b6r, r, _bn(N, r), _BKEY[m]))[0])
           for N in _B6N for r in ("B6-pub", "B6-recal", "B6-fit") for m in ("AUC", "Brier"))
need_true("expB6 Proposed ahead of B6-pub, recal, fit at every N on AUC and Brier", _pok and len(_pall) == 36
          and max(p for _, p in _pall) <= 0.007)
need_tex("expB6 Proposed sentence",
         f"The proposed model (AUC ${_f(_pa[0])}$--${_f(_pa[-1])}$, Brier ${_f(_pb[0])}$--${_f(_pb[-1])}$) is ahead of B6-pub, B6-recal, and B6-fit at every $N$ on both ($p\\le0.007$), "
         f"but on ECE not significantly ahead of B6-recal at $N{{=}}10$ and $20$ ($p{{=}}{_f(_bp(_bv(_b6r, 'Proposed', 10, 'ece_bin'), _bv(_b6r, 'B6-recal', _bn(10, 'B6-recal'), 'ece_bin'))[1], 2)}$, ${_f(_bp(_bv(_b6r, 'Proposed', 20, 'ece_bin'), _bv(_b6r, 'B6-recal', _bn(20, 'B6-recal'), 'ece_bin'))[1], 2)}$);")
_eqk = [_bp(_bv(_b6r, "B0-as-prior MAP", N, "auc"), _bv(_b6r, "B6-fit-p", N, "auc"))[0] for N in _B6N]
_eqe = [_bp(_bv(_b6r, "B0-as-prior MAP", N, "ece_bin"), _bv(_b6r, "B6-fit-p", N, "ece_bin"))[1] for N in _B6N]
_eqd = [_bp(_bv(_b6r, "Proposed-EM", N, "auc"), _bv(_b6r, "B6-fit", N, "auc"))[0] for N in _B6N]
need_true("expB6 hand-set prior: only N = 400 reaches p < 0.05 on ECE",
          [p < 0.05 for p in _eqe] == [False] * 5 + [True])
need_supp_later("expB6 hand-set prior, in the supplement",
         f"The order of Sec.~VII-B also holds with a hand-set prior (the B0 table, which shares the world's parents and response family, against B6-fit-p, $+{min(_eqk):.2f}$ to $+{max(_eqk):.2f}$ AUC), "
         f"though on ECE only $N{{=}}400$ reaches $p<0.05$.")
need_tex("expB6 unlabeled latent layer",
         f"with the latent layer unlabeled it still leads B6-fit by $+{min(_eqd):.2f}$ to $+{max(_eqd):.2f}$ AUC, keeping the elicited prior and the calibration split.")
# ---- large N and the worlds
_gv = lambda w, N, m, k="auc": _bv(_b6g, m, N, k, w).mean()
_wtop = _b6f.loc["standard"]
_wst = _b6f.loc["nonadd-1.0"]
need_true("expB6w world facts: joint evidence in the standard world, one top state enough in S1 strong",
          _wtop.p_high_top3_min == _wtop.p_high_top3_max and _wtop.p_high_top1_max < 0.04
          and abs(_wst.p_high_top1_min - _wst.p_high_top3_min) < 1e-12 and _wst.p_high_top1_max == _wst.p_high_top3_max)
need_tex("expB6w form depends on the world",
         f"This gap of form depends on the world: at $N{{=}}10,000$ B6-fit still trails here (${_f(_gv('standard', 10000, 'B6-fit'))}$ against ${_f(_gv('standard', 10000, 'Proposed'))}$) "
         f"but nearly catches up (${_f(_gv('nonadd-1.0', 10000, 'B6-fit'))}$ against ${_f(_gv('nonadd-1.0', 10000, 'Proposed'))}$) in S1 strong (Sec.~\\ref{{sec:misspec}}), "
         "where one top state suffices.")
need_supp_later("expB6w joint evidence, in the supplement", f"$P(T{{=}}\\text{{high}})$ is ${_f(_wtop.p_high_top3_max, 2)}$ with all three latent variables at their top state and at most ${_f(_wtop.p_high_top1_max, 2)}$ with one.")
need_supp_later("expB6w Noisy-OR fit, in the supplement", f"a Noisy-OR fits the true table to RMS ${_f(_wst.noisy_or_rms)}$ (${_f(_wtop.noisy_or_rms)}$ in the standard world).")
_RS5 = ("B6-pub", "B6-recal", "B6-fit", "B6-fit-r", "B6-fit-p")
_x = [_w1("Proposed", r, N, "AUC") for N in _B6N for r in _RS5]
need_true("expB6w Proposed ahead on AUC at every N against all five, p <= 0.007",
          len(_x) == 30 and all(d > 0 for d, _ in _x) and max(p for _, p in _x) <= 0.007)
_x = [_w1("Proposed", r, N, m) for N in (10, 20) for r in ("B6-pub", "B6-recal") for m in ("Brier", "ECE")]
need_true("expB6w B6-pub and recal beat Proposed on Brier and ECE at N <= 20",
          all(d > 0 and p < 0.05 for d, p in _x))
_x = [_w1("Proposed", "B6-fit-p", N, "Brier") for N in (10, 20)]
need_true("expB6w B6-fit-p beats Proposed on Brier at N <= 20, p <= 0.003",
          all(d > 0 and p <= 0.003 for d, p in _x))
need_true("expB6w B6-fit-p does not beat Proposed on ECE at N = 20 (the first two only)",
          _w1("Proposed", "B6-fit-p", 20, "ECE")[1] >= 0.05)
_x = [_w1("Proposed", "B6-fit-r", N, "Brier") for N in _B6N]
need_true("expB6w B6-fit-r level on Brier throughout, p >= 0.06", min(p for _, p in _x) >= 0.06)
_x = {N: _w1("Proposed", "B6-fit-r", N, "ECE") for N in _B6N}
need_true("expB6w B6-fit-r lower ECE at all but N = 50, 100, p <= 0.044",
          all(_x[N][0] > 0 and _x[N][1] <= 0.044 for N in (10, 20, 200, 400)) and all(_x[N][1] >= 0.05 for N in (50, 100)))
_x = [_w1("cMAP-full", r, N, "Brier") for N in (100, 200, 400) for r in _RS5]
need_true("expB6w cMAP-full leads on Brier from N = 100, p <= 0.007",
          len(_x) == 15 and all(d < 0 for d, _ in _x) and max(p for _, p in _x) <= 0.007)
_ax = _b6w[(_b6w.world == "nonadd-1.0") & (_b6w.method == "Proposed-adaptive")]
need_true("expB6w the additivity test selects the full table in all 20 from N = 100, not at N = 50",
          all(int((_ax[_ax.N == N].A_flag == True).sum()) == 20 for N in (100, 200, 400))
          and int((_ax[_ax.N == 50].A_flag == True).sum()) < 20)
need_true("expB6w the fallback rule equals cMAP-full from N = 100",
          all(np.array_equal(_bv(_b6w, "Proposed-adaptive", N, k, "nonadd-1.0"), _bv(_b6w, "cMAP-full", N, k, "nonadd-1.0"))
              for N in (100, 200, 400) for k in ("auc", "brier_bin", "ece_bin")))
_mo = _b6wb[(_b6wb.world == "nonadd-1.0") & (_b6wb.method == "B6-pub") & (_b6wb.N == 0)]
_mo2 = _b6wb[(_b6wb.world == "nonadd-0.5") & (_b6wb.method == "B6-pub") & (_b6wb.N == 0)]
_mo0 = _b6b[(_b6b.method == "B6-pub") & (_b6b.N == 0)]
need_true("expB6w the published output is the same in every world",
          float(_mo.sum_pred.sum()) == float(_mo2.sum_pred.sum()) and abs(float(_mo.sum_pred.sum()) - float(_mo0.sum_pred.sum())) < 1e-6)
_mout = float(_mo.sum_pred.sum()) / float(_mo.n.sum())
need_true("expB6w the mean output of B6-pub lies near the prevalence of S1 strong only",
          abs(_mout - _wst.prevalence) < 0.02 and abs(_mout - _b6f.loc["nonadd-0.5"].prevalence) > 0.3 and abs(_mout - _wtop.prevalence) > 0.5)
need_supp_later("expB6w S1 sentence",
         "With the expert prior also wrong in that world, Proposed stays ahead on AUC at every $N$ ($p\\le0.007$), but B6-pub, B6-recal, and B6-fit-p beat it on Brier at $N\\le20$, the first two also on ECE, "
         "and B6-fit-r is level on Brier throughout ($p\\ge0.06$), with lower ECE at all but $N{=}50$ and $100$ ($p\\le0.044$).")
need_supp_later("expB6w the rest of that world, in the supplement", f"the mean output of B6-pub (${_f(_mout, 2)}$) happens to lie near the prevalence (${_f(_wst.prevalence, 2)}$), and the full-table threat node, which the additivity test of Sec.~VII-G selects from $N{{=}}100$, leads on Brier from there ($p\\le0.007$).")
# ---- the diagonal sweep of the abstract model
_DG = sorted(x for x in _b6s["diag"].unique() if x > 0)
need_true("expB6 sweep covers 0.55 to 0.99", _DG[0] == 0.55 and _DG[-1] == 0.99 and len(_DG) == 8)
_sw = _b6s[_b6s["diag"] > 0].groupby("diag")[["macroAUC", "brier", "ece"]].mean()
_cal = _b6s[_b6s["diag"] == -1].sort_values("rep")
# diag = -1 is the calibrated model without discount, the comparator of the "Sensor fusion" paragraph
need_true("expB6 sweep comparator is the calibrated model of the paragraph (AUC 0.816, ECE 0.032)", len(_cal) == 20 and _f(_cal.macroAUC.mean()) == "0.816" and _f(_cal.ece.mean()) == "0.032")
_cmp = {c: _cal[c].values for c in ("macroAUC", "brier", "ece")}
_swp = {}
for _dg in _DG:
    _q = _b6s[_b6s["diag"] == _dg].sort_values("rep")
    _swp[_dg] = {c: _bp(_cmp[c], _q[c].values) for c in ("macroAUC", "brier", "ece")}
need_true("expB6 sweep: AUC and Brier gaps at every diagonal, p < 1e-9",
          all(_swp[g]["macroAUC"][0] > 0 and _swp[g]["brier"][0] < 0 and _swp[g]["macroAUC"][1] < 1e-9 and _swp[g]["brier"][1] < 1e-9 for g in _DG))
need_true("expB6 sweep: the ECE gap is not significant exactly at 0.60, 0.65, 0.70",
          [g for g in _DG if _swp[g]["ece"][1] >= 0.05] == [0.60, 0.65, 0.70])
need_tex("expB6 sweep sentence",
         f"A sweep of the abstract model's diagonal ($0.55$--$0.99$: AUC ${_f(_sw.macroAUC.min())}$--${_f(_sw.macroAUC.max())}$, Brier ${_f(_sw.brier.min())}$--${_f(_sw.brier.max())}$, ECE ${_f(_sw.ece.min())}$--${_f(_sw.ece.max())}$) "
         f"keeps the AUC and Brier gaps at every value ($p<10^{{-9}}$), while the ECE gap is not significant at $0.60$, $0.65$, and $0.70$ "
         f"($p{{=}}{_f(_swp[0.60]['ece'][1], 2)}$, ${_f(_swp[0.65]['ece'][1], 2)}$, ${_f(_swp[0.70]['ece'][1], 2)}$).")
need_tex("expB6 sweep range in Sec. VI-C", "sweeps it over $[0.55,0.99]$.")
# ---- wording of the contributions and of the baselines
for _lab, _sn in (("C1 sentence", "and add a rule, for labeled latent states, that tests additivity and monotonicity on the training data and falls back to a full-table threat node or an unconstrained latent node where they fail"),
                  ("C2 what it converges to", "which states what the constrained estimate is and what it converges to"),
                  ("C3 sentence", "replace the base framework's hand rule on RF silence by a latent emission state, through which an absent RF report is evidence weighted by the target class; and add an analysis of the double discount"),
                  ("C4 splits", "the RF node is evaluated under stricter splits, including a held-out flight mode, and under recorded interference"),
                  ("B0 sentence", "where the scalar rule of that framework has no cell-level counterpart"),
                  ("B6 definition", "and B6-fit estimates the form by maximum likelihood from the same $N$ scenarios and all evidence (Supplementary Material, Sec.~S4 adds two priors on it)."),
                  ("gap paragraph", "\\emph{First, the probabilities are heuristic.}"),
                  ("C4 sensitivity", "and a one-way sensitivity analysis (Sec.~\\ref{sec:tornado})."),
                  ("n1 sentence", "In operational terms, with a prior near the truth, the gains of claims (i) and (ii) are reached with tens of labeled scenarios and a modest calibration set.")):
    need_tex(f"expB6 {_lab}", _sn)
need_supp_later("expB6 the two priors on B6-fit, in the supplement", "B6-fit-r and B6-fit-p fit the form of B6-fit under a weak prior and under a prior at the published constants.")
for _sn in ("shows the comparison does not hinge on it", "not an artifact of the $0.94$ choice", "Added here", "Adopted:", "flat at $0.808$", "[0.70,0.99]", "evaluated on independent recordings",
            "and that it is consistent", "This gap of form reflects", "no cell-level CPT counterpart",
            "a trustworthy threat estimate", "could not previously quantify", "the robustness property C4 promises"):
    need_true(f"expB6 retired wording is gone: {_sn}", not HAVE_PAPER or _sn not in tex)
# ---- AR against Bebop on DroneRF: the file against the saved outputs
_zi = np.load(os.path.join(RES, "expI_cnn_dronerf.npz"), allow_pickle=True)
_ty, _tp = [], []
for _k in ("P3_0", "P3_1", "P3_2", "P3_3"):
    _yy, _sm = _zi[f"{_k}__te_y"], _zi[f"{_k}__test_softmax"]
    _mm = np.isin(_yy, (1, 2))
    _ty.append(_yy[_mm])
    _tp.append(1 + _sm[_mm][:, 1:3].argmax(1))
_ty, _tp = np.concatenate(_ty), np.concatenate(_tp)
_f1 = float(np.mean([2 * np.sum((_tp == c) & (_ty == c)) / (np.sum(_tp == c) + np.sum(_ty == c)) for c in (1, 2)]))
_tq = _tid[(_tid.split == "P3") & (_tid.instance == "pooled")]
need_true("expI_type_id.csv agrees with the saved outputs on P3", len(_tq) == 1 and abs(float(_tq.macro_f1_two_way.iloc[0]) - _f1) < 1e-12 and len(_ty) == 3300)
if HAVE_PAPER:
    print(f"expB6 checks run: {_n_expd - _n0b}")
assert _n_expd - _n0b > 0

# ---- the nuisance parent over all four reports, and the scope wording (abstract, Sec. I
# Scope, Sec. VII-F, Sec. VIII, Conclusion). Numbers are recomputed from the
# replication-level files of exp_jam_scaling.py, not read from its summary. The two web
# sources cannot be re-read here; only the sentence, its disclaimer and the bibliography
# entries are checked.
_n0j = _n_expd
_jr = _rt("expJ_raw.csv")
_jp = _rt("expJ_params.csv")
_jf = _rt("expJ_factors.csv")
_jc = _rt("expJ_calib_counts.csv")
_jl = _rt("expJ_latency.csv")
_jF = _rt("expF_corr_raw.csv")


def _jv(K, n, f, m, S=4):
    q = _jr[(_jr.sensors == S) & (_jr.K == K) & (_jr.n == n) & (_jr.fusion == f)].sort_values("seed")
    if len(q) != 20:
        fails.append(f"expJ: S={S}, K={K}, n={n}, {f} has {len(q)} rows, not 20")
    return q[m].values


def _jd(a, b):
    d = a - b
    d = d[~np.isnan(d)]
    return float(d.mean()), float(_st.ttest_1samp(d, 0.0).pvalue)


# path check against exp_correlated.py part (i)
_jz = _jr[_jr.sensors == 3].set_index(["seed", "fusion"])
_jy = _jF[_jF.severity == 1.0].set_index(["seed", "fusion"])
_jdev = [abs(float(_jz.loc[_k, _c]) - float(_jy.loc[_k, _c])) for _k in _jz.index
         for _c in ("rho", "acc", "macroAUC", "ece", "brier", "cwece", "acc_adv", "ece_adv", "auc_adv")]
need_true("expJ path check: three sensors, K = 2, s = 1 reproduces expF within 1e-9 over 540 values",
          len(_jdev) == 540 and max(_jdev) <= 1e-9)
# parameters
_jpp = {(int(x.sensors), int(x.K)): int(x.formula_extra) for x in _jp.itertuples()}
need_true("expJ parameter formula: 15 at three sensors, K = 2; 27, 54, 81 at four; the tables agree",
          _jpp[(3, 2)] == 15 and [_jpp[(4, _k)] for _k in (2, 3, 4)] == [27, 54, 81]
          and bool((_jp.formula_extra == _jp.tables_extra).all()))
# correlation, time, factors, samples
_jrho = [_jv(_k, 300, "naive-CI", "rho6").mean() for _k in (2, 3, 4)]
need_true("expJ rho: the largest pairwise correlation is at K = 2", _jrho[0] == max(_jrho))
_jt = _jl.assign(sensors=_jl.sensors.astype(str)).groupby(["sensors", "K", "fusion"]).q_mean_ms.mean()
need_true("expJ latency: 5 replications x 2,500 queries in every condition", len(_jl) == 13 * 5 and bool((_jl.n_queries == 2500).all()))
_jtl = [_jt[("4", _k, "W-latent")] for _k in (2, 3, 4)]
_jto = [_jt[("4", _k, "W-observed")] for _k in (2, 3, 4)]
_jtn = [_jt[("4", _k, "naive-CI")] for _k in (2, 3, 4)]
need_true("expJ latency: every mean update under 0.14 ms and the bound is tight", 0.13 < float(_jt.max()) < 0.14)
_jff = _jf[(_jf.sensors == 4) & (_jf.fusion == "W-latent")].set_index("K")
need_true("expJ largest factor with W is 48, 72, 96 and the largest overall at K = 4",
          [int(_jff.loc[_k].largest_with_W) for _k in (2, 3, 4)] == [48, 72, 96] and int(_jff.loc[4].largest) == 96)
_jn300 = _jc[(_jc.sensors == 4) & (_jc.K == 4) & (_jc.W >= 1) & (_jc.n == 300)].n_state.mean()
_jn1200 = _jc[(_jc.sensors == 4) & (_jc.K == 4) & (_jc.W >= 1) & (_jc.n == 1200)].n_state.mean()
need_true("expJ samples per jamming state: about 35 and about 140", abs(_jn300 - 35) < 1 and abs(_jn1200 - 140) < 2)
# recovery
_jwo = {_k: _jd(_jv(_k, 300, "W-observed", "ece_worst"), _jv(_k, 300, "naive-CI", "ece_worst")) for _k in (2, 4)}
_jwl = {_k: _jd(_jv(_k, 300, "W-latent", "ece_worst"), _jv(_k, 300, "naive-CI", "ece_worst")) for _k in (2, 4)}
need_true("expJ most severe state: both W models lower the ECE significantly at K = 2 and 4",
          all(_jwo[_k][0] < 0 and _jwo[_k][1] < 0.05 and _jwl[_k][0] < 0 and _jwl[_k][1] < 0.05 for _k in (2, 4)))
_jgo = [-_jd(_jv(_k, 300, "W-observed", "ece_adv"), _jv(_k, 300, "naive-CI", "ece_adv"))[0] for _k in (2, 3, 4)]
_jgl = [-_jd(_jv(_k, 300, "W-latent", "ece_adv"), _jv(_k, 300, "naive-CI", "ece_adv"))[0] for _k in (2, 3, 4)]
_jglp = _jd(_jv(4, 300, "W-latent", "ece_adv"), _jv(4, 300, "naive-CI", "ece_adv"))[1]
_jgd = _jd(_jv(4, 1200, "naive-CI", "ece_adv") - _jv(4, 1200, "W-observed", "ece_adv"),
           _jv(4, 300, "naive-CI", "ece_adv") - _jv(4, 300, "W-observed", "ece_adv"))
need_true("expJ the fixed rule: the ECE gain with W observed does not grow significantly at K = 4", _jgd[0] > 0 and _jgd[1] >= 0.05)
_jnb = _jd(_jv(4, 1200, "W-observed", "brier_adv"), _jv(4, 300, "W-observed", "brier_adv"))
_jna = _jd(_jv(4, 1200, "W-observed", "auc_adv"), _jv(4, 300, "W-observed", "auc_adv"))
need_true("expJ quadrupling n improves the degraded-subset Brier score and AUC, p < 0.001, with W observed only",
          _jnb[0] < 0 and _jnb[1] < 1e-3 and _jna[0] > 0 and _jna[1] < 1e-3
          and _jd(_jv(4, 1200, "W-latent", "brier_adv"), _jv(4, 300, "W-latent", "brier_adv"))[1] >= 1e-3
          and _jd(_jv(4, 1200, "W-latent", "auc_adv"), _jv(4, 300, "W-latent", "auc_adv"))[1] >= 1e-3)
# manuscript
need_tex("expJ VII-F first sentence",
         f"With $W$ a parent of all four reports and $|W|{{=}}2$--$4$ (RF reports lost to jamming, pairwise $\\rho$ up to ${_f(_jrho[0], 2)}$), "
         f"the cost is ${_jpp[(4, 2)]}(|W|{{-}}1)$ parameters and the mean update stays under $0.14$ ms.")
need_tex("expJ VII-F second sentence",
         f"At $|W|{{=}}4$, with about ${_jn300:.0f}$ calibration samples per jamming state, the most severe state's ECE falls by ${_f(-_jwo[4][0])}$ with $W$ observed but only by ${_f(-_jwl[4][0])}$ with $W$ marginalized.")
need_supp_later("expJ VII-F, quadrupling",
         f"with $W$ observed, quadrupling the calibration samples improves Brier and AUC on the degraded subset ($p<10^{{-3}}$) but not significantly the ECE gain ($p{{=}}{_f(_jgd[1], 2)}$);")
need_tex("expJ the cost of three sensors stays", f"The cost is ${_jpp[(3, 2)]}$ additional free parameters.")
need_tex("expJ Discussion sentence",
         "a shared nuisance parent handles correlated errors at a cost linear in $|W|$, its gain depends mainly on knowing the nuisance state at run time (Sec.~\\ref{sec:corrfail}),")
_jab = re.search(r"\\begin\{abstract\}(.*?)\\end\{abstract\}", tex).group(1) if HAVE_PAPER else ""
_jwc = len(_jab.split())
need_true("expJ abstract has at most 250 words", _jwc <= 250)
for _lab, _sn in (("abstract scope", "Synthetic experiments run over twenty replications in a fully specified generative Bayesian-network world."),
                  ("abstract probabilities", "but its probabilities and sensor-report rules are hand-set."),
                  ("abstract bound", "below its 0.0055 bound with error-free reports."),
                  ("conclusion", "In a fully specified generative BN world, with labeled latent states and a prior near the truth,"),
                  ("Scope", "\\emph{Scope.} The synthetic evaluation runs in one fully specified generative BN, the evaluation world of Sec.~\\ref{sec:world}."),
                  ("Scope, the other three nodes", "the EO/IR, acoustic, and radar nodes are calibrated in the evaluation world (Sec.~\\ref{sec:pubdata})."),
                  ("web sources sentence", "NATO's Technical Interoperability Exercise (TIE) 26 tested counter-drone systems alongside other countries' equipment in 2026 \\cite{nato2026tie}, and the SAPIENT standard defines what components exchange but not how they work internally \\cite{govuk2026sapient}; a fusion module could compute such a probability.")):
    need_tex(f"expJ {_lab}", _sn)
for _sn in ("conditional probability tables are hand-set", "the other three modalities remain future work", "synthetic generative-proxy world", "real-dataset instantiation is future work",
            "Two boundaries of the present study", "helps even without the free observation", "dynamic, neuro-symbolic", "restores calibration under correlated degradation"):
    need_true(f"expJ retired wording is gone: {_sn}", not HAVE_PAPER or _sn not in tex)
_jbib = open(os.path.join(PAPER, "references.bib"), encoding="utf-8").read() if HAVE_PAPER else ""
for _sn in ("@misc{nato2026tie,", "https://www.nato.int/en/multimedia/multimedia/videos/2026/06/15/nato-allies-and-industry-test-counter-drone-technologies",
            "@misc{govuk2026sapient,", "https://www.gov.uk/guidance/sapient-autonomous-sensor-system"):
    need_true(f"expJ bibliography has {_sn[:40]}", not HAVE_PAPER or _sn in _jbib)
need_true("expJ both web references carry the access date", not HAVE_PAPER or _jbib.count("Accessed: Nov. 6, 2026") == 2)
if HAVE_PAPER:
    print(f"expJ checks run: {_n_expd - _n0j}")
assert _n_expd - _n0j > 0

# ---- the two control studies of Sec. II. There are no numbers; only the sentence, the
# retired wording and the bibliography entries are checked.
_n0r = _n_expd
need_tex("expRW sentence",
         "Control studies treat neighboring concerns differently: sensor uncertainty as an unknown time-varying output gain, stabilized in probability by event-triggered feedback \\cite{sun2025sensitivity}, "
         "and UAV cooperation as a quadrotor formation steered to a scalar-field optimum under limited communication \\cite{jin2025sourceseeking}; "
         "here the uncertainty is a report table estimated from labeled scenarios, the output a threat posterior, and the UAVs the assessed targets.")
for _sn in ("nearest prior art", "most widely used ordinal", "swarm-level threat assessment remains future work", "within known relations", "stabilizing estimates"):
    need_true(f"expRW retired wording is gone: {_sn}", not HAVE_PAPER or _sn not in tex)
for _sn in ("@article{sun2025sensitivity,", "10.1109/TAC.2024.3508547", "volume  = {70}, number = {4}, pages = {2786--2793}, year = {2025},",
            "@article{jin2025sourceseeking,", "10.1109/TIE.2024.3423337", "volume  = {72}, number = {2}, pages = {1969--1979}, year = {2025},"):
    need_true(f"expRW bibliography has {_sn[:40]}", not HAVE_PAPER or _sn in _jbib)
if HAVE_PAPER:
    print(f"expRW checks run: {_n_expd - _n0r}")
assert _n_expd - _n0r > 0

# ---- the Supplementary Material and the figures redrawn from the same data. The numbers
# of the supplement are recomputed here and looked up in it.
import filecmp as _fc
_n0p = _n_expd
_supp_path = os.path.join(PAPER, "supplementary_material.tex")
_supp = (open(_supp_path, encoding="utf-8").read().replace("\n", " ").replace("{,}", ",")
         if HAVE_PAPER and os.path.exists(_supp_path) else "")
need_true("expSup supplementary_material.tex exists and is not empty", not HAVE_PAPER or bool(_supp))


def need_supp(label, snippet):
    global _n_expd
    _n_expd += 1
    if HAVE_PAPER and snippet not in _supp:
        fails.append(f"{label}: expected {snippet!r} in supplementary_material.tex")


for _l, _s in _supp_later:
    need_supp(_l, _s)
_supp_done = len(_supp_later)   # sentences registered after this point are checked at the end (block expP11)



# the world's noise level: the full paragraph is Sec. S1, the body keeps a summary. Recomputed from Exp-N.
_NT = (0.25, 0.40, 0.60)


def _nv(tau, N, m, k):
    return nz[(nz.tau == tau) & (nz.N == N) & (nz.method == m)].sort_values("rep")[k].values


_orc = [float(_nv(t, -1, "CPT-oracle", "acc").mean()) for t in _NT]
_nd = {(t, N, k): float((_nv(t, N, "Proposed", k) - _nv(t, N, "B1-MLE", k)).mean())
       for t in _NT for N in (10, 50, 200, 400) for k in ("acc", "macroAUC", "brier", "ece")}
_ot = f"oracle accuracy ${_orc[0]:.3f}$, ${_orc[1]:.3f}$, ${_orc[2]:.3f}$"
need_tex("expSup noise: oracle accuracy, body", _ot)
need_supp("expSup noise: oracle accuracy, supplement", _ot)
_e50 = [_nd[(t, 50, "ece")] for t in _NT]
need_tex("expSup noise: ECE margin, body", f"an ECE margin that grows with noise (${_e50[0]:.3f}$, ${_e50[1]:.3f}$, ${_e50[2]:.3f}$ at $N{{=}}50$)")
need_supp("expSup noise: ECE margin, supplement", f"(at $N{{=}}50$, ${_e50[0]:.3f}$, ${_e50[1]:.3f}$, and ${_e50[2]:.3f}$ as $\\tau_T$ rises)")
need_true("expSup noise: the ECE margin grows with noise at every N", all(_nd[(0.25, N, "ece")] > _nd[(0.40, N, "ece")] > _nd[(0.60, N, "ece")] for N in (10, 50, 200, 400)))
need_true("expSup noise: ahead of MLE on all four metrics, every N, all three worlds",
          all(_nd[(t, N, "acc")] > 0 and _nd[(t, N, "macroAUC")] > 0 and _nd[(t, N, "brier")] < 0 and _nd[(t, N, "ece")] < 0 for t in _NT for N in (10, 50, 200, 400)))
_ffa = [_nv(t, 10, "Proposed", "macroAUC") - _nv(t, 50, "B1-MLE", "macroAUC") for t in _NT]
_ffc = [_nv(t, 10, "Proposed", "acc") - _nv(t, 50, "B1-MLE", "acc") for t in _NT]
_ffam = [float(x.mean()) for x in _ffa]
need_true("expSup noise: five-fold macro-AUC, all p <= 0.01", all(_st.ttest_1samp(x, 0.0).pvalue <= 0.01 for x in _ffa))
_fft = f"(${min(_ffam):+.3f}$ to ${max(_ffam):+.3f}$, all $p\\le0.01$)"
need_tex("expSup noise: the body points to the supplement for that comparison", "but not its ECE lead at $\\tau_T{=}0.25$ (Supplementary Material, Sec.~S1).")
need_supp("expSup noise: five-fold crossing, supplement", _fft)
_ffp = [float(_st.ttest_1samp(x, 0.0).pvalue) for x in _ffc]
need_true("expSup noise: no significant accuracy difference in the two sharper worlds, ahead in the blurriest",
          _ffp[0] > 0.05 and _ffp[1] > 0.05 and _ffp[2] < 0.05 and _ffc[2].mean() > 0)
need_true("expSup noise: the main study's sharpness is 0.40", _ms.TAU_T == 0.40)
need_tex("expSup noise: the sharpness is named in the body", "Rebuilt at threat-response sharpness $\\tau_T\\in\\{0.25,0.40,0.60\\}$ ($0.40$ in the main study;")
need_supp("expSup noise: the world of the main study", f"The threat node's response sharpness $\\tau_T{{=}}0.40$ leaves deliberate overlap between adjacent threat levels, enough that the CPT oracle itself reaches only ${_orc[1]:.3f}$ accuracy")
need_supp("expSup noise: the section number", "\\section{Dependence on the world's noise level}")
need_true("expSup noise: it is the first section of the supplement, so Sec. S1", not HAVE_PAPER or re.findall(r"\\section\{([^}]*)\}", _supp)[:1] == ["Dependence on the world's noise level"])

# the smoothing paragraph: its numbers are recomputed from Exp-B


def _sv(m, N, k):
    return bl[(bl.method == m) & (bl.N == N)].sort_values("rep")[k].values


_b3d = {(N, k): _sv("Proposed", N, k) - _sv("uninformative MAP", N, k) for N in (10, 400) for k in ("macroAUC", "brier", "ece")}
_b3p = {x: float(_st.ttest_1samp(v, 0.0).pvalue) for x, v in _b3d.items()}
need_tex("expSup smoothing: B3 against MLE", f"reaches ECE ${_sv('uninformative MAP', 10, 'ece').mean():.3f}$ at $N{{=}}10$, worse than even unsmoothed MLE (${_sv('B1-MLE', 10, 'ece').mean():.3f}$)")
need_tex("expSup smoothing: at N=10", f"beats it by ${_b3d[(10, 'macroAUC')].mean():+.3f}$ macro-AUC and ${_b3d[(10, 'brier')].mean():+.3f}$ Brier (both $p<10^{{-7}}$)")
need_true("expSup smoothing: the p bound at N=10", _b3p[(10, "macroAUC")] < 1e-7 and _b3p[(10, "brier")] < 1e-7)
need_tex("expSup smoothing: at N=400", f"(at $N{{=}}400$, ${_b3d[(400, 'macroAUC')].mean():+.3f}$ macro-AUC and ${_b3d[(400, 'ece')].mean():+.3f}$ ECE, $p<10^{{-3}}$)")
need_true("expSup smoothing: the p bound at N=400", _b3p[(400, "macroAUC")] < 1e-3 and _b3p[(400, "ece")] < 1e-3)
_b5a = float((_sv("Proposed", 400, "acc") - _sv("logistic regression", 400, "acc")).mean())
_b5e = float(_sv("logistic regression", 400, "ece").mean() - _sv("Proposed", 400, "ece").mean())
need_true(f"expSup smoothing: B5 is three points of accuracy behind at N=400 ({_b5a:.4f})", 0.025 <= _b5a < 0.035)
need_tex("expSup smoothing: B5 ECE at N=400", f"at $N{{=}}400$ still by three points of accuracy and ${_sv('logistic regression', 400, 'ece').mean().round(3) - _sv('Proposed', 400, 'ece').mean().round(3):.3f}$ ECE")
need_true("expSup smoothing: B4 and B5 trail at every N", all(
    _sv(m, N, "acc").mean() < _sv("Proposed", N, "acc").mean() and _sv(m, N, "macroAUC").mean() < _sv("Proposed", N, "macroAUC").mean()
    and _sv(m, N, "brier").mean() > _sv("Proposed", N, "brier").mean() and _sv(m, N, "ece").mean() > _sv("Proposed", N, "ece").mean()
    for m in ("Laplace", "logistic regression") for N in (10, 50, 100, 200, 400)))

# the figures of the supplement: where the body points, which file the supplement draws, in which order
for _sn in ("(Supplementary Material, Sec.~S2 and Fig.~S1).",
            "(Supplementary Material, Fig.~S2) shows the knowledge-vs-data trade-off",
            "$0.507$; Supplementary Material, Sec.~S1)",
            "it lies on the flat plateau of the ESS sweep of Sec.~\\ref{sec:priorsens} and is not tuned per training set"):
    need_tex(f"expSup body locator: {_sn[:40]}", _sn)
need_true("expSup the body cites exactly Figs. S1 to S3, Secs. S1 to S5 and Tables S1 to S5", not HAVE_PAPER or set(re.findall(r"(?:Figs?\.|Secs?\.|Tables?)~S\d+", tex)) == {"Fig.~S1", "Fig.~S2", "Fig.~S3", "Sec.~S1", "Sec.~S2", "Sec.~S3", "Sec.~S4", "Sec.~S5", "Table~S1", "Table~S2", "Table~S3", "Table~S4", "Table~S5"})
need_true("expSup supplement draws the regime map, the ESS sweep and the tornado, in that order", not HAVE_PAPER or re.findall(r"\\includegraphics\[[^\]]*\]\{([^}]+)\}", _supp) == ["fig_expE_regime.png", "fig_expE_ess.png", "fig_expT_tornado_print.png"])
need_supp("expSup supplement locator", "Fig.~\\ref{s:regime} belongs to Sec.~VII-E(ii) of the article, Fig.~\\ref{s:ess} to Sec.~VII-E(iii), and Fig.~\\ref{s:tornado} to Sec.~VII-I.")
need_supp("expSup caption of the regime map", "Read with Sec.~VII-E of the article.")
need_supp("expSup author", "\\author{Moogeun Song\\\\ \\normalsize Division of Unmanned Aerial Vehicles (Drone System Major), Kyungwoon University,\\\\ \\normalsize Gumi 39160, Republic of Korea (e-mail: m.song@ikw.ac.kr)}")
for _sn in ("\\label{prop:cmap}", "\\label{app:proofs}", "\\label{fig:corrfail}", "\\label{tab:relvariants}", "\\label{tab:highrel}", "\\label{tab:misspec}", "\\label{tab:rfsplits}"):
    need_tex(f"expSup kept in the body: {_sn}", _sn)

# the figures: the same data, the copy in the manuscript folder is the file in results/
for _png, _src in (("exp1_scarcity_strip4.png", "expS_scarcity_raw.csv"), ("fig_expF_corr_print.png", "expF_corr_raw.csv"), ("fig_expT_tornado_print.png", "expT_summary.md"),
                 ("fig_expE_regime.png", "expE_prior_raw.csv"), ("fig_expE_ess.png", "expE_ess_raw.csv")):
    _rp, _pp = os.path.join(RES, _png), os.path.join(PAPER, _png)
    need_true(f"expSup figure {_png}: the copy in the manuscript folder is the file in results/", not HAVE_PAPER or (os.path.exists(_pp) and _fc.cmp(_rp, _pp, shallow=False)))
    need_true(f"expSup figure {_png} is not older than {_src}", not FRESH or os.path.getmtime(_rp) >= os.path.getmtime(os.path.join(RES, _src)))
for _png in ("exp1_scarcity_strip4.png", "fig_expF_corr_print.png"):
    need_tex(f"expSup main.tex draws {_png}", "{" + _png + "}")
for _png in ("{exp1_scarcity.png}", "{fig_expF_corr.png}", "{fig_expT_tornado.png}", "fig_expT_tornado_print", "fig:tornado", "fig_expE_regime", "fig_expE_ess", "fig:regime", "fig:ess", "Design Principles"):
    need_true(f"expSup retired from main.tex: {_png}", not HAVE_PAPER or _png not in tex)

# the clean-up: next steps only in the conclusion, what is not evaluated is said so
need_true("expSup no future-work wording in the body", not HAVE_PAPER or not re.search(r"future work|extended evaluation|left to future|follow-up study|deferred to future", tex))
need_true("expSup the next step appears once, in the conclusion", not HAVE_PAPER or (tex.count("next step") == 1 and tex.index("next step") > tex.index("\\section{Conclusion}")))
# what is not covered is said once, in the Limitations; the old scope statements are gone
for _sn in ("Latency on the embedded hardware of a fielded effector, where the platform sets the budget, is not evaluated here.",
            "dedicated dropout and spoofing sweeps not being evaluated here;",
            "they do \\emph{not} measure sim-to-real transfer.",
            "the other modalities are not calibrated on real recordings.",
            "and embedded-latency profiling belong to the full protocol and are not evaluated here;",
            "dedicated dropout/spoofing sweeps are not evaluated here)",
            "We evaluated the framework in the proxy world of Sec.~\\ref{sec:world}.",
            "so the reported gains measure the grounding and calibration machinery inside that world, not sim-to-real transfer."):
    need_true(f"expSup old scope statement is gone: {_sn[:40]}", not HAVE_PAPER or _sn not in tex)
for _sn in ("Fifth, swarms, sensor dropout, spoofing, and latency on embedded hardware are not covered.",
            "We evaluated the framework in the evaluation world of Sec.~\\ref{sec:world}.",
            "\\emph{Limitations.} First, the synthetic evaluation runs in one generative world (Sec.~\\ref{sec:world}), and only the RF likelihood is grounded in real recordings"):
    need_tex(f"expSup scope statement: {_sn[:40]}", _sn)

if HAVE_PAPER:
    print(f"expSup checks run: {_n_expd - _n0p}")
assert _n_expd - _n0p > 0

# ---- the noise-level results named by metric and recomputed, the scope of Sec. VI-D, the style of the whole text, [12]
_n0q = _n_expd
_NTN = (10, 50, 200, 400)
_np = {(t, N, k): float(_st.ttest_rel(_nv(t, N, "Proposed", k), _nv(t, N, "B1-MLE", k)).pvalue)
       for t in _NT for N in _NTN for k in ("acc", "macroAUC", "brier", "ece")}
_pab = [v for (t, N, k), v in _np.items() if k in ("macroAUC", "brier")]
need_true(f"expSup2 noise: macro-AUC and Brier significant in all 24 cells (largest p {max(_pab):.4f})", len(_pab) == 24 and max(_pab) <= 0.003)
need_tex("expSup2 noise: the metrics are named, body", "the grounded model beats unsmoothed MLE on macro-AUC and Brier at every training size tested in all three worlds (all $p\\le0.003$), with an ECE margin")
need_supp("expSup2 noise: the metrics are named, supplement", "The grounded model beats unsmoothed MLE on macro-AUC and Brier at every training size tested in all three worlds (all $p\\le0.003$).")
need_supp("expSup2 noise: the ECE margin, supplement", "Its ECE margin grows with noise at every training size tested")
_ns = sorted((t, N, k) for (t, N, k), v in _np.items() if v >= 0.05)
need_true(f"expSup2 noise: the cells not significant are the three named, of 48 ({_ns})", len(_np) == 48 and _ns == [(0.25, 10, "ece"), (0.25, 200, "acc"), (0.25, 400, "acc")])
need_supp("expSup2 noise: the three cells", f"At $\\tau_T{{=}}0.25$ three gaps are not significant: accuracy at $N{{=}}200$ and $400$ ($p{{=}}{_np[(0.25, 200, 'acc')]:.2f}$, ${_np[(0.25, 400, 'acc')]:.2f}$) and ECE at $N{{=}}10$ ($p{{=}}{_np[(0.25, 10, 'ece')]:.2f}$).")
_ma = [_nd[(t, 10, "acc")] for t in _NT]
_mu = [_nd[(t, 10, "macroAUC")] for t in _NT]
need_supp("expSup2 noise: the margins at N=10", f"the accuracy and macro-AUC margins at $N{{=}}10$ do not grow (${_ma[0]:+.3f}$, ${_ma[1]:+.3f}$, ${_ma[2]:+.3f}$ and ${_mu[0]:+.3f}$, ${_mu[1]:+.3f}$, ${_mu[2]:+.3f}$)")
need_true("expSup2 noise: those margins do not grow from the sharpest to the blurriest world", _ma[2] <= _ma[0] and _mu[2] <= _mu[0])
need_supp("expSup2 noise: the sizes tested", "and repeated the comparison at $N\\in\\{10,50,200,400\\}$.")
need_true("expSup2 noise: those are the sizes in the raw file", sorted(set(int(x) for x in nz.N) - {-1}) == list(_NTN))
need_supp("expSup2 noise: the comparison, supplement", "The comparison of Proposed at $N{=}10$ with MLE at $N{=}50$ also varies: Proposed stays ahead on macro-AUC in all three worlds")
_ffe = _nv(0.25, 10, "Proposed", "ece") - _nv(0.25, 50, "B1-MLE", "ece")
_ffep = float(_st.ttest_1samp(_ffe, 0.0).pvalue)
need_true("expSup2 noise: ECE unresolved at tau_T = 0.25", _ffep > 0.05)
need_supp("expSup2 noise: ECE at tau_T = 0.25, supplement", f"and ECE unresolved at $\\tau_T{{=}}0.25$ (${float(_ffe.mean()):+.3f}$, $p{{=}}{_ffep:.2f}$).")
for _sn in ("five-fold crossing", "since a noisier world", "the margin \\emph{grows}", "with a margin that grows with noise", "is the one part that varies"):
    need_true(f"expSup2 noise: retired wording is gone: {_sn}", not HAVE_PAPER or (_sn not in tex and _sn not in _supp))
# Sec. VI-D: what is not evaluated, and what Sec. VII-F does sweep
need_tex("expSup2 Sec. VII-F sweeps a low-SNR profile", "their confusion tables are mixed toward a common low-SNR profile")
need_true("expSup2 no script computes the Chan-Darwiche distance, as Sec. VI-D says",
          not any("darwiche" in open(_p, encoding="utf-8", errors="ignore").read().lower() for _p in glob.glob(os.path.join(HERE, "*.py")) if os.path.basename(_p) != "check_manuscript_numbers.py"))
need_true("expSup2 retired wording is gone: none of them evaluated here (the", not HAVE_PAPER or "none of them evaluated here (the" not in tex)
# the style of the whole text: --- only in the empty table cells, \textbf only in Algorithm 1, C1 to C4 as heads
_raw = re.sub(r"(?m)^\s*%.*$", "", open(TEX, encoding="utf-8").read()) if HAVE_PAPER else ""
need_true(f"expSup2 --- only in the twelve empty table cells ({_raw.count('---')} found)", not HAVE_PAPER or (_raw.count("---") == 12 and len(re.findall(r"&\s*---\s", _raw)) == 12))
need_true("expSup2 no em-dash character in main.tex", "\u2014" not in _raw)
need_true("expSup2 no --- and no em-dash character in the supplement", "---" not in _supp and "\u2014" not in _supp)
need_true("expSup2 \\textbf only in Algorithm 1", not HAVE_PAPER or re.findall(r"\\textbf\{([^}]*)\}", _raw) == ["E-step:", "M-step:"])
for _i, _h in ((1, "Threat node with a data check."), (2, "Scarce-data grounding."), (3, "Sensor-calibrated measurement layer."), (4, "Validation.")):
    need_tex(f"expSup2 head of C{_i}", f"\\item \\emph{{C{_i}: {_h}}}")
# the three long sentences stay split
for _sn in ("and~\\ref{sec:discussion}). Replacing the nine-parameter ordinal ICI threat node",
            "in \\emph{favor} of ICI). Removing reliability discounting",
            "at $N{=}50$ and $200$. The threat response stays monotone, but its additive level",
            "from the ICI family). Hostile maneuver is U-shaped in speed (S2). Edges $d\\to N$",
            "are absent from the model (S3). Test-only shifts change the class or emission mix"):
    need_tex(f"expSup2 long sentence stays split: {_sn[:40]}", _sn)
need_true("expSup2 [12] has its pages in the bibliography", not HAVE_PAPER or bool(re.search(r"@inproceedings\{altendorf2005monotonicity,.{0,260}pages\s*=\s*\{18--26\}", _jbib, re.S)))
_bbl = (open(os.path.join(PAPER, "main.bbl"), encoding="utf-8", errors="replace").read()
        if HAVE_PAPER and os.path.exists(os.path.join(PAPER, "main.bbl")) else "")
need_true("expSup2 [12] is printed with its pages", not HAVE_PAPER or bool(re.search(r"\\bibitem\{altendorf2005monotonicity\}.{0,400}?pp\. 18--26", _bbl, re.S)))
need_true("expSup2 the two removed references are not printed", not HAVE_PAPER or ("das2004weightedsum" not in _bbl and "diez2006canonical" not in _bbl))
if HAVE_PAPER:
    print(f"expSup2 checks run: {_n_expd - _n0q}")
assert _n_expd - _n0q > 0


# ---- the remaining result tables, cell by cell (Tables 8, 10 and 11 are checked in their own blocks)
_n0t = _n_expd


def _tbody(label):
    _i = _rawt.index("\\label{" + label + "}")
    _a = _rawt.index("\\toprule", _i)
    return _rawt[_a:_rawt.index("\\bottomrule", _a)].split("\n")


def _tcells(row):
    return [c for c in re.findall(r"\$[^$]*\$", row) if c != "$+$"]


def _tcmp(name, got, want):
    global _n_expd
    assert want
    _n_expd += len(want)
    if got != want:
        fails.append(f"expTab {name}: main.tex {got} != results {want}")


def _tpm(g):
    """Mean and 95% half-width as printed (t with n - 1 df; the subset AUC rows have 13 or 14 replications)."""
    _cn = int(g.count())
    _h = (2.093 if _cn == 20 else float(_st.t.ppf(0.975, _cn - 1))) * g.std() / np.sqrt(_cn)
    return f"${g.mean():.3f}_{{\\pm{_h:.3f}}}$".replace("\\pm0.", "\\pm.")


def _td3(x):
    return f"{x:.3f}"[1:]


if HAVE_PAPER:
    _rawt = open(TEX, encoding="utf-8").read()
    # Table 4
    _ts = pd.read_csv(os.path.join(RES, "expS_scarcity_raw.csv"))
    _tl = [l for l in _tbody("tab:scarcity") if re.match(r"\s*(10|50|200|400)\s*&", l)]
    _ti = 0
    for _k in ("acc", "ece", "macroAUC", "far@pd0.9"):
        for _tn in (10, 50, 200, 400):
            _tcmp(f"Table 4 {_k} N={_tn}", _tcells(_tl[_ti]),
                  [_tpm(_ts[(_ts.method == _m) & (_ts.N == _tn)][_k]) for _m in ("B0-heuristic", "B1-MLE", "B2-expert", "Proposed")])
            _ti += 1
    # Table 5
    _tl = [l for l in _tbody("tab:stdbase") if re.match(r"\s*(10|50|200|400)\s*&", l)]
    _tm = ("uninformative MAP", "Laplace", "logistic regression", "B0-as-prior MAP")
    for _ti, _tn in enumerate((10, 50, 200, 400)):
        _tv = lambda m, k: f"${_td3(bl[(bl.method == m) & (bl.N == _tn)][k].mean())}$"
        _tcmp(f"Table 5 N={_tn}", _tcells(_tl[_ti]), [_tv(m, k) for m in _tm for k in ("acc", "macroAUC", "ece")])
        _tcmp(f"Table 5 Brier N={_tn}", _tcells(_tl[_ti + 4]), [_tv(m, "brier") for m in _tm])
    _tr = [l for l in _tbody("tab:stdbase") if "reference Brier" in l or "B1 (MLE)" in l]
    for _tx, _m in ((_tr[0], "Proposed"), (_tr[1], "B1-MLE")):
        _tcmp(f"Table 5 reference {_m}", _tcells(_tx)[:4], [f"${_td3(bl[(bl.method == _m) & (bl.N == _tn)].brier.mean())}$" for _tn in (10, 50, 200, 400)])
    # Table 6, from the replication-level file
    _tf = pd.read_csv(os.path.join(RES, "expS_fusion_raw.csv"))
    _tb = _tbody("tab:sensors")
    for _lab, _c in (("Radar only", "radar only"), ("RF only", "RF only"), ("EO/IR only", "EO/IR only"), ("Acoustic only", "acoustic only"),
                     ("All, abstract", "all (abstract model)"), ("All, calibrated", "all (calibrated)"), ("All, calib.$+$rel.", "all (calib+reliab)")):
        _tl = [l for l in _tb if l.strip().startswith(_lab)]
        _d = _tf[_tf.config == _c]
        _tcmp(f"Table 6 {_lab}", _tcells(_tl[0]) if len(_tl) == 1 else [], [_tpm(_d[k]) for k in ("acc", "macroF1", "macroAUC", "brier", "ece")])
    for _head, _sub in (("\\emph{radio-silent drones", "silent-drone"), ("\\emph{far-range subset, all classes", "far-range"), ("\\emph{far-range radio-silent drones", "far-silent")):
        _ta = [i for i, l in enumerate(_tb) if _head in l]
        for _j, _v in zip((0, 3, 4) if _sub == "silent-drone" else (0, 1, 2), ("abstract", "calibrated", "calib+reliab")):
            _d = _tf[_tf.config == f"_subset_{_sub}_{_v}"]
            _tcmp(f"Table 6 {_sub} {_v}", [c for c in _tcells(_tb[_ta[0] + 1 + _j]) if c != "$+$"] if len(_ta) == 1 else [], [_tpm(_d[k]) for k in ("acc", "macroAUC", "brier", "ece")])
    _tfx = pd.read_csv(os.path.join(RES, "expRS_fusion.csv"))
    for _lab, _v in (("All, no RF", "no RF"), ("All, hand-set", "hand-set emission-aware")):
        _tl = [l for l in _tb if l.strip().startswith(_lab)]
        _d = _tfx[(_tfx.variant == _v) & (_tfx.subset == "all")]
        _tcmp(f"Table 6 {_lab}", _tcells(_tl[0]) if len(_tl) == 1 else [], [_tpm(_d[k]) for k in ("acc", "macroF1", "macroAUC", "brier", "ece")])
    _ta = [i for i, l in enumerate(_tb) if "\\emph{radio-silent drones" in l]
    for _j, _lab, _v in ((1, "no RF", "no RF"), (2, "hand-set", "hand-set emission-aware")):
        _d = _tfx[(_tfx.variant == _v) & (_tfx.subset == "radio-silent")]
        need_true(f"Table 6 silent-drone: the row of {_lab} is where it should be", len(_ta) == 1 and _lab in _tb[_ta[0] + 1 + _j])
        _tcmp(f"Table 6 silent-drone {_v}", [c for c in _tcells(_tb[_ta[0] + 1 + _j]) if c != "$+$"] if len(_ta) == 1 else [], [_tpm(_d[k]) for k in ("acc", "macroAUC", "brier", "ece")])
    # Table 7
    _tb = _tbody("tab:calibsize")
    _tc7 = pd.read_csv(os.path.join(RES, "expS_table_stats.csv"))
    _tc7 = _tc7[_tc7.table == "T5cs"].set_index("cell")
    for _lab, _c in (("abstract", "0"), ("estimated, $n{=}30$", "30"), ("estimated, $n{=}100$", "100"), ("estimated, $n{=}300$", "300"), ("estimated, $n{=}1000$", "1000"), ("sensor oracle", "-1")):
        _tl = [l for l in _tb if l.strip().startswith(_lab)]
        _r = _tc7.loc[_c]
        _tp = lambda x, h: f"${x:.3f}_{{\\pm{h:.3f}}}$".replace("\\pm0.", "\\pm.")
        _got = _tcells(_tl[0]) if len(_tl) == 1 else []
        _tcmp(f"Table 7 {_lab}", _got[1:] if _lab.startswith("estimated") else _got,
              [_tp(_r.acc, _r.acc_h), _tp(_r.macroAUC, _r.macroAUC_h), _tp(_r.brier, _r.brier_h), _tp(_r.ece, _r.ece_h)])
    # Table 9
    _tb = _tbody("tab:highrel")
    _th = pd.read_csv(os.path.join(RES, "expS_highclass_reliability.csv"))
    _row = lambda lab: [c.strip() for c in [l for l in _tb if l.strip().startswith(lab)][0].split("&")[1:]]
    _clean = lambda cells: [c.replace("\\\\", "").replace("\\midrule", "").strip() for c in cells]
    for _lab, _col in (("predicted", "predicted"), ("observed ", "observed"), ("oracle pred", "oracle_predicted"), ("oracle obs", "oracle_observed")):
        _tcmp(f"Table 9 {_lab.strip()}", _clean(_row(_lab)), [f"{x:.3f}" for x in _th[_col]])
    _tcmp("Table 9 n", _clean(_row("$n$")), [str(int(x)) for x in _th.n])
    # the CSV keeps four decimals, so the printed third decimal is compared within half a unit of it
    _tci = [float(x) for x in re.findall(r"\.\d+", " ".join(_clean(_row("$95\\%$ CI"))))]
    _twant = [x for a, b in zip(_th.ci_lo, _th.ci_hi) for x in (a, b)]
    _n_expd += len(_twant)
    if len(_tci) != len(_twant) or any(abs(a - b) > 0.00056 for a, b in zip(_tci, _twant)):
        fails.append(f"expTab Table 9 CI: main.tex {_tci} != results {_twant}")
    for _cl in _ms.STATES["T"]:
        _d = rel[rel["class"] == _cl]
        _tcmp(f"Table 9 {_cl}", _tcells([l for l in _tb if l.strip().startswith(_cl)][0]), [f"${x:+.3f}$" for x in _d.observed - _d.predicted])
    # Fig. 4: the copy in the manuscript folder is the file replot_rf_confusion.py draws
    _fig4 = os.path.join(PAPER, "fig4_dronerf_cm.png")
    need_true("expTab Fig. 4: fig4_dronerf_cm.png in the manuscript folder is results/rf_confusion.png",
              os.path.exists(_fig4) and _fc.cmp(_fig4, os.path.join(RES, "rf_confusion.png"), shallow=False))
    if DRAFT:
        print(f"expTab checks run: {_n_expd - _n0t}")
assert not HAVE_PAPER or _n_expd - _n0t > 0

# ---- the naive random split of Sec. VII-C, recounted from the two halves of its test set. Each
# half holds row-normalized matrices with add-0.5 smoothing, so the counts come back exactly:
# a cell is (c + 0.5) / (n + 2) in a row of n predictions. And the sentences on what the
# repository regenerates: the schematics and the tables of variables and notation are not results.
_n0q = _n_expd


def _naive_counts(row, top=681):
    for _n in range(1, top + 1):
        _c = row * (_n + 2) - 0.5
        _k = np.round(_c)
        if np.abs(_c - _k).max() < 2e-3 and _k.sum() == _n:
            return _k.astype(int)
    return np.zeros(len(row), int)


_naive_half = [pd.read_csv(os.path.join(RES, f"rf_cm_split{_naive_h}.csv"), index_col=0) for _naive_h in "AB"]
_naive_n = [np.array([_naive_counts(_naive_m.loc[_naive_k].values) for _naive_k in _naive_m.index]) for _naive_m in _naive_half]
_naive_all = _naive_n[0] + _naive_n[1]
_naive_acc = float(np.trace(_naive_all) / max(_naive_all.sum(), 1))
_naive_pct = f"{100 * _naive_acc:.1f}"
_naive_log = open(os.path.join(RES, "expG_splits_run.log"), encoding="utf-8").read()
_naive_mat = pd.read_csv(os.path.join(RES, "rf_confusion_matrix.csv"), index_col=0)
need_true("expRep naive split: each half holds the 681 predictions its log names",
          all(int(_naive_x.sum()) == 681 for _naive_x in _naive_n) and _naive_log.count("(n=681)") == 2)
need_true("expRep naive split: the halves add up to rf_confusion_matrix.csv",
          list(_naive_mat.index) == list(_naive_half[0].index) and (_naive_all.sum(1) > 0).all()
          and np.abs(_naive_all / np.maximum(_naive_all.sum(1, keepdims=True), 1) - _naive_mat.values).max() < 6e-5)
need_true(f"expRep naive split: the log's accuracy is the recounted {_naive_acc:.4f}", f"held-out accuracy {_naive_acc:.4f}" in _naive_log)
need_supp_later("expRep naive split sentence, in the supplement", f"split over segments-and-bands, with the checkpoint also chosen on its test part, yields ${_naive_pct}\\%$")
need_true("expRep naive split left the body", not HAVE_PAPER or "segments-and-bands" not in tex)
_naive_readme = re.findall(r"naive random[^.]{0,110}?(\d\d\.\d)%", open(os.path.join(HERE, "README.md"), encoding="utf-8").read())
need_true(f"expRep naive split in the README: both mentions are {_naive_pct}% ({_naive_readme})", len(_naive_readme) == 2 and set(_naive_readme) == {_naive_pct})
need_tex("expRep the footnote names the results tables", "to reproduce every results table and figure in this article")
need_tex("expRep Sec. VI-E names the results tables", "Every results table and figure is regenerated by a named script")
need_true("expRep no claim on every table and figure", not HAVE_PAPER or not re.search(r"every table and figure", tex, re.I))
if DRAFT:
    print(f"expRep checks run: {_n_expd - _n0q}")
assert _n_expd - _n0q > 0

# ---- sentences corrected against the cited sources, and wording those sources did not support
_n0c = _n_expd
for _cs in ("has turned inexpensive drones into a threat to critical infrastructure, public events, and military bases",
            "low speed that hampers Doppler processing, and low infrared and acoustic signatures",
            "Bayesian fusion of radar and electronic support measures (ESM) observations classifies air targets",
            "These works address air defense, battlefield targets, or general parameter learning; none couples",
            "a least-squares fit yields a minimally informative Dirichlet prior with specified means",
            "with optional Beta priors, is the direct antecedent of our C2-ii",
            "fits as many as half of the CPTs of practical networks well",
            "Fusing complementary modalities overcomes the weaknesses of the individual sensors",
            "raises the detection F-score by $2$ to $5$ points over the best single sensor",
            "radar/visual fusion before tracking suppresses ground clutter",
            "RF/acoustic fusion raises accuracy most at very low SNR",
            "Evidence fusion can weight each source by its sensor reliability",
            "each parent contributes through an independent mechanism and a deterministic function combines the effects",
            "improve learned parameters when data are scarce \\cite{guo2017fcqmap,zhou2016monotonic}",
            "particularly on training sets of fewer than ten examples \\cite{altendorf2005monotonicity}",
            "a BN structure for an unmanned aerial vehicle (UAV) to assess the threat of its own targets",
            "10-fold cross-validation of a deep network on these four classes reports $84.5\\%$"):
    need_tex(f"expCite sentence: {_cs[:44]}", _cs)
for _cs in ("a UAV-threat BN structure", "a reliability weight on the evidence", "marginalized automatically \\cite", "persistent asymmetric", "few pixels", "single-layer, single-target", "canonical formulation", "dates to Heckerman", "post-hoc projection", "associative operator",
            "materially outperforms", "-point F-score gain", "sim-to-real gap", "entirely from digital-twin", "propagate information", "higher four-class", "Each remains"):
    need_true(f"expCite retired from main.tex: {_cs}", not HAVE_PAPER or _cs not in tex)
need_true("expCite no author as the subject of a sentence", not HAVE_PAPER or not re.search(r"\\emph\{et al\.\}|[A-Z][a-z]+ and (van der )?[A-Z][a-z]+ \\cite\{|\b(Rijmen|Yuan|Mrad|Wu|Lee|Di|Ye|Guo) \\cite\{", tex))
if DRAFT:
    print(f"expCite checks run: {_n_expd - _n0c}")
assert _n_expd - _n0c > 0

# ---- what the numbers mean: a control, what the data identify, intervals over replications,
# the worked track over seeds. A number equal to a file can still carry the wrong claim, so these
# recompute the claim from the raw files, and the old wording must not come back.
_n0a = _n_expd
_rc, _rk = _rt("expRC_raw.csv"), _rt("expRC_contrasts.csv")
_no = _rc[_rc.variant == "no-RF"].sort_values("rep")
_sy = _rc[_rc.variant == "synthetic-RF"].sort_values("rep")
_e2 = _rt("expI_e2e_raw.csv")
_RCM = ("acc", "macroAUC", "brier", "ece")


def _rcq(proto, a, b="no-RF"):
    return _rk[(_rk.proto == proto) & (_rk.A == a) & (_rk.B == b)].iloc[0]


def _rcraw(proto, v):
    return _e2[(_e2.proto == proto) & (_e2.rf_likelihood == v)].sort_values("rep")


_rcp = ("P0", "P1", "P1c", "P2", "P3")
_rcbad = 0
for _p in _rcp:
    for _v in ("abstract", "calib-hard"):
        for _k in _RCM:
            _d = _rcraw(_p, _v)[_k].values - _no[_k].values
            _rcbad += int(abs(_d.mean() - _rcq(_p, _v)[f"d_{_k}"]) > 1e-12 or abs(_st.ttest_1samp(_d, 0.0).pvalue - _rcq(_p, _v)[f"p_{_k}"]) > 1e-9)
need_true("expAudit control: the contrasts are those of the raw files, five splits", len(_no) == 20 and len(_sy) == 20 and _rcbad == 0)
_fu = _rt("expS_fusion_raw.csv")
_fu3 = _fu[_fu.config == "all (calib+reliab)"].sort_values("rep")
need_true("expAudit control: the synthetic-RF variant is the main study's fusion row", all(np.allclose(_fu3[_k].values, _sy[_k].values, rtol=0, atol=1e-12) for _k in ("acc", "macroAUC", "ece")))
_p0a, _p0h, _syn = _rcq("P0", "abstract"), _rcq("P0", "calib-hard"), _rcq("-", "synthetic-RF")
need_tex("expAudit control: (i) is worse than the control",
         f"Variant (i) is worse than the control (${_p0a.d_macroAUC:+.3f}$ macro-AUC, ${_p0a.d_ece:+.3f}$ ECE), so (ii) leads it by")
need_true("expAudit control: (i) is worse than the control in every split", all(_rcq(_p, "abstract").d_macroAUC < 0 and _rcq(_p, "abstract").p_macroAUC < 1e-10 and _rcq(_p, "abstract").d_ece > 0 for _p in _rcp))
need_true("expAudit control: most of the gain of (ii) over (i) is that harm", abs(_p0a.d_macroAUC) > 0.5 * (_p0h.d_macroAUC - _p0a.d_macroAUC))
need_tex("expAudit control: (ii) over the control, P0",
         f"Over the control (accuracy/macro-AUC/ECE ${_no.acc.mean():.3f}/{_no.macroAUC.mean():.3f}/{_no.ece.mean():.3f}$), (ii) adds ${_p0h.d_macroAUC:+.4f}$ macro-AUC and ${_p0h.d_brier:+.4f}$ Brier ($p<10^{{-10}}$), with accuracy and ECE unresolved ($p{{=}}{_p0h.p_acc:.2f}$ and ${_p0h.p_ece:.2f}$); the synthetic RF table, with its discount, adds ${_syn.d_macroAUC:+.4f}$ and ${_syn.d_brier:+.4f}$.")
need_true("expAudit control: P0 p-values", _p0h.p_macroAUC < 1e-10 and _p0h.p_brier < 1e-10 and _p0h.p_acc > 0.05 and _p0h.p_ece > 0.05)
_hs = [_rcq(_p, "calib-hard") for _p in _rcp]
_ha, _hb = [float(_x.d_macroAUC) for _x in _hs], [float(_x.d_brier) for _x in _hs]
need_tex("expAudit control: (ii) over the control, five splits",
         f"a gain over the control of ${min(_ha):+.4f}$ to ${max(_ha):+.4f}$ macro-AUC and ${max(_hb):+.4f}$ to ${min(_hb):+.4f}$ Brier, with accuracy and ECE unresolved over replications ($p>0.10$);")
need_true("expAudit control: five-split p-values", all(_x.p_macroAUC < 1e-7 and _x.p_brier < 1e-7 and _x.p_acc > 0.10 and _x.p_ece > 0.10 for _x in _hs))
need_true("expAudit control: the abstract's about 0.004 and the conclusion's 0.006 to 0.007", all(0.0035 <= _x <= 0.0046 for _x in _ha) and f"{-max(_hb):.3f}" == "0.006" and f"{-min(_hb):.3f}" == "0.007")
need_tex("expAudit control: the conclusion", "the RF channel adds about $0.004$ macro-AUC and $-0.006$ to $-0.007$ Brier over no RF channel under segment, time-block, and held-out-flight-mode splits")
need_true("expAudit control: the one-CNN sentence gave way to the per-CNN p-values", not HAVE_PAPER or "so their $p$-values cover scenario draws, not training" not in tex)
need_tex("expAudit background", "Background and Phantom are recorded once each and keep time blocks, so background is never held out by recording: in the contrast that moves $P(T)$ the background side always comes from the recording that trained the CNN.")
need_tex("expAudit abstract: synthetic labels", "With real DroneRF spectrograms and synthetic threat labels, the RF channel")
need_true("expAudit what cannot be held out is said twice: once in Sec. VII-H, once in the Limitations", not HAVE_PAPER or (tex.count("recorded once") == 2 and tex.count("cannot be held out by recording") == 1 and tex.count("never held out by recording") == 1 and tex.index("never held out by recording") < tex.index("\\section{Discussion}") < tex.index("cannot be held out by recording")))

# what the unlabeled case identifies
_er, _erc, _ere = _rt("expEM_rank.csv").set_index("what"), _rt("expEM_reach.csv"), _rt("expEM_recoding.csv").iloc[0]
need_true("expAudit identifiability: rank 112 of 126 for Q and for the observed distribution, 72 of 72 with T known",
          _er.loc["Q, 126 free parameters", "rank"] == 112 and _er.loc["P(T, reports | inputs), 126 free parameters", "rank"] == 112
          and _er.loc["Q, T known, 72 free parameters", "rank"] == 72 and _er.loc["P(T, reports | inputs), T known, 72 free parameters", "rank"] == 72)
need_true("expAudit identifiability: the 14 recodings span the missing directions", _er.loc["recoding tangents", "rank"] == 14 and _er.loc["recoding tangents", "s_after_gap"] < 1e-12)
need_true("expAudit identifiability: a recoding leaves the observed distribution unchanged", _ere.largest_change_Q < 1e-12 and _ere.largest_change_observed < 1e-12 and _ere.smallest_entry_recoded > -1e-9 and max(_ere.largest_move_H, _ere.largest_move_C, _ere.largest_move_T) > 0.01)
need_true("expAudit identifiability: the search moves no entry by more than about 0.05", 0.04 < _erc.move.max() <= 0.05 and len(_erc) == 366)
need_tex("expAudit identifiability: rank sentence", "At the world's true tables the Jacobian of the observed distribution has rank $112$ of the $126$ free parameters, the $14$ missing dimensions being those of the recodings, and rank $72$ of $72$ if the threat table is known;")
need_supp("expAudit identifiability: reach", "Near the true tables our local search over recodings moves no entry by more than about $0.05$, and the constraints and the prior exclude state permutations.")
need_tex("expAudit identifiability: no guarantee", "the prior picks the point within it (Supplementary Material, Sec.~S4), and EM reaches only a stationary point.")
need_tex("expAudit identifiability: Proposition 1 does not carry over", "MAP-EM does not inherit Proposition~\\ref{prop:cmap}:")

# the latent layer with and without labels
_mp = _rt("expC_mapem_raw.csv")
_mp = _mp[(_mp.N == 200) & (_mp.variant == "final-only") & (_mp.iters == 12)]
_ccn = _rt("expC_consistency_raw.csv")
_ccn = _ccn[(_ccn.world == "standard") & (_ccn.N == 200) & (_ccn.estimator == "cMAP")]
_klu = "/".join(f"{_mp['wkl_' + _x].mean():.3f}" for _x in "HNCT")
_kll = "/".join(f"{_ccn[_ccn.node == _x].wkl.mean():.3f}" for _x in "HNCT")
need_true("expAudit latent: 20 replications behind the KL values", len(_mp) == 20 and len(_ccn[_ccn.node == "H"]) == 20)
need_tex("expAudit latent: KL", f"The column-weighted KL at $N{{=}}200$ is ${_klu}$ for $H/N/C/T$ against ${_kll}$ with labels (top states of the latent variables: Supplementary Material, Sec.~S4).")
_la = _rt("expEM_latent_raw.csv")


def _lav(m, n, x):
    _q = _la[(_la.method == m) & (_la.N == n) & (_la.node == x)].sort_values("rep")
    assert len(_q) == 20, (m, n, x)
    return _q


_pc = lambda m, n, x: f"{100 * _lav(m, n, x).acc.mean():.0f}"
need_supp("expAudit latent: top state of H and C",
         f"the top state of $H$ is right ${_pc('MAP-EM', 200, 'H')}\\%$ of the time against ${_pc('Proposed', 200, 'H')}\\%$ with labels (${_pc('prior only', 0, 'H')}\\%$ for the prior alone, ${_pc('true CPTs', 0, 'H')}\\%$ with the true tables), that of $C$ ${_pc('MAP-EM', 200, 'C')}\\%$ against ${_pc('Proposed', 200, 'C')}\\%$, while that of $N$ does not separate (${_pc('MAP-EM', 200, 'N')}\\%$ in both).")
need_true("expAudit latent: N does not separate, H and C do", _pc("MAP-EM", 200, "N") == _pc("Proposed", 200, "N")
          and _st.ttest_1samp(_lav("MAP-EM", 200, "N").acc.values - _lav("Proposed", 200, "N").acc.values, 0.0).pvalue > 0.05
          and all(_st.ttest_1samp(_lav("MAP-EM", 200, _x).acc.values - _lav("Proposed", 200, _x).acc.values, 0.0).pvalue < 0.05 for _x in "HC"))
_mm = _rt("expM_raw.csv")
_mmd = []
for _n in (50, 200):
    _a = _mm[(_mm.N == _n) & (_mm.method == "MAP-EM (latent)")].sort_values("rep")
    _b = _mm[(_mm.N == _n) & (_mm.method == "fully-observed")].sort_values("rep")
    _mmd += [float((_b.acc.values - _a.acc.values).mean()), float((_a.ece.values - _b.ece.values).mean())]
need_true(f"expAudit latent: 0.01 to 0.02 in accuracy and ECE {_mmd}", all(0.0095 <= _x < 0.0215 for _x in _mmd))
need_tex("expAudit latent: threat level survives, the layer does not", "threat-level metrics survive without latent labels ($0.01$ to $0.02$ in accuracy and ECE) but the latent layer does not.")
need_tex("expAudit latent: abstract", "Unlabeled, the latent layer costs 0.01--0.02 expected calibration error.")
need_tex("expAudit latent: C1 needs the labels", "C1 does not apply without them: MAP-EM learns $T$ as a table, and the data check of Sec.~\\ref{sec:misspec} counts labeled $H$, $N$, and $C$.")
need_true("expAudit latent: the data check counts labeled nodes", 'g.counts("T", data)' in open(os.path.join(HERE, "exp_additivity.py"), encoding="utf-8").read() and "g.counts(x, data)" in open(os.path.join(HERE, "exp_additivity.py"), encoding="utf-8").read())
need_tex("expAudit shared assumptions: labels", "The first two failures are detectable when $H$, $N$, and $C$ are labeled and data suffice")

# intervals over replications
_bs = _rt("expBS_intervals.csv")
_bsy = _bs[(_bs.world == "synthetic") & (_bs.variant == "pipeline")].reset_index(drop=True)
_bso = _bs[(_bs.world == "synthetic") & (_bs.variant == "CPT-oracle")]
need_tex("expAudit intervals: synthetic", f"outside intervals that resample the replications ($[{_bsy.boot_lo[3]:.3f},{_bsy.boot_hi[3]:.3f}]$ and $[{-_bsy.boot_hi[4]:.3f},{-_bsy.boot_lo[4]:.3f}]$; every oracle bin's contains zero)")
need_true("expAudit intervals: the two decision-relevant bins exclude zero, every oracle bin contains it", len(_bsy) == 5 and _bsy.boot_excludes_0[3] and _bsy.boot_excludes_0[4] and len(_bso) == 5 and not _bso.boot_excludes_0.any())
need_true("expAudit three low bins agree to within 0.013", 0.012 < _bsy.dev[:3].abs().max() < 0.013)
need_tex("expAudit three low bins", "The three low bins agree to within $0.013$.")
_bsm = _bs[_bs.world == "mixed"]
_bh, _bo = _bsm[_bsm.variant == "calib-hard"], _bsm[_bsm.variant == "CPT-oracle"]
need_true("expAudit intervals: mixed world, 2 of 25 for each, the top bin of P1 and P1c; binomial 18 and 1",
          len(_bh) == 25 and len(_bo) == 25 and int(_bh.boot_excludes_0.sum()) == 2 and int(_bo.boot_excludes_0.sum()) == 2
          and int(_bh.binomial_outside.sum()) == 18 and int(_bo.binomial_outside.sum()) == 1
          and all(sorted(_q[_q.boot_excludes_0].proto) == ["P1", "P1c"] and (_q[_q.boot_excludes_0].bin_lo == 0.75).all() for _q in (_bh, _bo)))
need_tex("expAudit intervals: mixed world, the body", "(Supplementary Material, Sec.~S4 gives the intervals and the comparison with the CPT oracle fed the same RF evidence).")
need_supp("expAudit intervals: mixed world", "but resampling replications leaves two of $25$ outside for each (the top bin under segment groups, with and without the size match).")

# the worked track over seeds, and the cells that move it
_ts, _tc = _rt("expTS_seeds.csv"), _rt("expTS_cells.csv")
_tsp = _rt("expRS_track_split.csv")
_tsd, _tsr = _tsp[_tsp.redrawn == "training data"].p_high, _tsp[_tsp.redrawn == "prior"].p_high
need_true("expAudit seeds: data and prior apart, 20 each, all above 0.5, seed 0 the same in all three", len(_tsd) == 20 and len(_tsr) == 20 and _tsd.min() > 0.5 and _tsr.min() > 0.5 and _tsp[_tsp.seed == 0].p_high.nunique() == 1)
need_true("expAudit seeds: both redrawn is the file of the seeds", np.allclose(_tsp[_tsp.redrawn == "both"].sort_values("seed").p_high.values, _ts.sort_values("seed").p_high.values, atol=1e-9))
need_supp("expAudit seeds, the two apart", f"over seeds 0 to 19 the data alone give $P(T{{=}}\\text{{high}})$ from ${_tsd.min():.2f}$ to ${_tsd.max():.2f}$ and the prior alone from ${_tsr.min():.2f}$ to ${_tsr.max():.2f}$.")
need_tex("expAudit seeds", f"(${_ts.p_high.min():.2f}$ to ${_ts.p_high.max():.2f}$ over 20 seeds that redraw the training data and the prior)")
need_supp_later("expAudit seeds, in the supplement", f"over seeds 0 to 19, which redraw the training data and the prior (the calibration split stays), the worked track's $P(T{{=}}\\text{{high}})$ averages ${_ts.p_high.mean():.2f}$, always above $0.5$, $P(T\\ge\\text{{medium}})$ stays within ${_ts.p_medium_up.min():.2f}$ to ${_ts.p_medium_up.max():.2f}$,")
need_true("expAudit seeds: 20, seed 0 the lowest, none below 0.5", len(_ts) == 20 and _ts.p_high.idxmin() == 0 and _ts.p_high.min() > 0.5)
need_tex("expAudit seeds: tornado sentence", f"redrawing the training data and the prior moves the posterior further (${_ts.p_high.min():.2f}$ to ${_ts.p_high.max():.2f}$, Sec.~\\ref{{sec:worked}}), again without crossing $0.5$.")
_m12, _m3 = _tc[_tc.swing.abs() > 1e-12], _tc[_tc.swing.abs() > 1e-3]
need_tex("expAudit cells", f"Only ${len(_m12)}$ of the ${len(_tc)}$ cells move the posterior at all under this evidence (${len(_m3)}$ by more than $0.001$).")
need_supp("expAudit cells, H and C", f"the cells that move the posterior include ${(_m12.node == 'H').sum()}$ of the ${(_tc.node == 'H').sum()}$ of $H$ and ${(_m12.node == 'C').sum()}$ of the ${(_tc.node == 'C').sum()}$ of $C$, whose parents the evidence fixes;")

# the factor of about five
_sc = _rt("expS_scarcity_raw.csv")
_SM = {"acc": 1, "macroF1": 1, "macroAUC": 1, "brier": -1, "ece": -1, "far@pd0.9": -1}


def _scd(a, Na, b, Nb):
    _x = {}
    for _k, _s in _SM.items():
        _A = _sc[(_sc.method == a) & (_sc.N == Na)].sort_values("rep")[_k].values
        _B = _sc[(_sc.method == b) & (_sc.N == Nb)].sort_values("rep")[_k].values
        assert len(_A) == 20 and len(_B) == 20
        _x[_k] = (float(_s * (_A - _B).mean()), float(_st.ttest_1samp(_A - _B, 0.0).pvalue), _A - _B)
    return _x


_d50, _d100, _b2 = _scd("Proposed", 10, "B1-MLE", 50), _scd("Proposed", 10, "B1-MLE", 100), _scd("B2-expert", 10, "B1-MLE", 50)
need_true("expAudit factor: N=10 not significantly worse than MLE at 50 on any metric, better on AUC, ECE, FAR", all(not (_v[0] < 0 and _v[1] < 0.05) for _v in _d50.values()) and all(_d50[_k][0] > 0 and _d50[_k][1] <= 0.05 for _k in ("macroAUC", "ece", "far@pd0.9")))
need_true("expAudit factor: the same tests separate N=10 from MLE at 100 on one metric of six", sorted(_k for _k, _v in _d100.items() if _v[1] < 0.05) == ["macroF1"])
need_true("expAudit factor: B2 not significantly worse than MLE at 50 on accuracy, Brier, ECE, FAR; it trails on macro-F1", all(not (_b2[_k][0] < 0 and _b2[_k][1] < 0.05) for _k in ("acc", "brier", "ece", "far@pd0.9")) and _b2["macroF1"][0] < 0 and _b2["macroF1"][1] < 0.05 and f"{_b2['macroF1'][1]:.2f}" == "0.05")


def _ci95(_x):
    _h = _st.t.ppf(0.975, 19) * _x.std(ddof=1) / np.sqrt(20)
    return f"$[{_x.mean() - _h:+.3f},{_x.mean() + _h:+.3f}]$"


need_tex("expAudit factor: first sentence", f"accuracy differs by ${_d50['acc'][2].mean():+.3f}$ ($95\\%$ CI {_ci95(_d50['acc'][2])}) and Brier by ${_d50['brier'][2].mean():+.3f}$ ({_ci95(_d50['brier'][2])}; Table~\\ref{{tab:stdbase}}).")
need_tex("expAudit factor: descriptive", "The same tests do not separate $N{=}10$ from MLE at $N{=}100$ on five of the six metrics, and the prior alone (B2) is not significantly worse than MLE at $N{=}50$ on accuracy, Brier, ECE, or FAR (it trails on macro-F1, $p{=}0.05$), so this comparison mainly measures the prior")
need_true("expAudit factor: no factor is quoted", not HAVE_PAPER or not re.search(r"factor of about|sample-efficiency factor|two-fold|five-fold", tex))
need_tex("expAudit factor: abstract", "the model at ten scenarios is not significantly worse than maximum likelihood at fifty and beats")
_d5100 = _scd("Proposed", 50, "B1-MLE", 100)
need_true("expAudit N=50 against MLE at 100: ECE p rounds to 0.001", f"{_d5100['ece'][1]:.3f}" == "0.001")
need_tex("expAudit N=50 against MLE at 100", "and ECE ($-0.023$, $p{=}0.001$), but MLE at $N{=}200$")

# the world, by exact inference
_gtw = _ms.ground_truth_bn()
_wt = [_gtw.query(["T"], {"cls": _c}).table for _c in range(3)]
need_tex("expAudit world: threat follows kinematics and context more than class",
         f"$P(T{{\\ge}}\\text{{medium}})$ is ${_wt[0][2:].sum():.2f}$, ${_wt[1][2:].sum():.2f}$, and ${_wt[2][2:].sum():.2f}$ for clutter, birds, and drones ($P(T{{=}}\\text{{high}})$: ${_wt[0][3]:.2f}$, ${_wt[1][3]:.2f}$, ${_wt[2][3]:.2f}$), and $\\mathrm{{emit}}$ is drawn independently of class, so a silent bird is non-cooperative with probability ${_gtw.cpts['N'].table[1, 1, 1]:.2f}$ against")
need_true("expAudit FAR: T >= medium is the positive class", "positive_from=2" in open(os.path.join(HERE, "metrics.py"), encoding="utf-8").read())
need_tex("expAudit FAR: stated", "with $T{\\ge}\\text{medium}$ as the positive class")
need_tex("expAudit what N counts", "The training size $N$ counts scenarios labeled with the threat level and, in the supervised case, with the three latent states (the table of the non-cooperative node")
need_tex("expAudit B0 is of the world's family", "B0 comes from the world's own ordinal-response family, so it is coarse but not structurally wrong;")
need_true("expAudit B0 is built by the world's response function", "_ordinal_cpt(\"H\", {\"v\": 0.7" in open(os.path.join(HERE, "model_spec.py"), encoding="utf-8").read())

# the adoption rule of the double discount
_ad = {}
for _w in ("standard", "structural"):
    _ad[_w] = (_paired(_dr, _w, "V1", "V0", "all", "brier"), _paired(_dr, _w, "V1", "V0", "all", "ece"),
               _sel(_dr, _w, "V1", "near")["ece"].mean() - _sel(_dr, _w, "V0", "near")["ece"].mean(),
               _sel(_dr, _w, "V1", "imminent")["ece"].mean() - _sel(_dr, _w, "V0", "imminent")["ece"].mean())
need_tex("expAudit adoption rule: the standard world",
         f"It fails (Brier ${_ptex(_ad['standard'][0][1], 2)}$, but ECE ${_ad['standard'][1][0]:+.4f}$, ${_ptex(_ad['standard'][1][1], 3)}$, and near- and imminent-range ECE does not fall), and")
need_tex("expAudit adoption rule: the structural world",
         f"the structural world meets none of the three conditions (Brier ${_ad['structural'][0][0]:+.4f}$, ${_ptex(_ad['structural'][0][1], 3)}$; ECE ${_ad['structural'][1][0]:+.4f}$, ${_ptex(_ad['structural'][1][1], 3)}$).")
need_true("expAudit adoption rule: what each world meets", _ad["standard"][0][1] >= 0.05 and _ad["standard"][1][0] > 0 and _ad["standard"][1][1] < 0.05 and _ad["structural"][0][0] > 0 and _ad["structural"][0][1] < 0.05 and _ad["structural"][1][0] > 0 and _ad["structural"][1][1] < 0.05)
_arv = open(os.path.join(HERE, "exp_reliability_variants.py"), encoding="utf-8").read()
need_true("expAudit adoption rule: its text names the standard world and reports the structural one regardless", "In the standard world, adopt V1" in _arv and "Report the structural-world results" in _arv and "regardless of the decision" in _arv)
need_true("expAudit adoption rule: the verdict of the summary", "## A7 decision (rule fixed before the run, standard world, n=300)" in open(os.path.join(RES, "expD_summary.md"), encoding="utf-8").read() and "VERDICT: KEEP V0" in open(os.path.join(RES, "expD_summary.md"), encoding="utf-8").read())
need_true("expAudit adoption rule: near- and imminent-range ECE falls in neither world", all(_ad[_w][2] > 0 and _ad[_w][3] > 0 for _w in _ad))
need_tex("expAudit adoption rule: stated in the manuscript", "Adoption rule: in the standard world the chance-corrected layer replaces the deployed one only if its Brier score and ECE are not significantly worse and its ECE falls at near and imminent range.")
need_tex("expAudit adoption rule: an empirical discount", "So $\\alpha_s$ of Eq.~\\eqref{eq:reliab} stays as an empirical discount: it takes the place of $P(\\text{reliable}\\mid\\boldsymbol{\\varphi}_s)$ in the network, but its fitted value is not an estimate of that probability, and on the true CPTs its ECE gain vanishes while its Brier gain stays")
_fus = tex[tex.find("\\subsubsection{Sensor fusion}"):tex.find("\\subsubsection{How much calibration data}")] if HAVE_PAPER else ""
need_true("expAudit adoption rule: said once in Sec. VII-B (the abstract and the conclusion repeat the result)", not HAVE_PAPER or ("not because the weight has a probabilistic basis" not in tex and _fus.count("empirical discount") == 1 and tex.count("empirical discount") == 3))
need_tex("expAudit adoption rule: the definition it refers to", "with $P(\\text{reliable}\\mid\\boldsymbol{\\varphi}_s)=\\alpha_s$ for context features")
need_true("expAudit diagnostics: no process wording in the body", not HAVE_PAPER or "added after the main run" not in tex)
need_supp("expAudit diagnostics, in the supplement", "These diagnostics on the true CPTs are not part of the adoption rule.")
need_true("expAudit diagnostics after the fact: the summary says so", "added after the main run; not part of the rule" in open(os.path.join(RES, "expD_summary.md"), encoding="utf-8").read())
need_supp("expAudit second conclusion scoped", "Second, in the standard world removing it leaves the Brier score unchanged")

# smaller statements
_lat = _rt("expL_latency.csv")
need_true("expAudit assembly: 2.2 ms is the total for the three range bins", f"{1e3 * _lat.assemble_s.mean():.1f}" == "2.2" and "for dv in range(ms.CARD[\"d\"])}" in open(os.path.join(HERE, "exp_latency.py"), encoding="utf-8").read())
need_tex("expAudit assembly", "Assembling the deployed networks costs $2.2$ ms in all for the three range bins at load time,")
_sol = open(os.path.join(RES, "expS_solver.txt"), encoding="utf-8").read()
need_true("expAudit solver tally", "'solved': 1100" in _sol and "'fallback': 0" in _sol and '("optimal", "optimal_inaccurate")' in open(os.path.join(HERE, "grounding.py"), encoding="utf-8").read())
need_supp_later("expAudit solver sentence, in the supplement", "all $1,100$ constrained solves of the main study ended with status optimal or optimal-inaccurate, with no fallback to the closed form.")
need_tex("expAudit the check script, as it is", "a check script there recomputes the quoted values from the result files and compares them with this text when given the LaTeX source.")
_fuk = lambda k: {c: float(_fu[_fu.config == c][k].mean()) for c in ("radar only", "RF only", "EO/IR only", "acoustic only")}
_fua = {k: float(_fu[_fu.config == "all (calibrated)"][k].mean()) for k in ("acc", "macroF1", "macroAUC", "brier", "ece")}
need_true("expAudit fusion against the best single sensor: ahead on four metrics, not on ECE",
          _fua["acc"] > max(_fuk("acc").values()) and _fua["macroF1"] > max(_fuk("macroF1").values()) and _fua["macroAUC"] > max(_fuk("macroAUC").values())
          and _fua["brier"] < min(_fuk("brier").values()) and _fua["ece"] > min(_fuk("ece").values()))
need_tex("expAudit fusion against the best single sensor", f"outperforms the best single sensor on accuracy, F1, AUC, and Brier, though not on ECE (${_fua['ece']:.3f}$ against ${min(_fuk('ece').values()):.3f}$--${max(_fuk('ece').values()):.3f}$;")
need_true("expAudit B5 and B1 settings", "C=1.0" in open(os.path.join(HERE, "exp_baselines.py"), encoding="utf-8").read() and "def mle_cpt(node, data, eps=1e-6)" in open(os.path.join(HERE, "grounding.py"), encoding="utf-8").read())
need_tex("expAudit B1 pseudo-count", "(a pseudo-count of $10^{-6}$ leaves empty columns uniform; overfits under scarcity)")
need_tex("expAudit B5 setting", "an $L_2$ multinomial logistic regression ($C{=}1$;")
need_tex("expAudit the p-values", "All $p$-values in this paper are unadjusted for multiplicity unless stated, so those near $0.05$ are descriptive;")
need_true("expAudit Holm values exist", "p_holm" in _rt("expB6_paired.csv").columns)
need_tex("expAudit Proposition 1 hypothesis", "and $\\theta^*$ has positive entries, then,")
need_tex("expAudit a projection", "and the limit points of the fit are Kullback--Leibler (KL) projections of the true threat table onto the family.")
need_supp_later("expAudit MLE comparison at an ESS of 0.2, in the supplement", "uniform $-0.0029$ Brier, $p\\le0.003$).")
# the accuracy of N=10 against MLE at N=50 over the ESS, and over the noise level: no equivalence is claimed
_ea = _rt("expA_ess_sensitivity.csv")
_ea50 = _ea[(_ea.method == "B1-MLE") & (_ea.N == 50)].sort_values("rep")
_eaq = {}
for _e in (1.0, 2.0, 5.0):
    _ea10 = _ea[(_ea.method == "Proposed") & (_ea.N == 10) & (_ea.ess == _e)].sort_values("rep")
    assert len(_ea10) == 20 and len(_ea50) == 20
    _eaq[_e] = {_k: (float((_ea10[_k].values - _ea50[_k].values).mean()), float(_st.ttest_1samp(_ea10[_k].values - _ea50[_k].values, 0.0).pvalue)) for _k in ("acc", "macroAUC", "ece")}
_eam, _eae = [_eaq[_e]["macroAUC"][0] for _e in _eaq], [_eaq[_e]["ece"][0] for _e in _eaq]
need_supp("expAudit the ESS sentence", f"keeps the macro-AUC advantage of Proposed at $N{{=}}10$ over MLE at $N{{=}}50$ significant at every value (${min(_eam):+.3f}$ to ${max(_eam):+.3f}$, all $p<10^{{-3}}$) with no significant accuracy difference, while the ECE gap varies (${max(_eae):+.3f}$ to ${min(_eae):+.3f}$, significant only at $\\alpha_0{{=}}2$).")
need_true("expAudit the ESS sentence: accuracy not significant at any value, macro-AUC at all, ECE only at 2", all(_eaq[_e]["acc"][1] > 0.05 and _eaq[_e]["macroAUC"][1] < 1e-3 for _e in _eaq) and [_eaq[_e]["ece"][1] < 0.05 for _e in (1.0, 2.0, 5.0)] == [False, True, False])
need_supp("expAudit noise: no significant accuracy difference, supplement", "with no significant accuracy difference in the two sharper worlds, a lead in the blurriest, and ECE unresolved")
# the contrasts with each CNN counted once, from the raw files
_NCNN = {"P1": 5, "P1c": 5, "P2": 5, "P3": 4}
need_true("expAudit per CNN: replication r uses CNN r mod the number of CNNs", "sources[r % len(sources)]" in open(os.path.join(HERE, "exp_rf_independent.py"), encoding="utf-8").read())
_pcn = _rt("expRC_per_cnn.csv")
_pcr = {}
for _p, _k in _NCNN.items():
    for _an, _bvn in (("calib-hard", "abstract"), ("calib-hard", "no-RF"), ("calib-soft", "calib-hard")):
        _a = _rcraw(_p, _an)
        _b = _no if _bvn == "no-RF" else _rcraw(_p, _bvn)
        _rec = {}
        for _m in _RCM:
            _g = pd.Series(_a[_m].values - _b[_m].values).groupby(np.arange(20) % _k).mean().values
            assert len(_g) == _k
            _rec[_m] = (float(_g.mean()), float(_st.ttest_1samp(_g, 0.0).pvalue), _g)
        _pcr[(_p, _an, _bvn)] = _rec
_pcbad = 0
for _, _q in _pcn.iterrows():
    for _m in _RCM:
        _r = _pcr[(_q.proto, _q.A, _q.B)][_m]
        _pcbad += int(abs(_r[0] - _q[f"d_{_m}"]) > 1e-12 or abs(_r[1] - _q[f"p_{_m}"]) > 1e-9)
need_true("expAudit per CNN: the file is what the raw files give, twelve rows", len(_pcn) == 12 and _pcbad == 0)
_pc1 = [_pcr[(_p, "calib-hard", "abstract")][_m][1] for _p in _NCNN for _m in ("macroAUC", "brier", "ece")]
_pc2 = [_pcr[(_p, "calib-hard", "no-RF")][_m][1] for _p in _NCNN for _m in ("macroAUC", "brier")]
_pc3 = [_pcr[(_p, "calib-soft", "calib-hard")][_m][1] for _p in _NCNN for _m in ("macroAUC", "brier")]
need_true(f"expAudit per CNN: (ii) over (i) p <= 0.003 (max {max(_pc1):.4f})", max(_pc1) <= 0.003 and f"{max(_pc1):.4f}" == "0.0024")
need_true(f"expAudit per CNN: over the control p <= 0.014 (max {max(_pc2):.4f})", max(_pc2) <= 0.0145 and f"{max(_pc2):.3f}" == "0.014")
need_true("expAudit per CNN: the soft variant runs from 0.016 to 0.094", f"{min(_pc3):.3f}" == "0.016" and f"{max(_pc3):.3f}" == "0.094" and max(_pc3) > 0.05)
need_true("expAudit per CNN: the sign of the gains holds for every CNN", all((_pcr[(_p, "calib-hard", "no-RF")]["macroAUC"][2] > 0).all() and (_pcr[(_p, "calib-hard", "no-RF")]["brier"][2] < 0).all() and (_pcr[(_p, "calib-hard", "abstract")]["macroAUC"][2] > 0).all() and (_pcr[(_p, "calib-hard", "abstract")]["ece"][2] < 0).all() for _p in _NCNN))
need_tex("expAudit per CNN: the sentence", "counting each of the four or five CNNs once, $p\\le0.003$ for (ii) over (i) and $p\\le0.014$ over the control.")
need_supp_later("expAudit per CNN, the replication-level p in the supplement", "the replications cycle through each split's four or five CNNs, so Sec.~VII-H counts each CNN once.")
# the pipeline against the oracle on the same resamples, from the raw files
_bpr = _rt("expBS_paired.csv")
_bidx = np.random.default_rng(20261106).integers(0, 20, size=(10000, 20))


def _bdev(df):
    _t = {_k: df.pivot(index="rep", columns="bin_lo", values=_k).sort_index(axis=1).values for _k in ("n", "sum_pred", "hits")}
    _bpo, _re = [], []
    for _b in range(5):
        _n, _s, _h = (_t[_k][:, _b] for _k in ("n", "sum_pred", "hits"))
        _bpo.append((_s.sum() - _h.sum()) / _n.sum())
        _ns = _n[_bidx].sum(axis=1).astype(float)
        _ns[_ns == 0] = np.nan
        _re.append((_s[_bidx].sum(axis=1) - _h[_bidx].sum(axis=1)) / _ns)
    return np.array(_bpo), np.array(_re)


_bsyn, _bmix = _rt("expBS_synthetic_bins.csv"), _rt("expI_highthreat_bins.csv")
_brows = []
for _w, _pr, _qa, _qo in [("synthetic", "-", _bsyn[_bsyn.variant == "pipeline"], _bsyn[_bsyn.variant == "CPT-oracle"])] + [("mixed", _p, _bmix[(_bmix.proto == _p) & (_bmix.variant == "calib-hard")], _bmix[(_bmix.proto == _p) & (_bmix.variant == "CPT-oracle")]) for _p in _rcp]:
    _bpa, _ra = _bdev(_qa)
    _bpo, _ro = _bdev(_qo)
    for _b in range(5):
        _ok = ~(np.isnan(_ra[_b]) | np.isnan(_ro[_b]))
        _d, _s = _ra[_b][_ok] - _ro[_b][_ok], np.abs(_ra[_b][_ok]) - np.abs(_ro[_b][_ok])
        _brows.append((_w, _pr, _b, _bpa[_b], _bpo[_b], *np.percentile(_d, [2.5, 97.5]), *np.percentile(_s, [2.5, 97.5])))
        # the interval of each alone, on the same resamples, stays what the intervals file says
need_true("expAudit paired: 30 rows, as the raw files give them", len(_bpr) == 30 and len(_brows) == 30 and all(
    abs(_x[3] - _y.dev_pipeline) < 1e-12 and abs(_x[4] - _y.dev_oracle) < 1e-12 and abs(_x[5] - _y.diff_lo) < 1e-12 and abs(_x[6] - _y.diff_hi) < 1e-12 and abs(_x[7] - _y.size_lo) < 1e-12 and abs(_x[8] - _y.size_hi) < 1e-12
    and _y.diff_excludes_0 == bool(_x[5] > 0 or _x[6] < 0) and _y.size_excludes_0 == bool(_x[7] > 0 or _x[8] < 0) for _x, (_, _y) in zip(_brows, _bpr.iterrows())))
_bpm = _bpr[_bpr.world == "mixed"]
_bpc = [int(_bpm[_bpm.bin_lo == _l].diff_excludes_0.sum()) for _l in (0.0, 0.1, 0.25, 0.5, 0.75)]
need_true(f"expAudit paired: signed differences outside by bin {_bpc}, sizes in none", _bpc == [0, 0, 4, 5, 0] and int(_bpm.size_excludes_0.sum()) == 0)
_bpd = _bpm[_bpm.bin_lo == 0.5]
need_true("expAudit paired: in (.50,.75] the pipeline's deviation is positive, the oracle's is not, and neither alone excludes zero", (_bpd.dev_pipeline > 0).all() and (_bpd.dev_oracle < 0).all() and not _bs[(_bs.world == "mixed") & (_bs.bin_lo == 0.5)].boot_excludes_0.any())
need_true("expAudit paired: the detail left the body for Sec. S4", not HAVE_PAPER or "the signed deviation differs in $(.50,.75]$ under all five splits" not in tex)
need_supp("expAudit paired: the sentence", f"Paired with the oracle on the same replications, the signed deviation differs in $(.50,.75]$ under all five splits (${_bpd['diff'].min():+.3f}$ to ${_bpd['diff'].max():+.3f}$: the pipeline is over-confident there and the oracle is not, though neither deviation alone excludes zero) and in $(.25,.50]$ under four, but in the top bin under none; its size differs in none of the $25$ bins.")
_bps = _bpr[_bpr.world == "synthetic"].reset_index(drop=True)
need_true("expAudit paired: in Table 9's world both decision-relevant differences exclude zero", _bps.diff_excludes_0[3] and _bps.diff_excludes_0[4])
need_tex("expAudit paired: Table 9", "paired on the same replications, both signed differences exclude zero, the difference in size only in the top bin).")
need_tex("expAudit added after the main results: the one pointer in the body", "Supplementary Material, Table~S5 lists the analyses added after the main results.")
need_supp_later("expAudit added after the main results: the table", "\\caption{Analyses added after the main results.}\\label{tab:s5}")
for _sn in ("No-RF control &", "Hand-set emission-aware table (diagonal $0.94$) &", "Its diagonal sweep, and the $0.75$ diagonal chosen after the fact as best on Brier &",
            "One estimated number per sensor &", "Identification check and latent-state comparison of MAP-EM &", "Replication-resampled and paired intervals &",
            "Per-CNN $p$-values &", "Worked track's seed range, split, and RF contrast &", "Comparison of B3 with the rule, and the rule read as $P(T{\\ge}\\text{medium})$ &",
            "B6 at $N{=}10,000$ and in the world S1 strong &", "Controls on B6's silence rule and on the softmax &", "Additivity test's false alarms at $N{\\le}20$ &",
            "Bound on the RF channel (error-free reports) &", "Spread and scaling of the reliability weights &", "Reliability diagnostics on the true CPTs &",
            "Root marginals counted on the calibration split &", "Adverse-subset confidence &", "Reversed-prior grid at $\\alpha_0{=}2$ &",
            "Factorial of prior mean and structure, weak uniform prior ($\\alpha_0{=}0.2$ chosen after the fact) &", "ESS of $0.2$ in S1 strong &",
            "Sweeps of B0 (and its $\\tau$ halved, chosen after the fact) and of B5 &", "Threshold of $0.25$ on real RF outputs &", "World with emission acting on drones only &"):
    need_supp_later(f"expAudit Table S5 row: {_sn[:40]}", _sn)
need_supp_later("expAudit Table S5: the rules", "were written into the scripts before the runs they judge; they were not registered externally, and the repository's history does not date them.")
need_tex("expAudit the pointer of N", "\\subsection{Simulation-to-CPT via MAP-EM (C2-iii)}\\label{sec:mapem}")
# the summaries of the new studies, rebuilt from their CSV files and compared whole
def _whole(name, lines):
    need_true(f"expAudit summary {name} is what its files give", open(os.path.join(RES, name), encoding="utf-8").read() == "\n".join(lines) + "\n")


_ev = next(_ast.literal_eval(_n.value) for _n in _ast.parse(open(os.path.join(HERE, "exp_sensitivity.py"), encoding="utf-8").read()).body if isinstance(_n, _ast.Assign) and getattr(_n.targets[0], "id", "") == "EVIDENCE")
_tqt = _gtw.query(["T"], _ev).table
_tq = (float(_gtw.query(["cls"], _ev).table[2]), float(_tqt[2:].sum()), float(_tqt[3]))
_tsn = [_x for _x in _ms.INTERNAL]
import dataio as _dio
_xnone = float(_dio.estimate_confusion(_dio.generate_dataset(_gtw, 300, seed=777))["Rf"].table[2, 1, _ev["Rf"]])
_rfq = _rt("expTS_rf_report.csv")
_rfc = _rfq[_rfq.model == "calibrated"]
_rfr = _rfc[_rfc.rf_report == "with"].sort_values("seed").p_high.values - _rfc[_rfc.rf_report == "without"].sort_values("seed").p_high.values
_tsl = (["# Exp-TS: worked track under other training seeds; cells that move its posterior", "",
    f"- true world: drone {_tq[0]:.3f}, P(T>=medium) {_tq[1]:.3f}, P(T=high) {_tq[2]:.3f}",
    f"- seed 0: drone {_ts.p_drone[0]:.3f}, P(T>=medium) {_ts.p_medium_up[0]:.3f}, P(T=high) {_ts.p_high[0]:.3f}",
    f"- seeds 0-19, P(T=high): mean {_ts.p_high.mean():.3f}, range {_ts.p_high.min():.3f} to {_ts.p_high.max():.3f}, below 0.5: {(_ts.p_high < 0.5).sum()}, above the true world's: {(_ts.p_high > _tq[2]).sum()}",
    f"- seeds 0-19, P(T>=medium): range {_ts.p_medium_up.min():.3f} to {_ts.p_medium_up.max():.3f}",
    f"- seeds 0-19, drone posterior: range {_ts.p_drone.min():.3f} to {_ts.p_drone.max():.3f}",
    f"- cells: {len(_tc)}; |swing| > 1e-12: {len(_m12)} ({', '.join(f'{_k} {(_m12.node == _k).sum()}' for _k in _tsn)}); |swing| > 1e-3: {len(_m3)} ({', '.join(f'{_k} {(_m3.node == _k).sum()}' for _k in _tsn)})",
    f"- cells per table: {', '.join(f'{_k} {(_tc.node == _k).sum()}' for _k in _tsn)}",
    "", "## the RF report on this evidence (none), with and without it", "| model | seed | RF report | P(drone) | P(silent) | P(T>=medium) | P(T=high) |", "|---|---|---|---|---|---|---|"]
    + [f"| {_q.model} | {_q.seed} | {_q.rf_report} | {_q.p_drone:.3f} | {_q.p_silent:.3f} | {_q.p_medium_up:.3f} | {_q.p_high:.3f} |" for _, _q in _rfq[(_rfq.model != "calibrated") | (_rfq.seed == 0)].iterrows()]
    + ["", f"- seeds 0-19, rise of P(T=high) from the RF report: {_rfr.min():+.3f} to {_rfr.max():+.3f}; below zero: {(_rfr < 0).sum()}",
       f"- P(Rf = none | drone, silent) in the estimated matrix (calibration seed 777): {_xnone:.4f}"])
_whole("expTS_summary.md", _tsl)
_eml = ["# Exp-EM: identifiability of the unlabeled case and quality of the latent posteriors", "", "## Jacobian rank at the true tables", "| what | parameters | rank | singular value at the rank | next singular value |", "|---|---|---|---|---|"]
for _, _q in _rt("expEM_rank.csv").iterrows():
    _eml.append(f"| {_q.what} | {_q.params} | {_q['rank']} | {_q.s_before_gap:.3g} | {_q.s_after_gap:.3g} |")
_eml.append("\nLast row: the next column gives the largest entry of J times the recoding tangents.")
_eml.append("\n## reach of a recoding near the true tables (SLSQP from R = I, lower bounds; permutations excluded)")
_eml.append(f"searches that ended infeasible and count as no move: {int((~_erc.feasible).sum())} of {len(_erc)}")
_eml.append("| table | largest move of any entry |\n|---|---|")
for _nm in ("H", "N", "C", "T"):
    _eml.append(f"| {_nm} | {_erc[_erc.node == _nm].move.max():.3f} |")
_eml.append(f"\nentries searched, both directions: {len(_erc)}")
_eml.append(f"\nthe recoding with the largest move: entries move by up to H {_ere.largest_move_H:.3f}, N {_ere.largest_move_N:.3f}, C {_ere.largest_move_C:.3f}, T {_ere.largest_move_T:.3f}; Q changes by at most {_ere.largest_change_Q:.1e} and P(T, reports | inputs) by at most {_ere.largest_change_observed:.1e}")
_eml.append("\n## latent posteriors on test scenarios (means over 20 replications)")
_eml.append("| method | N | H acc | N acc | C acc | H log loss | N log loss | C log loss |\n|---|---|---|---|---|---|---|---|")
for _m, _n in [("prior only", 0), ("true CPTs", 0), ("Proposed", 50), ("MAP-EM", 50), ("Proposed", 200), ("MAP-EM", 200)]:
    _eml.append(f"| {_m} | {_n} | " + " | ".join(f"{_lav(_m, _n, _x).acc.mean():.3f}" for _x in "HNC") + " | " + " | ".join(f"{_lav(_m, _n, _x).logloss.mean():.3f}" for _x in "HNC") + " |")
_eml.append("\n## MAP-EM minus Proposed (paired t over 20 replications)")
_eml.append("| N | node | d acc (p) | d log loss (p) |\n|---|---|---|---|")
for _n in (50, 200):
    for _x in "HNC":
        _parts = []
        for _k in ("acc", "logloss"):
            _d = _lav("MAP-EM", _n, _x)[_k].values - _lav("Proposed", _n, _x)[_k].values
            _parts.append(f"{_d.mean():+.3f} ({_st.ttest_1samp(_d, 0.0).pvalue:.2g})")
        _eml.append(f"| {_n} | {_x} | " + " | ".join(_parts) + " |")
_whole("expEM_summary.md", _eml)
_rcl = ["# Exp-RC: the mixed-world RF study against a control with no RF channel", "", f"- synthetic-RF equals expS_fusion_raw.csv on accuracy, macro-AUC and ECE over 20 replications: True", "\n## variants (means over 20 replications)", "| variant | acc | macro-AUC | Brier | ECE |\n|---|---|---|---|---|"]
for _nm, _q in (("no-RF", _no), ("synthetic-RF", _sy)):
    _rcl.append(f"| {_nm} | {_q.acc.mean():.4f} | {_q.macroAUC.mean():.4f} | {_q.brier.mean():.4f} | {_q.ece.mean():.4f} |")
for _p in _rcp:
    for _v in ("abstract", "calib-hard"):
        _q = _rcraw(_p, _v)
        _rcl.append(f"| {_p} {_v} | {_q.acc.mean():.4f} | {_q.macroAUC.mean():.4f} | {_q.brier.mean():.4f} | {_q.ece.mean():.4f} |")
_rcl += ["\n## contrasts (A - B, paired t over 20 replications)", "| split | A | B | d acc (p) | d macro-AUC (p) | d Brier (p) | d ECE (p) |\n|---|---|---|---|---|---|---|"]
for _, _q in _rk.iterrows():
    _a = _sy if _q.A == "synthetic-RF" else _rcraw(_q.proto, _q.A)
    _b = _no if _q.B == "no-RF" else _sy
    _cells = []
    for _m in _RCM:
        _d = _a[_m].values - _b[_m].values
        _cells.append(f"{_d.mean():+.4f} ({_st.ttest_1samp(_d, 0.0).pvalue:.2g})")
    _rcl.append(f"| {_q.proto} | {_q.A} | {_q.B} | " + " | ".join(_cells) + " |")
_rcl += ["\n## the same contrasts with each CNN counted once (mean over its replications, paired t over the CNNs)", "| split | CNNs | A | B | d acc (p) | d macro-AUC (p) | d Brier (p) | d ECE (p) |\n|---|---|---|---|---|---|---|---|"]
for _p, _k in _NCNN.items():
    for _an, _bvn in (("calib-hard", "abstract"), ("calib-hard", "no-RF"), ("calib-soft", "calib-hard")):
        _rcl.append(f"| {_p} | {_k} | {_an} | {_bvn} | " + " | ".join(f"{_pcr[(_p, _an, _bvn)][_m][0]:+.4f} ({_pcr[(_p, _an, _bvn)][_m][1]:.2g})" for _m in _RCM) + " |")
_whole("expRC_summary.md", _rcl)
_bsl = ["# Exp-BS: replication-level intervals for the binned reliability of P(T = high)", "", "- 10000 resamples of the 20 replications, default_rng(20261106); deviation = predicted - observed", "- pooled synthetic bins equal expR_reliability_classes.csv (high class): 5 bins", "",
        "## synthetic world", "| variant | bin | n | deviation | 95% replication interval | excludes 0 | outside binomial |", "|---|---|---|---|---|---|---|"]
for _, _q in _bs[_bs.world == "synthetic"].iterrows():
    _bsl.append(f"| {_q.variant} | ({_q.bin_lo:.2f},{_q.bin_hi:.2f}] | {_q.n} | {_q.dev:+.4f} | [{_q.boot_lo:+.4f}, {_q.boot_hi:+.4f}] | {_q.boot_excludes_0} | {_q.binomial_outside} |")
_bsl += ["", "## mixed world, bins excluding 0 (of 25 over five splits)"]
for _v in ("calib-hard", "CPT-oracle"):
    _q = _bs[(_bs.world == "mixed") & (_bs.variant == _v)]
    _bsl.append(f"- {_v}: replication intervals {int(_q.boot_excludes_0.sum())}/{len(_q)}; binomial intervals {int(_q.binomial_outside.sum())}/{len(_q)}")
_bsl += ["", "| split | variant | bin | deviation | 95% replication interval | excludes 0 | outside binomial |", "|---|---|---|---|---|---|---|"]
for _, _q in _bs[_bs.world == "mixed"].iterrows():
    _bsl.append(f"| {_q.proto} | {_q.variant} | ({_q.bin_lo:.2f},{_q.bin_hi:.2f}] | {_q.dev:+.4f} | [{_q.boot_lo:+.4f}, {_q.boot_hi:+.4f}] | {_q.boot_excludes_0} | {_q.binomial_outside} |")
_bsl += ["", "## pipeline minus CPT oracle on the same resampled replications", "signed = deviation of the pipeline - deviation of the oracle; size = |deviation of the pipeline| - |deviation of the oracle|", ""]
_bsl.append(f"- mixed world: signed difference excludes 0 in {int(_bpm.diff_excludes_0.sum())}/{len(_bpm)} bins, difference in size in {int(_bpm.size_excludes_0.sum())}/{len(_bpm)}")
for _l in (0.0, 0.1, 0.25, 0.5, 0.75):
    _q = _bpm[_bpm.bin_lo == _l]
    _bsl.append(f"- mixed world, bin from {_l:.2f}: signed {int(_q.diff_excludes_0.sum())}/{len(_q)}, size {int(_q.size_excludes_0.sum())}/{len(_q)}")
_bsl += ["", "| split | bin | pipeline | oracle | signed difference | 95% interval | excludes 0 | difference in size | 95% interval | excludes 0 |", "|---|---|---|---|---|---|---|---|---|---|"]
for _, _q in _bpr.iterrows():
    _bsl.append(f"| {_q.proto} | ({_q.bin_lo:.2f},{_q.bin_hi:.2f}] | {_q.dev_pipeline:+.4f} | {_q.dev_oracle:+.4f} | {_q['diff']:+.4f} | [{_q.diff_lo:+.4f}, {_q.diff_hi:+.4f}] | {_q.diff_excludes_0} | {_q.size_diff:+.4f} | [{_q.size_lo:+.4f}, {_q.size_hi:+.4f}] | {_q.size_excludes_0} |")
_whole("expBS_summary.md", _bsl)
# changed sentences with no number of their own
for _lab, _sn in (
        ("recoding sentence", "with $H$, $N$, and $C$ unlabeled, any recoding of a latent state by an invertible matrix with unit row sums, applied to its table and inversely to the threat table, leaves the distribution of the observed variables unchanged."),
        ("world: the lead-in", "Threat here follows kinematics and context more than class:"),
        ("world: what FAR counts", "The false-alarm rate (FAR) therefore counts threat-level errors on any target, not bird-versus-drone confusions."),
        ("what N counts: the budget", "these labels encode a judgment of intent in context that flight recordings do not carry, so $N$ is the budget we assume scarce."),
        ("what N counts: many labels", "with plentiful labels the gains are small, as at $N{=}400$."),
        ("notation: N in two senses", "(below, $N$ also counts training scenarios; context tells them apart)"),
        ("realistic expert: the reading", "An existing hand-set table should therefore be used as a prior and updated, not discarded."),
        ("labels for the rationale", "Labels are needed if the rationale, not only the threat level, is to be read."),
        ("Table 1 caption", "\\caption{Variables and states of the evaluation world (Sec.~\\ref{sec:world}).}"),
        ("Table 3 roots", "& --- (roots) & 10 (Sec.~\\ref{sec:world}) \\\\"),
        ("Fig. 2 caption", "Solid box $=$ observed, dashed box $=$ latent, ellipse $=$ inferred rationale variable or threat."),
        ("B6 inputs", "Inputs this world lacks stay at their benign state, and unlisted state scores are set at the ends of the range."),
        ("abstract: the condition", "Given labeled latent states and a prior near the truth,"),
        ("abstract: the rule", "On high threat it outranks the base framework's rule (AUC 0.876--0.890 against 0.752)."),
        ("M-step ends the display", "\\frac{\\bar N_{ijk}+\\alpha_{ijk}}{\\bar N_{ij}+\\alpha_{ij}}."),
        ):
    need_tex(f"expAudit {_lab}", _sn)
need_supp("expAudit what the repository holds", "the repository gives the per-range, oracle, and calibration-size breakdowns.")
need_supp("expAudit fit time", f"(${_lat.mapem_s.mean():.2f}$ s against ${_lat.ground_s.mean():.2f}$ s at $N{{=}}200$, a factor of ${_lat.mapem_s.mean() / _lat.ground_s.mean():.1f}$)")
# the old wording stays out, of the manuscript, the supplement and the README
_rm = open(os.path.join(HERE, "README.md"), encoding="utf-8").read()
for _x in ("five-fold", "five times fewer", "matches MLE at", "propagates to the decision", "propagates to the threat posterior", "equally under",
           "C2-iii is thus executed", "convenience, not a requirement", "about 0.01 expected", "nondegenerate for", "once per range bin", "status is logged",
           "returned optimal", "Every comparison in this paper", "The two low bins", "literature-informed", "complemented by the classwise",
           "which is what every comparison here reports", "cMAP$\\equiv$MAP", "BNPTL", "verifies that the values quoted here", "(see note)",
           "conservative, but directionally right", "$(v,\\dot r,d)$, airspace", "as already noted", "base rate $0.082$",
           "cannot be told apart", "fails in both worlds", "split seeds below cover training", "at parity", "come, respectively, from", "within a recording only", "thousands of trusted labels"):
    need_true(f"expAudit retired from main.tex and the supplement: {_x}", not HAVE_PAPER or (_x not in tex and _x not in _supp))
# the one place it may stand: the note on what the old run log says
_rmn = 'says "five-fold check" for what `results/expN_summary.md` calls Proposed(N=10) - MLE(N=50).'
need_true("expAudit README says what the old noise log means", _rmn in _rm)
for _x in ("five-fold", "matches unconstrained MLE", "status-logged", "before an amendment", "all optimal in the reported runs", "the gain holds under time-block", "cannot be told apart", "in both worlds (standard", "at parity"):
    need_true(f"expAudit retired from the README: {_x}", _x not in _rm.replace(_rmn, ""))
_nlv, _nsm = open(os.path.join(HERE, "exp_noise_level.py"), encoding="utf-8").read(), open(os.path.join(RES, "expN_summary.md"), encoding="utf-8").read()
need_true("expAudit the noise script says Proposed(N=10) - MLE(N=50)", "five-fold" not in _nlv and _nlv.count('"- Proposed(N=10) - MLE(N=50): "') == 1)
need_true("expAudit the noise summary says Proposed(N=10) - MLE(N=50)", "five-fold" not in _nsm and _nsm.count("- Proposed(N=10) - MLE(N=50): ") == 3)
need_true("expAudit model_spec.py comments say hand-set", "literature-informed" not in open(os.path.join(HERE, "model_spec.py"), encoding="utf-8").read())
_e2e = open(os.path.join(HERE, "exp_rf_e2e.py"), encoding="utf-8").read()
for _x in ("recording-level", "by RECORDING", "capture session", "session-level"):
    need_true(f"expAudit exp_rf_e2e.py says segment: {_x}", _x not in _e2e)
need_true("expAudit expG summary says segment", "segment-level split" in open(os.path.join(RES, "expG_summary.md"), encoding="utf-8").read() and "recording-level" not in open(os.path.join(RES, "expG_summary.md"), encoding="utf-8").read())
need_true("expAudit RESULTS.md is marked as the first results", open(os.path.join(RES, "RESULTS.md"), encoding="utf-8").read().split("\n", 1)[0] == "> First results of July 2026 (ESS = 8.0, oracle accuracy 0.604), from an earlier world and protocol, kept for the record. These are not the values of the current paper: see `README.md`, *Tables and figures*.")
_eo = open(os.path.join(HERE, "exp_eo_independent.py"), encoding="utf-8").read()
need_true("expAudit the infrared rule cites the hash of its first form", "6bf85fc1e3c2ca5f384e36f500d0b9772b9d9b1ca871feb8f47af6eb714c3cc4" in _eo and "is kept as the" not in _eo)
need_true("expAudit README gives the order of the gate runs", "which started at 10:47 on 2026-11-06 under the accuracy limit alone and ended at 10:59" in _rm and "Under the accuracy limit alone the gate would have passed" in _rm)
need_true("expAudit the first gate run passed under the accuracy limit", "pass: True" in open(os.path.join(RES, "expO_g2_run1", "g2_run1.log"), encoding="utf-8", errors="replace").read())
if DRAFT or not HAVE_PAPER:
    print(f"expAudit checks run: {_n_expd - _n0a}")
assert _n_expd - _n0a > 0

# ---- reasons, protocol, scope; new values from their files -----
_n0xf = _n_expd
_xfrq = lambda m, s, r: _rt("expTS_rf_report.csv").query("model == @m and seed == @s and rf_report == @r").iloc[0]
_xfc1, _xfc0 = _xfrq("calibrated", 0, "with"), _xfrq("calibrated", 0, "without")
_xft1, _xft0 = _xfrq("true world", -1, "with"), _xfrq("true world", -1, "without")
_xfa1, _xfa0 = _xfrq("abstract", 0, "with"), _xfrq("abstract", 0, "without")
need_true("expF worked track: 20 seeds, and seed 0 is the model of Sec. VII-A", len(_rfr) == 20 and abs(_xfc1.p_high - _ts.p_high[0]) < 1e-12 and abs(_xfc1.p_drone - _ts.p_drone[0]) < 1e-12)
need_true("expF worked track: the rows with the report are the seeds file", all(abs(_rfc[_rfc.rf_report == "with"].sort_values("seed")[_k].values - _ts.sort_values("seed")[_k].values).max() < 1e-12 for _k in ("p_drone", "p_medium_up", "p_high")))
need_true("expF worked track: the true-world row is the true world", abs(_xft1.p_high - _tq[2]) < 1e-12 and abs(_xft0.p_high - float(_gtw.query(["T"], {_k: _v for _k, _v in _ev.items() if _k != "Rf"}).table[3])) < 1e-12)
need_true("expF worked track: the report barely lowers the drone posterior and raises the threat in every seed", _xfc1.p_drone < _xfc0.p_drone and _xfc0.p_drone - _xfc1.p_drone < 0.05 and (_rfr > 0).all())
need_true("expF worked track: the abstract model ignores emission", abs(_xfa1.p_silent - _xfa0.p_silent) < 1e-12 and abs(_xfa1.p_high - _xfa0.p_high) < 0.002)
need_tex("expF worked track: the RF report", f"the RF ``none'' barely lowers the drone posterior (${_xfc0.p_drone:.2f}$ to ${_xfc1.p_drone:.2f}$), since a radio-silent drone also yields ``none'' in the estimated matrix (${_xnone:.2f}$), and it raises $P(\\text{{silent}})$ from the prior ${_xfc0.p_silent:.2f}$ to ${_xfc1.p_silent:.2f}$, so it raises the threat: $P(T{{=}}\\text{{high}})$ goes from ${_xfc0.p_high:.2f}$ to ${_xfc1.p_high:.2f}$ (true world: ${_xft0.p_high:.2f}$ to ${_xft1.p_high:.2f}$).")
need_supp_later("expF worked track: the RF report over seeds, in the supplement", f"and the RF ``none'' raises $P(T{{=}}\\text{{high}})$ by ${_rfr.min():+.2f}$ to ${_rfr.max():+.2f}$ (seed 0 is the lowest of the twenty).")
need_tex("expF worked track: the abstract model", f"($P(T{{=}}\\text{{high}}) = {_xfa1.p_high:.3f}$, $P(T \\ge \\text{{medium}}) = {_xfa1.p_medium_up:.2f}$): it ignores emission, so $P(\\text{{silent}})$ stays at ${_xfa1.p_silent:.2f}$ and $P(T{{=}}\\text{{high}})$ barely moves with or without the RF report (${_xfa1.p_high:.3f}$, ${_xfa0.p_high:.3f}$);")

# the controls: each file against its source, the summary whole, the sentences from the files
_xrs, _xrc, _xrm, _xrt, _xrl, _xrw = (_rt(f"expRS_{_x}.csv") for _x in ("b6_silent", "b6_control", "softmax", "transfer", "flags", "weights"))
_xsb = _rt("expB6_subsets.csv")
for _, _q in _xrs.iterrows():
    _qq = _xsb[(_xsb.subset == _q.subset) & (_xsb.method == _q.method) & (_xsb.N == _q.N)]
    need_true(f"expF B6 silence file: {_q.subset} {_q.method} {_q.N}", _qq.rep.nunique() == 20 and abs(_qq.auc.mean() - _q.auc) < 1e-12 and abs(_qq.brier_bin.mean() - _q.brier) < 1e-12 and abs(_qq.mean_p.mean() - _q.mean_p) < 1e-12)
_xg = lambda s, m, n, k: float(_xrs.query("subset == @s and method == @m and N == @n")[k].iloc[0])
_xsd, _xsn = "silent-drone", "silent-report-nondrone"
need_tex("expF B6 on the silent subsets", f"On radio-silent drones B6-pub's mean output is ${_xg(_xsd, 'B6-pub', 0, 'mean_p'):.3f}$ against ${_xg(_xsd, 'CPT-oracle', 0, 'mean_p'):.3f}$ for the CPT oracle and ${_xg(_xsd, 'Proposed', 200, 'mean_p'):.3f}$ for Proposed at $N{{=}}200$ (Brier ${_xg(_xsd, 'B6-pub', 0, 'brier'):.3f}$, ${_xg(_xsd, 'CPT-oracle', 0, 'brier'):.3f}$, ${_xg(_xsd, 'Proposed', 200, 'brier'):.3f}$); its hand rule on silence explains only part of that")
need_supp("expF B6 on silent reports from non-drones", f"on silent reports from non-drones, the mean output of B6-pub is ${_xg(_xsn, 'B6-pub', 0, 'mean_p'):.3f}$ against ${_xg(_xsn, 'CPT-oracle', 0, 'mean_p'):.3f}$ for the CPT oracle and ${_xg(_xsn, 'Proposed', 200, 'mean_p'):.3f}$ for Proposed at $N{{=}}200$ (Brier ${_xg(_xsn, 'B6-pub', 0, 'brier'):.3f}$, ${_xg(_xsn, 'CPT-oracle', 0, 'brier'):.3f}$, ${_xg(_xsn, 'Proposed', 200, 'brier'):.3f}$).")
_xci = lambda s, k: float(_xrc[_xrc.subset == s][k].mean())
_xno = "RF report not none"
need_true("expF B6 control: 20 replications and the published means match the subsets file", len(_xrc) == 60 and abs(_xci(_xsd, "mean_published") - _xg(_xsd, "B6-pub", 0, "mean_p")) < 5e-4 and abs(_xci(_xsn, "mean_published") - _xg(_xsn, "B6-pub", 0, "mean_p")) < 5e-4)
need_true("expF B6 control: silence off lowers the mean but it stays above the observed rate", all(_xci(_s, "observed_high") < _xci(_s, "mean_silence_off") < _xci(_s, "mean_published") for _s in (_xsd, _xsn)))
need_true("expF B6 control: on other reports the input changes nothing", abs(_xci(_xno, "mean_silence_off") - _xci(_xno, "mean_published")) < 1e-12)
need_true("expF B6 rescaled keeps the ranking", all(f"{_xg(_s, 'B6-recal', 0, 'auc'):.3f}" == f"{_xg(_s, 'B6-pub', 0, 'auc'):.3f}" for _s in (_xsd, _xsn)))
need_supp("expF B6 control", f"B6-pub's hand rule on silence explains only part of its radio-silent errors: with the silence input off the means fall to ${_xci(_xsd, 'mean_silence_off'):.3f}$ on radio-silent drones and ${_xci(_xsn, 'mean_silence_off'):.3f}$ on those silent reports, still above the observed ${_xci(_xsd, 'observed_high'):.3f}$ and ${_xci(_xsn, 'observed_high'):.3f}$, and on other reports the mean is ${_xci(_xno, 'mean_silence_off'):.3f}$ (observed ${_xci(_xno, 'observed_high'):.3f}$).")
_xrsm = open(os.path.join(RES, "expRS_summary.md"), encoding="utf-8", newline="").read()
_xctl = re.search(r"moving up to (\d\.\d+) of softmax mass among the drone models moves P\(T\) by at most (\S+) \(300 scenarios\)", _xrsm)
need_true("expF softmax: 20 runs; moving mass among the drone models leaves P(T) alone", len(_xrm) == 20 and _xctl is not None and float(_xctl.group(1)) > 0.5 and float(_xctl.group(2)) < 1e-7)
_xcc = np.load(os.path.join(RES, "expG_cnn.npz"), allow_pickle=True)
_xp0 = _xrm[_xrm.proto == "P0"].iloc[0]
_xts = _xcc["test_softmax"]
need_true("expF softmax: the first split's row is its CNN", abs(_xts.max(axis=1).mean() - _xp0.mean_top) < 1e-6 and abs(np.median(np.maximum(_xts[:, 0], 1 - _xts[:, 0])) - _xp0.median_bg_split) < 1e-6)
need_true("expF softmax: on the test pool the split between background and drones has a median above 0.99 in every run", _xrm.median_bg_split.min() > 0.99)
_xyb = _xcc["test_y"]
need_true("expF the first split's test background: 200 spectrograms, 20 to a segment, two errors", (_xyb == 0).sum() == 200 and all(_c % 20 == 0 for _c in np.bincount(_xyb)) and ((_xyb == 0) & (_xts.argmax(axis=1) != 0)).sum() == 2)
need_true("expF the sentence on the difference between machines is gone", not HAVE_PAPER or "of the size of the difference between machines" not in tex)
need_tex("expF tolerance, what it is", "Monte Carlo values can differ across machines and library versions by about $0.001$ to $0.004$, as in one independent re-run of an earlier version of the code; seeds and library versions are pinned.")
_xdd = _rt("expI_dd_reliability.csv")
for _, _q in _xrt.iterrows():
    _qq = _xdd[(_xdd.fit == _q.fit) & (_xdd.cond == _q.cond)].sort_values("seed")
    _qd = _qq.ll_transfer.values - _qq.ll_none.values
    need_true(f"expF transfer file: {_q.fit} to {_q.cond}", len(_qq) == 5 and abs(_qd.mean() - _q.gain_over_no_discount) < 1e-12 and abs(_st.ttest_1samp(_qd, 0.0).pvalue - _q.p) < 1e-9)
_xtr = lambda f, c: _xrt.query("fit == @f and cond == @c").iloc[0]
_xtw, _xtb1, _xtb2 = _xtr("clean+BOTH", "WIFI"), _xtr("clean+WIFI", "BLUE"), _xtr("clean+BOTH", "BLUE")
need_true("expF transfer: no direction toward a milder condition is significant", all(_q.p > 0.05 for _q in (_xtw, _xtb1, _xtb2)))
need_supp("expF transfer: toward a milder condition", f"Toward a milder one the results are mixed (${_xtw.gain_over_no_discount:+.3f}$ under Wi-Fi, $p{{=}}{_xtw.p:.2f}$; ${_xtb1.gain_over_no_discount:+.3f}$ and ${_xtb2.gain_over_no_discount:+.3f}$ under Bluetooth, $p\\ge{min(_xtb1.p, _xtb2.p):.2f}$), which five seeds cannot separate.")
_xfq = pd.concat([_rt("expB6w_flags_raw.csv"), _rt("expW_add_diag_raw.csv")])
_xfq = _xfq[(_xfq.test == "A") & (_xfq.node == "T") & _xfq.world.isin(("standard", "nonadd-0.5", "nonadd-1.0"))]
_xfg = _xfq.groupby(["world", "N"]).flag.agg(["sum", "count"]).reset_index()
need_true("expF flags file is what the two raw files give", len(_xfg) == len(_xrl) == 17 and (_xfg["sum"].values == _xrl.flagged.values).all() and (_xfg["count"].values == 20).all())
_xfl = lambda w, n: int(_xrl.query("world == @w and N == @n").flagged.iloc[0])
need_tex("expF additivity test at small N", f"(weak and strong: {_xfl('nonadd-0.5', 50)} and {_xfl('nonadd-1.0', 50)} of 20 at $N{{=}}50$; {_xfl('nonadd-0.5', 20)} and {_xfl('nonadd-1.0', 20)} at $N{{=}}20$; {_xfl('nonadd-0.5', 10)} and {_xfl('nonadd-1.0', 10)} at $N{{=}}10$)")
_xsmall = max(_xfl(_w, _n) for _w in ("nonadd-0.5", "nonadd-1.0") for _n in (10, 20))
need_tex("expF additivity test in the Discussion", "The first two failures are detectable when $H$, $N$, and $C$ are labeled and data suffice, so")
need_true("expF additivity test: at most 7 of 20 at N <= 20 and, on a weak violation, at N = 50", _xsmall == 7)
_xs2 = _rt("expW_add_diag_raw.csv")
need_true("expF the S2 test was measured from N=50 only", _xs2[_xs2.world == "H-v-U"].N.min() == 50 and "H-v-U" not in set(_rt("expB6w_flags_raw.csv").world))
need_tex("expF the S2 test", "but never in the standard world, that test being measured from $N{=}50$ only.")
_xwr = lambda s, r: _xrw.query("sensor == @s and range == @r").iloc[0]
need_true("expF weights: 20 splits each", all(_xrw[_xrw.sensor == _s].highest_here.sum() == 20 for _s in ("Rr", "Re", "Ra", "Rf")))
_xrfw = _xrw[_xrw.sensor == "Rf"]
need_supp("expF weights vary with the calibration split", f"radar's far range is its highest fitted weight in ${_xwr('Rr', 'far').highest_here}$ of $20$ splits, and RF, with no range effect, spans ${_xrfw['min'].min():.3f}$ to ${_xrfw['max'].max():.2f}$.")
need_true("expF weights: RF has no range effect in the world", "Rf" not in getattr(_ms, "SENSOR_DEGRADE", {"Rf": 0}) or not any(_ms.SENSOR_DEGRADE["Rf"]))
need_tex("expF weights: direction, not size", "the fitted weights follow the direction of the world's range degradation but not its size, and they vary with the calibration split (Supplementary Material, Sec.~S4).")
need_tex("expF Table 2: the weight", "$\\alpha_s$ & fitted weight in place of $P(\\mathrm{Rel}_s{=}\\text{reliable}\\mid\\text{context})$ \\\\")
# the reversed prior at the deployed ESS
_xed = _rt("expED_raw.csv")
_xev = lambda m, n, l, k: (_xed[(_xed.method == m) & (_xed.N == n) & ((_xed.lam == l) if l is not None else _xed.lam.isna())].sort_values("seed")[k]).values
_xed2 = {(_n, _k): (lambda _d: (_d.mean(), _st.ttest_1samp(_d, 0.0).pvalue))(_xev("Proposed", _n, 1.0, _k) - _xev("MLE", _n, None, _k)) for _n in (10, 50, 200, 400) for _k in ("acc", "brier", "ece")}
need_true("expF reversed prior at alpha0 = 2: 20 seeds, the same grid", len(_xev("Proposed", 50, 1.0, "acc")) == 20 and sorted(_xed.N.unique()) == [10, 50, 200, 400] and sorted(_xed.lam.dropna().unique()) == [0.0, 0.25, 0.5, 0.75, 1.0])
need_true("expF reversed prior: accuracy recovers by N=200; Brier and ECE stay worse at N=400", _xed2[(50, "acc")][1] < 0.05 and _xed2[(200, "acc")][1] > 0.05 and _xed2[(400, "acc")][1] > 0.05 and _xed2[(50, "brier")][0] > 0 and _xed2[(50, "brier")][1] < 0.05 and all(_xed2[(400, _k)][0] > 0 and _xed2[(400, _k)][1] < 1e-3 for _k in ("brier", "ece")))
need_tex("expF reversed prior at the deployed ESS", f"At the deployed $\\alpha_0{{=}}2$ the proposed model under the fully reversed prior trails MLE at $N{{=}}50$ by ${-_xed2[(50, 'acc')][0]:.3f}$ accuracy and ${_xed2[(50, 'brier')][0]:.3f}$ Brier; accuracy recovers by $N{{=}}200$, but Brier and ECE stay worse at $N{{=}}400$ (${_xed2[(400, 'brier')][0]:+.3f}$, ${_xed2[(400, 'ece')][0]:+.3f}$, $p<10^{{-3}}$).")
# the summaries, whole
_xl = ["# Exp-RS: controls behind sentences that give a reason", "", "## (a) the rule of [5] on RF silence", "| subset | method | N | AUC | Brier | mean output |", "|---|---|---|---|---|---|"]
for _, _q in _xrs.iterrows():
    _xl.append(f"| {_q.subset} | {_q.method} | {_q.N} | {_q.auc:.3f} | {_q.brier:.3f} | {_q.mean_p:.3f} |")
_xl += ["", "Control: the published rule with the silence input off (same test sets, means over 20 replications)", "| subset | scenarios | observed high | mean, published | mean, silence off | Brier, published | Brier, silence off |", "|---|---|---|---|---|---|---|"]
for _s in (_xsd, _xsn, _xno):
    _xl.append(f"| {_s} | {_xci(_s, 'n'):.1f} | {_xci(_s, 'observed_high'):.3f} | {_xci(_s, 'mean_published'):.3f} | {_xci(_s, 'mean_silence_off'):.3f} | {_xci(_s, 'brier_published'):.3f} | {_xci(_s, 'brier_silence_off'):.3f} |")
_xl += ["", "## (b) softmax concentration", "| split | runs | mean top | share top > 0.9 | median max(background, drones) | smallest of that median |", "|---|---|---|---|---|---|"]
for _xpr in ("P0", "P1", "P1c", "P2", "P3"):
    _q = _xrm[_xrm.proto == _xpr]
    _xl.append(f"| {_xpr} | {len(_q)} | {_q.mean_top.mean():.3f} | {_q['share_top_above_0.9'].mean():.3f} | {_q.median_bg_split.mean():.4f} | {_q.median_bg_split.min():.4f} |")
_xl += [f"- runs: mean top from {_xrm.mean_top.min():.3f} to {_xrm.mean_top.max():.3f}; median of max(background, drones) at least {_xrm.median_bg_split.min():.4f}",
        f"- control: moving up to {_xctl.group(1) if _xctl else ''} of softmax mass among the drone models moves P(T) by at most {_xctl.group(2) if _xctl else ''} (300 scenarios)"]
_xl += ["", "## (c) binary-context weight of DroneDetect against no discount (five seeds)", "| fitted on | applied to | gain (nats) | p | log-lik., transfer | log-lik., own weight |", "|---|---|---|---|---|---|"]
for _, _q in _xrt.iterrows():
    _xl.append(f"| {_q.fit} | {_q.cond} | {_q.gain_over_no_discount:+.3f} | {_q.p:.2g} | {_q.ll_transfer:.3f} | {_q.ll_own:.3f} |")
_xl += ["", "## (d) additivity test on the threat node, flagged of 20", "| world | N | flagged |", "|---|---|---|"]
for _, _q in _xrl.iterrows():
    _xl.append(f"| {_q.world} | {_q.N} | {_q.flagged}/{_q.reps} |")
_xl += ["", "## (e) fitted reliability weights, calibration seeds 5000 + r, n = 300", "| sensor | range | mean | min | max | highest at this range (of 20) |", "|---|---|---|---|---|---|"]
for _, _q in _xrw.iterrows():
    _xl.append(f"| {_q.sensor} | {_q['range']} | {_q['mean']:.3f} | {_q['min']:.3f} | {_q['max']:.3f} | {_q.highest_here} |")
_YF = ("acc", "macroAUC", "brier", "ece")
_yfb = _rt("expRS_rf_bound.csv")
_ypr = lambda a, b, k: (lambda _d: (float(_d.mean()), float(_st.ttest_1samp(_d, 0.0).pvalue)))(a[k].values - b[k].values)
_ya = lambda n: _yfb[_yfb.variant == n].sort_values("rep")
_xl += ["", "## (f) the RF channel's upper bound (first split, 20 replications; hard and soft are the rows of expG_rf_e2e_raw.csv, none the row of expRC_raw.csv)", "| variant | accuracy | macro-AUC | Brier | ECE |", "|---|---|---|---|---|"]
for _n in ("none", "hard", "soft", "error-free split"):
    _q = _ya(_n)
    _xl.append(f"| {_n} | {_q.acc.mean():.4f} | {_q.macroAUC.mean():.4f} | {_q.brier.mean():.4f} | {_q.ece.mean():.4f} |")
_xl += ["", "| contrast | accuracy (p) | macro-AUC (p) | Brier (p) | ECE (p) |", "|---|---|---|---|---|"]
for _a, _b in (("hard", "none"), ("soft", "none"), ("error-free split", "none"), ("soft", "hard"), ("error-free split", "hard")):
    _xl.append(f"| {_a} - {_b} | " + " | ".join(f"{_v[0]:+.4f} ({_v[1]:.2g})" for _v in [_ypr(_ya(_a), _ya(_b), _k) for _k in _YF]) + " |")
_ysx = _rt("expRS_split.csv")
_yw0 = 1 - float(_gtw.cpts["cls"].table[2]) * float(_gtw.cpts["emit"].table[0])
_xl += ["", f"## (g) the split on what the scenarios draw (background pool {_yw0:.4f}, each drone model {(1 - _yw0) / 3:.4f}) and the test background", "| split | run | background spectrograms | segments | errors | pool median | scenario median | scenario below 0.9 | scenario mean top | background median | background below 0.9 |", "|---|---|---|---|---|---|---|---|---|---|---|"]
for _, _q in _ysx.iterrows():
    _xl.append(f"| {_q.proto} | {_q.run} | {_q.background_spectrograms} | {_q.background_segments} | {_q.background_errors} | {_q.pool_median:.4f} | {_q.scenario_median:.4f} | {_q['scenario_below_0.9']:.3f} | {_q.scenario_mean_top:.3f} | {_q.background_median:.4f} | {_q['background_below_0.9']:.3f} |")
_xl += [f"- runs: scenario median from {_ysx.scenario_median.min():.3f} to {_ysx.scenario_median.max():.3f}; scenario share below 0.9 from {_ysx['scenario_below_0.9'].min():.3f} to {_ysx['scenario_below_0.9'].max():.3f}; scenario mean top from {_ysx.scenario_mean_top.min():.3f} to {_ysx.scenario_mean_top.max():.3f}",
        f"- test background: {_ysx.background_segments.min()} to {_ysx.background_segments.max()} segments, {_ysx.background_spectrograms.min()} to {_ysx.background_spectrograms.max()} spectrograms, {_ysx.background_errors.min()} to {_ysx.background_errors.max()} errors"]
_yrl = _rt("expRS_rel_scale.csv")
_yrl["mult"] = _yrl["mult"].astype(str)
_yb1 = _yrl[_yrl["mult"] == "1.0"].sort_values("seed")
_yfx = _rt("expF_rel_raw.csv")
_yfx = _yfx[(_yfx.part == "miscal") & (_yfx.world == "nominal")]
_xl += ["", "## (h) the fitted weights scaled, minus the fitted weights (20 seeds)", "| factor | ECE, mean | accuracy (p) | macro-AUC (p) | Brier (p) | ECE (p) |", "|---|---|---|---|---|---|"]
for _m in (1.5, 1.25, 0.75, 0.5):
    _q = _yfx[_yfx.mult == _m].sort_values("seed")
    _xl.append(f"| {_m:g} | {_q.ece.mean():.4f} | " + " | ".join(f"{_v[0]:+.4f} ({_v[1]:.2g})" for _v in [_ypr(_q, _yb1, _k) for _k in _YF]) + " |")
for _m in ("0.25", "0.0", "reports hidden"):
    _q = _yrl[_yrl["mult"] == _m].sort_values("seed")
    _xl.append(f"| {_m} | {_q.ece.mean():.4f} | " + " | ".join(f"{_v[0]:+.4f} ({_v[1]:.2g})" for _v in [_ypr(_q, _yb1, _k) for _k in _YF]) + " |")
_xl.append(f"- fitted weights: ECE {_yb1.ece.mean():.4f}, macro-AUC {_yb1.macroAUC.mean():.4f}, Brier {_yb1.brier.mean():.4f}, accuracy {_yb1.acc.mean():.4f}")
_yrt = _rt("expRS_roots.csv")
_YM = ("acc", "macroF1", "macroAUC", "brier", "ece", "cwece", "far@pd0.9")
_yrq = lambda m, n, r: _yrt[(_yrt.method == m) & (_yrt.N == n) & (_yrt.roots == r)].sort_values("rep")
_YCASES = (("Proposed", 10), ("Proposed", 50), ("Proposed", 200), ("B1-MLE", 50))
_xl += ["", "## (i) the root marginals of cls and emit, minus the specification (20 replications)", "| method | N | roots | " + " | ".join(_YM) + " |", "|---|---|---|" + "---|" * len(_YM)]
for _m, _n in _YCASES:
    for _r in ("calibration split", "training rows"):
        _xl.append(f"| {_m} | {_n} | {_r} | " + " | ".join(f"{_v[0]:+.4f} ({_v[1]:.2g})" for _v in [_ypr(_yrq(_m, _n, _r), _yrq(_m, _n, "specification"), _k) for _k in _YM]) + " |")
_xl.append(f"- calibration split: the largest mean change of any metric is {max(abs(_ypr(_yrq(_m, _n, 'calibration split'), _yrq(_m, _n, 'specification'), _k)[0]) for _m, _n in _YCASES for _k in _YM):.4f}")
for _r in ("specification", "calibration split", "training rows"):
    _xl.append(f"- Proposed at N=10 minus MLE at N=50, roots from {_r}: " + ", ".join(f"{_k} {_v[0]:+.4f} (p={_v[1]:.2g})" for _k, _v in zip(_YF, [_ypr(_yrq("Proposed", 10, _r), _yrq("B1-MLE", 50, _r), _k) for _k in _YF])))
_yax = _rt("expRS_adverse.csv")
_xl += ["", "## (j) adverse subset, naive model (20 seeds; accuracy and ECE are the rows of expF_corr_raw.csv)", "| severity | mean top probability | accuracy | top minus accuracy | ECE | accuracy, reports hidden |", "|---|---|---|---|---|---|"]
for _s in sorted(_yax.severity.unique().tolist()):
    _q = _yax[_yax.severity == _s]
    _xl.append(f"| {_s:g} | {_q.mean_top.mean():.3f} | {_q.acc.mean():.3f} | {(_q.mean_top - _q.acc).mean():+.3f} | {_q.ece.mean():.3f} | {_q.acc_reports_hidden.mean():.3f} |")
_Zcell = lambda a, b, k: (lambda _d: f"{_d.mean():+.4f} ({_st.ttest_1samp(_d, 0.0).pvalue:.2g})")((lambda _e: _e[~np.isnan(_e)])(a.sort_values(a.columns[0])[k].values - b.sort_values(b.columns[0])[k].values))
_zF = _rt("expRS_fusion.csv")
_zsel = lambda v, s: _zF[(_zF.variant == v) & (_zF.subset == s)].sort_values("rep")
_xl += ["", "## (k) calibrated fusion against two controls (N = 200, 20 replications; abstract, calibrated and calib+rel are the rows of expS_fusion_raw.csv)", "| subset | variant | accuracy | macro-F1 | macro-AUC | Brier | ECE |", "|---|---|---|---|---|---|---|"]
for _s in ("all", "radio-silent"):
    for _v in ("abstract", "no RF", "no RF+rel", "hand-set emission-aware", "calibrated", "calib+rel"):
        _q = _zsel(_v, _s)
        _xl.append(f"| {_s} | {_v} | {_q.acc.mean():.4f} | " + (f"{_q.macroF1.mean():.4f}" if _s == "all" else "-") + f" | {_q.macroAUC.mean():.4f} | {_q.brier.mean():.4f} | {_q.ece.mean():.4f} |")
_xl += ["", "| contrast | subset | accuracy (p) | macro-AUC (p) | Brier (p) | ECE (p) |", "|---|---|---|---|---|---|"]
for _a, _b in (("no RF", "abstract"), ("calibrated", "no RF"), ("calib+rel", "no RF+rel"), ("calibrated", "abstract"), ("hand-set emission-aware", "abstract"), ("calibrated", "hand-set emission-aware"), ("calib+rel", "hand-set emission-aware"), ("calib+rel", "calibrated")):
    for _s in ("all", "radio-silent"):
        _xl.append(f"| {_a} - {_b} | {_s} | " + " | ".join(_Zcell(_zsel(_a, _s)[["rep", _k]], _zsel(_b, _s)[["rep", _k]], _k) for _k in _YF) + " |")
_zR = _rt("expRS_rule_b3.csv")
_zrm = lambda m, N: _zR[(_zR.method == m) & (_zR.N == N)].sort_values("rep")
_zr6 = _zrm("B6-pub", 0)
_xl += ["", "## (l) a network with a uniform prior mean against the rule of [5] (20 replications)",
        f"- the rule on T = high: AUC {_zr6.auc_high.mean():.4f}, Brier {_zr6.brier_high.mean():.4f}; scores above 0.75 are high-threat {_zr6.high_above.sum() / _zr6.n_above.sum():.4f} of the time ({int(_zr6.n_above.sum())} scenarios)",
        f"- the rule read as P(T >= medium): AUC {_zr6.auc_ge_medium.mean():.4f}, Brier {_zr6.brier_ge_medium.mean():.4f}; scores above 0.75 are at T >= medium {_zr6.ge_medium_above.sum() / _zr6.n_above.sum():.4f} of the time; constant forecast at the base rate {_zr6.base_ge_medium.mean():.4f}: Brier {(_zr6.base_ge_medium * (1 - _zr6.base_ge_medium)).mean():.4f}",
        "", "| N | B3 AUC, T = high | minus the rule (p) | B3 Brier, T = high | minus the rule (p) | Proposed AUC, T = high | Proposed AUC, T >= medium | Proposed Brier, T >= medium |", "|---|---|---|---|---|---|---|---|"]
for _n in (10, 20, 50, 100, 200, 400):
    _a, _c = _zrm("B3", _n), _zrm("Proposed", _n)
    _xl.append(f"| {_n} | {_a.auc_high.mean():.4f} | {_Zcell(_a[['rep', 'auc_high']], _zr6[['rep', 'auc_high']], 'auc_high')} | {_a.brier_high.mean():.4f} | {_Zcell(_a[['rep', 'brier_high']], _zr6[['rep', 'brier_high']], 'brier_high')} | {_c.auc_high.mean():.4f} | {_c.auc_ge_medium.mean():.4f} | {_c.brier_ge_medium.mean():.4f} |")
_zB = _rt("expB_baselines_raw.csv")
_zbq = lambda m, N: _zB[(_zB.method == m) & (_zB.N == N)].sort_values("rep")
_xl += ["", "| N | contrast with B0-as-prior MAP | ECE (p) | macro-AUC (p) |", "|---|---|---|---|"]
for _n in (10, 50):
    for _m in ("Proposed", "expert MAP (full table)"):
        _xl.append(f"| {_n} | {_m} | {_Zcell(_zbq(_m, _n)[['rep', 'ece']], _zbq('B0-as-prior MAP', _n)[['rep', 'ece']], 'ece')} | {_Zcell(_zbq(_m, _n)[['rep', 'macroAUC']], _zbq('B0-as-prior MAP', _n)[['rep', 'macroAUC']], 'macroAUC')} |")
_zT = _rt("expRS_track_split.csv")
_xl += ["", "## (m) P(T = high) of the worked track, with the training data and the prior redrawn apart (seeds 0 to 19, calibration seed 777)", "| redrawn | mean | min | max |", "|---|---|---|---|"]
for _k in ("both", "training data", "prior"):
    _q = _zT[_zT.redrawn == _k]
    _xl.append(f"| {_k} | {_q.p_high.mean():.4f} | {_q.p_high.min():.4f} | {_q.p_high.max():.4f} |")
_zfs = _rt("expRS_flags_small.csv")
_xl += ["", "## (n) false alarms of the additivity test at small N, standard world (B = 199)", "| N | flagged | reps |", "|---|---|---|"]
for _n, _q in _zfs.groupby("N"):
    _xl.append(f"| {_n} | {int(_q.flag.sum())} | {len(_q)} |")
_ze = _rt("expED_raw.csv")
_zml = lambda N: _ze[(_ze.method == "MLE") & (_ze.N == N)].sort_values("seed")
_zpr = lambda lam, N: _ze[(_ze.method == "Proposed") & (_ze.lam == lam) & (_ze.N == N)].sort_values("seed")
_zG = _rt("expRS_lambda.csv")
_xl += ["", "## (o) the reversed-prior grid at alpha0 = 2, from expED_raw.csv (20 seeds; test seed 999, calibration seed 777)", "| method | lambda | N | accuracy | macro-AUC | Brier | ECE |", "|---|---|---|---|---|---|---|"]
for _, _q in _zG.iterrows():
    _xl.append(f"| {_q.method} | " + ("-" if _q.method == "MLE" else f"{_q.lam:g}") + f" | {_q.N} | {_q.acc:.4f} | {_q.macroAUC:.4f} | {_q.brier:.4f} | {_q.ece:.4f} |")
_xl += ["", "| lambda | contrast | accuracy (p) | macro-AUC (p) | Brier (p) | ECE (p) |", "|---|---|---|---|---|---|"]
for _l in (0.0, 0.25, 0.5, 0.75, 1.0):
    for _n, _m in ((10, 50), (50, 50), (10, 10), (200, 200), (400, 400)):
        _xl.append(f"| {_l:g} | Proposed at N={_n} minus MLE at N={_m} | " + " | ".join(_Zcell(_zpr(_l, _n)[["seed", _k]], _zml(_m)[["seed", _k]], _k) for _k in _YF) + " |")
_zck = _rt("expS_calibsize_raw.csv")
_zka, _zk30 = _zck[_zck.source == "abstract"].sort_values("rep"), _zck[(_zck.source == "estimated") & (_zck.calib_n == 30)].sort_values("rep")
_xl += ["", "## (p) contrasts read from the result files", "- 30 labeled scenarios minus the abstract model: " + ", ".join(f"{_m} {_Zcell(_zk30[['rep', _m]], _zka[['rep', _m]], _m)}" for _m in _YF)]
_zsc = _rt("expS_scarcity_raw.csv")
_zsq = lambda m, N: _zsc[(_zsc.method == m) & (_zsc.N == N)].sort_values("rep")
_xl += ["", "| N | Proposed - B0: macro-AUC (p) | ECE (p) | FAR (p) | Proposed - MLE: FAR (p) |", "|---|---|---|---|---|"]
for _n in (10, 20, 50, 100, 200, 400):
    _xl.append(f"| {_n} | " + " | ".join(_Zcell(_zsq("Proposed", _n)[["rep", _m]], _zsq("B0-heuristic", _n)[["rep", _m]], _m) for _m in ("macroAUC", "ece", "far@pd0.9")) + f" | {_Zcell(_zsq('Proposed', _n)[['rep', 'far@pd0.9']], _zsq('B1-MLE', _n)[['rep', 'far@pd0.9']], 'far@pd0.9')} |")
_xl += ["", "- the fitted discount minus none: " + ", ".join(f"{_m} {_Zcell(_zsel('calib+rel', 'all')[['rep', _m]], _zsel('calibrated', 'all')[['rep', _m]], _m)}" for _m in ("brier", "ece")) + f"; radio-silent Brier {_Zcell(_zsel('calib+rel', 'radio-silent')[['rep', 'brier']], _zsel('calibrated', 'radio-silent')[['rep', 'brier']], 'brier')}"]
_zH = _rt("expRS_handset.csv")
_zHD = (0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.94, 0.99)
_zHS = ("all", "radio-silent drones", "drones", "non-drones", "RF none", "RF none, non-drones")
_zH4 = ("acc", "macroAUC", "brier", "ece")
_zhv = lambda v, d, s: _zH[(_zH.variant == v) & (_zH.subset == s) & (_zH.diagonal.isna() if d is None else _zH.diagonal == d)].sort_values("rep")


def _zqc(a, b, k):
    _d = a[k].values - b[k].values
    _d = _d[~np.isnan(_d)]
    return f"{_d.mean():+.4f} ({_st.ttest_1samp(_d, 0.0).pvalue:.2g})" if len(_d) >= 10 else f"n/a ({len(_d)} of 20)"


_xl += ["", "## (q) the hand-set emission-aware table swept over its diagonal (N = 200, no discount, 20 replications; at 0.94 the rows of expRS_fusion.csv, calibrated the row n = 300 of expS_calibsize_raw.csv)", "| diagonal | accuracy | macro-F1 | macro-AUC | Brier | ECE | FAR |", "|---|---|---|---|---|---|---|"]
_zq0 = _zhv("calibrated", None, "all")
_xl.append(f"| calibrated | {_zq0.acc.mean():.4f} | {_zq0.macroF1.mean():.4f} | {_zq0.macroAUC.mean():.4f} | {_zq0.brier.mean():.4f} | {_zq0.ece.mean():.4f} | {_zq0.far.mean():.4f} |")
for _d in _zHD:
    _zq0 = _zhv("hand-set", _d, "all")
    _xl.append(f"| {_d:.2f} | {_zq0.acc.mean():.4f} | {_zq0.macroF1.mean():.4f} | {_zq0.macroAUC.mean():.4f} | {_zq0.brier.mean():.4f} | {_zq0.ece.mean():.4f} | {_zq0.far.mean():.4f} |")
_xl += ["", "Calibrated minus hand-set", "| subset | scenarios per replication | diagonal | accuracy (p) | macro-AUC (p) | Brier (p) | ECE (p) |", "|---|---|---|---|---|---|---|"]
for _s in _zHS:
    for _d in _zHD:
        _zq0, _zq1 = _zhv("calibrated", None, _s), _zhv("hand-set", _d, _s)
        _xl.append(f"| {_s} | {_zq0.n.mean():.0f} | {_d:.2f} | " + " | ".join(_zqc(_zq0, _zq1, _k) for _k in _zH4) + " |")
_xl += ["", "The mixed tables: is the RF row the cause?", "| contrast | subset | accuracy (p) | macro-AUC (p) | Brier (p) | ECE (p) |", "|---|---|---|---|---|---|"]
for _s in ("all", "radio-silent drones"):
    _xl.append(f"| estimated, RF row hand-set 0.94 - calibrated | {_s} | " + " | ".join(_zqc(_zhv("estimated, RF row hand-set", 0.94, _s), _zhv("calibrated", None, _s), _k) for _k in _zH4) + " |")
    _xl.append(f"| hand-set 0.94, RF row estimated - hand-set 0.94 | {_s} | " + " | ".join(_zqc(_zhv("hand-set, RF row estimated", 0.94, _s), _zhv("hand-set", 0.94, _s), _k) for _k in _zH4) + " |")
_xl += ["", "The estimated matrices' correct-report probability (mean over the classes, 20 calibration splits, n = 300)", "| sensor | mean | smallest | largest |", "|---|---|---|---|"]
_zlev = {_s: _zH[(_zH.variant == "estimated, correct-report probability") & (_zH.subset == _s)].acc for _s in ("Rr", "Re", "Ra")}
for _s in ("Rr", "Re", "Ra"):
    _xl.append(f"| {_s} | {_zlev[_s].mean():.4f} | {_zlev[_s].min():.4f} | {_zlev[_s].max():.4f} |")
_xl += ["", "By calibration size, minus hand-set at 0.75", "| matrices | accuracy (p) | macro-AUC (p) | Brier (p) | ECE (p) |", "|---|---|---|---|---|"]
_zq1 = _zhv("hand-set", 0.75, "all")
for _s, _zq0 in [("abstract", _zck[_zck.source == "abstract"])] + [(f"estimated, n = {_n}", _zck[(_zck.source == "estimated") & (_zck.calib_n == _n)]) for _n in (30, 100, 300, 1000)] + [("sensor oracle", _zck[_zck.source == "oracle-marginal"])]:
    _xl.append(f"| {_s} | " + " | ".join(_zqc(_zq0.sort_values("rep"), _zq1, _k) for _k in _zH4) + " |")
_xl += ["", "At one level: the emission-aware RF row, no RF channel and the emission-blind RF row, all with the same diagonal", "| contrast | subset | diagonal | accuracy (p) | macro-AUC (p) | Brier (p) | ECE (p) |", "|---|---|---|---|---|---|---|"]
for _zu, _zw in (("hand-set", "abstract"), ("no RF", "abstract"), ("hand-set", "no RF")):
    for _s in ("all", "radio-silent drones", "non-drones"):
        for _d in _zHD:
            _xl.append(f"| {_zu} - {_zw} | {_s} | {_d:.2f} | " + " | ".join(_zqc(_zhv(_zu, _d, _s), _zhv(_zw, _d, _s), _k) for _k in _zH4) + " |")
_whole("expRS_summary.md", _xl)
_xl = ["# Exp-ED: adversarial prior at the deployed ESS (alpha0 = 2)", "", f"- {_xed.seed.nunique()} seeds; test seed 999, calibration seed 777, as exp_prior_ess.py part (i); added after the main results", "", "## means", "| lambda | N | method | acc | macro-AUC | Brier | ECE |", "|---|---|---|---|---|---|---|"]
for _n in (10, 50, 200, 400):
    _q = _xed[(_xed.method == "MLE") & (_xed.N == _n)]
    _xl.append(f"| - | {_n} | MLE | {_q.acc.mean():.3f} | {_q.macroAUC.mean():.3f} | {_q.brier.mean():.3f} | {_q.ece.mean():.3f} |")
for _l in (0.0, 0.25, 0.5, 0.75, 1.0):
    for _n in (10, 50, 200, 400):
        for _m in ("MAP", "cMAP", "Proposed"):
            _q = _xed[(_xed.lam == _l) & (_xed.N == _n) & (_xed.method == _m)]
            _xl.append(f"| {_l:g} | {_n} | {_m} | {_q.acc.mean():.3f} | {_q.macroAUC.mean():.3f} | {_q.brier.mean():.3f} | {_q.ece.mean():.3f} |")
_xl += ["", "## fully reversed prior (lambda = 1) minus MLE at the same N (paired t over the seeds)", "| N | method | d acc (p) | d macro-AUC (p) | d Brier (p) | d ECE (p) |", "|---|---|---|---|---|---|"]
for _n in (10, 50, 200, 400):
    for _m in ("MAP", "cMAP", "Proposed"):
        _c = []
        for _k in ("acc", "macroAUC", "brier", "ece"):
            _d = _xev(_m, _n, 1.0, _k) - _xev("MLE", _n, None, _k)
            _c.append(f"{_d.mean():+.3f} ({_st.ttest_1samp(_d, 0.0).pvalue:.2g})")
        _xl.append(f"| {_n} | {_m} | " + " | ".join(_c) + " |")
_whole("expED_summary.md", _xl)

# the minor values
_xsc = _rt("expS_scarcity_raw.csv")
_xpair = lambda a, b, k: (lambda _d: (float(_d.mean()), float(_st.ttest_1samp(_d, 0.0).pvalue)))(a.sort_values("rep")[k].values - b.sort_values("rep")[k].values)
_xcw = _xpair(_xsc[(_xsc.method == "Proposed") & (_xsc.N == 10)], _xsc[(_xsc.method == "B1-MLE") & (_xsc.N == 50)], "cwece")
need_tex("expF classwise ECE beside the top-label one", f"classwise ${_xcw[0]:+.3f}$, $p{{=}}{_xcw[1]:.2f}$) and FAR")
_xbb = _rt("expB_baselines_raw.csv")
_xb0 = {_n: _xpair(_xbb[(_xbb.method == "B0-as-prior MAP") & (_xbb.N == _n)], _xbb[(_xbb.method == "B1-MLE") & (_xbb.N == _n)], "ece") for _n in (10, 50, 80, 100, 200, 400)}
need_true("expF the B0-mean prior leads MLE on ECE only at N=50, 80 and 100", {_n for _n, _v in _xb0.items() if _v[0] < 0 and _v[1] < 0.05} == {50, 80, 100} and not any(_v[0] > 0 and _v[1] < 0.05 for _v in _xb0.values()))
need_tex("expF the B0-mean prior on ECE", f"on ECE only at $N{{=}}50$, $80$, and $100$ ($p{{=}}{_xb0[50][1]:.2f}$, ${_xb0[80][1]:.3f}$, ${_xb0[100][1]:.3f}$)")
_xfu = _rt("expS_fusion_raw.csv")
_xfa, _xfb = _xfu[_xfu.config == "all (calibrated)"], _xfu[_xfu.config == "all (abstract model)"]
_xfd = {_k: _xpair(_xfa, _xfb, _k) for _k in ("acc", "macroF1", "macroAUC", "brier", "ece", "far@pd0.9")}
need_true("expF calibrated fusion beats the abstract model on five metrics, significantly", all(_xfd[_k][1] < 0.05 for _k in ("acc", "macroAUC", "brier", "ece", "far@pd0.9")) and _xfd["acc"][0] > 0 and _xfd["macroAUC"][0] > 0 and all(_xfd[_k][0] < 0 for _k in ("brier", "ece", "far@pd0.9")) and _xfd["macroF1"][1] > 0.05)
need_tex("expF the metrics of the fusion sentence", f"the calibrated fusion surpasses it on accuracy, AUC, Brier, ECE, and FAR (macro-F1 ${_xfd['macroF1'][0]:+.3f}$, $p{{=}}{_xfd['macroF1'][1]:.2f}$): AUC")
_xbc = {(_m, _n): {_k: _xpair(_xsc[(_xsc.method == "Proposed") & (_xsc.N == _n)], _xbb[(_xbb.method == _m) & (_xbb.N == _n)], _k) for _k in ("acc", "macroAUC", "brier", "ece")} for _m in ("uninformative MAP", "Laplace", "logistic regression") for _n in (10, 50, 100, 200, 400)}
need_true("expF the abstract: ahead of B3, B4, B5 on macro-AUC, Brier and ECE at every N", all(_v["macroAUC"][0] > 0 and _v["brier"][0] < 0 and _v["ece"][0] < 0 and all(_v[_k][1] < 0.05 for _k in ("macroAUC", "brier", "ece")) for _v in _xbc.values()))
need_true("expF the abstract: accuracy at N=400 is not significant against B3 and B4", _xbc[("uninformative MAP", 400)]["acc"][1] > 0.05 and _xbc[("Laplace", 400)]["acc"][1] > 0.05)
need_tex("expF the abstract: the metrics", "and beats full-table uninformative-prior and discriminative baselines on macro-AUC and Brier at every size.")
_xco = _rt("expF_corr_raw.csv")
_xauc = _xco[_xco.severity == 1.0].groupby("fusion").auc_adv.mean()
need_tex("expF nuisance parent: AUC", f"though its AUC barely moves (${_xauc['W-observed']:.3f}$ against ${_xauc['naive-CI']:.3f}$);")
need_tex("expF nuisance parent: labels", "and $W$-conditional matrices are estimated from the same split, with $W$ labeled there and $P(W)$ given.")
need_tex("expF P(e)", "$P(e)$ is a weak detector of correlated degradation,")
_xe = _rt("expE_prior_raw.csv")
_xem = lambda m, k: float(_xe[(_xe.method == m) & (_xe.N == 400) & (((_xe.lam == 1.0) & (_xe.ess == 8.0)) if m != "MLE" else True)][k].mean())
need_supp("expF cMAP, MAP and MLE on this study's test set", f"cMAP's accuracy recovers by $N{{\\approx}}400$ (${_xem('cMAP', 'acc'):.3f}$ vs.\\ MLE ${_xem('MLE', 'acc'):.3f}$ on that study's fixed test set; MAP ${_xem('MAP', 'acc'):.3f}$), \\emph{{but calibration does not}} (ECE ${_xem('cMAP', 'ece'):.3f}$; MAP ${_xem('MAP', 'ece'):.3f}$; MLE ${_xem('MLE', 'ece'):.3f}$)")
# the constraint study in Sec. VII-B
_xcn = _rt("exp_constraints_raw.csv")
_xcs = open(os.path.join(HERE, "exp_constraints.py"), encoding="utf-8").read()
need_true("expF constraint study: 8 seeds, ESS 8, N = 20 and 50, one test and one calibration set", _xcn.seed.nunique() == 8 and sorted(_xcn.N.unique()) == [20, 50] and "ESS = 8.0" in _xcs and _xcs.count("dataio.generate_dataset(gt, 2500, seed=999)") == 1 and _xcs.count("dataio.generate_dataset(gt, 300, seed=777)") == 1)
need_tex("expF constraint study: its conditions", "(a separate study: $8$ seeds, $\\alpha_0{=}8$, $N{=}20$ and $50$, one fixed test and calibration set)")
_xcd = {(_z, _n): (lambda _d: (float(_d.mean()), float(_st.ttest_1samp(_d, 0.0).pvalue)))(_xcn[(_xcn.prior_noise == _z) & (_xcn.N == _n) & (_xcn.method == "cMAP")].sort_values("seed").ece.values - _xcn[(_xcn.prior_noise == _z) & (_xcn.N == _n) & (_xcn.method == "MAP")].sort_values("seed").ece.values) for _z in (0.15, 0.5) for _n in (20, 50)}
_xca = lambda z: _xcn[_xcn.prior_noise == z].groupby(["method", "N"]).acc.mean()
_xcv = lambda z: _xcn[(_xcn.prior_noise == z) & (_xcn.method == "MAP")].groupby("N").T_mono_violation.mean()
need_true("expF constraint study: cMAP removes every violation", (_xcn[_xcn.method == "cMAP"].T_mono_violation == 0).all())
need_true("expF constraint study: the calibration change at noise 0.5 is not resolved", all(_xcd[(0.5, _n)][1] > 0.05 for _n in (20, 50)) and all(f"{_xcd[(0.5, _n)][0]:+.3f}" == "+0.005" for _n in (20, 50)))
need_true("expF constraint study: |dECE| at noise 0.15 is at most 0.001", all(abs(_xcd[(0.15, _n)][0]) <= 0.001 for _n in (20, 50)))
need_tex("expF constraint study: noise 0.5", f"(accuracy ${_xca(0.5).min():.3f}$--${_xca(0.5).max():.3f}$, below the $0.434$ majority rate), with no resolved calibration change (Supplementary Material, Sec.~S4).")
need_supp("expF constraint study: noise 0.5, the change", f"at noise $0.5$ admissibility comes with no resolved calibration change ($\\mathrm{{ECE}}$ $+0.005$, $p{{=}}{_xcd[(0.5, 20)][1]:.2f}$ and ${_xcd[(0.5, 50)][1]:.2f}$).")
need_supp("expF constraint study: noise 0.15", f"(accuracy ${_xca(0.15).min():.3f}$--${_xca(0.15).max():.3f}$)")
need_tex("expF constraint study: violations", f"(in ${100 * _xcv(0.15).min():.1f}$--${100 * _xcv(0.15).max():.1f}\\%$ of the covering-pair thresholds at elicitation noise $0.15$ and in ${100 * _xcv(0.5).min():.1f}$--${100 * _xcv(0.5).max():.1f}\\%$ at noise $0.5$)")
need_tex("expF claim (iv), its scope", "and claim (iv) holds at elicitation noise $0.15$ but not at $0.5$ in the separate study above ($\\alpha_0{=}8$, $8$ seeds); Sec.~\\ref{sec:priorsens} maps its limits.")
need_tex("expF constraints in the body", "Admissible tables are therefore a structural guarantee, not a performance fix;")
# Sec. S2 of the supplement: the regime numbers that left the body
_xrg = lambda l, e, n, m, k: _xe[(_xe.lam == l) & (_xe.ess == e) & (_xe.N == n) & (_xe.method == m)].sort_values("seed")[k].values
_xrd = {(_l, _e, _n, _k): float((_xrg(_l, _e, _n, "cMAP", _k) - _xrg(_l, _e, _n, "MAP", _k)).mean()) for _l in (0.0, 0.25, 0.5, 0.75, 1.0) for _e in (8.0, 50.0) for _n in (10, 50, 200, 400) for _k in ("ece", "brier")}
_xr5e, _xr5b = min(-_v for _kk, _v in _xrd.items() if _kk[0] == 0.5 and _kk[3] == "ece"), min(-_v for _kk, _v in _xrd.items() if _kk[0] == 0.5 and _kk[3] == "brier")
_xr7e, _xr7b = max(-_v for _kk, _v in _xrd.items() if _kk[0] == 0.75 and _kk[3] == "ece"), max(-_v for _kk, _v in _xrd.items() if _kk[0] == 0.75 and _kk[3] == "brier")
need_supp("expF S2: half-reversed prior", f"($\\lambda{{=}}0.5$: down to ${_xr5e:.3f}$ ECE and ${_xr5b:.3f}$ Brier,")
need_supp("expF S2: strongly reversed prior", f"($\\lambda{{=}}0.75$: up to ${_xr7e:+.3f}$ ECE and ${_xr7b:+.3f}$ Brier,")
need_supp("expF S2: the two metrics part", f"top-label ECE improves by ${-_xrd[(1.0, 50.0, 50, 'ece')]:+.3f}$ while Brier degrades by ${-_xrd[(1.0, 50.0, 50, 'brier')]:.3f}$")
_xr1 = lambda m, k: float(_xrg(1.0, 50.0, 50, m, k).mean())
need_supp("expF S2: MAP and cMAP against the reference", f"(accuracy ${_xr1('MAP', 'acc'):.3f}$, Brier ${_xr1('MAP', 'brier'):.3f}$) while cMAP falls below it on both (${_xr1('cMAP', 'acc'):.3f}$, ${_xr1('cMAP', 'brier'):.3f}$)")
for _sn in ("(Supplementary Material, Sec.~S2 and Fig.~S1)", "The constraints always restore admissibility, but whether they \\emph{help} depends on the metric",
            "At $\\lambda{=}1$ and small $N$, cMAP falls below a constant predictor of the threat marginal on accuracy and Brier,",
            "and the proposed ordinal-ICI threat node fails the same way at $\\alpha_0{=}50$ (accuracy $0.408$ and Brier $0.670$--$0.676$, against $0.434$ and $0.654$)."):
    need_tex(f"expF VII-E(ii) keeps: {_sn[:40]}", _sn)
for _sn in ("the induced violation rate falls from $0.55$--$0.87$ under MAP to exactly $0$ under cMAP in every $\\lambda{=}1$ cell.", "cMAP is below the reference on accuracy and Brier in all four, MAP in only one.",
            "($N\\ge200$ at $\\alpha_0{=}8$, $N{=}400$ at $\\alpha_0{=}50$)", "a top-label ECE of $0.019$--$0.022$ that reflects that collapse rather than good calibration."):
    need_supp(f"expF VII-E(ii), the detail in S2: {_sn[:40]}", _sn)
for _sn in ("\\section{When constraints help, and on which metric}", "Fig.~S1 therefore gives four maps, top-label ECE and Brier at $\\alpha_0{=}8$ and $50$."):
    need_supp(f"expF supplement: {_sn[:40]}", _sn)
# wording: what now stands, whole sentences
for _lab, _sn in (("second need", "\\emph{Second, the sensor model is a set of hand rules.} The framework scores sensor reports directly (RF silence, for instance, raises the threat)"),
                  ("stand-in", "The \\emph{abstract} sensor model is a stand-in of ours, since \\cite{song2026base} has no report-level sensor model and scores RF silence by a hand rule;"),
                  ("B6", "reading its output as $P(T{=}\\text{high})$ is our assumption. It reads the kinematic, RF, radar, and airspace inputs (no input node of"),
                  ("conclusion", "keeping the base framework's structure while grounding its hand-set probabilities in priors and data and replacing its hand rules on sensor reports."),
                  ("Algorithm 1", "\\State data check (not in the main runs): test additivity and monotonicity; where flagged, full table or no constraint (Sec.~\\ref{sec:misspec})"),
                  ("with enough data", "With enough data both failures show in the training data:"),
                  ("abstract", "Keeping that structure, we adopt an ordinal independence-of-causal-influence threat node,"), ("Sec. I", "We address both gaps"), ("VII-D", "In this world a reported high-threat probability"),
                  ("conclusion, the grounding", "We grounded a hierarchical C-UAS Bayesian network in expert priors, monotonicity constraints, and simulated scenarios (C1--C2)"),
                  ("conclusion, the result", "the grounded model at ten scenarios is not significantly worse than maximum likelihood at fifty and ranks high threat better than the base framework's rule; it breaks where additivity or a monotonicity constraint fails, and a data check on the training data flags both"),
                  ("VII-B", "Unless stated, values are means over $20$ independent replications"), ("VI-E", "The ESS is $\\alpha_0{=}2$ per column in the main study (other values are stated where used),"),
                  ("label budget", "Every BN method sees the same $N$ training scenarios and the same $n{=}300$ sensor-calibration split, with no validation labels; B5 sees only the $N$ scenarios."),
                  ("B5", "(it cannot use the calibration split of the BN methods)"),
                  ("a sentence", "the form the BN uses (Sec.~\\ref{sec:rfe2e} estimates the deployed table on the calibration split):"), ("a sentence", "which may arise because both are Parrot platforms."),
                  ("Proposition 1", "and it differs from the constrained maximum-likelihood estimate by $O(1/N)$."), ("Appendix A", "so the variational inequalities of the two programs put them $O(1/N)$ apart, with a constant that grows as $\\min_{jk}P(j)\\theta^*_{jk}$ shrinks."),
                  ("r", "with $r$ the number of reports and $\\beta$ (e.g., $0.5$) preventing impossible off-diagonals;"), ("K", "over $K{=}4$ levels"),
                  ("tables with intervals", "Tables~\\ref{tab:scarcity}, \\ref{tab:sensors}, and~\\ref{tab:calibsize} report means with $95\\%$ confidence intervals"),
                  ("Table 3", "The generating ordinal-response weights, the base confusion tables, and the rounded weights of the B0 surrogate are in \\texttt{model\\_spec.py}"),
                  ("VI-A", "whose structure and main parameters are given in Table~\\ref{tab:world} and whose complete specification, with the base confusion tables, is in \\texttt{model\\_spec.py}"),
                  ("VII-I", "\\subsection{One-Way Sensitivity and Its Envelope}")):
    need_tex(f"expF {_lab}", _sn)
_xT = _ms.PARENTS["T"]
_xtc = int(np.prod([_ms.CARD[_p] for _p in _xT]))
need_true("expF the threat table has 18 columns, so 36 pseudo-observations at alpha0 = 2", _xtc == 18)
_xhc = int(np.prod([_ms.CARD[_p] for _p in _ms.PARENTS["H"]]))
need_true("expF H has 27 columns, so 54 pseudo-observations at alpha0 = 2", _xhc == 27)
need_tex("expF ESS per column", f"so $H$ carries ${2 * _xhc}$ pseudo-observations over its ${_xhc}$ columns and the threat table ${2 * _xtc}$ over its ${_xtc}$, more than the training set at $N{{=}}10$ and $20$;")
need_true("expF the rules of the study and the drones-only check: not in the body", not HAVE_PAPER or ("Rules described as fixed in advance" not in tex and "we did not run that check" not in tex))
need_true("expF the rule reads kinematics, RF, radar and airspace", "def map_states(d, rdot, v, z, rf, rr," in open(os.path.join(HERE, "exp_base_rule.py"), encoding="utf-8").read())
if HAVE_PAPER:
    _xf1, _xf2 = (open(os.path.join(PAPER, _xfn), encoding="utf-8").read() for _xfn in ("fig1_framework.tex", "fig2_bn.tex"))
    need_true("expF Fig. 1 says envelope", "with a one-at-a-time envelope" in _xf1 and "credal" not in _xf1)
    need_true("expF Fig. 2 labels the edge from X^(1) to C with d", "\\draw[fl] (X1.north) -- node[el,pos=0.42,above=0.5pt] {$d$} (C);" in _xf2)
    _xfar = tex.find("The false-alarm block of Table~\\ref{tab:scarcity}")
    need_true("expF the false-alarm paragraph ends the part on scarcity", 0 < tex.find("\\subsubsection{Grounding under scarcity}") < _xfar < tex.find("\\subsubsection{Standard baselines}"))
    need_true("expF Fig. 1 names simulated scenarios, not a digital twin", "{Simulated scenarios}" in _xf1 and "twin" not in _xf1.lower())
# the old wording stays out
for _x in ("contributes little", "reads the RF", "near-one-hot", "over-discount", "closes both gaps", "close both gaps", "certify", "exactly what the fitted", "physical consequence", "especially after ICI", "PRE-REGISTERED", "pre-registered",
           "Credal Robustness", "credal bounds", "every reported metric", "for every reported run", "All values are means", "In practice a reported", "O(1/N_", "useful but partial", "transfers in three", "consumes directly", "abstract sensor model.", "heuristic probabilities", "interpretability while"):
    need_true(f"expF retired from main.tex and the supplement: {_x}", not HAVE_PAPER or (_x not in tex and _x not in _supp))
need_true("expFin the README names no run log left out of the repository", "expD_run.log" not in _rm and "expB6w_run.log" not in _rm)
for _x in ("contributes little", "near-one-hot", "over-discount", "closes both gaps", "certify", "exactly what the fitted", "physical consequence", "PRE-REGISTERED", "pre-registered", "credal bounds", "every reported metric", "for all reported runs", "±0.001", "and both failures show"):
    need_true(f"expF retired from the README: {_x}", _x not in _rm)
_xpy = [_f for _f in glob.glob(os.path.join(HERE, "*.py")) if os.path.basename(_f) != os.path.basename(__file__)]
need_true("expF scripts scanned", len(_xpy) > 40)
for _xfn in _xpy:
    need_true(f"expF {os.path.basename(_xfn)} no longer says pre-registered or pre-registration", "pre-regist" not in open(_xfn, encoding="utf-8").read().lower())
for _xfn in glob.glob(os.path.join(RES, "*_summary.md")):
    need_true(f"expF {os.path.basename(_xfn)} no longer says pre-registered or pre-registration", "pre-regist" not in open(_xfn, encoding="utf-8", errors="replace").read().lower())
need_true("expFin the README says neither pre-registered nor pre-registration", "pre-regist" not in _rm.lower())
need_true("expFin main.tex and the supplement say neither pre-registered nor pre-registration", not HAVE_PAPER or ("pre-regist" not in tex.lower() and "pre-regist" not in _supp.lower()))
need_true("expFin exp_eo_independent.py says the run stands outside the rule fixed in advance", "outside the rule fixed in advance" in open(os.path.join(HERE, "exp_eo_independent.py"), encoding="utf-8").read())
_xpl = open(os.path.join(HERE, "pipeline.py"), encoding="utf-8").read()
need_true("expF pipeline header says what the manuscript says", all(_x not in _xpl for _x in ("Noisy-MAX for T", "Noisy-MAX T", "fixed a\npriori", "post\nhoc", "exactly the same N")) and "every BN method sees the\nsame N labelled scenarios and the same sensor-calibration split" in _xpl.replace("\r\n", "\n"))
need_true("expF significance script says the ESS is fixed", "ESS selected" not in open(os.path.join(HERE, "exp_significance.py"), encoding="utf-8").read())
for _x in ("python exp_calibsize_ci.py", "python exp_ece_bins.py", "python exp_ess_sensitivity.py", "python exp_prior_deployed_ess.py", "python exp_reason_checks.py"):
    need_true(f"expF README run list: {_x}", _x in _rm)
for _x in ("expTS_rf_report.csv", "expRS_summary.md", "expRS_weights.csv", "expED_summary.md", "exp_constraints_raw.csv"):
    need_true(f"expF README file list: {_x}", _x in _rm)
if DRAFT or not HAVE_PAPER:
    print(f"expF checks run: {_n_expd - _n0xf}")
assert _n_expd - _n0xf > 0

# ---- what a value's population is, labels, scope: second set of controls -----
_n0yg = _n_expd
# the RF channel's upper bound
_yb = _rt("expRS_rf_bound.csv")
_yv = lambda a, k: _yb[_yb.variant == a].sort_values("rep")[k].values
_yp = lambda a, b, k: (lambda _d: (float(_d.mean()), float(_st.ttest_1samp(_d, 0.0).pvalue)))(_yv(a, k) - _yv(b, k))
_ygx, _yrc = _rt("expG_rf_e2e_raw.csv").sort_values("rep"), _rt("expRC_raw.csv").sort_values("rep")
need_true("expG bound: hard and soft are the rows of the main run, none is the control", len(_yb) == 80 and all(np.allclose(_yv(_a, _k), _r[_k].values, atol=1e-9) for _a, _r in (("hard", _ygx[_ygx.rf_likelihood == "calib-hard"]), ("soft", _ygx[_ygx.rf_likelihood == "calib-soft"]), ("none", _yrc[_yrc.variant == "no-RF"])) for _k in ("macroAUC", "brier", "ece", "acc")))
_yrn, _yrh = (_yp("error-free split", "none", "macroAUC"), _yp("error-free split", "none", "brier")), (_yp("error-free split", "hard", "macroAUC"), _yp("error-free split", "hard", "brier"))
need_true("expG bound: the hard report takes most of it, and what is left is small but resolved", _yp("hard", "none", "macroAUC")[0] / _yrn[0][0] > 0.5 and _yp("hard", "none", "brier")[0] / _yrn[1][0] > 0.5 and all(_q[1] < 1e-3 for _q in _yrh))
_ysrc = open(os.path.join(HERE, "exp_reason_checks.py"), encoding="utf-8").read()
need_true("expG the script says error-free split", "error-free split" in _ysrc and "emission revealed" not in _ysrc and "emission state revealed" not in _ysrc)
# the split on what the scenarios draw: the file against the CNN outputs
_ysp = _rt("expRS_split.csv")
_ywb = 1 - float(_gtw.cpts["cls"].table[2]) * float(_gtw.cpts["emit"].table[0])
_yim = np.load(os.path.join(RES, "expI_cnn_dronerf.npz"), allow_pickle=False)
need_true("expG split file: all 20 runs", len(_ysp) == 20 and abs(_ywb - 0.8075) < 1e-9)
for _, _q in _ysp.iterrows():
    _ys, _yyy = (_xcc["test_softmax"], _xcc["test_y"]) if _q.proto == "P0" else (_yim[f"{_q.proto}_{_q.run}__test_softmax"], _yim[f"{_q.proto}_{_q.run}__te_y"])
    _ysl = np.maximum(_ys[:, 0], 1 - _ys[:, 0])
    _yn = np.bincount(_yyy, minlength=4)
    _yw = np.where(_yyy == 0, _ywb / _yn[0], ((1 - _ywb) / 3) / _yn[_yyy])
    _yo = np.argsort(_ysl)
    need_true(f"expG split file: {_q.proto} {_q.run}", abs(_yw.sum() - 1) < 1e-9 and _yn[0] == _q.background_spectrograms and _yn[0] == 20 * _q.background_segments and ((_yyy == 0) & (_ys.argmax(1) != 0)).sum() == _q.background_errors
              and abs(_ysl[_yo][np.searchsorted(np.cumsum(_yw[_yo]), 0.5)] - _q.scenario_median) < 1e-6 and abs(_yw[_ysl < 0.9].sum() - _q["scenario_below_0.9"]) < 1e-9 and abs((_yw * _ys.max(axis=1)).sum() - _q.scenario_mean_top) < 1e-6)
need_supp("expG soft evidence, the mass among the models", "Moving the softmax's mass among the drone models changes $P(T)$ by under $10^{-7}$.")
need_tex("expG soft evidence, the reason and the bound", f"because only the softmax's mass on background moves $P(T)$ and the hard report already takes most of what the channel can give: error-free reports add ${_yrn[0][0]:+.4f}$ macro-AUC and ${_yrn[1][0]:+.4f}$ Brier over the control, only ${_yrh[0][0]:+.4f}$ and ${_yrh[1][0]:+.4f}$ over (ii).")
need_supp("expG soft evidence, the population", f"The softmax is not certain: on what the scenarios draw, its top probability averages ${_ysp.scenario_mean_top.min():.2f}$ to ${_ysp.scenario_mean_top.max():.2f}$ across runs and the split falls below $0.9$ in ${100 * _ysp['scenario_below_0.9'].min():.0f}$ to ${100 * _ysp['scenario_below_0.9'].max():.0f}\\%$.")
need_supp_later("expG the test background of every split, in the supplement", f"is tested in each split on only ${_ysp.background_segments.min()}$ to ${_ysp.background_segments.max()}$ segments (${_ysp.background_spectrograms.min()}$ to ${_ysp.background_spectrograms.max()}$ spectrograms).")
need_supp_later("expG VII-C, in the supplement", "yields $84.9\\%$, higher than the segment split, with a different training share and checkpoint rule as well.")
# the weights scaled
_yfr = _rt("expF_rel_raw.csv")
_yfr = _yfr[(_yfr.part == "miscal") & (_yfr.world == "nominal")]
_yrs = _rt("expRS_rel_scale.csv")
_yrs["mult"] = _yrs["mult"].astype(str)
_ybs = _yrs[_yrs["mult"] == "1.0"].sort_values("seed")
need_true("expG scale: 20 seeds, and factor 1 is the sweep's own row", len(_ybs) == 20 and all(np.allclose(_ybs[_k].values, _yfr[_yfr.mult == 1.0].sort_values("seed")[_k].values, atol=1e-9) for _k in ("acc", "macroAUC", "brier", "ece")))
need_true("expG scale: factor 0 is the reports left out", all(np.allclose(_yrs[_yrs["mult"] == "0.0"].sort_values("seed")[_k].values, _yrs[_yrs["mult"] == "reports hidden"].sort_values("seed")[_k].values, atol=1e-9) for _k in ("acc", "macroAUC", "brier", "ece")))
_yds = lambda _q, _k: (lambda _d: (float(_d.mean()), float(_st.ttest_1samp(_d, 0.0).pvalue)))(_q.sort_values("seed")[_k].values - _ybs[_k].values)
_yhalf, _yquart = _yfr[_yfr.mult == 0.5], _yrs[_yrs["mult"] == "0.25"]
need_true("expG scale: halving costs Brier and macro-AUC, both below 1e-8", _yds(_yhalf, "brier")[0] > 0 and _yds(_yhalf, "macroAUC")[0] < 0 and _yds(_yhalf, "brier")[1] < 1e-8 and _yds(_yhalf, "macroAUC")[1] < 1e-8)
need_true("expG scale: within 50% ECE moves by at most 0.004", all(abs(_yds(_yfr[_yfr.mult == _m], "ece")[0]) <= 0.004 for _m in (0.5, 0.75, 1.25, 1.5)))
need_tex("expG scale: the sentence", f"but halving them costs ${_yds(_yhalf, 'brier')[0]:.3f}$ Brier and ${-_yds(_yhalf, 'macroAUC')[0]:.3f}$ macro-AUC (both $p<10^{{-8}}$), and at a quarter of the fitted value ECE rises by ${_yds(_yquart, 'ece')[0]:.3f}$ ($p{{=}}{_yds(_yquart, 'ece')[1]:.3f}$), so ECE alone understates what a mis-set weight costs.")
# the root marginals
_yro = _rt("expRS_roots.csv")
_ysc = _rt("expS_scarcity_raw.csv")
_YMET = ("acc", "macroF1", "macroAUC", "brier", "ece", "cwece", "far@pd0.9")
_yrr = lambda _m, _n, _r: _yro[(_yro.method == _m) & (_yro.N == _n) & (_yro.roots == _r)].sort_values("rep")
need_true("expG roots: the specification rows are the main study's", all(np.allclose(_yrr(_m, _n, "specification")[_k].values, _ysc[(_ysc.method == _m) & (_ysc.N == _n)].sort_values("rep")[_k].values, atol=1e-9) for _m, _n in (("Proposed", 10), ("Proposed", 50), ("Proposed", 200), ("B1-MLE", 50)) for _k in _YMET))
_ybig = max(abs((_yrr(_m, _n, "calibration split")[_k].values - _yrr(_m, _n, "specification")[_k].values).mean()) for _m, _n in (("Proposed", 10), ("Proposed", 50), ("Proposed", 200), ("B1-MLE", 50)) for _k in _YMET)
need_true("expG roots: seven metrics, the largest change on the calibration split is under 0.002", 0.001 < _ybig < 0.002)
need_true("expG roots: the training rows carry the class and the emission state, and the table of N uses them", {"cls", "emit"} <= set(_dio.generate_dataset(_gtw, 2, seed=0)[0]) and set(_ms.PARENTS["N"]) == {"cls", "emit"})
need_true("expG roots: MAP-EM rows leave them out", 'KEEP = ["v", "rdot", "d", "z", "Rr", "Re", "Ra", "Rf", "T"]' in open(os.path.join(HERE, "exp_mapem.py"), encoding="utf-8").read())
need_tex("expG roots: the sentence", "are supplied to every compared model from the generative specification (Supplementary Material, Sec.~S4).")
need_supp_later("expG roots, in the supplement", "counting them on the calibration split instead of reading them from the specification changes no metric by more than $0.002$.")
# the adverse subset
_yad = _rt("expRS_adverse.csv").sort_values(["severity", "seed"])
_ycr = _rt("expF_corr_raw.csv")
_ycr = _ycr[_ycr.fusion == "naive-CI"].sort_values(["severity", "seed"])
need_true("expG adverse subset: accuracy and ECE are the rows of the main file", len(_yad) == len(_ycr) == 100 and np.allclose(_yad.acc.values, _ycr.acc_adv.values, atol=1e-12) and np.allclose(_yad.ece.values, _ycr.ece_adv.values, atol=1e-9))
_ysev = _yad.groupby("severity")[["mean_top", "acc", "acc_reports_hidden", "ece"]].mean()
need_true("expG adverse subset: confidence holds, accuracy falls, the gap grows, reports hidden does not move", _ysev.mean_top.max() - _ysev.mean_top.min() < 0.01 and (_ysev.acc.diff().dropna() < 0).all() and ((_ysev.mean_top - _ysev.acc).diff().dropna() > 0).all() and _ysev.acc_reports_hidden.max() - _ysev.acc_reports_hidden.min() < 0.005)
need_tex("expG adverse subset: the sentence", f"the failure is overconfidence (mean confidence stays near ${_ysev.mean_top.iloc[-1]:.2f}$ while accuracy falls from ${_ysev.acc.iloc[0]:.2f}$ to ${_ysev.acc.iloc[-1]:.2f}$), and kinematic and context evidence alone holds ${_ysev.acc_reports_hidden.iloc[-1]:.2f}$ at every severity.")
# wording that now stands
for _lab, _sn in (("G3 what N counts", "the table of the non-cooperative node also uses each scenario's true class and emission state, which the world knows by construction)"),
                  ("h1", "per-class paired DeLong tests are significant in $5$--$17$ of $20$ replications depending on the class, and the scenario-level paired bootstrap CI of the delta excludes zero in $18/20$ replications."),
                  ("G3 MAP-EM", "so $H,N,C$, $\\mathrm{cls}$, and $\\mathrm{emit}$ are latent)"),
                  ("a sentence", "and scores RF silence by a hand rule; in it, every report names the true class with probability $0.94$ (emission ignored), and because that diagonal is a design choice,"),
                  ("a sentence", "and does not tie those scores to the actual detection,"),
                  ("a sentence", "In the learned model ($N{=}200$, seed 0, Sec.~\\ref{sec:simresults}) the RF ``none'' barely lowers"),
                  ("second sentence", "The evidence as a whole concentrates the threat node on its upper levels:"),
                  ("a sentence", "so their rows vary only through the prior, calibration, and test draws;"),
                  ("a sentence", "The discriminative baseline (B5), without the calibration split, trails on the same evidence and scenarios at every size, at $N{=}400$ still by three points of accuracy and $0.027$ ECE, consistent with the premise of Sec.~\\ref{sec:intro}."),
                  ("a sentence", "(no input node of \\cite{song2026base} is an EO/IR or acoustic report; it names EO/IR as a source of its kinematic and RCS features)."),
                  ("VII-D", "the high class is the \\emph{best} calibrated on average (Supplementary Material, Sec.~S4), so the decomposition matters more than the aggregate."),
                  ("I-B", "Labeled C-UAS threat data are scarce (no public dataset we found labels threat, Sec.~\\ref{sec:pubdata}), so")):
    need_tex(f"expG {_lab}", _sn)
_yhc = _rt("expR_reliability_classes.csv")
_yhs = _yhc.pivot_table(index="class", columns="bin", values="n", aggfunc="sum")
need_true("expG the high class has most of its mass in the near-zero bin", (_yhs.loc["high"] / _yhs.loc["high"].sum())["(0.00,0.10]"] > 0.5)
need_true("expG credal: gone from the sensitivity script and its summary", "credal" not in open(os.path.join(HERE, "exp_sensitivity.py"), encoding="utf-8").read() and "credal" not in open(os.path.join(RES, "expT_summary.md"), encoding="utf-8").read())
need_true("expG credal: the summary has the envelope lines and the rows the figure reads", "## one-at-a-time envelope (top-5, +-0.1)" in open(os.path.join(RES, "expT_summary.md"), encoding="utf-8").read() and "invariant over the envelope at the 0.5 decision level" in open(os.path.join(RES, "expT_summary.md"), encoding="utf-8").read())
# the old wording stays out
for _x in ("revealing the emission state", "that split is nearly certain", "conservative by construction", "not identifiable from a labeled scenario file", "more leakage-optimistic", "because most of its mass", "its measurement probabilities", "kinematic evidence chain", "overconfident agreement", "genuinely scarce", "supports the premise", "take no EO/IR", "the first split tests", "only through the prior and test draws", "credal bounds", "credal set"):
    need_true(f"expG retired from main.tex and the supplement: {_x}", not HAVE_PAPER or (_x not in tex and _x not in _supp))
    need_true(f"expG retired from the README: {_x}", _x not in _rm)
for _x in ("are the rows", "Supervised training rows carry the threat", "changes no metric by"):
    pass
for _x in ("Supervised training rows carry the threat\nlevel, the three latent states, and the true class and emission state", "each split tests it on 5 to 10 segments", "only +0.0009 and -0.0014 over the hard report", "halving the weights costs 0.008 Brier and 0.005 macro-AUC"):
    need_true(f"expG README: {_x[:40]}", _x in _rm)
for _x in ("expRS_rf_bound.csv", "expRS_split.csv", "expRS_rel_scale.csv", "expRS_roots.csv", "expRS_adverse.csv"):
    need_true(f"expG README file list: {_x}", _x in _rm)
if DRAFT or not HAVE_PAPER:
    print(f"expG checks run: {_n_expd - _n0yg}")
assert _n_expd - _n0yg > 0

# ---- what each comparison is against, where the prior stops carrying the result, what the abstract adds -----
_n0zh = _n_expd
_zp = lambda a, b: (lambda _d: (float(_d.mean()), float(_st.ttest_1samp(_d, 0.0).pvalue)))((lambda _e: _e[~np.isnan(_e)])(a - b))
_zv = lambda v, s, k: _zsel(v, s)[k].values
_zMET = ("acc", "macroAUC", "brier", "ece")
_zref = _rt("expS_fusion_raw.csv")
need_true("expH fusion: six variants, two subsets, 20 replications", len(_zF) == 240 and _zF.groupby(["variant", "subset"]).size().eq(20).all())
for _v, _c, _zcs in (("abstract", "all (abstract model)", "_subset_silent-drone_abstract"), ("calibrated", "all (calibrated)", "_subset_silent-drone_calibrated"), ("calib+rel", "all (calib+reliab)", "_subset_silent-drone_calib+reliab")):
    for _s, _znm in (("all", _c), ("radio-silent", _zcs)):
        _q = _zref[_zref.config == _znm].sort_values("rep")
        need_true(f"expH fusion: {_v}, {_s} are the rows of the main study", all(np.allclose(_zv(_v, _s, _k), _q[_k].values, atol=1e-12, equal_nan=True) for _k in _zMET))
need_true("expH fusion: the controls are what the text says", "(0.47, 0.47, 0.06) if (c == 2 and e == 0) else (0.03, 0.03, 0.94)" in _ysrc and '("no RF", {"sensors_on": no_rf, "sensor_cpts": est, "alpha": None})' in _ysrc and list(_ms.STATES["Rf"])[2] == "none")
_zsa = _zp(_zv("calibrated", "radio-silent", "acc"), _zv("no RF", "radio-silent", "acc"))
_zna = _zp(_zv("no RF", "radio-silent", "acc"), _zv("abstract", "radio-silent", "acc"))
need_true("expH radio silence: the abstract model is worse than no channel, calibrated level with it", _zna[0] > 0.08 and _zna[1] < 1e-6 and _zsa[1] > 0.5)
need_tex("expH the abstract model against no RF channel", f"is worse than no RF channel on radio-silent drones (accuracy ${_zv('abstract', 'radio-silent', 'acc').mean():.3f}$ against ${_zv('no RF', 'radio-silent', 'acc').mean():.3f}$, calibrated ${_zsa[0]:+.3f}$, $p{{=}}{_zsa[1]:.2f}$;")
need_true("expH no apology for the stand-in", not HAVE_PAPER or ("removes the harm" not in tex and "what the stand-in gets wrong" not in tex))
_zdp = {_k: _zp(_zv("calib+rel", "all", _k), _zv("no RF+rel", "all", _k)) for _k in _zMET}
_zds = {_k: _zp(_zv("calib+rel", "radio-silent", _k), _zv("no RF+rel", "radio-silent", _k)) for _k in _zMET}
need_true("expH the deployed RF channel: accuracy and ECE unresolved overall, Brier resolved on radio-silent drones", _zdp["acc"][1] > 0.05 and _zdp["ece"][1] > 0.05 and _zds["brier"][1] < 1e-5 and _zdp["macroAUC"][1] < 1e-5 and _zdp["brier"][1] < 1e-5)
_zdn = _zp(_zv("calibrated", "radio-silent", "macroAUC"), _zv("no RF", "radio-silent", "macroAUC"))
need_true("expH the fitted discount is on both sides of this contrast", _zdp == {_k: _zp(_zv("calib+rel", "all", _k), _zv("no RF+rel", "all", _k)) for _k in _zMET})
need_true("expH the deployed channel: macro-AUC and Brier p < 1e-8", _zdp["macroAUC"][1] < 1e-8 and _zdp["brier"][1] < 1e-8)
need_tex("expH the deployed channel", f"Its RF channel, read through the latent emission state, adds information over the same fusion without RF: with the fitted discount on both sides, ${_zdp['macroAUC'][0]:+.3f}$ macro-AUC and ${_zdp['brier'][0]:+.3f}$ Brier overall (both $p<10^{{-8}}$), with accuracy and ECE unresolved ($p{{=}}{_zdp['acc'][1]:.2f}$, ${_zdp['ece'][1]:.2f}$), and on radio-silent drones ${_zds['brier'][0]:+.3f}$ Brier ($p<10^{{-5}}$) with macro-AUC ${_zds['macroAUC'][0]:+.3f}$ ($p{{=}}{_zds['macroAUC'][1]:.2f}$; ${_zdn[0]:+.3f}$, $p{{=}}{_zdn[1]:.3f}$, without the discount).")
_zh = {_k: _zp(_zv("calibrated", "all", _k), _zv("hand-set emission-aware", "all", _k)) for _k in _zMET}
_zhs = {_k: _zp(_zv("calibrated", "radio-silent", _k), _zv("hand-set emission-aware", "radio-silent", _k)) for _k in _zMET}
need_true("expH hand-set: ahead overall (p < 1e-5), behind on radio-silent drones (p <= 0.014)", _zh["macroAUC"][0] > 0 and _zh["brier"][0] < 0 and _zh["ece"][0] < 0 and all(_zh[_k][1] < 1e-5 for _k in ("macroAUC", "brier", "ece")) and _zhs["acc"][0] < 0 and _zhs["macroAUC"][0] < 0 and _zhs["brier"][0] > 0 and all(_zhs[_k][1] <= 0.014 for _k in ("acc", "macroAUC", "brier")))
need_tex("expH the hand-set table", f"Against a hand-set emission-aware table (the abstract diagonal, with RF reporting ``none'' at $0.06$ for emitting drones and at $0.94$ otherwise), estimated matrices without the discount gain overall (${_zh['macroAUC'][0]:+.3f}$ macro-AUC, ${_zh['brier'][0]:+.3f}$ Brier, ${_zh['ece'][0]:+.3f}$ ECE, all $p<10^{{-5}}$) but not on radio-silent drones (accuracy ${_zhs['acc'][0]:+.3f}$, macro-AUC ${_zhs['macroAUC'][0]:+.3f}$, Brier ${_zhs['brier'][0]:+.3f}$, all $p\\le0.014$).")
need_tex("expH Table 6 caption", "The no-RF and hand-set (emission-aware, diagonal $0.94$) rows use no reliability discount.")
# why the deployed weight stays
_zdc = {_k: _zp(_zv("calib+rel", "all", _k), _zv("calibrated", "all", _k)) for _k in _zMET}
_zdcs = _zp(_zv("calib+rel", "radio-silent", "brier"), _zv("calibrated", "radio-silent", "brier"))
need_true("expH the average effect is small, favorable on Brier, and costs on radio-silent drones", -0.002 < _zdc["brier"][0] < 0 and _zdc["brier"][1] < 1e-3 and _zdc["ece"][0] < 0 and _zdcs[0] > 0 and _zdcs[1] < 1e-3)
need_true("expH the average effect sentence left the body (Sec. S4 keeps the values)", not HAVE_PAPER or "Its small average effect" not in tex)
_zb6 = _rt("expB6_raw.csv")
need_true("expH rule file: the rule and Proposed are the rows of the study of the rule", np.allclose(_zr6.auc_high.values, _zb6[(_zb6.method == "B6-pub") & (_zb6.N == 0)].sort_values("rep").auc.values, atol=1e-12) and np.allclose(_zr6.brier_high.values, _zb6[(_zb6.method == "B6-pub") & (_zb6.N == 0)].sort_values("rep").brier_bin.values, atol=1e-12) and all(np.allclose(_zrm("Proposed", _n).auc_ge_medium.values, _zb6[(_zb6.method == "Proposed") & (_zb6.N == _n)].sort_values("rep").auc_ge_medium.values, atol=1e-12) for _n in (10, 20, 50, 100, 200, 400)))
need_true("expH rule file: B3 is the uninformative MAP of the baselines", all(np.allclose(_zrm("B3", _n).macroAUC.values, _zbq("uninformative MAP", _n).macroAUC.values, atol=1e-12) for _n in (10, 50, 100, 200, 400)) and "g.map_cpt(n, data, unif[n], pl.ESS_FIXED)" in _ysrc)
_zb3 = {_n: _zp(_zrm("B3", _n).auc_high.values, _zr6.auc_high.values) for _n in (10, 20, 50, 100, 200, 400)}
need_true("expH B3: no better than the rule on AUC at N <= 20, ahead from N = 50", _zb3[10][1] > 0.05 and _zb3[20][1] > 0.05 and all(_zb3[_n][0] > 0 and _zb3[_n][1] < 1e-10 for _n in (50, 100, 200, 400)))
need_supp_later("expH B3, in the supplement", f"With a uniform prior mean, a network (B3, which keeps the graph and the latent labels) does not beat the published rule on AUC at $N{{\\le}}20$ (${_zrm('B3', 10).auc_high.mean():.3f}$ and ${_zrm('B3', 20).auc_high.mean():.3f}$ against ${_zr6.auc_high.mean():.3f}$; $p{{=}}{_zb3[10][1]:.2f}$, ${_zb3[20][1]:.3f}$) and passes it from $N{{=}}50$ (${_zb3[50][0]:+.3f}$, $p<10^{{-10}}$).")
_zmo = float(_zr6.ge_medium_above.sum() / _zr6.n_above.sum())
_zmb, _zmc = float(_zr6.brier_ge_medium.mean()), float((_zr6.base_ge_medium * (1 - _zr6.base_ge_medium)).mean())
need_true("expH the other reading: its Brier is a constant forecast's, within 0.001", abs(_zmb - _zmc) < 0.001 and int(_zr6.n_above.sum()) > 1000)
_zpm = [float(_zrm("Proposed", _n).auc_ge_medium.mean()) for _n in (10, 20, 50, 100, 200, 400)]
need_supp("expH the other reading", f"Read as $P(T{{\\ge}}\\text{{medium}})$ instead, the rule's scores above $0.75$ are right ${100 * _zmo:.0f}\\%$ of the time and its Brier score (${_zmb:.3f}$) equals a constant forecast's, with AUC ${_zr6.auc_ge_medium.mean():.3f}$ against ${min(_zpm):.3f}$--${max(_zpm):.3f}$ for Proposed.")
_zbp, _zbe = _zp(_zbq("Proposed", 10).ece.values, _zbq("B0-as-prior MAP", 10).ece.values), _zp(_zbq("expert MAP (full table)", 10).ece.values, _zbq("B0-as-prior MAP", 10).ece.values)
need_tex("expH what the better prior adds", f"What the elicited prior adds over it is calibration ($\\Delta$ECE ${_zbp[0]:+.3f}$, $p{{=}}{_zbp[1]:.3f}$; ${_zbe[0]:+.3f}$, $p{{=}}{_zbe[1]:.3f}$ with the same full-table structure),")
need_true("expH B0 comes from the world's response family with rounded weights", "def heuristic_cpts" in open(os.path.join(HERE, "model_spec.py"), encoding="utf-8").read() and all(_x in _ms.PARENTS for _x in ("H", "N", "C", "T")))
_zq = {_k: _zp(_zpr(0.25, 10)[_k].values, _zml(50)[_k].values) for _k in _zMET}
_zq50 = {_k: _zp(_zpr(0.25, 50)[_k].values, _zml(50)[_k].values) for _k in _zMET}
need_true("expH half effects at N = 10: worse than MLE at N = 50 on accuracy, Brier and ECE, p <= 0.02, ahead on macro-AUC", _zq["acc"][0] < 0 and _zq["brier"][0] > 0 and _zq["ece"][0] > 0 and all(_zq[_k][1] <= 0.02 for _k in ("acc", "brier", "ece")) and _zq["macroAUC"][0] > 0)
need_true("expH half effects at N = 50: ahead of MLE at N = 50 on macro-AUC, Brier and ECE, p <= 0.025, accuracy unresolved", _zq50["macroAUC"][0] > 0 and _zq50["brier"][0] < 0 and _zq50["ece"][0] < 0 and all(_zq50[_k][1] <= 0.025 for _k in ("macroAUC", "brier", "ece")) and _zq50["acc"][1] > 0.05)
need_tex("expH half effects", f"With the right direction but half the true effects ($\\lambda{{=}}0.25$), the model at $N{{=}}10$ keeps a macro-AUC lead over MLE at $N{{=}}50$ (${_zq['macroAUC'][0]:+.3f}$) but trails by ${-_zq['acc'][0]:.3f}$ accuracy, ${_zq['brier'][0]:.3f}$ Brier, and ${_zq['ece'][0]:.3f}$ ECE ($p\\le0.02$), though at $N{{=}}50$ it still leads MLE at $N{{=}}50$ on macro-AUC, Brier, and ECE ($p\\le0.025$).")
_zq5 = {_n: _zp(_zpr(0.5, _n).brier.values, _zml(_n).brier.values) for _n in (10, 50, 200, 400)}
need_true("expH no believed trend: Brier worse than MLE at N = 10, 200, 400 (p <= 0.03), not at 50", all(_zq5[_n][0] > 0 and _zq5[_n][1] <= 0.03 for _n in (10, 200, 400)) and _zq5[50][1] > 0.05)
need_tex("expH no trend", "With no believed trend ($\\lambda{=}0.5$) its Brier is worse than MLE's at $N{=}10$, $200$, and $400$ ($p\\le0.03$); Supplementary Material, Table~S1 gives the grid.")
need_true("expH lambda: 0.25 halves the slope, 0.5 removes it", "(1 - lam) * x + lam * (1 - x)" in open(os.path.join(HERE, "exp_prior_ess.py"), encoding="utf-8").read())
need_true("expH the status of claim (iv) no longer quotes the half-effects prior", not HAVE_PAPER or "already loses the headline result" not in tex)
# Table S1
_zfm = lambda x: f"{x:.3f}"[1:] if x < 1 else f"{x:.3f}"
need_true("expH the grid file is the raw file's means", all(abs(_q[_k] - (_zml(int(_q.N)) if _q.method == "MLE" else _zpr(_q.lam, int(_q.N)))[_k].mean()) < 1e-12 for _, _q in _zG.iterrows() for _k in _zMET) and len(_zG) == 24)
for _lab, _zgr in [("MLE", _zG[_zG.method == "MLE"])] + [(f"$\\lambda{{=}}{_l:g}$", _zG[(_zG.method == "Proposed") & (_zG.lam == _l)]) for _l in (0.0, 0.25, 0.5, 0.75, 1.0)]:
    need_supp(f"expH Table S1: {_lab}", _lab + " & " + " & ".join(" & ".join(_zfm(float(_v)) for _v in _zgr[_k].values) for _k in _zMET) + " \\\\")
_zpf = lambda p: f"$p{{=}}{p:.2f}$" if p >= 0.0095 else ("$p{=}" + f"{p:.3f}" + "$" if p >= 0.00095 else "$p<10^{-3}$")
for _l in (0.0, 0.25, 0.5, 0.75, 1.0):
    _c = [_zp(_zpr(_l, 10)[_k].values, _zml(50)[_k].values) for _k in _zMET]
    need_supp(f"expH Table S1: lambda {_l:g}, N = 10 minus MLE at N = 50", f"$\\lambda{{=}}{_l:g}$ & " + " & ".join(f"\\multicolumn{{4}}{{c}}{{${_v[0]:+.3f}$ ({_zpf(_v[1])})}}" for _v in _c) + " \\\\")
need_true("expH the supplement's sections, in order", not HAVE_PAPER or re.findall(r"\\section\{([^}]*)\}", _supp) == ["Dependence on the world's noise level", "When constraints help, and on which metric", "The reversed-prior grid at the deployed ESS", "Further detail of the article's sections", "The comparators' free values", "Analyses added after the main results", "Figures"])
need_tex("expH the body points to S1 of the supplement's table", "Supplementary Material, Table~S1 gives the grid.")
need_tex("expH the abstract: what is added", "We add a proposition on the constrained estimate's meaning and limit, a data check on additivity and monotonicity, a latent emission state replacing the hand rule on radio silence, and an analysis of reliability weights.")
need_tex("expH C1", "that tests additivity and monotonicity on the training data and falls back")
need_true("expH the abstract and C1 hold no limitation clauses", not HAVE_PAPER or ("loses the first result" not in tex and "with little power up to" not in tex))
_zas = open(os.path.join(RES, "expW_add_summary.md"), encoding="utf-8").read()
_zam = re.search(r"\| nonadd-0\.5 \| 50 \| [0-9.]+ \| [0-9.]+ \| [0-9.]+ \| - \| [-+0-9.e()]+ \([-+0-9.e]+\) \| \+([0-9.]+) \(([0-9.e+-]+)\) \| [0-9.]+ \| \+([0-9.]+) \(([0-9.e+-]+)\) \|", _zas)
need_true("expH the row of weak S1 at N = 50 in the summary of the adaptive rule", _zam is not None and _xfl("nonadd-0.5", 50) == 7)
need_tex("expH VII-G", f"At $N{{=}}50$ the test flags a weak violation in {_xfl('nonadd-0.5', 50)} of 20 replications, and the switch is worse than the full table on Brier ($+{_zam.group(1)}$, $p{{=}}{float(_zam.group(2)):.3f}$) and than the fixed model on ECE ($+{_zam.group(3)}$, $p{{=}}{float(_zam.group(4)):.3f}$).")
need_tex("expH VII-A", "In the learned model ($N{=}200$, seed 0, Sec.~\\ref{sec:simresults})")
_zc30 = {_k: _zp(_zk30[_k].values, _zka[_k].values) for _k in _zMET}
need_true("expH Brier resolved, macro-AUC and ECE not", _zc30["brier"][1] < 0.01 and _zc30["macroAUC"][1] > 0.05 and _zc30["ece"][1] > 0.05)
need_tex("expH a sentence", f"$30$ labeled scenarios already improve on the abstract model on Brier (${_zc30['brier'][0]:+.3f}$, $p{{=}}{_zc30['brier'][1]:.3f}$), with AUC and ECE unresolved ($p{{=}}{_zc30['macroAUC'][1]:.3f}$, ${_zc30['ece'][1]:.3f}$), and")
need_tex("expH the prior's intercept", "by perturbing its weights with $18\\%$ multiplicative Gaussian noise and its intercept with additive noise of s.d.\\ $0.09$,")
need_true("expH the code's intercept noise is 0.09", "bias = bias + noise * 0.5 * rng.standard_normal()" in open(os.path.join(HERE, "model_spec.py"), encoding="utf-8").read() and "def expert_prior_cpts(rng, noise=0.18):" in open(os.path.join(HERE, "model_spec.py"), encoding="utf-8").read())
need_tex("expH kappa", "a sharpness $\\kappa>0.3$ (the fit's lower bound), and per-parent")
need_true("expH the code's lower bound is 0.3", "scale = np.logaddexp(0.0, x[o]) + 0.3" in open(os.path.join(HERE, "grounding.py"), encoding="utf-8").read())
_zbs = _rt("expBS_paired.csv")
_zbs = _zbs[_zbs.world == "synthetic"]
need_true("expH signed differences exclude zero in both bins, size only in the top", bool(_zbs[_zbs.bin_lo == 0.5].diff_excludes_0.iloc[0]) and bool(_zbs[_zbs.bin_lo == 0.75].diff_excludes_0.iloc[0]) and bool(_zbs[_zbs.bin_lo == 0.75].size_excludes_0.iloc[0]) and not bool(_zbs[_zbs.bin_lo == 0.5].size_excludes_0.iloc[0]))
# the FAR paragraph and the status of the claims
_zfb = {_n: _zp(_zsq("Proposed", _n)["far@pd0.9"].values, _zsq("B0-heuristic", _n)["far@pd0.9"].values) for _n in (10, 20, 50, 100, 200, 400)}
_zfm2 = {_n: _zp(_zsq("Proposed", _n)["far@pd0.9"].values, _zsq("B1-MLE", _n)["far@pd0.9"].values) for _n in (10, 20, 50, 100, 200, 400)}
need_true("expH FAR below B0 at N = 20 (p < 0.05) and from N = 50 (p < 1e-3), not at N = 10", _zfb[20][1] < 0.05 and all(_zfb[_n][1] < 1e-3 for _n in (50, 100, 200, 400)) and _zfb[10][1] > 0.05)
need_tex("expH the FAR paragraph", f"the grounded model lowers $\\mathrm{{FAR}}$ below the heuristic surrogate B0 at $N{{=}}20$ (${_zfb[20][0]:+.3f}$, $p{{=}}{_zfb[20][1]:.2f}$) and from $N{{=}}50$ on (${max(_zfb[_n][0] for _n in (50, 100, 200, 400)):+.3f}$ to ${min(_zfb[_n][0] for _n in (50, 100, 200, 400)):+.3f}$, all $p<10^{{-3}}$),")
need_tex("expH against MLE", f"is far worse when data are scarce (${_zsq('B1-MLE', 10)['far@pd0.9'].mean():.3f}$ at $N{{=}}10$) but level from $N{{=}}200$ ($p{{=}}{_zfm2[200][1]:.2f}$, ${_zfm2[400][1]:.2f}$).")
need_true("expH FAR against MLE resolved up to N = 100, not from 200", all(_zfm2[_n][0] < 0 and _zfm2[_n][1] < 0.05 for _n in (10, 20, 50, 100)) and all(_zfm2[_n][1] > 0.05 for _n in (200, 400)))
_zb10 = {_k: _zp(_zsq("Proposed", 10)[_k].values, _zsq("B0-heuristic", 10)[_k].values) for _k in ("macroAUC", "ece", "far@pd0.9")}
_zb20 = {_k: _zp(_zsq("Proposed", 20)[_k].values, _zsq("B0-heuristic", 20)[_k].values) for _k in ("macroAUC", "ece", "far@pd0.9")}
need_true("expH against B0, resolved from N = 20 on all three, not at N = 10", all(_zb20[_k][1] < 0.05 for _k in _zb20) and all(_zb10[_k][1] > 0.05 for _k in _zb10))
need_tex("expH the status of claims (i) and (ii)", f"The evidence to this point supports claim (i) of Sec.~\\ref{{sec:experiments}} from $N{{=}}20$ on against B0 (at $N{{=}}10$, macro-AUC, ECE, and FAR have $p{{=}}{_zb10['macroAUC'][1]:.2f}$, ${_zb10['ece'][1]:.2f}$, ${_zb10['far@pd0.9'][1]:.2f}$), against uniform-mean full tables on macro-AUC and Brier, and on FAR against MLE up to $N{{=}}100$. It supports claim (ii) against single sensors on all but ECE, against the abstract model on all but macro-F1, and against fusion without RF on macro-AUC and Brier.")
need_tex("expH a sentence", "as the positive class and the score threshold set on the evaluated scenarios);")
need_true("expH the code sets it on the scores it evaluates", "thr_candidates = np.unique(score)[::-1]" in open(os.path.join(HERE, "metrics.py"), encoding="utf-8").read())
# with the runs at small N
need_true("expH 40 runs at small N, standard world, B = 199, the flag is p < 0.05", len(_zfs) == 40 and set(_zfs.world) == {"standard"} and sorted(set(_zfs.N)) == [10, 20] and ((_zfs.p < 0.05) == _zfs.flag).all() and "B_BOOT = 199" in open(os.path.join(HERE, "exp_additivity.py"), encoding="utf-8").read())
_zf10, _zf20 = int(_zfs[_zfs.N == 10].flag.sum()), int(_zfs[_zfs.N == 20].flag.sum())
need_true("expH the flags by size", _zf10 == 1 and _zf20 == 0 and _wA0 == 1 and _wflags(_waA, "standard", 1000)[0] == 1)
need_tex("expH a sentence", f"$N{{=}}10$) and in {_wnum[_wA0 + _zf10 + _zf20]} of {_wA0n + len(_zfs)} standard-world replications (one each at $N{{=}}10$ and $1,000$), and flags the $H$ constraint")
need_tex("expH a sentence", "and the high-threat call at an illustrative $0.5$ level are invariant over it,")
need_tex("expH Sec. IV-A", "nine for our threat node, of which eight are identifiable (the sharpness $\\kappa$ is confounded with the scale of $\\theta$ and $\\beta$, so we report the identifiable count), versus $54$ for the full table.")
need_true("expH accuracy with the reports hidden is the same at every severity", _yax.groupby("severity").acc_reports_hidden.mean().nunique() == 1 and f"{_yax.acc_reports_hidden.mean():.2f}" == "0.53")
need_tex("expH a sentence", f"and the correlated failures studied there leave the kinematic and context inputs error-free (alone they hold ${_yax.acc_reports_hidden.mean():.2f}$ accuracy at every severity).")
# the README's range of the re-run; the rest of the README
need_true("expH README: what the re-run is", "we have no other measurement across machines" in _rm)
for _x in (f"with no RF channel the radio-silent accuracy is {_zv('no RF', 'radio-silent', 'acc').mean():.3f} (calibrated {_zsa[0]:+.3f}, p={_zsa[1]:.2f})",
           "does not beat the published rule on AUC at N≤20", f"the rule's scores above 0.75 are right {100 * _zmo:.0f}% of the time",
           "With the right direction but half the true effects (lambda=0.25), the model at N=10 keeps a macro-AUC lead over MLE at N=50 (+0.021)", "at most 7 of 20 at N≤20 and, on a weak violation, at N=50", "in two of 100 standard-world replications",
           f"the data alone {_tsd.min():.3f} to {_tsd.max():.3f}, the prior alone {_tsr.min():.3f} to {_tsr.max():.3f}", "`replot_print_size.py` (Figs. 5 and S3)", "| Fig. S3 | `replot_print_size.py` |", "| Table 6, the no-RF and hand-set rows | `exp_reason_checks.py` | `expRS_fusion.csv` |"):
    need_true(f"expH README: {_x[:50]}", _x in _rm)
for _x in ("expRS_fusion.csv", "expRS_rule_b3.csv", "expRS_track_split.csv", "expRS_flags_small.csv", "expRS_lambda.csv"):
    need_true(f"expH README file list: {_x}", _x in _rm)
# the old wording stays out
for _x in ("raising radio-silent-drone accuracy", "equal knowledge", "equal data", "held equal", "training seeds 0", "with little power at", "replacing its hand-set"):
    need_true(f"expH retired from main.tex and the supplement: {_x}", not HAVE_PAPER or (_x not in tex and _x not in _supp))
    need_true(f"expH retired from the README: {_x}", _x not in _rm)
if DRAFT or not HAVE_PAPER:
    print(f"expH checks run: {_n_expd - _n0zh}")
assert _n_expd - _n0zh > 0

# ---- the hand-set table over its diagonal: what estimating the matrices buys, and at which level -----
_n0zi = _n_expd
need_true("expHS file: 33 tables, six subsets, 20 replications, three levels", len(_zH) == 20 * 33 * 6 + 60 and _zH.rep.nunique() == 20)
for _v, _d, _zwr in (("hand-set", 0.94, "hand-set emission-aware"), ("abstract", 0.94, "abstract"), ("calibrated", None, "calibrated")):
    for _s, _x in (("all", "all"), ("radio-silent drones", "radio-silent")):
        need_true(f"expHS the rows at 0.94 are those of expRS_fusion.csv: {_v}, {_s}", all(np.allclose(_zhv(_v, _d, _s)[_k].values, _zv(_zwr, _x, _k), atol=1e-12, equal_nan=True) for _k in _zH4))
_zkn = lambda n: _zck[(_zck.source == "estimated") & (_zck.calib_n == n)].sort_values("rep")
need_true("expHS calibrated is the row n = 300 of the calibration-size file", all(np.allclose(_zhv("calibrated", None, "all")[_k].values, _zkn(300)[_k].values, atol=1e-12) for _k in _zH4))
_zsig = open(os.path.join(HERE, "exp_significance.py"), encoding="utf-8").read()
need_true("expHS the table is what the text says", "(hi, hi, off) if (c == 2 and e == 0) else (lo, lo, d)" in _ysrc and '("hand-set", d, {"sensor_cpts": {**ab, "Rf": hand_rf(d)}, "alpha": None})' in _ysrc and '("no RF", d, {"sensors_on": no_rf, "sensor_cpts": ab, "alpha": None})' in _ysrc)
need_true("expHS no row of Table 7 uses a reliability discount", 'met, _ = mt.evaluate(cpts, test, sensor_mode="abstract")' in _zsig and "met, _ = mt.evaluate(cpts, test, sensor_cpts=None, alpha=None)" in _zsig and "met, _ = mt.evaluate(cpts, test, sensor_cpts=est_n, alpha=None)" in _zsig)
# the level
need_tex("expHS the level", f"That diagonal lies far above the sensors' estimated accuracy (${_zlev['Rr'].mean():.2f}$, ${_zlev['Re'].mean():.2f}$, ${_zlev['Ra'].mean():.2f}$ for radar, EO/IR, acoustic);")
need_true("expHS the table's 0.94 lies above every split's estimate", max(_v.max() for _v in _zlev.values()) < 0.94)
# the sweep, all scenarios
_zqh = lambda d, s, k: _zp(_zhv("calibrated", None, s)[k].values, _zhv("hand-set", d, s)[k].values)
_zti = [_zqh(_d, "all", _k) for _d in (0.70, 0.75, 0.80) for _k in ("acc", "macroAUC", "brier")]
need_true("expHS ties at 0.70 to 0.80 on accuracy, macro-AUC and Brier, p >= 0.18", min(_v[1] for _v in _zti) >= 0.18)
need_true("expHS the other diagonals lose on Brier, p <= 0.008", all(_zqh(_d, "all", "brier")[0] < 0 and _zqh(_d, "all", "brier")[1] <= 0.008 for _d in (0.55, 0.60, 0.65, 0.85, 0.90, 0.94, 0.99)))
_zt75 = _zqh(0.75, "all", "ece")
need_true("expHS at 0.75 the table's ECE is lower", _zt75[0] > 0 and _zt75[1] < 0.01)
need_true("expHS do no better: at 0.70 to 0.80 calibrated is not significantly ahead on any of the four metrics", all(not (_zqh(_d, "all", _k)[1] < 0.05 and (_zqh(_d, "all", _k)[0] > 0) == (_k in ("acc", "macroAUC"))) for _d in (0.70, 0.75, 0.80) for _k in _zH4))
_zmx = {_k: _zp(_zhv("estimated, RF row hand-set", 0.94, "all")[_k].values, _zhv("calibrated", None, "all")[_k].values) for _k in _zH4}
need_true("expHS with the other three estimated the RF row changes no overall metric, p >= 0.08", min(_v[1] for _v in _zmx.values()) >= 0.08)
need_tex("expHS the sweep", "swept from $0.55$ to $0.99$, the table ties the estimated matrices at $0.70$ to $0.80$ on accuracy, macro-AUC, and Brier ($p\\ge0.18$) and loses on Brier elsewhere ($p\\le0.008$), so estimating supplies the level without choosing it (Supplementary Material, Sec.~S4 and Table~S2).")
# the selection effect
_zup = (0.75, 0.80, 0.85, 0.90, 0.94, 0.99)
need_true("expHS from 0.75 up the table leads on radio-silent drones, accuracy and Brier, p <= 0.007; not at 0.70", all(_zqh(_d, "radio-silent drones", "acc")[0] < 0 and _zqh(_d, "radio-silent drones", "brier")[0] > 0 and max(_zqh(_d, "radio-silent drones", "acc")[1], _zqh(_d, "radio-silent drones", "brier")[1]) <= 0.007 for _d in _zup) and min(_zqh(0.70, "radio-silent drones", "acc")[1], _zqh(0.70, "radio-silent drones", "brier")[1]) > 0.05)
need_true("expHS from 0.75 up it trails on non-drones, Brier, p <= 0.03", all(_zqh(_d, "non-drones", "brier")[0] < 0 and _zqh(_d, "non-drones", "brier")[1] <= 0.03 for _d in _zup))
need_true("expHS RF none: ties at 0.75 and 0.80 on Brier, p >= 0.47, trails above them, p <= 0.004", min(_zqh(_d, "RF none", "brier")[1] for _d in (0.75, 0.80)) >= 0.47 and all(_zqh(_d, "RF none", "brier")[0] < 0 and _zqh(_d, "RF none", "brier")[1] <= 0.004 for _d in (0.85, 0.90, 0.94, 0.99)))
need_true("expHS RF none is what its name says", "rfr == 2" in _ysrc and "(rfr == 2) & (cls_ != 2)" in _ysrc)
need_supp("expHS the selection effect, in the supplement", "From $0.75$ up the table leads on radio-silent drones and trails on non-drones")
# what estimating supplies
_zc75 = lambda q, k: _zp(q.sort_values("rep")[k].values, _zhv("hand-set", 0.75, "all")[k].values)
need_true("expHS about 300 labeled scenarios: n = 300 ties the table at 0.75 on accuracy, macro-AUC and Brier, n = 100 does not", min(_zc75(_zkn(300), _k)[1] for _k in ("acc", "macroAUC", "brier")) >= 0.18 and max(_zc75(_zkn(100), _k)[1] for _k in ("acc", "macroAUC", "brier")) < 0.05)
need_tex("expHS what estimating supplies", "so estimating supplies the level without choosing it")
# one level
_zlab = {_k: _zp(_zhv("hand-set", 0.75, "all")[_k].values, _zhv("abstract", 0.75, "all")[_k].values) for _k in _zH4}
_zlna = {_k: _zp(_zhv("no RF", 0.75, "all")[_k].values, _zhv("abstract", 0.75, "all")[_k].values) for _k in _zH4}
_zlan = {_k: _zp(_zhv("hand-set", 0.75, "all")[_k].values, _zhv("no RF", 0.75, "all")[_k].values) for _k in _zH4}
need_true("expHS most of the gain on the abstract model is the blind row's harm", all(_zlna[_k][0] / _zlab[_k][0] > 0.5 for _k in ("macroAUC", "brier")))
need_true("expHS above no RF channel: p < 1e-8", max(_zlan["macroAUC"][1], _zlan["brier"][1]) < 1e-8)
need_supp("expHS one level", f"($p<10^{{-8}}$; ${_zlab['macroAUC'][0]:+.3f}$ and ${_zlab['brier'][0]:+.3f}$ on the abstract model in all)")
# How much calibration data
_zorc = _zck[_zck.source == "oracle-marginal"]
_zco = {_k: _zc75(_zorc, _k) for _k in _zH4}
need_true("expHS 0.75 ties the oracle on accuracy, macro-AUC and Brier, p >= 0.14", min(_zco[_k][1] for _k in ("acc", "macroAUC", "brier")) >= 0.13 and _zco["ece"][0] > 0)
need_true("expHS 0.75 is ahead of n <= 100 on all four metrics, p <= 0.05", all(((_zc75(_zkn(_n), _k)[0] < 0) == (_k in ("acc", "macroAUC"))) and _zc75(_zkn(_n), _k)[1] <= 0.05 for _n in (30, 100) for _k in _zH4))
need_true("expHS 0.75 is the best of the sweep on Brier", min(_zHD, key=lambda _d: _zhv("hand-set", _d, "all").brier.mean()) == 0.75)
need_supp_later("expHS the calibration sweep, in the supplement", f"it ties the sensor oracle on accuracy, macro-AUC, and Brier ($p\\ge0.14$), with ECE lower by ${_zco['ece'][0]:.3f}$ ($p{{=}}{_zco['ece'][1]:.2f}$), and is ahead of the matrices at $n\\le100$ on all four metrics ($p\\le0.05$).")
# Table 7 and the other places
_zci = lambda x: f"${x.mean():.3f}_{{\\pm{x.std(ddof=1) / np.sqrt(len(x)) * _st.t.ppf(0.975, len(x) - 1):.3f}}}$".replace("\\pm0.", "\\pm.")
_zh75 = [_zci(_zhv("hand-set", 0.75, "all")[_k].values) for _k in _zH4]
need_supp_later("expHS the table at 0.75 in the format of Table 7, in the supplement", f"has accuracy {_zh75[0]}, macro-AUC {_zh75[1]}, Brier {_zh75[2]}, ECE {_zh75[3]}, and macro-F1 ${_zhv('hand-set', 0.75, 'all').macroF1.mean():.3f}$;")
need_true("expHS Table 7 holds no hand-set or one-number row", not HAVE_PAPER or ("hand-set, $0.75$" not in tex and "one number, $n{=}100$" not in tex))
need_tex("expHS Table 7 caption", "Macro-F1 is flat across this sweep ($0.481$--$0.488$) and is omitted. No row uses a reliability discount.")
need_tex("expHS claim (ii)", "(ii) fusing the sensors through estimated confusion matrices improves on single sensors and on the abstract sensor model, and its RF channel, read through the latent emission state, adds information over fusion without RF;")
need_true("expHS the abstract's range is the sweep's ties", f"{_zlev['Rr'].mean():.2f}" == "0.69" and f"{_zlev['Ra'].mean():.2f}" == "0.76")
_zps = lambda p: f"{p:.2f}"[1:] if p >= 0.0095 else (f"{p:.3f}"[1:] if p >= 0.00095 else "$<$.001")
_zcf = lambda s, d, k: "$" + f"{_zqh(d, s, k)[0]:+.4f}".replace("0.", ".", 1) + f"$ ({_zps(_zqh(d, s, k)[1])})"
# Table S2 and the paragraph of the supplement
for _d in _zHD:
    need_supp(f"expHS Table S2: {_d:.2f}", f"{_d:.2f} & " + " & ".join(_zcf(_s, _d, _k) for _s, _k in (("all", "acc"), ("all", "macroAUC"), ("all", "brier"), ("all", "ece"), ("radio-silent drones", "acc"), ("radio-silent drones", "brier"), ("non-drones", "brier"), ("RF none", "brier"))) + " \\\\")
need_supp("expHS Table S2 caption", "Estimated matrices ($n{=}300$, $N{=}200$, no discount) minus the hand-set emission-aware table, by the table's diagonal: mean of $20$ paired replications, with its $p$-value.")
need_supp("expHS S4 levels", f"is ${_zlev['Rr'].mean():.2f}$, ${_zlev['Re'].mean():.2f}$, and ${_zlev['Ra'].mean():.2f}$ for radar, EO/IR, and acoustic (${_zlev['Rr'].min():.2f}$--${_zlev['Rr'].max():.2f}$, ${_zlev['Re'].min():.2f}$--${_zlev['Re'].max():.2f}$, and ${_zlev['Ra'].min():.2f}$--${_zlev['Ra'].max():.2f}$ over the $20$ calibration splits).")
_zmr = {_k: _zp(_zhv("hand-set, RF row estimated", 0.94, "all")[_k].values, _zhv("hand-set", 0.94, "all")[_k].values) for _k in _zH4}
need_true("expHS S4 mixed tables: the p-values the text quotes", min(_zmx[_k][1] for _k in ("macroAUC", "brier", "ece")) >= 0.42 and min(_zmr[_k][1] for _k in ("macroAUC", "brier", "ece")) >= 0.07)
need_supp("expHS S4 mixed tables", f"(accuracy ${_zmx['acc'][0]:+.4f}$, $p{{=}}{_zmx['acc'][1]:.2f}$; macro-AUC, Brier, and ECE $p\\ge0.42$); with the three at $0.94$ and the RF row estimated, accuracy rises by ${_zmr['acc'][0]:.4f}$ ($p{{=}}{_zmr['acc'][1]:.2f}$) and macro-AUC, Brier, and ECE do not move ($p\\ge0.07$).")
need_supp("expHS S4 non-drones", f"(Brier ${-_zqh(0.75, 'non-drones', 'brier')[0]:+.3f}$ at $0.75$, ${-_zqh(0.94, 'non-drones', 'brier')[0]:+.3f}$ at $0.94$, $p\\le0.03$).")
need_true("expHS S4 RF none and non-drones: trails from 0.75", all(_zqh(_d, "RF none, non-drones", "brier")[0] < 0 and _zqh(_d, "RF none, non-drones", "brier")[1] < 0.01 for _d in _zup))
need_supp("expHS S4 RF none", f"over all such scenarios the table ties at $0.75$ and $0.80$ ($p\\ge0.47$) and trails above them (Brier, $p\\le0.004$), and among those that are non-drones it trails from $0.75$ (Brier ${-_zqh(0.75, 'RF none, non-drones', 'brier')[0]:+.3f}$ at $0.75$, $p{{=}}{_zqh(0.75, 'RF none, non-drones', 'brier')[1]:.3f}$).")
need_supp("expHS S4 one level", f"no RF channel gains ${_zlna['macroAUC'][0]:+.3f}$ macro-AUC and ${_zlna['brier'][0]:+.3f}$ Brier on it, and the emission-aware row a further ${_zlan['macroAUC'][0]:+.3f}$ and ${_zlan['brier'][0]:+.3f}$ ($p<10^{{-8}}$; ${_zlab['macroAUC'][0]:+.3f}$ and ${_zlab['brier'][0]:+.3f}$ on the abstract model in all), close to the deployed channel's ${_zdp['macroAUC'][0]:+.3f}$ and ${_zdp['brier'][0]:+.3f}$ in Sec.~VII-B.")
# what moved to Sec. S4 keeps its pointer in the body
need_supp("expHS reliability layer, in the supplement", "two-thirds of the radio-silent Brier loss arises at far range")
need_tex("expHS pointer: the top states", "(top states of the latent variables: Supplementary Material, Sec.~S4). Labels are needed")
need_tex("expHS pointer: the ESS repeat", "The $N{=}10$-vs-$N{=}50$ ECE gap is significant only at $\\alpha_0{=}2$ of $\\{1,2,5\\}$ (Supplementary Material, Sec.~S4).")
need_tex("expHS pointer: the rule on silence", "its hand rule on silence explains only part of that (Supplementary Material, Sec.~S4, which also sets B6 under a hand-set prior and B3 against the rule).")
need_tex("expHS Table 9", "The lower block of Table~\\ref{tab:highrel} shows every class over-confident in the $(.50,.75]$ bin;")
# the README
for _x in (f"With the fitted discount on both sides, the deployed RF channel adds {_zdp['macroAUC'][0]:+.3f} macro-AUC",
           "That table's diagonal, 0.94, lies far above the sensors' estimated accuracy (mean correct-report probability 0.69, 0.71, 0.76 for radar, EO/IR, acoustic).",
           "it ties the estimated matrices at 0.70 to 0.80 on accuracy, macro-AUC and Brier (p≥0.18; at 0.75 its ECE is lower by 0.003, p=0.008) and loses on Brier elsewhere (p≤0.008), so estimating supplies the level without choosing it.",
           "so that lead comes from conditioning on the latent class.",
           "| Supplementary Material Sec. S4 (the hand-set table at 0.75) and Table S2 | `exp_reason_checks.py` | `expRS_handset.csv` |",
           "the hand-set table at diagonal 0.75 and the sweep of its diagonal (Supplementary Material Sec. S4, Table S2)",
           "Over the abstract model's own diagonal sweep 0.55–0.99"):
    need_true(f"expHS README: {_x[:50]}", _x in _rm)
need_true("expHS README: the non-drone value", f"Brier {-_zqh(0.75, 'non-drones', 'brier')[0]:+.3f} at 0.75, {-_zqh(0.94, 'non-drones', 'brier')[0]:+.3f} at 0.94, p≤0.03" in _rm)
# the old wording stays out
for _x in ("Estimated matrices beat a hand-set", "abstract-fusion, and hand-set emission-aware baselines", "fair controls", "structures such threat"):
    need_true(f"expHS retired from main.tex and the supplement: {_x}", not HAVE_PAPER or (_x not in tex and _x not in _supp))
    need_true(f"expHS retired from the README: {_x}", _x not in _rm)
if DRAFT or not HAVE_PAPER:
    print(f"expHS checks run: {_n_expd - _n0zi}")
assert _n_expd - _n0zi > 0

# ---- the last touches: two sentences, the notation of the journal's checklist, the headers of the scripts -----
_n0zj = _n_expd
need_tex("expFin a sentence", "and $\\sim\\!100$--$300$ come within $0.002$ AUC of the \\emph{sensor oracle} that knows the true range-marginal matrices.")
_zo = float(_zck[_zck.source == "oracle-marginal"].macroAUC.mean())
need_true("expFin n = 100 and 300 are within 0.002 AUC of the oracle, n = 30 is not", all(abs(float(_zkn(_n).macroAUC.mean()) - _zo) <= 0.002 for _n in (100, 300)) and abs(float(_zkn(30).macroAUC.mean()) - _zo) > 0.002)
need_supp("expFin the range-conditional matrix, in the supplement", "it overtakes the deployed layer only in that world and only with $n{=}1000$ labeled scenarios")
for _x in ("\\mathit{cls}", "\\mathit{emit}", "P_{fa}", "\\sigma(w_s", "$\\lambda_s$", "\\{\\lambda_s\\}", "$\\pi_s$", "$\\gamma{=}", "1-\\gamma$", "reach the \\emph{sensor-oracle}", "and the range-conditional matrix."):
    need_true(f"expFin retired from main.tex: {_x}", not HAVE_PAPER or _x not in tex)
need_true("expFin every context vector is bold", not HAVE_PAPER or (tex.count("\\varphi") == tex.count("\\boldsymbol{\\varphi}") == 23))
need_true("expFin the coefficient vector is bold", not HAVE_PAPER or tex.count("\\mathbf{w}_s^\\top") == 2)
need_true("expFin the other vectors are bold", not HAVE_PAPER or (tex.count("\\boldsymbol{\\lambda}_s") == 2 and tex.count("\\boldsymbol{\\pi}_s") == 2 and tex.count("\\boldsymbol{\\gamma}") == 4))
need_true("expFin no multi-letter math italic in the supplement", not HAVE_PAPER or "\\mathit{" not in _supp)
# the headers of the scripts
for _zfn, _zxs in (("analyze_subsets.py", ("(N4)", "(N6)", "rows for Table 5")), ("exp_calibsize_ci.py", ("Table-5",)), ("exp_ess_sensitivity.py", ("Table-4", "Fig. 5", "fold-reduction")),
                ("exp_significance.py", ("Tables 3-5", "(Table 3 + Fig 3)", "## Table 3 (", "reliability (Table 4)", "sweep (Table 5)", "Table 5-style", '"\\n## Table 4 (acc/F1')), ("model_spec.py", ("measured sensor confusion", "literature-informed")),
                ("grounding.py", ("8 for the threat node vs 54",)), ("dataio.py", ("B0's 'abstract sensor model'", "ignoring measured confusion")), ("exp_base_rule.py", ("  adjustments of Table 3.", "  Table 2 weights:", "  Table 3 rules:", "  Table 6 worked example:"))):
    _zt = open(os.path.join(HERE, _zfn), encoding="utf-8").read()
    for _x in _zxs:
        need_true(f"expFin {_zfn} no longer says {_x}", _x not in _zt)
_zst = open(os.path.join(RES, "expS_summary.md"), encoding="utf-8").read()
need_true("expFin the summary of the main study has the headings the script writes", _zst.count("## Table 4 (acc/ECE/AUC mean +- 95%CI half)") == 1 and _zst.count("## Table 6 (acc/F1/AUC/Brier/ECE/cwECE/FAR mean +- CIhalf)") == 1 and "## Table 3" not in _zst)
_zss = open(os.path.join(HERE, "exp_significance.py"), encoding="utf-8").read()
need_true("expFin the script writes those headings", '"\\n## Table 4 (acc/ECE/AUC mean +- 95%CI half)"' in _zss and '"\\n## Table 6 (acc/F1/AUC/Brier/ECE/cwECE/FAR mean +- CIhalf)"' in _zss)
if DRAFT or not HAVE_PAPER:
    print(f"expFin checks run: {_n_expd - _n0zj}")
assert _n_expd - _n0zj > 0

# ---- the comparators' free values, added after the main results: exp_comparators.py -----
_n0cm = _n_expd
_cmf, _cmz, _cmb5, _cmsc, _cmer, _cmw, _cms, _cms1 = (_rt(f"expCM_{_x}.csv") for _x in ("factorial", "b0", "b5", "scalar", "node_errors", "world", "sens", "s1"))
_cmNS = (10, 20, 50, 80, 100, 200, 400)
need_true("expCM factorial: seven sizes, 20 replications per cell", tuple(sorted(_cmf.N.unique())) == _cmNS and _cmf.groupby(["N", "mean", "structure", "a0"]).size().eq(20).all())
_cmP = lambda N: _cmf[(_cmf["mean"] == "Proposed") & (_cmf.N == N)].sort_values("rep")
_cmA = lambda m, s, a, N: _cmf[(_cmf["mean"] == m) & (_cmf.structure == s) & np.isclose(_cmf.a0, a) & (_cmf.N == N)].sort_values("rep")
_cmM = lambda N: _cmA("none", "full", 0.0, N)
_cmU = lambda N, a=0.2: _cmA("uniform", "constrained+ICI", a, N)
_cmd = lambda x, y, k: _zp(x[k].values, y[k].values)
# the rows the main study has
_cmS = _rt("expS_scarcity_raw.csv")
_cmSq = lambda m, N: _cmS[(_cmS.method == m) & (_cmS.N == N)].sort_values("rep")
need_true("expCM Proposed and MLE are the rows of the main study", all(np.allclose(_cmP(_n)[_k].values, _cmSq("Proposed", _n)[_k].values, atol=1e-12) and np.allclose(_cmM(_n)[_k].values, _cmSq("B1-MLE", _n)[_k].values, atol=1e-12) for _n in sorted(set(_cmS.N) & set(_cmNS)) for _k in ("acc", "macroAUC", "brier", "ece")))
_cmB = _rt("expB_baselines_raw.csv")
_cmBN = sorted(set(_cmB.N) & set(_cmNS))
need_true("expCM the main study's sizes are in the grid", len(set(_cmS.N) & set(_cmNS)) >= 5 and len(_cmBN) >= 5)
need_true("expCM uniform, full, ESS 2 is B3 of the main study", all(np.allclose(_cmA("uniform", "full", 2.0, _n)[_k].values, _cmB[(_cmB.method == "uninformative MAP") & (_cmB.N == _n)].sort_values("rep")[_k].values, atol=1e-12) for _n in _cmBN for _k in ("macroAUC", "brier", "ece")))
need_true("expCM elicited, constrained+ICI, ESS 2 is Proposed", all(np.allclose(_cmA("elicited", "constrained+ICI", 2.0, _n)[_k].values, _cmP(_n)[_k].values, atol=1e-12) for _n in _cmNS for _k in ("macroAUC", "brier", "ece")))
# the result at N = 10, the factor at N = 50
_cma, _cmb, _cmc, _cme = (_cmd(_cmU(10), _cmM(50), _k) for _k in ("acc", "brier", "macroAUC", "ece"))
need_true("expCM the weak uniform arm at 10 against MLE at 50: p <= 0.022 on accuracy and Brier", _cma[1] <= 0.022 and _cmb[1] <= 0.022)
need_tex("expCM the first result, its pointer", "so this comparison mainly measures the prior (Supplementary Material, Sec.~S5 replaces it by a weak uniform one).")
_cmq = {_k: _cmd(_cmU(50), _cmM(100), _k) for _k in ("macroAUC", "brier", "ece")}
need_true("expCM the weak uniform arm at 50 against MLE at 100: p <= 0.016", max(_v[1] for _v in _cmq.values()) <= 0.016)
need_true("expCM the weak uniform arm at 50 left the body for Sec. S5", not HAVE_PAPER or "That factor needs no elicited knowledge" not in tex)
_cm200 = {_k: _cmd(_cmU(50), _cmM(200), _k) for _k in ("acc", "brier")}
need_true("expCM MLE at 200 overtakes that arm at 50", _cm200["acc"][1] < 1e-3 and _cm200["brier"][1] < 1e-3)
need_supp("expCM S5 the arm against MLE", f"At $N{{=}}10$ it trails MLE at $N{{=}}50$ on accuracy and Brier (${_cma[0]:+.3f}$, ${_cmb[0]:+.3f}$, $p\\le0.022$) and leads on ECE (${_cme[0]:+.3f}$, $p{{=}}{_cme[1]:.3f}$), with macro-AUC level (${_cmc[0]:+.3f}$, $p{{=}}{_cmc[1]:.2f}$). At $N{{=}}50$ it beats MLE at $N{{=}}100$ (macro-AUC ${_cmq['macroAUC'][0]:+.3f}$, Brier ${_cmq['brier'][0]:+.3f}$, ECE ${_cmq['ece'][0]:+.3f}$, $p\\le0.016$), and MLE at $N{{=}}200$ overtakes it (accuracy ${_cm200['acc'][0]:+.3f}$, Brier ${_cm200['brier'][0]:+.3f}$, $p<10^{{-3}}$).")
need_true("expCM from N = 50 the arm at 0.1, 0.2, 0.5 is within 0.005 macro-AUC", all(abs(_cmd(_cmP(_n), _cmU(_n, _a), "macroAUC")[0]) <= 0.005 for _n in _cmNS if _n >= 50 for _a in (0.1, 0.2, 0.5)))
need_true("expCM at N = 20 it is not (so the claim starts at 50)", abs(_cmd(_cmP(20), _cmU(20), "macroAUC")[0]) > 0.01)
need_true("expCM the abstract and the conclusion hold no comparison chosen after the fact", not HAVE_PAPER or all(_x not in _jab and _x not in tex[tex.index("\\section{Conclusion}"):tex.index("\\appendices")] for _x in ("that takes an informative prior", "weak uniform prior", "0.005 macro-AUC", "half the true effects", "0.70 to 0.80", "hand-set emission-aware", "$\\tau$ halved", "diagonal")))
need_true("expCM the abstract's numbers on the rule are the study's", f"AUC {_f(_pa[0])} to {_f(_pa[-1])} against {_f(_a_pub.mean())}" == "AUC 0.876 to 0.890 against 0.752")
need_tex("expCM abstract, the baselines", "full-table uninformative-prior and discriminative baselines on macro-AUC and Brier at every size.")
# the result of C3 in the abstract and the conclusion: the latent emission state and the double discount
need_true("expCM C3: on radio-silent drones the RF channel, with the fitted discount on both sides, lowers Brier by 0.024 (p < 1e-5)", f"{-_zds['brier'][0]:.3f}" == "0.024" and _zds["brier"][1] < 1e-5)
need_true("expCM C3: the chance-corrected layer calibrates worse in the standard world (ECE, p < 0.05)", _ad["standard"][1][0] > 0 and _ad["standard"][1][1] < 0.05)
need_tex("expCM C3, abstract", "The emission state lets the radio-frequency (RF) channel cut the Brier score of radio-silent drones by 0.024; the weights discount twice, but stay an empirical discount since a chance-corrected alternative calibrates worse.")
need_true("expCM the abstract defines RF at its first use, once", not HAVE_PAPER or (re.search(r"\\begin\{abstract\}(.*?)\\end\{abstract\}", tex, re.S).group(1).count("(RF)") == 1
          and re.search(r"\\begin\{abstract\}(.*?)\\end\{abstract\}", tex, re.S).group(1).index("radio-frequency (RF)") < re.search(r"\\begin\{abstract\}(.*?)\\end\{abstract\}", tex, re.S).group(1).index("RF channel adds")))
need_tex("expCM C3, conclusion", "Through the latent emission state the RF channel lowers the Brier score of radio-silent drones by $0.024$, and the reliability weights, which discount twice, stay as an empirical discount because a chance-corrected alternative calibrates worse (C3).")
need_tex("expCM VI-C", "with a \\emph{uniform} prior mean at the same ESS, a heavy setting for a uniform mean (Supplementary Material, Table~S3 varies it),")
need_tex("expCM VI-C, B5", "an $L_2$ multinomial logistic regression ($C{=}1$; Supplementary Material, Sec.~S5 varies it)")
need_tex("expCM VII-E", "Because the prior carries the result at small $N$, we stress-test it")
need_tex("expCM heading", "\\subsubsection{Standard baselines}")
need_tex("expCM Discussion", "so the comparative gains, the structure's gain without elicited knowledge (Sec.~\\ref{sec:simresults}) among them, rest in part on sharing that structure with the world: where additivity fails (S1), unsmoothed MLE beats the model at $N{=}200$ (Table~\\ref{tab:misspec}).")
need_tex("expCM VII-C says what was done", "We calibrate the RF node on the public DroneRF benchmark \\cite{alsad2019dronerf}.")
need_true("expCM VII-C: no contrast between specification and execution", not HAVE_PAPER or "from specification to execution" not in tex)
need_true("expCM Discussion: in S1 weak and strong the best alternative at N = 200 is MLE, significantly better", all(_wjudge(_w, 200)[0] == "B1-MLE" and _wjudge(_w, 200)[1] > 0 and _wjudge(_w, 200)[2] < 0.05 for _w in ("nonadd-0.5", "nonadd-1.0")))
need_true("expCM Discussion: the old attribution is gone", not HAVE_PAPER or "is the gain of sharing it with the world" not in tex)
# full tables and BDeu against Proposed, over every setting
_cmg = [(_m, _a, _n, _k) for _m in ("uniform", "BDeu") for _a in sorted(_cmf[_cmf["mean"] == _m].a0.unique()) for _n in _cmNS for _k in ("macroAUC", "brier")]
_cmgp = [_cmd(_cmP(_n), _cmA(_m, "full", _a, _n), _k) for _m, _a, _n, _k in _cmg]
need_true("expCM 168 contrasts on macro-AUC and Brier: all favor Proposed, p < 0.001", len(_cmg) == 168 and all((_v[0] > 0 if _g[3] == "macroAUC" else _v[0] < 0) and _v[1] < 1e-3 for _g, _v in zip(_cmg, _cmgp)))
need_true("expCM the grids: uniform 0.01 to 2, BDeu 1 to 20", sorted(_cmf[_cmf["mean"] == "uniform"].a0.unique()) == [0.01, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0] and sorted(_cmf[_cmf["mean"] == "BDeu"].a0.unique()) == [1.0, 2.0, 5.0, 10.0, 20.0] and sorted(_cmf[_cmf["mean"] == "elicited"].a0.unique()) == [0.2, 0.5, 1.0, 2.0, 5.0])
_cmu5 = {_n: _cmd(_cmP(_n), _cmA("uniform", "full", 0.5, _n), "ece") for _n in _cmNS}
_cmb5e = {_n: _cmd(_cmP(_n), _cmA("BDeu", "full", 5.0, _n), "ece") for _n in _cmNS}
need_true("expCM ECE: uniform at 0.5 ties at every N, p >= 0.06; BDeu at 5, p >= 0.05", min(_v[1] for _v in _cmu5.values()) >= 0.06 and min(_v[1] for _v in _cmb5e.values()) >= 0.05)
need_tex("expCM the free values of the baselines", "Over the free values of these baselines (a uniform ESS from $0.01$ to $2$, BDeu, and B5's $C$ from $0.001$ to $10$), none closes the macro-AUC or Brier gap ($p<10^{-3}$), while a lighter uniform prior closes the ECE gap at every $N$ and stronger regularization closes B5's at $N{=}50$ (Supplementary Material, Sec.~S5 and Table~S3).")
need_supp("expCM S5 the lighter prior", f"A full table with a uniform mean at $\\alpha_0{{=}}0.5$ ties Proposed on ECE at every $N$ ($p\\ge0.06$; at $N{{=}}50$ Proposed is ${_cmu5[50][0]:+.3f}$), and so does BDeu at a total of $5$ ($p\\ge0.05$); no uniform ESS from $0.01$ to $2$ and no BDeu total from $1$ to $20$ ties it on macro-AUC or Brier at any $N$ ($p<10^{{-3}}$ over $168$ contrasts).")
# B5
_cmC = sorted(_cmb5.C.unique())
need_true("expCM B5: C from 0.001 to 10 at N = 10, 50, 200, 400, 20 replications", _cmC[0] == 0.001 and _cmC[-1] == 10.0 and len(_cmC) == 9 and sorted(_cmb5.N.unique()) == [10, 50, 200, 400] and _cmb5.groupby(["C", "N"]).size().eq(20).all())
_cmb5v = lambda C, N: _cmb5[np.isclose(_cmb5.C, C) & (_cmb5.N == N)].sort_values("rep")
need_true("expCM B5 at C = 1 is B5 of the main study", all(np.allclose(_cmb5v(1.0, _n)[_k].values, _cmB[(_cmB.method == "logistic regression") & (_cmB.N == _n)].sort_values("rep")[_k].values, atol=1e-12) for _n in sorted(set(_cmBN) & {10, 50, 200, 400}) for _k in ("macroAUC", "brier", "ece")))
need_true("expCM B5: the ECE gap at N = 50 closes at C <= 0.1, p >= 0.46", min(_cmd(_cmP(50), _cmb5v(_c, 50), "ece")[1] for _c in _cmC if _c <= 0.1) >= 0.455)
need_true("expCM B5: and it holds at C >= 0.3 there", all(_cmd(_cmP(50), _cmb5v(_c, 50), "ece")[1] < 0.05 for _c in _cmC if _c > 0.1))
need_true("expCM B5: at no C at N = 10, 200, 400, p <= 0.031", max(_cmd(_cmP(_n), _cmb5v(_c, _n), "ece")[1] for _n in (10, 200, 400) for _c in _cmC) <= 0.0315)
need_true("expCM B5: macro-AUC and Brier hold at every C and N, p < 1e-5", all(_cmd(_cmP(_n), _cmb5v(_c, _n), _k)[1] < 1e-5 and ((_cmd(_cmP(_n), _cmb5v(_c, _n), _k)[0] > 0) == (_k == "macroAUC")) for _n in (10, 50, 200, 400) for _c in _cmC for _k in ("macroAUC", "brier")))
need_true("expCM B5: the sweep sentence left the body for Sec. S5", not HAVE_PAPER or "at $C{\\le}0.1$ its ECE gap" not in tex)
need_supp("expCM S5 B5", "The logistic baseline's ECE gap closes at $C\\le0.1$ at $N{=}50$ ($p\\ge0.46$), and at no $C$ at $N{=}10$, $200$, or $400$ ($p\\le0.031$); its macro-AUC and Brier gaps hold at every $C$ from $0.001$ to $10$ ($p<10^{-5}$).")
# the walks
_cmwk = lambda n, m, s, a: _cmA(m, s, a, n)
_cmw1 = [_cmd(_cmwk(10, "elicited", "full", 2.0), _cmwk(10, "uniform", "full", 2.0), "macroAUC"), _cmd(_cmwk(10, "elicited", "constrained", 2.0), _cmwk(10, "elicited", "full", 2.0), "macroAUC"), _cmd(_cmwk(10, "elicited", "constrained+ICI", 2.0), _cmwk(10, "elicited", "constrained", 2.0), "macroAUC")]
_cmw2 = [_cmd(_cmwk(10, "uniform", "constrained", 2.0), _cmwk(10, "uniform", "full", 2.0), "macroAUC"), _cmd(_cmwk(10, "uniform", "constrained+ICI", 2.0), _cmwk(10, "uniform", "constrained", 2.0), "macroAUC"), _cmd(_cmwk(10, "elicited", "constrained+ICI", 2.0), _cmwk(10, "uniform", "constrained+ICI", 2.0), "macroAUC")]
_cme80 = _cmd(_cmwk(80, "elicited", "full", 2.0), _cmwk(80, "uniform", "full", 2.0), "ece")
_cme80s = _cmd(_cmP(80), _cmU(80), "ece")
need_supp("expCM S5 the walks", f"In that walk's order at $\\alpha_0{{=}}2$ and $N{{=}}10$ the elicited mean gains ${_cmw1[0][0]:+.3f}$ macro-AUC, the constraints ${_cmw1[1][0]:+.3f}$, and the ICI node ${_cmw1[2][0]:+.3f}$; with the structure first the constraints gain ${_cmw2[0][0]:+.3f}$, the ICI node ${_cmw2[1][0]:+.3f}$, and the elicited mean ${_cmw2[2][0]:+.3f}$. At $N{{=}}80$ the elicited mean moves ECE by ${_cme80[0]:+.3f}$ in the first order and by ${_cme80s[0]:+.3f}$ on the weak uniform prior with the same structure.")
_cmp80 = {_k: _cmd(_cmP(80), _cmU(80), _k) for _k in ("macroAUC", "brier", "ece")}
_cms80 = {_k: _cmd(_cmU(80), _cmM(80), _k) for _k in ("macroAUC", "brier", "ece")}
_cms10 = {_k: _cmd(_cmU(10), _cmM(10), _k) for _k in ("macroAUC", "brier", "ece")}
_cmp10 = {_k: _cmd(_cmP(10), _cmU(10), _k) for _k in ("macroAUC", "acc", "brier")}
need_true("expCM the walk: p >= 0.07 at N = 80, p <= 0.013 and 0.017 at N = 10", min(_v[1] for _v in _cmp80.values()) >= 0.07 and max(_v[1] for _v in _cms10.values()) <= 0.013 and max(_v[1] for _v in _cmp10.values()) <= 0.017)
need_tex("expCM the walk", f"That walk is one order at one ESS; taken with the structure first, on a weak uniform prior, the elicited mean adds little from $N{{=}}50$ on (within $0.005$ macro-AUC) but still adds ${_cmp10['macroAUC'][0]:+.3f}$ macro-AUC and ${_cmp10['brier'][0]:+.3f}$ Brier at $N{{=}}10$ ($p\\le0.017$; Supplementary Material, Sec.~S5 and Table~S3). The structure's gain rests on the monotone additivity the model shares with this world (Secs.~\\ref{{sec:misspec}} and~\\ref{{sec:discussion}}).")
need_supp_later("expCM the walk, in the supplement", f"With the structure first on that prior ($\\alpha_0{{=}}0.2$), the structure gains ${_cms80['macroAUC'][0]:+.3f}$ macro-AUC, ${_cms80['brier'][0]:+.3f}$ Brier, and ${_cms80['ece'][0]:+.3f}$ ECE on MLE at $N{{=}}80$ and the elicited mean adds ${_cmp80['macroAUC'][0]:+.3f}$, $0.000$, and ${_cmp80['ece'][0]:+.3f}$ ($p\\ge0.07$); at $N{{=}}10$ the structure gains ${_cms10['macroAUC'][0]:+.3f}$, ${_cms10['brier'][0]:+.3f}$, ${_cms10['ece'][0]:+.3f}$ ($p\\le0.013$) and the elicited mean adds ${_cmp10['macroAUC'][0]:+.3f}$ macro-AUC, ${_cmp10['acc'][0]:+.3f}$ accuracy, and ${_cmp10['brier'][0]:+.3f}$ Brier ($p\\le0.017$).")
need_true("expCM the elicited mean adds 0.000 Brier at N = 80", f"{abs(_cmp80['brier'][0]):.3f}" == "0.000")
need_tex("expCM the walk, its head", "In an ablation at $N{=}80$ over the $20$ replications, removing prior and structure together (MLE) is the only large regression")
need_true("expCM the arm's within-0.005 sentence is in the supplement", not HAVE_PAPER or "That arm at $\\alpha_0$ from $0.1$ to $0.5$ stays within $0.005$ macro-AUC of Proposed from $N{=}50$ on." in _supp)
# Table S3
_cmd3 = lambda x: f"{x:.3f}".replace("0.", ".", 1)
_cmNS3 = (10, 20, 50, 80, 200, 400)
for _lab, _m, _s, _a in (("MLE", "none", "full", 0.0), ("uniform, full, $2$ (B3)", "uniform", "full", 2.0), ("uniform, full, $0.5$", "uniform", "full", 0.5), ("BDeu, full, total $5$", "BDeu", "full", 5.0), ("uniform, constrained, $0.2$", "uniform", "constrained", 0.2), ("uniform, constr.+ICI, $0.2$", "uniform", "constrained+ICI", 0.2), ("uniform, constr.+ICI, $2$", "uniform", "constrained+ICI", 2.0), ("elicited, full, $2$", "elicited", "full", 2.0), ("elicited, constrained, $2$", "elicited", "constrained", 2.0), ("elicited, constr.+ICI, $2$ (Proposed)", "elicited", "constrained+ICI", 2.0)):
    need_supp(f"expCM Table S3: {_lab}", _lab + " & " + " & ".join(_cmd3(_cmA(_m, _s, _a, _n)[_k].mean()) for _k in ("macroAUC", "brier", "ece") for _n in _cmNS3) + " \\\\")
for _lab, _k in (("Accuracy", "acc"), ("Macro-AUC", "macroAUC"), ("Brier", "brier"), ("ECE", "ece"), ("FAR", "far@pd0.9")):
    _cc = []
    for _n in _cmNS3:
        _v = _cmd(_cmP(_n), _cmU(_n), _k)
        _cc.append(f"${_v[0]:+.3f}$".replace("0.", ".", 1) + " (" + ("$<$.001" if _v[1] < 0.001 else f"{_v[1]:.3f}".replace("0.", ".", 1)) + ")")
    need_supp(f"expCM Table S3, lower block: {_lab}", _lab + " & " + " & ".join(_cc) + " \\\\")
need_supp("expCM Table S3 caption", "Values of $\\alpha_0$ other than $2$ and the BDeu total come from a grid (uniform $0.01$ to $2$, elicited $0.2$ to $5$, BDeu $1$ to $20$; the repository's \\texttt{expCM\\_factorial.csv} has all of it, $N{=}100$ included).")
# B0
need_true("expCM B0 sweep: six factors, alone and at seven sizes, 20 replications", sorted(_cmz.f.unique()) == [0.3, 0.4, 0.5, 0.6, 0.8, 1.0] and _cmz.groupby(["f", "use", "N"]).size().eq(20).all() and sorted(_cmz[_cmz.use == "prior"].N.unique()) == list(_cmNS))
_cmB0 = lambda f, use, N=0: _cmz[np.isclose(_cmz.f, f) & (_cmz.use == use) & (_cmz.N == N)].sort_values("rep")
need_true("expCM f = 1 as a prior is the realistic expert of the main study", all(np.allclose(_cmB0(1.0, "prior", _n)[_k].values, _cmB[(_cmB.method == "B0-as-prior MAP") & (_cmB.N == _n)].sort_values("rep")[_k].values, atol=1e-12) for _n in _cmBN for _k in ("macroAUC", "brier", "ece")))
need_true("expCM f = 1 alone is B0 of the main study", all(np.allclose(_cmB0(1.0, "alone")[_k].values, _cmSq("B0-heuristic", 10)[_k].values, atol=1e-12) for _k in ("macroAUC", "brier", "ece", "far@pd0.9")))
_cmht = open(os.path.join(HERE, "model_spec.py"), encoding="utf-8").read()
need_tex("expCM B0", "with rounded weights and a flattened response ($\\tau{=}0.95/0.9/0.95/1.0$ for $H/N/C/T$ against the world's $0.28$ and $0.40$; Supplementary Material, Table~S4 sweeps it); it stands in")
_cmhe = lambda N, k: _cmd(_cmP(N), _cmB0(0.5, "alone"), k)
_cmef = [_cmhe(_n, "ece") for _n in (10, 20, 50, 80, 100)]
_cmff = [_cmhe(_n, "far@pd0.9") for _n in (10, 20)]
need_true("expCM halved B0: Proposed trails on ECE up to N = 100 (p < 0.05), not at 200", all(_v[0] > 0 and _v[1] < 0.05 for _v in _cmef) and _cmhe(200, "ece")[1] > 0.05)
need_true("expCM halved B0: Proposed trails on FAR at N <= 20 (p <= 0.01), ties to 100 (p >= 0.3), leads from 200 (p <= 0.003)", all(_v[0] > 0 and _v[1] <= 0.01 for _v in _cmff) and all(_cmhe(_n, "far@pd0.9")[1] >= 0.3 for _n in (50, 80, 100)) and all(_cmhe(_n, "far@pd0.9")[0] < 0 and _cmhe(_n, "far@pd0.9")[1] <= 0.003 for _n in (200, 400)))
need_true("expCM halved B0: Proposed leads on accuracy from 20, Brier from 50, p <= 0.007", all(_cmhe(_n, "acc")[0] > 0 and _cmhe(_n, "acc")[1] <= 0.007 for _n in (20, 50, 80, 100, 200, 400)) and all(_cmhe(_n, "brier")[0] < 0 and _cmhe(_n, "brier")[1] <= 0.007 for _n in (50, 80, 100, 200, 400)))
need_true("expCM the halved B0 left the body for Sec. S5", not HAVE_PAPER or ("$\\tau$ halved" not in tex and "B0 as set here" not in tex and "as set here" not in tex))
need_tex("expCM FAR paragraph, its head", "supports claim~(i) of Sec.~\\ref{sec:experiments}: at a fixed $P_d{=}0.9$")
_cmha = _cmB0(0.5, "alone")
_cmhp = {_n: _cmB0(0.5, "prior", _n) for _n in (10, 50)}
_cmpe = {_n: _cmd(_cmP(_n), _cmhp[_n], "ece") for _n in (10, 50)}
need_true("expCM the halved B0 as a prior has the lower ECE at N = 10 and 50, p <= 0.034", all(_v[0] > 0 and _v[1] <= 0.034 for _v in _cmpe.values()))
need_supp_later("expCM VII-B-3, in the supplement", f"With its $\\tau$ halved ($f{{=}}0.5$), B0 alone has ECE ${_cmha.ece.mean():.3f}$ and FAR ${_cmha['far@pd0.9'].mean():.3f}$ with no data.")
need_supp_later("expCM VII-B-3, as a prior, in the supplement", f"against Proposed, it has the lower ECE at $N{{=}}10$ and $50$ (${_cmhp[10].ece.mean():.3f}$, ${_cmhp[50].ece.mean():.3f}$ against ${_cmP(10).ece.mean():.3f}$, ${_cmP(50).ece.mean():.3f}$; $p\\le0.034$).")
_cmhm = {_k: _cmd(_cmhp[10], _cmM(50), _k) for _k in ("macroAUC", "brier", "ece", "far@pd0.9", "acc")}
_cm1m = _cmd(_cmB0(1.0, "prior", 10), _cmM(50), "brier")
need_true("expCM the halved B0 at 10 beats MLE at 50 on four metrics, p < 0.001; accuracy p = 0.10", all(_cmhm[_k][1] < 1e-3 for _k in ("macroAUC", "brier", "ece", "far@pd0.9")) and _cmhm["macroAUC"][0] > 0 and all(_cmhm[_k][0] < 0 for _k in ("brier", "ece", "far@pd0.9")) and f"{_cmhm['acc'][1]:.2f}" == "0.10")
need_tex("expCM the operational reading", f"and at $N{{=}}10$ the B0 prior trails MLE at $N{{=}}50$ on Brier (${_cm1m[0]:+.3f}$, $p{{=}}{_cm1m[1]:.2f}$).")
need_supp("expCM S5 the halved B0", f"Proposed minus that B0 has ECE ${_cmef[0][0]:+.3f}$, ${_cmef[1][0]:+.3f}$, ${_cmef[2][0]:+.3f}$, ${_cmef[3][0]:+.3f}$, and ${_cmef[4][0]:+.3f}$ at $N{{=}}10$, $20$, $50$, $80$, and $100$ ($p<0.05$), and FAR ${_cmff[0][0]:+.3f}$ and ${_cmff[1][0]:+.3f}$ at $N{{=}}10$ and $20$ ($p\\le0.01$); Proposed ties it on FAR from $N{{=}}50$ to $100$ ($p\\ge0.3$) and leads on accuracy from $N{{=}}20$, on Brier from $N{{=}}50$, and on FAR from $N{{=}}200$ (${_cmhe(200, 'far@pd0.9')[0]:+.3f}$, ${_cmhe(400, 'far@pd0.9')[0]:+.3f}$; $p\\le0.007$). As a prior at $N{{=}}10$, the halved B0 against MLE at $N{{=}}50$ has macro-AUC ${_cmhm['macroAUC'][0]:+.3f}$, Brier ${_cmhm['brier'][0]:+.3f}$, ECE ${_cmhm['ece'][0]:+.3f}$, and FAR ${_cmhm['far@pd0.9'][0]:+.3f}$ ($p<10^{{-3}}$), with accuracy ${_cmhm['acc'][0]:+.3f}$ ($p{{=}}{_cmhm['acc'][1]:.2f}$);")
# Table S4
_cmK5 = ("acc", "macroAUC", "brier", "ece", "far@pd0.9")
for _v in (1.0, 0.8, 0.6, 0.5, 0.4, 0.3):
    need_supp(f"expCM Table S4: f = {_v:g}", f"{_v:g} & " + " & ".join(_cmd3(_q[_k].mean()) for _q in (_cmB0(_v, "alone"), _cmB0(_v, "prior", 10), _cmB0(_v, "prior", 50)) for _k in _cmK5) + " \\\\")
need_supp("expCM Table S4: Proposed", "Proposed & " + " & ".join([""] * 5) + " & " + " & ".join(_cmd3(_cmP(_n)[_k].mean()) for _n in (10, 50) for _k in _cmK5) + " \\\\")
# one number per sensor
need_true("expCM one number: four sizes, two layers, 20 replications", sorted(_cmsc.n.unique()) == [30, 100, 300, 1000] and _cmsc.groupby(["n", "layer"]).size().eq(20).all())
_cm1 = lambda n, l: _cmsc[(_cmsc.n == n) & (_cmsc.layer == l)].sort_values("rep")
_cmCk = _rt("expS_calibsize_raw.csv")
need_true("expCM one number: the matrix rows are the calibration-size rows of the main study", all(np.allclose(_cm1(_n, "matrix")[_k].values, _zkn(_n)[_k].values, atol=1e-12) for _n in (30, 100, 300, 1000) for _k in ("macroAUC", "brier", "ece")))
_cm1d = lambda n, k: _cmd(_cm1(n, "one number"), _cm1(n, "matrix"), k)
_cm30 = {_k: _cm1d(30, _k) for _k in ("macroAUC", "brier", "ece")}
need_true("expCM one number at 30 beats the matrices, p < 0.001", all(_v[1] < 1e-3 for _v in _cm30.values()) and _cm30["macroAUC"][0] > 0 and _cm30["brier"][0] < 0 and _cm30["ece"][0] < 0)
need_true("expCM one number ties on Brier from 100, p >= 0.24", min(_cm1d(_n, "brier")[1] for _n in (100, 300, 1000)) >= 0.24)
need_true("expCM the one-number sentence left the body for Sec. S5", not HAVE_PAPER or "one correct-report probability per sensor in that table" not in tex)
_cm1v = [_zci(_cm1(100, "one number")[_k].values) for _k in _zH4]
need_supp_later("expCM the one-number row of Table 7, in the supplement", f"per sensor at $n{{=}}100$ (Sec.~S5) it has {_cm1v[0]}, {_cm1v[1]}, {_cm1v[2]}, and {_cm1v[3]}.")
need_true("expCM macro-F1 of that row lies within the caption's range", 0.481 <= round(_cm1(100, "one number").macroF1.mean(), 3) <= 0.488)
_cm13 = {_k: _cmd(_cm1(100, "one number"), _cm1(300, "matrix"), _k) for _k in ("macroAUC", "brier")}
need_true("expCM one number at 100 is within 0.001 macro-AUC of the matrices at 300", abs(_cm13["macroAUC"][0]) <= 0.001)
need_true("expCM How much calibration data: the one-number sentence left the body", not HAVE_PAPER or "One number per sensor at $n{=}100$" not in tex)
need_supp("expCM S5 one number", f"Minus the estimated matrices at the same $n$: at $n{{=}}30$, macro-AUC ${_cm30['macroAUC'][0]:+.3f}$, Brier ${_cm30['brier'][0]:+.3f}$, ECE ${_cm30['ece'][0]:+.3f}$ ($p<10^{{-3}}$); at $n{{=}}100$, Brier ${_cm1d(100, 'brier')[0]:+.3f}$ ($p{{=}}{_cm1d(100, 'brier')[1]:.2f}$), ECE ${_cm1d(100, 'ece')[0]:+.3f}$ ($p{{=}}{_cm1d(100, 'ece')[1]:.2f}$); at $n{{=}}300$, ECE ${_cm1d(300, 'ece')[0]:+.3f}$ ($p{{=}}{_cm1d(300, 'ece')[1]:.2f}$), Brier $p{{=}}{_cm1d(300, 'brier')[1]:.2f}$; at $n{{=}}1000$, macro-AUC ${_cm1d(1000, 'macroAUC')[0]:+.4f}$ ($p{{=}}{_cm1d(1000, 'macroAUC')[1]:.2f}$), ECE ${_cm1d(1000, 'ece')[0]:+.3f}$ ($p{{=}}{_cm1d(1000, 'ece')[1]:.3f}$), Brier $p{{=}}{_cm1d(1000, 'brier')[1]:.2f}$. Against the matrices at $n{{=}}300$, one number at $n{{=}}100$ has macro-AUC ${_cm13['macroAUC'][0]:+.4f}$ ($p{{=}}{_cm13['macroAUC'][1]:.2f}$) and Brier ${_cm13['brier'][0]:+.4f}$ ($p{{=}}{_cm13['brier'][1]:.2f}$).")
# the other subset
_cmnd = {_k: _zp(_zhv("calibrated", None, "non-drones")[_k].values, _zhv("abstract", 0.94, "non-drones")[_k].values) for _k in ("acc", "macroAUC", "brier")}
need_true("expCM on non-drones the abstract model leads, p < 0.001", all(_v[1] < 1e-3 for _v in _cmnd.values()) and _cmnd["acc"][0] < 0 and _cmnd["macroAUC"][0] < 0 and _cmnd["brier"][0] > 0)
need_tex("expCM a sentence", f"On non-drones the abstract model leads (calibrated accuracy ${_cmnd['acc'][0]:+.3f}$, macro-AUC ${_cmnd['macroAUC'][0]:+.3f}$, Brier ${_cmnd['brier'][0]:+.3f}$, $p<10^{{-3}}$); both subsets condition on the latent class.")
# the margin
_cmx = _cmSq("Proposed", 10)
_cmy = _cmSq("B1-MLE", 50)
_cmwx = {_k: float(_st.wilcoxon(_cmx[_k].values - _cmy[_k].values).pvalue) for _k in ("ece", "far@pd0.9")}
need_tex("expCM a sentence", f"and, at the margin, on ECE ($0.077$ vs.\\ $0.107$, $p{{=}}0.05$, Wilcoxon ${_cmwx['ece']:.3f}$; classwise ${_xcw[0]:+.3f}$, $p{{=}}{_xcw[1]:.2f}$) and FAR ($p{{=}}0.05$, Wilcoxon ${_cmwx['far@pd0.9']:.3f}$);")
# the CNN errors under the scenarios' mix
_cmmix = float(_gtw.query(["cls"]).table[2] * _gtw.query(["emit"]).table[0])
need_true("expCM background is 81% of what the scenarios draw, each drone model 6.4%", f"{100 * (1 - _cmmix):.0f}" == "81" and f"{100 * _cmmix / 3:.1f}" == "6.4")
need_true("expCM 20 folds or seeds over five split types", len(_cmer) == 20 and list(_cmer.proto.unique()) == ["P0", "P1", "P1c", "P2", "P3"])
_cmeg = _cmer.groupby("proto", sort=False).mean(numeric_only=True)
_cmsh, _cmdb, _cmpl = 100 * _cmeg.scenario_share_models, 100 * _cmeg.scenario_drone_background, 100 * _cmeg.pool_share_models
need_tex("expCM the bound", f"the whole channel is bounded by ${_yrn[0][0]:+.4f}$ (the error-free reports above; Supplementary Material, Sec.~S5 breaks down the CNN errors).")
need_true("expCM the abstract's 0.0055 is the error-free split", f"{_yrn[0][0]:.4f}" == "0.0055")
need_supp("expCM S5 errors", "drone-model confusions are " + ", ".join(f"${_x:.0f}$" for _x in _cmsh[:-1]) + f", and ${_cmsh.iloc[-1]:.0f}\\%$ of the errors (" + ", ".join(f"${_x:.0f}$" for _x in _cmpl[:-1]) + f", and ${_cmpl.iloc[-1]:.0f}\\%$ of the test pool), and drone/background errors " + ", ".join(f"${_x:.1f}$" for _x in _cmdb[:-1]) + f", and ${_cmdb.iloc[-1]:.1f}\\%$ of scenarios; in the hardest folds the share falls to ${100 * _cmer.scenario_share_models.min():.0f}\\%$ and those errors reach ${100 * _cmer.scenario_drone_background.max():.1f}\\%$.")
need_tex("expCM (ii)", "(ii) the calibrated pipeline, a CNN hard report through the calibration-split confusion estimate, with no discount on RF;")
need_tex("expCM the synthetic table", "the synthetic RF table, with its discount, adds")
need_true("expCM variant (ii) applies no RF weight in the code", "\"Rr\", \"Re\", \"Ra\"" in open(os.path.join(HERE, "exp_rf_e2e.py"), encoding="utf-8").read() or "'Rr', 'Re', 'Ra'" in open(os.path.join(HERE, "exp_rf_e2e.py"), encoding="utf-8").read())
need_tex("expCM Table 11 caption", "\\caption{Real RF data under stricter splits and interference (no RF: macro-AUC of (ii) minus a control with no RF channel).}")
# the infrared gate
_cmig = _rt("expO_g2_bootstrap.csv")
need_true("expCM both infrared intervals reach the limit", (_cmig.lo <= 0.35).all() and (_cmig.hi >= 0.35).all() and len(_cmig) == 2)
need_tex("expCM C4 stops at the posterior", "and under recorded interference, and DroneRF outputs reach the threat posterior (Sec.~\\ref{sec:rfe2e}).")
_cmv = {_k: _paired(_dr, "standard", "V0", "V3", "all", _k) for _k in ("brier", "ece")}
_cmt = {_k: _paired(_dt, "standard", "V0|true-internal", "V3|true-internal", "all", _k) for _k in ("brier", "ece")}
need_true("expCM on the true CPTs the ECE part is unresolved and the Brier part stays", _cmt["ece"][1] > 0.05 and _cmt["brier"][0] < 0 and _cmt["brier"][1] < 1e-3)
_cmvs = _paired(_dr, "standard", "V0", "V3", "silent-drone", "brier")
need_true("expCM the sentence on the deployed weight and the untested matrix are gone", not HAVE_PAPER or ("We keep the deployed weight because" not in tex and "which we did not test" not in tex))
need_supp("expCM the blind row costs the most", "the abstract model's emission-blind RF row costs the most:")
need_supp("expCM S4", f"The same goes for the deployed layer's ECE gain over no discount (${_cmv['ece'][0]:+.4f}$, $p{{=}}{_cmv['ece'][1]:.2f}$): on the true CPTs it vanishes (${_cmt['ece'][0]:+.4f}$, $p{{=}}{_cmt['ece'][1]:.2f}$), while its Brier gain stays (${_cmv['brier'][0]:+.4f}$; ${_cmt['brier'][0]:+.4f}$, $p<10^{{-3}}$).")
need_tex("expCM V-B", "Sensor reliability enters through a parent $\\mathrm{Rel}_s\\in\\{\\text{reliable},\\text{unreliable}\\}$, which gives the confusion-matrix CPT when reliable")
need_tex("expCM a sentence", "$P(e)$, which can be monitored (Sec.~\\ref{sec:corrfail}).")
# the world, and the run with emission on drones only
_cmwv = lambda w, c, e, k: float(_cmw[(_cmw.world == w) & (_cmw.cls == c) & (_cmw.emit == e)][k].iloc[0])
need_true("expCM the world's values are what inference on the network gives", abs(_cmwv("standard", "bird", "silent", "p_N_top") - _gtw.cpts["N"].table[1, 1, 1]) < 1e-9 and abs(_cmwv("standard", "drone", "emitting", "p_N_top") - _gtw.cpts["N"].table[2, 0, 1]) < 1e-9)
need_tex("expCM VI-A", f"so a silent bird is non-cooperative with probability ${_cmwv('standard', 'bird', 'silent', 'p_N_top'):.2f}$ against ${_cmwv('standard', 'drone', 'emitting', 'p_N_top'):.2f}$ for an emitting drone, and its $P(T{{=}}\\text{{high}})$ is ${_cmwv('standard', 'bird', 'silent', 'p_high'):.2f}$ against ${_cmwv('standard', 'drone', 'emitting', 'p_high'):.2f}$ (Supplementary Material, Sec.~S5 lets emission act on drones only).")
need_true("expCM in the new world a silent bird is an emitting bird", abs(_cmwv("emit-drone-only", "bird", "silent", "p_high") - _cmwv("emit-drone-only", "bird", "emitting", "p_high")) < 1e-12 and abs(_cmwv("emit-drone-only", "drone", "silent", "p_high") - _cmwv("standard", "drone", "silent", "p_high")) < 1e-12)
_cmsv = lambda w, N, m, k, s=None: (lambda _q: _q[_q.subset.isna() if s is None else _q.subset == s].sort_values("rep")[k].values)(_cms[(_cms.world == w) & (_cms.N == N) & (_cms.method == m)])
_cmexc = {(_w, _n): float((_cmsv(_w, _n, "Proposed", "brier") - _cmsv(_w, -1, "CPT-oracle", "brier")).mean()) for _w in ("standard", "emit-drone-only") for _n in (50, 200)}
_cmgr = float((_cmsv("emit-drone-only", -1, "CPT-oracle", "brier") - _cmsv("emit-drone-only", -1, "Bayes-oracle", "brier")).mean())
_cmalt = {_n: _zp(_cmsv("emit-drone-only", _n, "Proposed", "brier"), _cmsv("emit-drone-only", _n, "cMAP-full", "brier")) for _n in (50, 200)}
need_true("expCM standard rows are the main study's", all(np.allclose(_cmsv("standard", _n, "Proposed", "brier"), _cmSq("Proposed", _n).brier.values, atol=1e-12) for _n in (50, 200)))
need_true("expCM Proposed holds: no alternative significantly better, excess at most twice the standard world's", all(_cmexc[("emit-drone-only", _n)] <= 2 * _cmexc[("standard", _n)] for _n in (50, 200)) and all(_v[0] < 0 and _v[1] < 0.01 for _v in _cmalt.values()) and all(_zp(_cmsv("emit-drone-only", _n, "Proposed", "brier"), _cmsv("emit-drone-only", _n, _m, "brier"))[0] < 0 for _n in (50, 200) for _m in ("MAP", "B1-MLE", "B5-logistic")))
_cm105 = {_k: _zp(_cmsv("emit-drone-only", 10, "Proposed", _k), _cmsv("emit-drone-only", 50, "B1-MLE", _k)) for _k in _cmK5}
need_true("expCM at 10 against MLE at 50: macro-AUC and ECE resolved, p <= 0.004; the rest p >= 0.39", _cm105["macroAUC"][1] <= 0.004 and _cm105["ece"][1] <= 0.004 and min(_cm105[_k][1] for _k in ("acc", "brier", "far@pd0.9")) >= 0.385)
_cmch = {_k: _zp(_cmsv("emit-drone-only", 200, "channel: calib+rel", _k, "all"), _cmsv("emit-drone-only", 200, "channel: no RF+rel", _k, "all")) for _k in ("macroAUC", "brier", "ece")}
need_true("expCM the RF channel: p < 1e-10 on macro-AUC and Brier", _cmch["macroAUC"][1] < 1e-10 and _cmch["brier"][1] < 1e-10)
_cmcs = {_w: _zp(_cmsv(_w, 200, "channel: calib+rel", "brier", "radio-silent drones"), _cmsv(_w, 200, "channel: no RF+rel", "brier", "radio-silent drones")) for _w in ("standard", "emit-drone-only")}
need_tex("expCM VII-G", "The structure-free logistic baseline trails Proposed throughout, and with emission acting on drones only (Supplementary Material, Sec.~S5) Proposed holds at both sizes.")
need_supp("expCM S5 emission", f"so $P(T{{=}}\\text{{high}})$ is ${_cmwv('emit-drone-only', 'bird', 'silent', 'p_high'):.3f}$ for a silent bird and ${_cmwv('emit-drone-only', 'drone', 'emitting', 'p_high'):.3f}$ for an emitting drone (standard world: ${_cmwv('standard', 'bird', 'silent', 'p_high'):.3f}$, ${_cmwv('standard', 'drone', 'emitting', 'p_high'):.3f}$). With the conventions of Sec.~VII-G, the graph loses ${_cmgr:.3f}$ Brier, Proposed's excess is ${_cmexc[('emit-drone-only', 50)]:.3f}$ and ${_cmexc[('emit-drone-only', 200)]:.3f}$ at $N{{=}}50$ and $200$ (standard world ${_cmexc[('standard', 50)]:.3f}$, ${_cmexc[('standard', 200)]:.3f}$), and the best alternative, cMAP on the full table, trails by ${-_cmalt[50][0]:.3f}$ and ${-_cmalt[200][0]:.3f}$. Proposed at $N{{=}}10$ against MLE at $N{{=}}50$ has macro-AUC ${_cm105['macroAUC'][0]:+.3f}$, ECE ${_cm105['ece'][0]:+.3f}$ ($p\\le0.004$), accuracy ${_cm105['acc'][0]:+.3f}$, Brier ${_cm105['brier'][0]:+.3f}$, and FAR ${_cm105['far@pd0.9'][0]:+.3f}$ ($p\\ge0.39$). The RF channel adds ${_cmch['macroAUC'][0]:+.3f}$ macro-AUC and ${_cmch['brier'][0]:+.3f}$ Brier ($p<10^{{-10}}$), at ECE ${_cmch['ece'][0]:+.3f}$ ($p{{=}}{_cmch['ece'][1]:.2f}$), and on radio-silent drones ${_cmcs['emit-drone-only'][0]:+.3f}$ Brier ($p<10^{{-3}}$; ${_cmcs['standard'][0]:+.3f}$ in the standard world).")
# S1
_cms1v = lambda w, N, m, k, mean=None, a=None: (lambda _q: _q.sort_values("rep")[k].values)(_cms1[(_cms1.world == w) & (_cms1.N == N) & (_cms1.method == m) & ((_cms1["mean"] == mean) if mean else True) & (np.isclose(_cms1.a0, a) if a is not None else True)])
_cms1d = {(_w, _k): _zp(_cms1v(_w, 10, "Proposed", _k), _cms1v(_w, 50, "MLE", _k)) for _w in ("nonadd-0.5", "nonadd-1.0") for _k in ("acc", "brier")}
need_true("expCM at N = 10 against MLE at 50: p <= 0.009", max(_v[1] for _v in _cms1d.values()) <= 0.009 and all(_cms1d[(_w, "acc")][0] < 0 and _cms1d[(_w, "brier")][0] > 0 for _w in ("nonadd-0.5", "nonadd-1.0")))
need_tex("expCM the sentence", f"It breaks where its own assumptions fail (S1, S2), and the result at $N{{=}}10$ does not survive S1: against MLE at $N{{=}}50$ Proposed loses ${-_cms1d[('nonadd-0.5', 'acc')][0]:.3f}$ accuracy and ${_cms1d[('nonadd-0.5', 'brier')][0]:.3f}$ Brier in weak S1 and ${-_cms1d[('nonadd-1.0', 'acc')][0]:.3f}$ and ${_cms1d[('nonadd-1.0', 'brier')][0]:.3f}$ in strong S1 ($p\\le0.009$). With enough data both failures show in the training data:")
_cmft = lambda mean, a: _zp(_cms1v("nonadd-1.0", 200, "constrained full table", "brier", mean, a), _cms1v("nonadd-1.0", 200, "MLE", "brier"))
_cmf2, _cmu2, _cmf02, _cmu02 = _cmft("elicited", 2.0), _cmft("uniform", 2.0), _cmft("elicited", 0.2), _cmft("uniform", 0.2)
need_true("expCM the row at ESS 2 is the adaptive rule's table in strong S1", abs(_cmf2[0] - _wml[0]) < 1e-9)
need_true("expCM ESS 2 loses to MLE, ESS 0.2 beats it, p <= 0.003", _cmf2[0] > 0 and _cmu2[0] > 0 and _cmf02[0] < 0 and _cmu02[0] < 0 and max(_cmf02[1], _cmu02[1]) <= 0.003 and max(_cmf2[1], _cmu2[1]) <= 0.003)
need_tex("expCM the reason", f"though in strong S1 unsmoothed MLE beats that full table at an ESS of $2$ per column whatever its mean (elicited ${_cmf2[0]:+.3f}$, uniform ${_cmu2[0]:+.3f}$ Brier; Supplementary Material, Sec.~S5 lowers the ESS).")
need_supp_later("expCM the reason, in the supplement", f"and an ESS of $0.2$ reverses that (elicited ${_cmf02[0]:+.4f}$, uniform ${_cmu02[0]:+.4f}$ Brier, $p\\le0.003$).")
need_supp("expCM a sentence", "Rebuilding the world at another $\\tau_T$ also rebuilds the expert prior, which reads the same value, so in each world the prior knows that world's sharpness.")
need_true("expCM the prior reads the same value in the code", "ms.TAU_T" in open(os.path.join(HERE, "exp_noise_level.py"), encoding="utf-8").read())
need_tex("expCM a sentence", "The ESS $\\alpha_{0,ij}$ (expert confidence, small per column in the spirit of a minimally informative prior \\cite{kelly2011dirichlet}; Sec.~\\ref{sec:repro}) gives the posterior mean")
need_tex("expCM the noise level", "Proposed at $N{=}10$ keeps its macro-AUC lead over MLE at $N{=}50$ in all three worlds, but not its ECE lead at $\\tau_T{=}0.25$ (Supplementary Material, Sec.~S1).")
need_tex("expCM a sentence", "in the separate study above ($\\alpha_0{=}8$, $8$ seeds);")
need_tex("expCM CPT", "The conditional probability tables (CPTs) must be \\emph{statistically grounded}")
need_tex("expCM MAP", "(ii) constrained maximum a posteriori (MAP) estimation under monotonicity")
need_tex("expCM KL", "are Kullback--Leibler (KL) projections of the true threat table onto the family.")
need_true("expCM the first uses are the spelled-out ones", not HAVE_PAPER or (tex.find("(CPTs)") < tex.find("CPTs ") + 200 and tex.find("Kullback--Leibler (KL)") == tex.find("KL") - len("Kullback--Leibler (")))
if HAVE_PAPER:
    need_true("expCM Fig. 2 sets cls and emit upright", "(\\mathrm{cls},\\mathrm{emit})" in open(os.path.join(PAPER, "fig2_bn.tex"), encoding="utf-8").read())
if HAVE_PAPER and os.path.exists(os.path.join(PAPER, "main.aux")):
    _cmaux = open(os.path.join(PAPER, "main.aux"), encoding="utf-8", errors="replace").read()
    need_true("expCM the supplement's Eq. (11) is the chance-corrected weight", "\\newlabel{eq:chance}{{11}" in _cmaux and "the chance-corrected ones of Eq.~(11)" in _supp)
# the spectrograms, from the recordings
_cmps = open(os.path.join(HERE, "prepare_dronerf_four.py"), encoding="utf-8").read()
need_true("expCM the parameters in the script", all(_x in _cmps for _x in ("CHUNK_SAMPLES = 1_000_000", "MAX_CHUNKS = 10", "NFFT = 512", "HOP = 2048", "SIZE = 128", 'window="hann"', "np.log10(np.abs(z).astype(np.float32) + 1e-8)")))
for _x in ("Each file gives up to 10 chunks of 1,000,000 samples; each chunk goes through `scipy.signal.stft` with a 512-sample Hann window every 2048 samples (one-sided), 20 log10(|Z| + 1e-8), a bilinear resize to 128 x 128, and zero mean and unit standard deviation per spectrogram.",
           "python prepare_dronerf_four.py --raw /path/DroneRF --out /path/droneRF_four.npz --order data_derived/droneRF_four_order.csv",
           "`prepare_dronerf_four.py` — builds `droneRF_four.npz` from the DroneRF recordings"):
    need_true(f"expCM README: {_x[:50]}", _x in _rm)
need_tex("expCM VII-C", "on $128{\\times}128$ RF spectrograms ($512$-sample Hann windows every $2048$ samples, in dB) over four classes")
need_tex("expCM VI-E", "and \\texttt{prepare\\_dronerf\\_four.py} rebuilds the DroneRF spectrograms from the recordings;")
_cmord = pd.read_csv(os.path.join(HERE, "data_derived", "droneRF_four_order.csv"))
need_true("expCM the row order: 4,540 rows, no repeat", len(_cmord) == 4540 and not _cmord.duplicated().any())
_cmrb = open(os.path.join(RES, "expCM_dronerf_rebuild.txt"), encoding="utf-8").read()
need_true("expCM the rebuild is what the README says", "identical bit for bit 4539 of 4540" in _cmrb and "labels the same: True" in _cmrb and "4,539 of the 4,540 spectrograms equal that file bit for bit, and the other differs in one cell by one float32 step of its dB value (2e-7 after normalization; `results/expCM_dronerf_rebuild.txt`)" in _rm)
_cmrq, _cmlk = (open(os.path.join(HERE, _x), encoding="utf-8").read() for _x in ("requirements.txt", "requirements.lock"))
need_true("expCM requirements: av and openpyxl, with their versions; the torch line names the scripts", all(_x in _cmrq for _x in ("\nav ", "\nopenpyxl ", "exp_rf_e2e.py")) and "only for rf_confusion.py" not in _cmrq and "av==18.0.0\n" in _cmlk.replace("\r\n", "\n") and "openpyxl==3.1.5\n" in _cmlk.replace("\r\n", "\n"))
# the headers of the scripts, and the design note
for _zfn, _zxs in (("exp_noise_level.py", ("owed the reader",)), ("exp_rf_e2e.py", ("critique",)), ("exp_baselines.py", ("settle both questions", "default of BN parameter learning", "author-constructed"))):
    _zt = open(os.path.join(HERE, _zfn), encoding="utf-8").read()
    for _x in _zxs:
        need_true(f"expCM {_zfn} no longer says {_x}", _x not in _zt)
need_true("expCM the README does not call the design note", "VALIDATION_OPTIONS" not in _rm)
# the README, on the same claims
for _x in ("so this comparison mainly measures the prior; Supplementary Material Sec. S5 replaces it by a weak uniform one.",
           "At N=50 the proposed model beats MLE at N=100 on macro-AUC (+0.014) and ECE (−0.023), but MLE at N=200 overtakes it.",
           "A lighter uniform prior closes the ECE gap (alpha_0=0.5: p>=0.06 at every N), but no ESS from 0.01 to 2 and no BDeu total from 1 to 20 closes the macro-AUC or Brier gap (p<1e-3; Table S3).",
           "but at N=10 it trails MLE at N=50 on Brier (+0.019, p=0.02). B0 is flat (tau 0.95/0.9/0.95/1.0 for H/N/C/T against the world's 0.28 and 0.40); Supplementary Material Sec. S5 and Table S4 vary its flatness.",
           "- **Ablation path (N=80, one change per step, one order, alpha_0=2):**",
           "one correct-report probability per sensor and the emission-aware RF row at one level are in Supplementary Material Secs. S4 and S5.",
           "so the fitted weight of Eq. (10) stays as an empirical discount: it takes the place of P(reliable | context) in the network, but its fitted value is not an estimate of that probability.",
           "and the result at N=10 does not survive S1:",
           "(elicited +0.012, uniform +0.008 Brier; Supplementary Material Sec. S5 lowers the ESS).",
           "Under the scenarios' mix (81% background) drone-model confusions are 45 to 64% of the CNN errors per split type (77 to 83% of the test pool)",
           "session-bootstrap interval [0.35, 0.48]",
           "| `exp_comparators.py` | `expCM_summary.md`,"):
    need_true(f"expCM README: {_x[:50]}", _x in _rm)
# the old wording stays out
for _x in ("the default of BN parameter learning", "smoothing alone explains almost none", "the factor belongs to the near-truth prior", "the ablation measures elicited knowledge", "come from the prior mean", "because the prior is wrong", "mainly owing to that prior", "on macro-AUC, Brier, and calibration at every training size", "from about $300$ labeled scenarios", "We keep the deployed weight on three grounds", "modeled explicitly",
           "isolates what smoothing alone buys", "Smoothing is not the mechanism", "ordering of mechanisms", "dominant mechanism", "Is it the knowledge or the smoothing", "Most CNN errors", "a rule fixed in advance set aside", "as flat as ours"):
    need_true(f"expCM retired from main.tex and the supplement: {_x}", not HAVE_PAPER or (_x.lower() not in tex.lower() and _x.lower() not in _supp.lower()))
for _x in ("smoothing alone explains almost none", "from about 300 labeled", "because the prior is wrong", "mainly owing to that prior", "modeled explicitly", "three grounds"):
    need_true(f"expCM retired from the README: {_x}", _x.lower() not in _rm.lower())
if DRAFT or not HAVE_PAPER:
    print(f"expCM checks run: {_n_expd - _n0cm}")
assert _n_expd - _n0cm > 0


# ---- the plain manuscript: what was not used is not described, the process of the study is in one table of the supplement -----
_n0p11 = _n_expd
# the sentences that moved to the supplement and were registered after the supplement was read
for _l, _s in _supp_later[_supp_done:]:
    need_supp(_l, _s)
_supp_late_n = len(_supp_later) - _supp_done
need_true(f"expP11 every sentence registered for the supplement was looked up ({len(_supp_later)}, of them after the first pass {_supp_late_n}; 42 expected)",
          _supp_late_n == 42)
_p11body = "\n".join(re.sub(r"(?<!\\)%.*$", "", _l) for _l in open(TEX, encoding="utf-8").read().split("\n")) if HAVE_PAPER else ""
_P11_WORDS = (r"AirSim", r"Colosseum", r"Gazebo", r"digital[\s~-]*twin", r"proxy", r"sim-to-real", r"full\s+design", r"full\s+protocol", r"intended",
              r"added\s+after", r"after\s+the\s+fact", r"fixed\s+in\s+advance", r"fixed\s+before", r"not\s+evaluated\s+here", r"not\s+registered")
# the one exception: the sentence of Sec. VI-E that points to the table, by its title
_P11_POINTER = "Supplementary Material, Table~S5 lists the analyses added after the main results."
need_true("expP11 the pointer to the table of added analyses appears once", not HAVE_PAPER or _p11body.replace("\n", " ").count(_P11_POINTER) == 1)
_p11scan = re.sub(r"\\cite\{[^}]*\}", r"\\cite{}", _p11body.replace("\n", " ")).replace(_P11_POINTER, "")   # a BibTeX key is not text
for _pat in _P11_WORDS:
    _m = re.search(_pat, _p11scan, re.I)
    need_true(f"expP11 main.tex, comments aside, does not say {_pat!r}" + (f" (found {_m.group(0)!r})" if _m else ""), not HAVE_PAPER or _m is None)
# the supplement: none of the first six words anywhere; the process words only in the section of the added analyses
_p11s = re.sub(r"(?m)(?<!\\)%.*$", "", open(_supp_path, encoding="utf-8").read()) if HAVE_PAPER and os.path.exists(_supp_path) else ""   # a missing supplement is reported by expSup, not a crash
_p11a = _p11s.find("\\section{Analyses added after the main results}")
_p11b = _p11s.find("\\section{Figures}")
need_true("expP11 the supplement has the section of added analyses just before the figures", not HAVE_PAPER or 0 < _p11a < _p11b)
_p11out = _p11s[:_p11a] + _p11s[_p11b:] if HAVE_PAPER else ""
for _pat in _P11_WORDS[:9]:
    _m = re.search(_pat, _p11s, re.I)
    need_true(f"expP11 supplement does not say {_pat!r}" + (f" (found {_m.group(0)!r})" if _m else ""), not HAVE_PAPER or _m is None)
for _pat in _P11_WORDS[9:] + (r"chosen\s+after", r"in\s+advance", r"reported\s+regardless"):
    _m = re.search(_pat, _p11out, re.I)
    need_true(f"expP11 supplement, outside the table of added analyses, does not say {_pat!r}" + (f" (found {_m.group(0)!r})" if _m else ""), not HAVE_PAPER or _m is None)
for _pat in (r"chosen\s+after", r"in\s+advance", r"reported\s+regardless", r"amended", r"as\s+set\s+here", r"we\s+did\s+not"):
    _m = re.search(_pat, _p11scan, re.I)
    need_true(f"expP11 main.tex does not say {_pat!r}" + (f" (found {_m.group(0)!r})" if _m else ""), not HAVE_PAPER or _m is None)
# Table 1 lists the evaluation world only; Fig. 1 names simulated scenarios
_p11t1 = re.search(r"\\label\{tab:vars\}(.*?)\\end\{tabularx\}", _p11body, re.S)
need_true("expP11 Table 1 holds the evaluation world's variables only", not HAVE_PAPER or (_p11t1 is not None and all(_x not in _p11t1.group(1) for _x in ("Altitude", "RCS", "Formation", "Weather", "spoofed", "swarm"))
          and all(_x in _p11t1.group(1) for _x in ("Speed $v$", "Range-rate $\\dot r$", "CPA $d$", "Class $\\mathrm{cls}$", "Emission $\\mathrm{emit}$", "Airspace $z$", "Radar; EO/IR", "Acoustic; RF"))))
need_true("expP11 Table 1's states are those of model_spec.py", not HAVE_PAPER or all(("\\{" + ",".join(_ms.STATES[_k]).replace("nofly", "no-fly").replace("noncoop", "non-coop") + "\\}") in _p11t1.group(1) for _k in ("v", "rdot", "d", "cls", "emit", "z", "T", "Rr", "Re", "Ra", "Rf")))
# the references: five left with their sentences, the rest renumbered in the order of first citation
_p11bbl = open(os.path.join(PAPER, "main.bbl"), encoding="utf-8", errors="replace").read() if HAVE_PAPER else ""
_p11keys = re.findall(r"\\bibitem\{([^}]+)\}", _p11bbl)
_p11cites = []
for _c in re.findall(r"\\cite\{([^}]+)\}", _p11body):
    for _k in _c.split(","):
        if _k.strip() not in _p11cites:
            _p11cites.append(_k.strip())
need_true(f"expP11 the bibliography has 48 entries ({len(_p11keys)})", not HAVE_PAPER or len(_p11keys) == 48)
need_true("expP11 the references are numbered in the order of first citation", not HAVE_PAPER or _p11keys == _p11cites)
for _k in ("shah2018airsim", "koziarski2023simreal", "cooke1991experts", "tobin2017domain", "chan2005distance"):
    need_true(f"expP11 {_k} is neither cited nor printed", not HAVE_PAPER or (_k not in _p11cites and _k not in _p11keys))
need_true("expP11 the base framework stays [5]", not HAVE_PAPER or _p11keys.index("song2026base") == 4)
need_true("expP11 the one source kept for training a BN on simulated data stays", not HAVE_PAPER or "ademujimi2022digitaltwin" in _p11cites)
need_true("expP11 the Chan-Darwiche distance is not described", not HAVE_PAPER or "Chan--Darwiche" not in _p11body)
# the README, after the manuscript: no simulator, no proxy, no protocol that was not run; the infrared check stays, outside the paper
_p11rm = open(os.path.join(HERE, "README.md"), encoding="utf-8").read()
for _pat in _P11_WORDS[:6] + (r"not\s+evaluated\s+as\s+specified", r"factor\s+of\s+about", r"two-fold", r"loses\s+the\s+headline", r"reported\s+regardless", r"probabilistic\s+basis", r"fixed\s+in\s+advance\s+flags", r"as\s+set\s+here", r"needs\s+an\s+informative\s+prior"):
    _m = re.search(_pat, _p11rm, re.I)
    need_true(f"expP11 the README does not say {_pat!r}" + (f" (found {_m.group(0)!r})" if _m else ""), _m is None)
need_true("expP11 the README points the infrared check to its own Key results, not to a section of the paper", all("Sec. VI-B" not in _l for _l in _p11rm.split("\n") if "infrared" in _l or "EO/IR gate" in _l)
          and sum("Key results, EO/IR gate" in _l for _l in _p11rm.split("\n")) == 2)
need_true("expP11 the README keeps the infrared result with the values of expO", f"balanced accuracy {_obn:.2f}, session-bootstrap interval {_oci(_ogb, _ONN)}, and {_obc:.2f}, {_oci(_ogb, _OCN)}, against 0.25 for no information and a limit of 0.35 that both intervals reach" in _p11rm)
need_true("expP11 the README names the evaluation world as the paper does", "These hold in the evaluation world of Sec. VI-A," in _p11rm and "The synthetic evaluation runs in one fully specified generative BN, the\nevaluation world of Sec. VI-A;" in _p11rm)
if DRAFT:
    print(f"expP11 checks run: {_n_expd - _n0p11}")
assert _n_expd - _n0p11 > 0

need_true("expP11 no sentence is registered for the supplement after its last lookup", len(_supp_later) == _supp_done + _supp_late_n)
# ---- verdict (last: every check above must have had a chance to run) -----
if __name__ == "__main__":
    if fails:
        print("MISMATCHES:")
        for f in fails:
            print(" -", f)
        sys.exit(1)
    print("OK: the tables and the quoted numbers match the files in results/." if HAVE_PAPER
          else "OK: the files in results/ pass. No manuscript was compared.")

