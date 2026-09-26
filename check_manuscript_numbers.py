"""
Consistency gate: verify that the headline numbers quoted in main.tex match
the results/ files (single source of truth). Exits nonzero on any mismatch.
Run after any experiment rerun and before building the submission PDF.
"""
from __future__ import annotations
import re, sys, os, glob
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")

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
    ("exp_constraints.csv", "exp_constraints.py"),
    ("expA_ess_sensitivity.csv", "exp_ess_sensitivity.py"),
    ("expA_baselines.csv", "exp_ess_sensitivity.py"),
    ("expB_baselines_raw.csv", "exp_baselines.py"),
    ("expN_noise_raw.csv", "exp_noise_level.py"),
    ("expL_latency.csv", "exp_latency.py"),
    ("expR_reliability_classes.csv", "exp_reliability_classes.py"),
    ("expK_ece_bins.csv", "exp_ece_bins.py"),
]

# ---- banned-phrase gate: wording retired in earlier revisions must not
# reappear in the manuscript or the companion documents.
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
        "the ESS was chosen from the Fig. 5 plateau, so say so",
    r"credal\s+intervals?\s+propagated":
        "the analysis is a one-at-a-time envelope",
    r"measured\s+multi-sensor\s+confusion":
        "synthetic-world matrices are estimated, not measured",
    r"(now\s+contains|tracks\s+the\s+frozen|are\s+included\s+in)"
    r"[^.]{0,40}released\s+repository":
        "claims about the public repository are not true until it is pushed",
    r"54-entry": "the threat table has 72 entries and 54 free parameters",
    r"\blabelled\b": "the manuscript uses US spelling (labeled)",
    r"fast enough for (online|real-time)":
        "the per-update inference cost is measured -- quote it",
}

# These two are about the manuscript's voice, not about facts: a response
# letter is supposed to name the original submission and to argue both
# directions of a result. Applied to main.tex only.
BANNED_MANUSCRIPT_ONLY = {
    r"original submission|previous submission|in the revision":
        "review history must not appear in the manuscript body",
    r"honest reading|we state both|we report both directions":
        "response-letter tone does not belong in the manuscript",
}

# ---- submission-readiness: the manuscript cites a release tag, so that tag
# must exist and be pushed before the paper goes out.
RELEASE_TAG = "v2.0"
if RELEASE_TAG in tex:
    import subprocess
    try:
        local = subprocess.run(["git", "tag", "--list", RELEASE_TAG],
                               cwd=HERE, capture_output=True, text=True,
                               timeout=30).stdout.strip()
        remote = subprocess.run(["git", "ls-remote", "--tags", "origin",
                                 RELEASE_TAG], cwd=HERE, capture_output=True,
                                text=True, timeout=60).stdout.strip()
    except Exception as exc:
        local = remote = ""
        fails.append(f"release: could not query git ({exc})")
    if not local:
        fails.append(f"release: the manuscript cites tag {RELEASE_TAG}, "
                     "which does not exist locally -- create it at the "
                     "commit that produced these results")
    elif not remote:
        fails.append(f"release: tag {RELEASE_TAG} exists locally but is not "
                     "pushed -- the manuscript's reproducibility claim is "
                     "not yet true")
DOCS = [os.path.join(HERE, "README.md")]
def live_letter():
    """The response letter that would be submitted. The name carries a
    timestamp, so archived copies can sit beside it; take the newest and
    check only that one, or every stale quotation in an old copy would
    fail the gate."""
    cands = glob.glob(os.path.join(PAPER, "peer review",
                                   "ResponseToReviewers*.md"))
    return max(cands, key=os.path.getmtime) if cands else None


LETTER = live_letter() if HAVE_PAPER else None
if HAVE_PAPER:
    DOCS = [TEX,
            os.path.join(PAPER, "cover_letter.tex"),
            os.path.join(PAPER, "differences_from_prior.tex")] + DOCS
    if LETTER:
        DOCS.append(LETTER)
for doc in DOCS:
    if not os.path.exists(doc):
        fails.append(f"banned-phrase: {doc} missing")
        continue
    body = open(doc, encoding="utf-8").read()
    patterns = dict(BANNED)
    if os.path.abspath(doc) == os.path.abspath(TEX):
        patterns.update(BANNED_MANUSCRIPT_ONLY)
    for pattern, why in patterns.items():
        m = re.search(pattern, body, re.I)
        if m:
            fails.append(f"banned wording \"{m.group(0)}\" in "
                         f"{os.path.basename(doc)} -- {why}")

# ---- quotation gate: when the response letter quotes the manuscript, the
# sentence has to still be there. This has drifted three times, and a
# reviewer reading the letter beside the paper is the one who notices.
def _norm_quote(t):
    t = re.sub(r"\$[^$]*\$", " ", t)
    t = re.sub(r"\\(?:emph|textbf|texttt|mathrm|text)\{([^{}]*)\}", r"\1", t)
    t = re.sub(r"\\[a-zA-Z]+\*?", " ", t)
    for _a, _b in (("~", " "), ("{", " "), ("}", " "), ("---", " "),
                   ("--", " "), ("\u2014", " "), ("\u2013", " "),
                   ("\u2019", "'"), ("`", ""), ("*", "")):
        t = t.replace(_a, _b)
    t = re.sub(r"[^a-z0-9' ]+", " ", t.lower())
    return re.sub(r"\s+", " ", t).strip()


_LETTERS = [LETTER] if LETTER else []
if _LETTERS:
    _texn = _norm_quote(open(TEX, encoding="utf-8").read())
    _BOUND = ("**Action.**", "**Response.**", "**Concern.**", "###", "\n\n",
              "\u2026")
    for _lp in _LETTERS:
        _let = open(_lp, encoding="utf-8").read()
        _skip = []
        for _m in re.finditer(r"\*\*Concern\.\*\*", _let):
            _e = _let.find("\n\n", _m.end())
            _skip.append((_m.start(), _e if _e > 0 else len(_let)))
        for _m in re.finditer(r'"([^"\n]{40,260})"', _let):
            _q, _p = _m.group(1), _m.start()
            if any(_a <= _p <= _b for _a, _b in _skip):
                continue
            if any(_t in _q for _t in _BOUND) or len(_q.split()) < 8:
                continue
            if _norm_quote(_q) not in _texn:
                fails.append(
                    f"quotation: {os.path.basename(_lp)} quotes "
                    f"\"{_q[:60]}...\" as manuscript text, but main.tex "
                    "does not contain it")

# ---- build gate: the template assets the class file loads must exist, and
# the LaTeX logs must be free of errors, missing files and undefined
# references. A missing asset does not stop pdflatex in nonstopmode -- it
# silently substitutes a draft placeholder box -- so only the log catches it.
TEMPLATE_ASSETS = ["ieeeaccess.cls", "spotcolor.sty", "IEEEtran.bst",
                   "logo.png", "notaglinelogo.png", "bullet.png"]
for asset in (TEMPLATE_ASSETS if HAVE_PAPER else []):
    if not os.path.exists(os.path.join(PAPER, asset)):
        fails.append(f"template: {asset} is missing from the manuscript "
                     "folder -- the build will substitute a placeholder")

LOG_PATTERNS = [
    (r"^! ", "LaTeX error"),
    (r"not found", "missing input file"),
    (r"Citation `[^']+' .*undefined", "undefined citation"),
    (r"Reference `[^']+' .*undefined", "undefined reference"),
]
for log_name in (("main.log", "main_diff.log")
                 if HAVE_PAPER else ()):
    log_path = os.path.join(PAPER, log_name)
    if not os.path.exists(log_path):
        fails.append(f"build: {log_name} missing -- rebuild the document")
        continue
    body = open(log_path, encoding="utf-8", errors="replace").read()
    for pattern, why in LOG_PATTERNS:
        hits = re.findall(pattern, body, re.M)
        if hits:
            fails.append(f"build: {log_name} reports {len(hits)} "
                         f"{why}(s), first: {hits[0].strip()[:60]!r}")

# ---- (a) control-character gate: LaTeX written through a non-raw Python
# string silently loses its backslash ("\\texttt" -> TAB + "exttt"), which
# builds without error and prints garbage. Tabs are legal in .py sources and
# in latexdiff's own preamble, so only the submission documents are swept.
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

# ---- (c) artifact currency: the marked-up source must have been generated
# from the final manuscript, and every PDF must be newer than its source.
DIFF_TEX = os.path.join(PAPER, "main_diff.tex")
if not HAVE_PAPER:
    pass
elif os.path.exists(DIFF_TEX):
    head = open(DIFF_TEX, encoding="utf-8", errors="replace").read(4000)
    m = re.search(r"%DIF ADD main\.tex (.+)", head)
    if not m:
        fails.append("diff: main_diff.tex has no %DIF ADD header -- "
                     "regenerate it with latexdiff")
    else:
        import email.utils, time as _time
        stamp = m.group(1).strip()
        try:
            gen = _time.mktime(_time.strptime(stamp, "%a %b %d %H:%M:%S %Y"))
        except ValueError:
            gen = None
            fails.append(f"diff: unparsable latexdiff timestamp {stamp!r}")
        # latexdiff records whole seconds, so allow sub-second slack
        if gen is not None and gen + 2 < os.path.getmtime(TEX):
            fails.append("diff: main_diff.tex was generated from a main.tex "
                         f"of {stamp}, older than the current manuscript -- "
                         "rerun latexdiff, then rebuild")
else:
    fails.append("diff: main_diff.tex missing")

# ---- artifact-freshness gate: a PDF that is actually uploaded must not be
# older than the .tex it is built from.
PDF_PAIRS = [(os.path.join(PAPER, stem + ".pdf"),
              os.path.join(PAPER, stem + ".tex"))
             for stem in ("main", "main_diff", "main_highlight",
                          "main_review", "cover_letter",
                          "differences_from_prior")
             if os.path.exists(os.path.join(PAPER, stem + ".tex"))]
for pdf_path, tex_path in (PDF_PAIRS if HAVE_PAPER else []):
    if not os.path.exists(pdf_path):
        fails.append(f"artifact: {os.path.basename(pdf_path)} missing")
    elif os.path.getmtime(pdf_path) < os.path.getmtime(tex_path):
        fails.append(f"artifact: {os.path.basename(pdf_path)} is older than "
                     f"its .tex -- rebuild it before submission")
for rf, script in RESULT_FILES:
    path = os.path.join(RES, rf)
    if not os.path.exists(path):
        fails.append(f"freshness: {rf} missing")
        continue
    need_after = max(core_mtime, os.path.getmtime(os.path.join(HERE, script)))
    if os.path.getmtime(path) < need_after:
        fails.append(f"freshness: {rf} is OLDER than the code that makes it "
                     f"-- rerun {script}")


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
need("E2 lam1 ess8 N400 cMAP acc", x8[x8.method == "cMAP"]["acc"].mean())
need("E2 lam1 ess8 N400 cMAP ece", x8[x8.method == "cMAP"]["ece"].mean())

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

# ---- new-reviewer round: standard baselines and the noise sweep ----------
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

# ---- new-reviewer round: measured cost and the per-class decomposition ---
lat = pd.read_csv(os.path.join(RES, "expL_latency.csv"))
nq = int(lat.n_queries.sum())
if HAVE_PAPER and f"{nq//1000},{nq%1000:03d}" not in tex:
    fails.append(f"expL: expected the timed-query count {nq:,} in main.tex")
need("expL mean query ms", lat.q_mean_ms.mean())
need("expL p95 query ms", lat.q_p95_ms.mean())
need("expL max query ms", lat.q_max_ms.max())
need("expL assemble ms", lat.assemble_s.mean() * 1e3, 1)
need("expL MAP-EM s", lat.mapem_s.mean(), 2)
need("expL grounding s", lat.ground_s.mean(), 2)
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

# ---- verdict (last: every check above must have had a chance to run) -----
if fails:
    print("MISMATCHES:")
    for f in fails:
        print(" -", f)
    sys.exit(1)
print(f"OK: all {len(ts)}-cell tables and quoted headline numbers "
      "match results/ files.")

