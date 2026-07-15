# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Cross-model comparison report for the statistical models.

Run after the models have been fitted (``python
scripts/fit_statistical_model.py all``). Produces, under
``output/statistical_models/comparison/``:

- ``itt_vs_joint_tau.csv`` — per-outcome tau from the LRPITT single-outcome
  fits alongside tau_k from the LRPITT12 joint (consistency check), on the shared
  (non-floored) outcomes W, R, E, L, B.
- ``triangulation_consistency.csv`` — per-outcome cross-*design* consistency of the
  randomised on-intervention effect (single-outcome ITT ``tau`` vs waitlist-crossover
  t2 arm contrast ``tau_t2`` vs gain-factor ``beta_trt``): whether the analyses agree
  in direction and their intervals overlap on the shared logit scale. A consistency
  check, never a pooled estimate — the analyses share the same trial data (issue #230
  §6).
- ``tau_forest.png`` — forest plot of the LRPITT12 joint taus, overlaid with
  the LRPITT single-outcome taus on those shared outcomes.
- ``mechanism_forest.png`` — forest plot of the marginal slope of each
  mechanism GP (LRP56 R->W, LRP57 E->W, LRP58 L->W). Slopes are computed
  from each model's actual posterior ``f_mech`` samples on its own
  ``logit(mech_post)`` grid (not a shared dummy grid).

Requires ``trace.nc`` and ``tau_summary.csv`` under
``output/statistical_models/models/{model_id}-{config}/`` for each model.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import shutil

import arviz as az
import matplotlib.pyplot as plt
import xarray as xr
import numpy as np
import pandas as pd

from language_reading_predictors import paths as _paths
from language_reading_predictors.statistical_models.definitions import (
    MODEL_REGISTRY,
    Status,
)
from language_reading_predictors.statistical_models.measures import (
    MEASURES,
    ROPE_DELTA_PROB,
)
from language_reading_predictors.statistical_models.preprocessing import (
    load_and_prepare,
    logit_safe,
)

# Heavily-floored outcomes whose exploratory ITT headline is the binary off-floor
# effect (a risk difference), not the graded logit tau shown in the joint. Any of
# these appearing in a forest/CSV of graded taus is flagged so the artefact does
# not misrepresent it.
FLOORED_SYMBOLS: frozenset[str] = frozenset(ROPE_DELTA_PROB)


# Single-outcome ITT models (LRPITT suite, #119) overlaid on the LRPITT12 joint, on
# the outcomes the joint also carries. The floored outcomes P (lrp-rli-itt-009) and N
# (lrp-rli-itt-011) are excluded from the graded overlay: their exploratory headline is the
# binary off-floor effect, read from their own reports rather than compared to the
# joint's graded tau. F/T have standalone ITTs but are not outcomes in this joint fit.
ITT_IDS: list[tuple[str, str]] = [
    ("lrp-rli-itt-010", "W"),
    ("lrp-rli-itt-005", "R"),
    ("lrp-rli-itt-006", "E"),
    ("lrp-rli-itt-007", "L"),
    ("lrp-rli-itt-008", "B"),
]
MECH_IDS: list[tuple[str, str]] = [("lrp-rli-mech-056", "R"), ("lrp-rli-mech-057", "E"), ("lrp-rli-mech-058", "L")]
JOINT_ID = "lrp-rli-itt-012"

# Mechanism models compared by PSIS-LOO: the LRP58 baseline (L -> W) against the
# interaction extensions on the *same* word-reading outcome/rows (a like-for-like
# elpd comparison). LRP70 is intentionally excluded: it was repurposed to a
# ``growth``-kind joint growth-curve model (posterior dims child/wave/outcome, no
# ``obs_id``), so it is a different dataset and a growth-vs-mechanism LOO would be
# the cross-dataset comparison ``_loo_compare`` forbids. The celf mechanism model
# is deferred pending a DAG review (see ``lrp-rli-mech-071.py``).
LOO_COMPARE_IDS: list[str] = ["lrp-rli-mech-058", "lrp-rli-mech-071"]

# Phonics route (LRP72): the interaction model vs its no-interaction baseline,
# same decoding outcome — a clean nested PSIS-LOO test of the L x B interaction.
# NOT comparable to the LOO_COMPARE_IDS set (different outcome: decoding vs W).
PHONICS_LOO_IDS: list[str] = ["lrp-rli-mech-072", "lrp-rli-mech-172"]

# Age moderation (LRP73): interaction vs no-interaction baseline, same word-reading
# outcome — a clean nested PSIS-LOO test of the L x age interaction.
AGE_LOO_IDS: list[str] = ["lrp-rli-mech-073", "lrp-rli-mech-173"]

# Dose-response (LRP77, #104 Phase 2): the period-varying dose model vs its
# pooled-dose comparator, same word-reading outcome and rows — a nested PSIS-LOO
# test of whether the dose-gain slope varies by period.
DOSE_LOO_IDS: list[str] = ["lrp-rli-dose-077", "lrp-rli-dose-277"]

# DiD period-resolved letter-sound dose (LRPDID07, #135): the period-varying dose
# model vs its pooled-dose comparator, same letter-sound outcome and rows — a
# nested PSIS-LOO test of whether the L dose-gain slope varies by period (the
# DAG-clean DiD analogue of the LRP77 word-reading test; never conditions on the
# IS collider attend_cumul).
DID_DOSE_LOO_IDS: list[str] = ["lrp-rli-did-007", "lrp-rli-did-107"]

# Mediation family (#84 + the 2026-07 expansion): every g-formula route to word
# reading, compared on the shared response (words-out-of-test-length) scale. The
# mediation fits carry no PSIS-LOO (the g-formula is not a pointwise-likelihood
# model), so this is a decomposition-summary comparison, not a nested-LOO test.
# ``(model_id, human label, indirect-effect row)`` — the row is the model's
# headline indirect effect (NIE for single mediators, NIE_joint for the two-mediator
# blocks, IIE for the interventional analogue).
MEDIATION_IDS: list[tuple[str, str, str]] = [
    ("lrp-rli-med-059", "L — letter sounds", "NIE"),
    ("lrp-rli-med-062", "L+B — code route (composite)", "NIE"),
    ("lrp-rli-med-064", "{L,E} — joint block", "NIE_joint"),
    ("lrp-rli-med-066", "{L,B} — joint (parallel)", "NIE_joint"),
    ("lrp-rli-med-075", "L→B→W — sequential", "NIE_joint"),
    ("lrp-rli-med-068", "TE — taught expressive vocab", "NIE"),
    ("lrp-rli-med-080", "TR — taught receptive vocab", "NIE"),
    ("lrp-rli-med-074", "N — nonword decoding (floored)", "NIE"),
    ("lrp-rli-med-076", "L — longitudinal (t2→t4)", "NIE"),
    ("lrp-rli-med-078", "L — interventional (IIE)", "IIE"),
]


def _run_dir(model_id: str, config: str) -> str:
    return os.path.join(str(_paths.stat_models_dir()), f"{model_id}-{config}")


def _gate_status(model_id: str, config: str) -> str:
    """Convergence-gate verdict for a fitted run (issue #274 item 3).

    Reads ``diagnostics_summary.json`` from the run directory and returns
    ``"PASS"`` / ``"REVIEW"`` / ``"MISSING"``. A ``REVIEW`` fit "is not
    interpretable — fix the model, do not report it" (METHODS.md), so a tau/slope
    from such a run must never enter the comparison forests unmarked.
    """
    path = os.path.join(_run_dir(model_id, config), "diagnostics_summary.json")
    if not os.path.exists(path):
        return "MISSING"
    try:
        with open(path) as f:
            payload = json.load(f)
    except Exception:  # pragma: no cover - defensive
        return "MISSING"
    return "PASS" if payload.get("passed") is True else "REVIEW"


def _gate_ok(model_id: str, config: str) -> bool:
    """True only if the run exists and its gate passed."""
    return _gate_status(model_id, config) == "PASS"


# ---------------------------------------------------------------------------
# ITT vs joint
# ---------------------------------------------------------------------------


def build_itt_vs_joint(config: str) -> pd.DataFrame | None:
    rows: list[dict] = []
    for model_id, outcome in ITT_IDS:
        tau_path = os.path.join(_run_dir(model_id, config), "tau_summary.csv")
        if not os.path.exists(tau_path):
            return None
        df = pd.read_csv(tau_path)
        rows.append(
            {
                "config": config,
                "outcome": outcome,
                "source": model_id,
                "floored": outcome in FLOORED_SYMBOLS,
                "converged": _gate_ok(model_id, config),
                "tau_median": df["tau_logit_median"].iloc[0],
                "tau_lo": df["tau_logit_lo"].iloc[0],
                "tau_hi": df["tau_logit_hi"].iloc[0],
            }
        )
    joint_path = os.path.join(_run_dir(JOINT_ID, config), "tau_summary.csv")
    if not os.path.exists(joint_path):
        return None
    joint = pd.read_csv(joint_path)
    joint_ok = _gate_ok(JOINT_ID, config)
    for _, row in joint.iterrows():
        rows.append(
            {
                "config": config,
                "outcome": row["outcome"],
                "source": JOINT_ID,
                "floored": row["outcome"] in FLOORED_SYMBOLS,
                "converged": joint_ok,
                "tau_median": row["tau_median"],
                "tau_lo": row["tau_lo"],
                "tau_hi": row["tau_hi"],
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Cross-design triangulation (#230 §6)
# ---------------------------------------------------------------------------

# Per shared graded outcome, the three analyses on the same logit scale:
# single-outcome ITT tau, waitlist-crossover randomised t2 arm contrast, and gain-factor
# on-intervention beta_trt. These are not statistically independent designs: they
# reuse the same trial data, and the DiD additionally imposes crossover/history
# assumptions. The registry-derived catalogue avoids silently omitting newly completed
# suites (the previous hand-written list missed TR, TE and F).
_TRIANGULATION_OUTCOME_ORDER: tuple[str, ...] = (
    "W",
    "R",
    "E",
    "L",
    "B",
    "TR",
    "TE",
    "F",
)


def _triangulation_outcomes() -> list[tuple[str, str, str, str]]:
    """Build the complete non-floored ITT/DiD/gain-factor catalogue.

    Select only the primary member of each family: model-of-record ITTs, ordinary
    waitlist-crossover arm-by-wave DiDs (not dose or heterogeneity variants), and
    full-sample gain-factor models (not treated-only companions). Outcomes must be
    present in all three families and must share the graded logit estimand.
    """
    primary_itts = {
        definition.model_id: definition
        for definition in MODEL_REGISTRY.values()
        if definition.kind == "itt" and definition.status is Status.MODEL_OF_RECORD
    }
    by_kind: dict[str, dict[str, str]] = {"itt": {}, "did": {}, "gain_factors": {}}
    for definition in MODEL_REGISTRY.values():
        outcome = definition.outcome
        if outcome is None or definition.floored:
            continue
        if definition.model_id in primary_itts:
            by_kind["itt"][outcome] = definition.model_id
        elif (
            definition.kind == "did"
            and definition.status is Status.ROBUSTNESS
            and definition.base in primary_itts
        ):
            by_kind["did"][outcome] = definition.model_id
        elif (
            definition.kind == "gain_factors"
            and definition.status is Status.ASSOCIATION
            and definition.base is None
        ):
            by_kind["gain_factors"][outcome] = definition.model_id

    shared = set.intersection(*(set(ids) for ids in by_kind.values()))
    ordered = [o for o in _TRIANGULATION_OUTCOME_ORDER if o in shared]
    ordered.extend(sorted(shared.difference(ordered)))
    return [
        (
            outcome,
            by_kind["itt"][outcome],
            by_kind["did"][outcome],
            by_kind["gain_factors"][outcome],
        )
        for outcome in ordered
    ]


TRIANGULATION_OUTCOMES: list[tuple[str, str, str, str]] = _triangulation_outcomes()


def _itt_effect(model_id: str, config: str) -> dict | None:
    """Single-outcome ITT tau (logit) from ``tau_summary.csv``, or None if absent."""
    p = os.path.join(_run_dir(model_id, config), "tau_summary.csv")
    if not os.path.exists(p):
        return None
    df = pd.read_csv(p)
    return {
        "term": "tau",
        "median": float(df["tau_logit_median"].iloc[0]),
        "lo": float(df["tau_logit_lo"].iloc[0]),
        "hi": float(df["tau_logit_hi"].iloc[0]),
        "prob_pos": float(df["prob_tau_pos"].iloc[0]),
    }


def _did_effect(model_id: str, config: str) -> dict | None:
    """Randomised t2 arm contrast from ``did_summary.csv``, or None.

    New arm-by-wave fits report this as ``tau_t2``. The legacy ``delta`` fallback
    keeps previously fitted artefacts readable; in that constrained model ``delta``
    was forced to equal the t2 arm contrast, so it is labelled explicitly rather than
    silently treated as a result from the redesigned model.
    """
    p = os.path.join(_run_dir(model_id, config), "did_summary.csv")
    if not os.path.exists(p):
        return None
    df = pd.read_csv(p)
    if "tau_t2_median" in df.columns:
        return {
            "term": "tau_t2",
            "median": float(df["tau_t2_median"].iloc[0]),
            "lo": float(df["tau_t2_lo"].iloc[0]),
            "hi": float(df["tau_t2_hi"].iloc[0]),
            "prob_pos": float(df["prob_tau_t2_pos"].iloc[0]),
        }
    return {
        "term": "legacy_delta_constrained_to_t2",
        "median": float(df["delta_median"].iloc[0]),
        "lo": float(df["delta_lo"].iloc[0]),
        "hi": float(df["delta_hi"].iloc[0]),
        "prob_pos": float(df["prob_delta_pos"].iloc[0]),
    }


def _gain_factor_effect(model_id: str, config: str) -> dict | None:
    """Gain-factor on-intervention ``beta_trt`` (logit) from ``factor_summary.csv``."""
    p = os.path.join(_run_dir(model_id, config), "factor_summary.csv")
    if not os.path.exists(p):
        return None
    df = pd.read_csv(p)
    row = df[df["term"] == "beta_trt"]
    if row.empty:
        return None
    return {
        "term": "beta_trt",
        "median": float(row["median"].iloc[0]),
        "lo": float(row["lo"].iloc[0]),
        "hi": float(row["hi"].iloc[0]),
        "prob_pos": float(row["prob_positive"].iloc[0]),
    }


_TRIANGULATION_DESIGNS: tuple[tuple[str, str], ...] = (
    ("itt", "ITT tau"),
    ("did", "crossover-model t2 arm contrast"),
    ("gf", "gain-factor beta_trt"),
)


def build_triangulation(config: str) -> pd.DataFrame | None:
    """Per-outcome cross-design consistency of the randomised on-intervention effect.

    For each shared graded outcome reads the logit-scale treatment effect from up to
    three analyses — single-outcome ITT (``tau``), the waitlist-crossover model's
    randomised t2 arm contrast (``tau_t2``), and gain-factor ANCOVA (``beta_trt``) — and
    reports whether they **agree in direction** (all favour the same side of zero) and
    whether their credible intervals **mutually overlap**. ``consistent`` is true when
    both hold. The direction/overlap verdict is computed over the *converged* designs
    (``diagnostics_summary.json`` gate PASS); it is left blank (``pd.NA``) when fewer
    than two converged designs are available.

    This is a **consistency check, not a pooled estimate**: the three analyses share
    the same trial data, so their effects must never be averaged into one headline —
    the value is in whether distinct model specifications triangulate on the same story
    (issue #230 §6). Returns ``None`` if no outcome has at least two design summaries.

    Two interpretation caveats (#295 review), why the flags are read qualitatively:

    - **The three logit effects are on the same *scale* but not the same conditioning
      set.** ITT ``tau`` adjusts for own baseline + age; the gain-factor ``beta_trt``
      additionally adjusts for ability, upstream DAG skills and the exogenous
      confounders; the crossover model's ``tau_t2`` is the same randomised t2 arm
      contrast estimated in a longitudinal arm-by-wave likelihood. Because the ITT
      and crossover fits share the t2 comparison, agreement is a parameterisation
      check rather than independent evidence.
      Because the logit link is **non-collapsible**, a more-adjusted conditional
      log-odds effect is expected to be systematically larger in magnitude even under an
      identical truth. Direction agreement is robust to this, but ``intervals_overlap``
      compares *magnitudes*, so a ``consistent = False`` driven by non-overlap can be a
      conditioning-set artefact rather than a genuine design disagreement — read it
      alongside the direction flag, not on its own.
    - **Direction is a coarse sign check at the 0.5 boundary.** Two essentially-null
      designs with opposite-sign medians (``prob_pos`` e.g. 0.55 and 0.45) are marked
      ``direction_agree = False`` even though neither shows a real signal; a
      ``direction_agree = False`` with wide, overlapping intervals is a null-result
      artefact, not a contradiction. (``prob_pos == 0.5`` exactly is treated as agreeing
      with either side, which is harmless.)
    """
    readers = {"itt": _itt_effect, "did": _did_effect, "gf": _gain_factor_effect}
    rows: list[dict] = []
    for outcome, itt_id, did_id, gf_id in TRIANGULATION_OUTCOMES:
        ids = {"itt": itt_id, "did": did_id, "gf": gf_id}
        ests: dict[str, dict] = {}
        for key, model_id in ids.items():
            e = readers[key](model_id, config)
            if e is not None:
                e["source"] = model_id
                e["converged"] = _gate_ok(model_id, config)
                ests[key] = e
        if len(ests) < 2:
            continue  # nothing to triangulate for this outcome
        converged = {k: v for k, v in ests.items() if v["converged"]}
        assessable = len(converged) >= 2
        use = converged if assessable else ests
        probs = [v["prob_pos"] for v in use.values()]
        los = [v["lo"] for v in use.values()]
        his = [v["hi"] for v in use.values()]
        direction_agree = all(p >= 0.5 for p in probs) or all(p <= 0.5 for p in probs)
        intervals_overlap = max(los) <= min(his)
        row: dict = {
            "config": config,
            "outcome": outcome,
            "n_designs": len(ests),
            "n_converged": len(converged),
            "all_converged": len(converged) == len(ests),
            "direction_agree": direction_agree if assessable else pd.NA,
            "intervals_overlap": intervals_overlap if assessable else pd.NA,
            "consistent": (direction_agree and intervals_overlap) if assessable else pd.NA,
        }
        for key, _label in _TRIANGULATION_DESIGNS:
            v = ests.get(key)
            row[f"{key}_source"] = v["source"] if v else ""
            row[f"{key}_term"] = v["term"] if v else ""
            row[f"{key}_median"] = v["median"] if v else pd.NA
            row[f"{key}_lo"] = v["lo"] if v else pd.NA
            row[f"{key}_hi"] = v["hi"] if v else pd.NA
            row[f"{key}_prob_pos"] = v["prob_pos"] if v else pd.NA
            row[f"{key}_converged"] = v["converged"] if v else pd.NA
        rows.append(row)
    return pd.DataFrame(rows) if rows else None


# ---------------------------------------------------------------------------
# Tau forest plot
# ---------------------------------------------------------------------------


def tau_forest(config: str, out_path: str) -> bool:
    """Forest plot of the LRPITT12 joint taus, overlaid with the LRPITT single-outcome
    fits on the shared (non-floored) outcomes."""
    joint_path = os.path.join(_run_dir(JOINT_ID, config), "tau_summary.csv")
    if not os.path.exists(joint_path):
        return False  # joint run not fitted — main() reports the skip
    joint = pd.read_csv(joint_path)
    # Gate-awareness (issue #274 review): a REVIEW fit "is not interpretable"
    # (METHODS.md), so mark any non-converged run feeding this forest rather than
    # plotting its tau unflagged.
    joint_ok = _gate_ok(JOINT_ID, config)
    uni: dict[str, tuple[float, float, float]] = {}
    uni_review: set[str] = set()
    for model_id, outcome in ITT_IDS:
        p = os.path.join(_run_dir(model_id, config), "tau_summary.csv")
        if not os.path.exists(p):
            continue
        df = pd.read_csv(p)
        uni[outcome] = (
            float(df["tau_logit_median"].iloc[0]),
            float(df["tau_logit_lo"].iloc[0]),
            float(df["tau_logit_hi"].iloc[0]),
        )
        if not _gate_ok(model_id, config):
            uni_review.add(outcome)

    outcomes = list(joint["outcome"].values)
    y = np.arange(len(outcomes))

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.errorbar(
        joint["tau_median"].values,
        y,
        xerr=[
            joint["tau_median"].values - joint["tau_lo"].values,
            joint["tau_hi"].values - joint["tau_median"].values,
        ],
        fmt="o",
        color="#1f77b4",
        label="LRPITT12 (joint)" + ("" if joint_ok else " — REVIEW: not converged"),
        capsize=3,
    )
    # Univariate overlay, offset vertically for readability.
    uni_y = []
    uni_mean = []
    uni_lo = []
    uni_hi = []
    for k, s in enumerate(outcomes):
        if s in uni:
            m, lo, hi = uni[s]
            uni_y.append(y[k] - 0.25)
            uni_mean.append(m)
            uni_lo.append(m - lo)
            uni_hi.append(hi - m)
    if uni_y:
        ax.errorbar(
            uni_mean,
            uni_y,
            xerr=[uni_lo, uni_hi],
            fmt="s",
            color="#ff7f0e",
            label="LRPITT (single-outcome)",
            capsize=3,
        )
    ax.axvline(0.0, color="k", lw=0.75, ls="--")
    ax.set_yticks(y)
    # Flag heavily-floored outcomes (P/N): the graded logit tau shown here is NOT
    # their exploratory headline — the binary off-floor risk difference, read
    # from their own reports. Marking them keeps the forest from misrepresenting P.
    floored_present = [s for s in outcomes if s in FLOORED_SYMBOLS]

    def _ylabel(s: str) -> str:
        return s + (" †" if s in FLOORED_SYMBOLS else "") + (" ‡" if s in uni_review else "")

    ax.set_yticklabels([_ylabel(s) for s in outcomes])
    _caption = []
    if floored_present:
        _caption.append(
            "† floored outcome — graded τ shown; post-hoc exploratory headline is "
            "the binary off-floor risk difference (see model report)"
        )
    if uni_review or not joint_ok:
        _caption.append(
            "‡ single-outcome fit did not pass the convergence gate (REVIEW — not "
            "interpretable)" + ("; the joint fit is REVIEW too" if not joint_ok else "")
        )
    if _caption:
        ax.text(
            0.99,
            -0.14,
            "\n".join(_caption),
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=7.5,
            color="#555555",
        )
    ax.invert_yaxis()
    ax.set_xlabel(r"$\tau$ (logit scale, coefficient on $G=1$ = intervention; positive = benefit)")
    ax.set_title("Treatment effect by outcome")
    ax.legend(loc="lower left", fontsize=9)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    return True


# ---------------------------------------------------------------------------
# Mechanism forest plot — real-grid slopes
# ---------------------------------------------------------------------------


def _mechanism_slope_distribution(
    trace: xr.DataTree, mech_logit: np.ndarray
) -> np.ndarray:
    """Posterior draws of the average slope of ``f_mech`` over ``mech_logit``.

    Uses ``np.gradient`` per draw along the sorted grid and averages across
    the grid, returning a 1-D array of per-draw mean slopes. Input arrays
    must be aligned (one row of ``f_mech`` per ``mech_logit`` entry).
    """
    if "f_mech" not in trace.posterior:
        raise ValueError("Trace has no f_mech variable.")

    f = trace.posterior["f_mech"].stack(sample=("chain", "draw")).values  # (n, S)

    order = np.argsort(mech_logit)
    x = mech_logit[order]
    f_ord = f[order]

    # Drop rows with duplicate x (np.gradient requires monotone x).
    keep = np.concatenate([[True], np.diff(x) > 0])
    x = x[keep]
    f_ord = f_ord[keep]

    # per-draw gradient over the unique grid, then mean.
    grad = np.gradient(f_ord, x, axis=0)  # (n, S)
    slopes = grad.mean(axis=0)  # (S,)
    return slopes


def mechanism_forest(config: str, out_path: str) -> bool:
    labels: list[str] = []
    means: list[float] = []
    los: list[float] = []
    his: list[float] = []

    # Shared prepared (all phases) so we can rebuild the mech_logit vector per model.
    prepared = load_and_prepare(phase_mode="all")

    for model_id, sym in MECH_IDS:
        nc = os.path.join(_run_dir(model_id, config), "trace.nc")
        if not os.path.exists(nc):
            return False
        trace = az.from_netcdf(nc)
        if "f_mech" not in trace.posterior:
            continue

        # Rebuild the logit vector used at fit time. Rows with missing
        # outcome_post or mechanism_post are dropped by the mechanism factory,
        # so filter prepared to rows where both W_post and mech_post are
        # observed (and match the trace's obs_id length).
        #
        # CAVEAT: the factory ALSO drops rows with missing *confounder* post-scores
        # (lrp-rli-mech-057 adjusts for R; lrp-rli-mech-058 for E and R). This keep-mask does not model
        # that, so if confounder-only missingness ever occurs the reconstructed
        # length will not match the trace and the guard below skips the model. That
        # skip is a *silent drop of the model from the persisted forest/CSV* — the
        # warning is deliberately explicit about that so a reader notices a missing
        # row rather than assuming the model was never fitted.
        mech_post = prepared.post_counts[sym]
        w_post = prepared.post_counts["W"]
        keep = ~(np.isnan(mech_post) | np.isnan(w_post))
        mech_logit = logit_safe(mech_post[keep], MEASURES[sym].n_trials)

        if mech_logit.shape[0] != trace.posterior.sizes["obs_id"]:
            # Skip rather than silently misalign. Most likely cause: the factory
            # dropped rows for missing confounder post-scores that this simplified
            # keep-mask (outcome + mechanism only) does not account for.
            print(
                f"[warn] {model_id}: reconstructed mech_logit size "
                f"({mech_logit.shape[0]}) != trace obs_id size "
                f"({trace.posterior.sizes['obs_id']}) — likely confounder-only "
                "missingness the keep-mask does not model. DROPPING this model "
                "from the persisted mechanism forest AND its CSV (it will be "
                "absent from the artefact, not merely un-plotted)."
            )
            continue

        slopes = _mechanism_slope_distribution(trace, mech_logit)
        means.append(float(np.mean(slopes)))
        los.append(float(np.quantile(slopes, 0.025)))
        his.append(float(np.quantile(slopes, 0.975)))
        labels.append(f"{model_id} ({sym}->W)")

    if not labels:
        return False

    y = np.arange(len(labels))
    plt.figure(figsize=(7, 3.5))
    plt.errorbar(
        means,
        y,
        xerr=[np.array(means) - np.array(los), np.array(his) - np.array(means)],
        fmt="o",
        color="#1f77b4",
        capsize=3,
    )
    plt.yticks(y, labels)
    plt.gca().invert_yaxis()
    plt.axvline(0.0, color="k", lw=0.75, ls="--")
    plt.xlabel("Mean slope of $f^{\\mathrm{mech}}$ (logit scale)")
    plt.title("Mechanism-model average slopes")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()

    # Also write the underlying numbers. Record the config so a dev-config rerun
    # does not silently masquerade as the reporting-config artefact.
    pd.DataFrame(
        {
            "config": config,
            "model": labels,
            "slope_mean": means,
            "slope_lo": los,
            "slope_hi": his,
        }
    ).to_csv(out_path.replace(".png", ".csv"), index=False)
    return True


# ---------------------------------------------------------------------------
# Mechanism LOO comparison (LRP58 baseline vs interaction models)
# ---------------------------------------------------------------------------


_ROW_KEY_CANDIDATES: tuple[tuple[str, ...], ...] = (
    ("subject_id", "phase_idx"),
    ("subject_id", "period"),
    ("subject_id", "phase"),
    ("subject_id", "timepoint"),
    ("child_idx", "phase_idx"),
    ("child_idx", "period"),
    ("child_idx", "phase"),
    ("child_idx", "timepoint"),
    ("child_idx", "wave"),
)


def _meaningful_obs_id(trace: az.InferenceData, n_obs: int) -> np.ndarray | None:
    """Return a meaningful trace ``obs_id`` coordinate, not PyMC's 0..n-1 default."""
    for group_name in ("log_likelihood", "observed_data", "constant_data", "posterior"):
        group = getattr(trace, group_name, None)
        if group is None or "obs_id" not in group.coords:
            continue
        values = np.asarray(group.coords["obs_id"].values)
        if values.ndim != 1 or values.size != n_obs:
            continue
        if np.issubdtype(values.dtype, np.integer) and np.array_equal(
            values, np.arange(n_obs)
        ):
            continue
        return values
    return None


def _shared_row_identities(
    traces: dict[str, az.InferenceData], sizes: dict[str, int]
) -> tuple[dict[str, np.ndarray] | None, str | None]:
    """Extract an ordered analysis-row identity shared by every trace.

    A non-positional ``obs_id`` coordinate is preferred. Current PyMC traces use a
    positional coordinate, so the normal route is a stable pair in ``constant_data``
    such as ``(child_idx, phase_idx)`` or ``(child_idx, period)``. Those pairs identify
    each child-period row and catch both membership and ordering changes. If no common
    identity is present, return ``None``: equal row counts alone are not enough to
    justify an ``elpd_diff``.
    """
    coords = {mid: _meaningful_obs_id(trace, sizes[mid]) for mid, trace in traces.items()}
    if all(coord is not None for coord in coords.values()):
        return {mid: np.asarray(coord) for mid, coord in coords.items()}, "obs_id"

    constants = {mid: getattr(trace, "constant_data", None) for mid, trace in traces.items()}
    if any(dataset is None for dataset in constants.values()):
        return None, None
    common_row_vars = set.intersection(
        *(
            {
                name
                for name, data in dataset.data_vars.items()
                if data.dims == ("obs_id",)
            }
            for dataset in constants.values()
        )
    )

    for keys in _ROW_KEY_CANDIDATES:
        if not set(keys).issubset(common_row_vars):
            continue
        # ``child_idx`` is re-derived after filtering, so on its own it is weaker
        # than a raw subject id. Strengthen it with every shared row-level constant
        # (age, group, exposure, baseline/outcome transforms, etc.). A difference in
        # any shared fingerprint field triggers the safe per-model fallback.
        fingerprint_keys = list(keys) + sorted(common_row_vars.difference(keys))
        identities: dict[str, np.ndarray] = {}
        for mid, constant in constants.items():
            columns = []
            for key in fingerprint_keys:
                data = constant[key]
                if data.dims != ("obs_id",) or data.sizes["obs_id"] != sizes[mid]:
                    break
                columns.append(np.asarray(data.values))
            else:
                identities[mid] = np.column_stack(columns)
                continue
            break
        if len(identities) == len(traces):
            return identities, f"{'+'.join(keys)}+shared-constant-fingerprint"
    return None, None


def _write_separate_loo(
    traces: dict[str, az.InferenceData],
    sizes: dict[str, int],
    config: str,
    out_path: str,
    *,
    reason: str,
) -> None:
    """Write per-model LOO estimates when a shared-row delta is not identified."""
    rows = []
    for mid, trace in traces.items():
        loo = az.loo(trace)
        rows.append(
            {
                "config": config,
                "model": mid,
                "n_obs": sizes[mid],
                "comparison_valid": False,
                "comparison_reason": reason,
                "elpd_loo": float(loo.elpd),
                "se": float(loo.se),
                "p_loo": float(loo.p),
            }
        )
    pd.DataFrame(rows).to_csv(out_path, index=False)


def _loo_compare(ids: list[str], config: str, out_path: str) -> bool:
    """Write ``az.compare`` over the fitted models in ``ids`` (LOO).

    Loads every *convergence-gate-passing* model in ``ids`` that has a trace with a
    ``log_likelihood`` group. ``az.compare`` is only a like-for-like elpd-difference
    when the models share the same observations in the same order, so row identities
    are verified from trace coordinates or stable ``constant_data`` keys. If identity
    cannot be proved, it falls back to a clearly marked per-model ``elpd_loo`` table
    rather than emitting a misleading delta. Returns False if fewer than two eligible
    models are available.
    """
    traces: dict[str, az.InferenceData] = {}
    for mid in ids:
        if not _gate_ok(mid, config):
            print(
                f"[warn] {mid}: convergence gate {_gate_status(mid, config)}; "
                "excluding from LOO comparison."
            )
            continue
        nc = os.path.join(_run_dir(mid, config), "trace.nc")
        if not os.path.exists(nc):
            continue
        t = az.from_netcdf(nc)
        # arviz 1.x returns a DataTree whose ``.groups`` is a tuple of paths
        # like "/log_likelihood" (0.x exposed a ``.groups()`` method of bare
        # names) — normalise to leaf names for the membership test.
        group_names = {g.rstrip("/").split("/")[-1] for g in t.groups}
        if "log_likelihood" not in group_names:
            print(f"[warn] {mid}: trace has no log_likelihood group; skipping.")
            continue
        traces[mid] = t

    if len(traces) < 2:
        return False

    # A trace whose posterior has no ``obs_id`` dim is not a single-outcome
    # observation model (e.g. a repurposed ``growth``-kind fit with child/wave/
    # outcome dims) — it cannot participate in a shared-observation LOO, so drop
    # it with a clear warning rather than crashing on the KeyError.
    sizes: dict[str, int] = {}
    for mid, t in list(traces.items()):
        if "obs_id" not in t.posterior.sizes:
            print(
                f"[warn] {mid}: posterior has no 'obs_id' dim "
                f"(dims={tuple(t.posterior.sizes)}); not an observation-level "
                "model — skipping from this LOO comparison."
            )
            del traces[mid]
            continue
        sizes[mid] = t.posterior.sizes["obs_id"]

    if len(traces) < 2:
        return False

    if len(set(sizes.values())) != 1:
        reason = f"observation counts differ: {sizes}"
        print(
            f"[warn] models do not share observations ({sizes}); "
            "writing per-model elpd_loo instead of az.compare deltas."
        )
        _write_separate_loo(traces, sizes, config, out_path, reason=reason)
        return True

    identities, identity_source = _shared_row_identities(traces, sizes)
    if identities is None:
        reason = "ordered analysis-row identity unavailable"
        print(
            "[warn] models have equal observation counts but no common ordered row "
            "identity; writing per-model elpd_loo instead of az.compare deltas."
        )
        _write_separate_loo(traces, sizes, config, out_path, reason=reason)
        return True

    reference_id = next(iter(traces))
    reference = identities[reference_id]
    mismatched = [
        mid for mid, identity in identities.items() if not np.array_equal(identity, reference)
    ]
    if mismatched:
        reason = (
            f"ordered analysis rows differ on {identity_source}: "
            f"{reference_id} vs {', '.join(mismatched)}"
        )
        print(
            f"[warn] {reason}; writing per-model elpd_loo instead of az.compare "
            "deltas."
        )
        _write_separate_loo(traces, sizes, config, out_path, reason=reason)
        return True

    cmp = az.compare(traces)  # ic="loo" by default
    cmp = cmp.copy()
    cmp.insert(0, "config", config)  # record the tier that produced the row
    cmp.insert(1, "row_identity", identity_source)
    cmp.insert(2, "comparison_valid", True)
    cmp.to_csv(out_path)
    return True


def mechanism_loo_compare(config: str, out_path: str) -> bool:
    """LOO comparison of the LRP58 baseline against its interaction extensions."""
    return _loo_compare(LOO_COMPARE_IDS, config, out_path)


def phonics_route_loo_compare(config: str, out_path: str) -> bool:
    """LOO comparison of LRP72 against its no-interaction baseline (isolates L x B)."""
    return _loo_compare(PHONICS_LOO_IDS, config, out_path)


def age_moderation_loo_compare(config: str, out_path: str) -> bool:
    """LOO comparison of LRP73 against its no-interaction baseline (isolates L x age)."""
    return _loo_compare(AGE_LOO_IDS, config, out_path)


def dose_response_loo_compare(config: str, out_path: str) -> bool:
    """LOO comparison of LRP77 against its pooled-dose comparator (does dose vary by period?)."""
    return _loo_compare(DOSE_LOO_IDS, config, out_path)


def did_dose_loo_compare(config: str, out_path: str) -> bool:
    """LOO comparison of LRPDID07 vs its pooled comparator (does the L dose vary by period?)."""
    written = _loo_compare(DID_DOSE_LOO_IDS, config, out_path)
    if not written:
        return False

    # Model reports render from their own run directories. Keep the shared comparison
    # artefact beside both fitted reports so the result cannot silently disappear from
    # an otherwise complete render. The CSV explicitly marks whether an elpd_diff was
    # valid; row-mismatch fallbacks remain auditable rather than masquerading as ranks.
    for model_id in DID_DOSE_LOO_IDS:
        run_dir = _run_dir(model_id, config)
        if os.path.isdir(run_dir):
            destination = os.path.join(run_dir, "did_dose_loo_compare.csv")
            if os.path.abspath(out_path) != os.path.abspath(destination):
                shutil.copyfile(out_path, destination)
    return True


# ---------------------------------------------------------------------------
# Mediation family — indirect-effect comparison (decomposition summaries)
# ---------------------------------------------------------------------------


def build_mediation_family(config: str) -> pd.DataFrame | None:
    """One row per mediation route: its headline indirect effect + total + direct,
    on the shared words scale, read from each model's ``mediation_summary.csv``."""
    rows: list[dict] = []
    for model_id, label, indirect_row in MEDIATION_IDS:
        path = os.path.join(_run_dir(model_id, config), "mediation_summary.csv")
        if not os.path.exists(path):
            continue
        df = pd.read_csv(path).set_index("quantity")
        if indirect_row not in df.index:
            continue
        ind = df.loc[indirect_row]
        direct_row = "IDE" if indirect_row == "IIE" else "NDE"
        total = df.loc["total"] if "total" in df.index else None
        direct = df.loc[direct_row] if direct_row in df.index else None

        # Median-first (#268); fall back to the mean column for pre-#268 CSVs.
        def _w(series) -> float:
            if series is None:
                return float("nan")
            return float(series.get("words_median", series.get("words_mean")))

        rows.append(
            {
                "config": config,
                "model": model_id,
                "route": label,
                "estimand": indirect_row,
                "converged": _gate_ok(model_id, config),
                "indirect_words": _w(ind),
                "indirect_lo": float(ind["words_lo"]),
                "indirect_hi": float(ind["words_hi"]),
                "indirect_prob_pos": float(ind["prob_pos"]),
                "total_words": _w(total),
                "direct_words": _w(direct),
            }
        )
    if not rows:
        return None
    return pd.DataFrame(rows)


def mediation_family_forest(df: pd.DataFrame, out_path: str) -> bool:
    """Forest of every mediation route's indirect effect on the words scale."""
    if df is None or df.empty:
        return False
    d = df.iloc[::-1].reset_index(drop=True)  # top-to-bottom = table order
    y = np.arange(len(d))
    fig, ax = plt.subplots(figsize=(8, 0.55 * len(d) + 1.6))
    strong = d["indirect_prob_pos"] >= 0.97
    colors = ["#1f77b4" if s else "#9ecae1" for s in strong]
    ax.errorbar(
        d["indirect_words"], y,
        xerr=[d["indirect_words"] - d["indirect_lo"], d["indirect_hi"] - d["indirect_words"]],
        fmt="none", ecolor="#666666", capsize=3, zorder=1,
    )
    ax.scatter(d["indirect_words"], y, c=colors, s=45, zorder=2)
    for k in y:
        ax.annotate(f"P={d['indirect_prob_pos'][k]:.2f}",
                    (d["indirect_hi"][k], y[k]), textcoords="offset points",
                    xytext=(6, 0), va="center", fontsize=7.5, color="#555555")
    ax.axvline(0.0, color="k", lw=0.75, ls="--")
    ax.set_yticks(y)
    ax.set_yticklabels(
        [f"{r}  [{e}]" for r, e in zip(d["route"], d["estimand"], strict=True)],
        fontsize=8.5,
    )
    ax.set_xlabel("Indirect effect on word reading (words out of test length; positive = route carries benefit)")
    ax.set_title("Mediation family — routes to word reading (g-formula; all ID-2 adjusted associations)")
    ax.text(0.99, -0.13, "Darker = strong (P≥0.97). All estimands GA-confounded; letter-sound routes only.",
            transform=ax.transAxes, ha="right", va="top", fontsize=7.5, color="#555555")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    return True


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _report_gate_status(config: str) -> None:
    """Print a convergence-gate roll-call for every fitted run of this config.

    Any REVIEW fit whose tau/slope feeds a comparison below is thereby surfaced
    (issue #274 item 3): the comparison CSVs carry a per-row ``converged`` flag,
    and this roll-call names the offenders up front so a non-converged fit's
    numbers cannot slip into the forests unnoticed. A REVIEW fit "is not
    interpretable — fix the model, do not report it" (METHODS.md).
    """
    models_dir = str(_paths.stat_models_dir())
    suffix = f"-{config}"
    run_dirs = sorted(
        d
        for d in glob.glob(os.path.join(models_dir, f"*{suffix}"))
        if os.path.isdir(d) and os.path.exists(os.path.join(d, "config.json"))
    )
    if not run_dirs:
        return
    review: list[str] = []
    missing: list[str] = []
    for d in run_dirs:
        model_id = os.path.basename(d)[: -len(suffix)]
        status = _gate_status(model_id, config)
        if status == "REVIEW":
            review.append(model_id)
        elif status == "MISSING":
            missing.append(model_id)
    n = len(run_dirs)
    n_pass = n - len(review) - len(missing)
    print(f"\nConvergence gate ({config}): {n_pass}/{n} PASS")
    if review:
        print("  ⚠ REVIEW (not interpretable — flagged in the comparison CSVs):")
        for model_id in review:
            print(f"      {model_id}")
    if missing:
        print(f"  ({len(missing)} run(s) with no diagnostics_summary.json)")
    print()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="dev")
    parser.add_argument(
        "--out",
        default=None,
        help="Comparison output dir (default: <output-root>/statistical_models/comparison).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help=(
            "Override the output root for this run (highest precedence, above "
            "DSE_LRP_OUTPUT_DIR); the relative layout is unchanged. Default: "
            "repo-local output/."
        ),
    )
    args = parser.parse_args()

    _paths.set_output_root(args.output_dir)
    print(f"Output root: {_paths.describe_output_root()}")
    args.out = args.out or str(_paths.stat_comparison_dir())
    os.makedirs(args.out, exist_ok=True)

    _report_gate_status(args.config)

    itt_joint = build_itt_vs_joint(args.config)
    if itt_joint is not None:
        path = os.path.join(args.out, "itt_vs_joint_tau.csv")
        itt_joint.to_csv(path, index=False)
        print(f"Wrote {path}")
    else:
        print("Skipping ITT-vs-joint comparison: one or more runs missing.")

    triangulation = build_triangulation(args.config)
    if triangulation is not None:
        path = os.path.join(args.out, "triangulation_consistency.csv")
        triangulation.to_csv(path, index=False)
        print(f"Wrote {path}")
    else:
        print(
            "Skipping cross-design triangulation: fewer than two designs available "
            "for every outcome."
        )

    tau_forest_path = os.path.join(args.out, "tau_forest.png")
    if tau_forest(args.config, tau_forest_path):
        print(f"Wrote {tau_forest_path}")
    else:
        print("Skipping tau forest: joint run missing.")

    mech_forest_path = os.path.join(args.out, "mechanism_forest.png")
    if mechanism_forest(args.config, mech_forest_path):
        print(f"Wrote {mech_forest_path}")
    else:
        print("Skipping mechanism forest: one or more mechanism runs missing.")

    loo_compare_path = os.path.join(args.out, "mechanism_loo_compare.csv")
    if mechanism_loo_compare(args.config, loo_compare_path):
        print(f"Wrote {loo_compare_path}")
    else:
        print("Skipping mechanism LOO compare: fewer than two mechanism runs available.")

    phonics_path = os.path.join(args.out, "phonics_route_loo_compare.csv")
    if phonics_route_loo_compare(args.config, phonics_path):
        print(f"Wrote {phonics_path}")
    else:
        print("Skipping phonics-route LOO compare: LRP72 / LRP72base runs missing.")

    age_path = os.path.join(args.out, "age_moderation_loo_compare.csv")
    if age_moderation_loo_compare(args.config, age_path):
        print(f"Wrote {age_path}")
    else:
        print("Skipping age-moderation LOO compare: LRP73 / LRP73base runs missing.")

    dose_path = os.path.join(args.out, "dose_response_loo_compare.csv")
    if dose_response_loo_compare(args.config, dose_path):
        print(f"Wrote {dose_path}")
    else:
        print("Skipping dose-response LOO compare: LRP77 / LRP77base runs missing.")

    did_dose_path = os.path.join(args.out, "did_dose_loo_compare.csv")
    if did_dose_loo_compare(args.config, did_dose_path):
        print(f"Wrote {did_dose_path}")
    else:
        print("Skipping DiD L-dose LOO compare: LRPDID07 / LRPDID07base runs missing.")

    med_family = build_mediation_family(args.config)
    if med_family is not None:
        med_csv = os.path.join(args.out, "mediation_family.csv")
        med_family.to_csv(med_csv, index=False)
        print(f"Wrote {med_csv}")
        med_png = os.path.join(args.out, "mediation_family_forest.png")
        if mediation_family_forest(med_family, med_png):
            print(f"Wrote {med_png}")
    else:
        print("Skipping mediation-family comparison: no mediation runs found.")


if __name__ == "__main__":
    main()
