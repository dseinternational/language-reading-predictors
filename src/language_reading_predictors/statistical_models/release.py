# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""The release decision that gates the findings-first key-findings box (#392 P1).

Before this module, ``reporting.generate_key_findings`` checked only the sampling
gate. A fit could therefore publish an unqualified cause-and-effect headline while a
robustness gap the report *itself* diagnoses further down remained unresolved — a
``tau`` prior-data conflict from power-scaling, or a P/N floor-rule model whose
required treatment-prior grid was absent. ``_results_floored.qmd`` already refused to
release under exactly those conditions, so the findings box and the prose below it
could contradict each other, with the box winning the reader's attention.

The policy implemented here is the **evidence-bound withhold** signed off on
2026-08-05 (option A of the three offered on #392): a diagnosed robustness gap
suppresses the causal headline until provenance-validated sensitivity evidence covers
the model. Two consequences follow that are worth stating plainly.

**A conflict is not automatically a withhold.** The suite's effect priors are
zero-centred and conservative, so a prior-data conflict means the prior *attenuates* a
real effect, not that it invents one. The case worth gating is where the prior
out-works the data. :func:`classify_tau_sensitivity` therefore separates a *prior-data
conflict* — both prior and likelihood move the posterior, released with a note saying
the size is a lower bound and the direction is the reliable part — from a
*prior-dominant* posterior, where the prior moves it and the data do not, which is
withheld without evidence. Both classes come from ArviZ's own predicate rather than a
rule of our own, so a fit's release class and the psense table printed in its report
cannot disagree.

**Absence of measurement is a gap, not a pass.** A fit with no ``psense_summary.csv``
has not been measured clean; it has not been measured. #381 named that distinction as
its central meta-finding, and ``tau_psense_status`` is already fail-closed for the
floor gate on the same reasoning, so an unavailable diagnosis withholds here too. The
reason string distinguishes the two cases, so a fit that withholds only because
power-scaling was never run is repaired by
``scripts/regenerate_psense.py`` followed by ``scripts/regenerate_key_findings.py``,
with no refit.

**Scope.** :data:`GATED_KINDS` is the authoritative list, and it is exactly the
families whose findings box publishes a randomisation-anchored causal claim: ``itt``
(``tau``, including the floored P/N primaries), ``joint`` (``tau``, vector-valued —
one randomised effect per jointly-fitted outcome, aggregated worst-first), ``did``
(``tau_t2``, or the dose model's own focal slope), ``gain_factors`` (``beta_trt``) and
``level_factors`` (the plan's focal t2 term: ``d_grp_time[t2]``, the change in the
adjusted arm gap from t1 to t2 under the t1-referenced parameterisation (#552), or
``b_grp_time[1]`` on a stored pre-#552 fit / the free comparator — the only
randomised element either way).

Everything else is out, and for one of two reasons. The observational families report
adjusted associations, which their reports already label as such. A treated-only
``gain_factors`` companion is in a gated family but has no randomised term at all, and
a ``gain_factors`` moderation variant's ``beta_trt`` is never released as causal — see
:func:`gate_applies` for both.

#392 reviewed ITT and left the mirroring onto the others to a follow-up; Frank ruled the
uniform extension on 2026-08-05 after it was measured, because the case that never bit
in ITT does bite outside it. **No ITT fit is prior-dominant; eight fits across ``did``,
``gain_factors`` and ``level_factors`` were when the gate landed**, and every one of
them was publishing an unqualified causal headline. Same defect, same treatment. All
eight have since been resolved by their family treatment-prior sweeps' trace-backed
evidence: level (#389/#488), did (#390/#489), and the two ``gain_factors`` off-floor
fits (#391 — swept against their post-respecification refits, since the #391
findings 2+3 respec changed the primaries first). ``joint`` was added in review:
it publishes a causal headline too, and its ``itt-012`` fit has three prior-attenuated
outcomes that the box said nothing about.

The ITT six-cell grid itself stays ITT-only — it is bound to the registered ITT
floor rule and :func:`sensitivity.evaluate_floor_sensitivity`'s provenance
machinery — but since #575 finding 10d the *policy* is family-uniform:
``gain_factors``' off-floor fits route through
:func:`_gain_offfloor_decision`, the same decision shape keyed on their own
estimand-matched evidence (the family treatment-prior sweep, per-fit
trace-validated by :func:`_standard_sweep_evidence`). Any non-clean
power-scaling verdict on the off-floor risk difference — the release-with-note
prior-data-conflict class included — now requires that evidence before the
findings box may speak, closing the gap this paragraph used to record.

**Tiering.** The policy applies uniformly across base ITT models, adjusted-robustness
models and outcomes outside the standard 44-cell sweep. That was the default offered
alongside option A; the graded alternative discussed earlier on #392 (withhold for
primary, qualify-never-withhold for adjusted robustness) is a one-line change to
:data:`_WITHHOLD_TIERS` if it is preferred.
"""

from __future__ import annotations

import json
import os
from contextlib import suppress
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from collections.abc import Callable
from typing import Any, Literal, Mapping

import numpy as np
import pandas as pd

from language_reading_predictors.statistical_models.sensitivity import (
    FLOOR_SENSITIVITY_FILENAME,
    STANDARD_SENSITIVITY_FILENAME,
    evaluate_floor_sensitivity,
    load_primary_floor_reference,
    tau_psense_status,
)

__all__ = [
    "GATED_KINDS",
    "GROWTH_INFLUENCE_TRACE_FILENAME",
    "JOINT_MECHANISM_MARGINAL_COVERAGE_FLOORS",
    "JOINT_MECHANISM_WAVE_MARGINAL_PPC",
    "JOINT_MECHANISM_WAVE_PSENSE",
    "JOINT_MECHANISM_WAVE_TRACE",
    "MEDIATION_T3_TRACE_FILENAME",
    "PSENSE_THRESHOLD",
    "RELEASE_DECISION_FILENAME",
    "PublicationStatus",
    "ReleaseDecision",
    "ReleaseEvaluation",
    "ReleaseStage",
    "TauSensitivityClass",
    "causal_term_for",
    "classify_tau_sensitivity",
    "evaluate_publication",
    "gate_applies",
    "evaluate_itt_release",
    "evaluate_release",
    "write_release_decision",
]

#: ArviZ's default power-scaling flag threshold. A parameter is "sensitive" when
#: either its prior or its likelihood power-scaling statistic reaches this.
PSENSE_THRESHOLD = 0.05

#: How far the declared joint contrast's direction probability may move between a
#: factorised parent and its LKJ residual-correlation companion before the pairing
#: is reported as materially changing the conclusion (2026-08-23 joint audit,
#: finding 2). 0.05 on P(> 0) is a conclusion-level rule: on the three registered
#: pairs the observed shifts are 0.006, 0.006 and 0.006, so it flags a change of
#: reading rather than the Monte Carlo noise between two independently sampled
#: fits. The contrast *interval* is deliberately not thresholded — moving it is
#: exactly what the companion exists to do.
_CONTRAST_DIRECTION_SHIFT = 0.05

#: How far the implied cross-outcome posterior correlation must move between a
#: factorised parent and its companion before the contrast's width change is
#: attributed to covariance at all (2026-08-24 review of the joint audit). Both
#: correlations are read off equal-tailed interval widths, and a factorised
#: parent's is *structurally* zero — its outcomes share no parameter — so the
#: parent's measured value is this approximation's own noise floor. On the three
#: registered pairs it is -0.011, -0.010 and -0.019 against exact posterior-draw
#: values of -0.003, -0.006 and -0.006, so a band just above that floor separates
#: Monte Carlo noise from a covariance correction.
_AME_CORRELATION_NOISE = 0.03

#: Trace backing the natural-effect temporal-ordering sensitivity.  The fixed
#: basename is part of the release contract: a table is not independently
#: auditable when its posterior exists only in memory during the fit.
MEDIATION_T3_TRACE_FILENAME = "trace_mediation_t3_sensitivity.nc"

GROWTH_INFLUENCE_TRACE_FILENAME = "trace_growth_influence_sensitivity.nc"
"""Trace backing the growth family's high-Pareto observation-cell refit."""

#: Per-wave artefact names for the joint-mechanism levels design. Every published
#: wave carries the same three files, so the release check requires one uniform
#: bundle instead of special-casing the wave hosting the fit-level artefacts
#: (2026-08-23 joint-mechanism follow-up review, finding 1).
JOINT_MECHANISM_WAVE_TRACE = "trace_wave_t{timepoint}.nc"
JOINT_MECHANISM_WAVE_MARGINAL_PPC = "ppc_summary_marginal_t{timepoint}"
JOINT_MECHANISM_WAVE_PSENSE = "psense_wave_t{timepoint}"

#: Predeclared new-child predictive-adequacy floors for that design, pooled and per
#: outcome. The ordinary conditional check is saturated by construction and PSIS-LOO
#: is deliberately not computed, so the marginal check is the only one that can fail
#: — and "can fail" needs a stated threshold rather than a reader's judgement. Set
#: below nominal because the check is deliberately conservative (a redrawn residual
#: widens the interval), and breaching it *qualifies* a release rather than
#: withholding it: substantive misfit is information about the model, not evidence
#: that the sampler failed.
JOINT_MECHANISM_MARGINAL_COVERAGE_FLOORS: dict[int, float] = {50: 0.35, 90: 0.75}

TauSensitivityClass = Literal[
    "clear", "prior_data_conflict", "prior_dominant", "unavailable"
]

ReleaseStatus = Literal["release", "qualify", "withhold"]

#: Model tiers to which the withhold applies. Uniform by decision; narrowing this
#: set is how a graded policy would be expressed.
_WITHHOLD_TIERS = frozenset({"primary", "adjusted_robustness", "off_grid"})

#: Families the gate covers, each keyed to the term its causal headline rests on.
#: A family is here only if a randomised contrast identifies its headline — the
#: observational families report adjusted associations, which the reports already
#: label as such, and gating those on prior sensitivity would say nothing a reader
#: does not already know from the label.
GATED_KINDS = frozenset({"itt", "joint", "did", "gain_factors", "level_factors"})


def gate_applies(config: Mapping[str, Any]) -> bool:
    """Is this fit in scope for the robustness gate at all?

    Family membership is necessary but not sufficient. A ``gain_factors`` **treated-only**
    companion keeps every row on intervention, so the treatment indicator is constant,
    ``build_gain_factors_model`` drops ``beta_trt`` and every interaction naming it, and
    the fit's own resolved plan states it has no randomised contrast. There is no causal
    headline to gate, and no ``beta_trt`` row for power-scaling to measure.

    That distinction is load-bearing rather than tidy-minded. Without it the gate reads
    the absent term as an *unmeasured* one and withholds all eight companions — the
    fail-closed rule doing real damage, because "not measured" and "structurally not
    present" are the same absence to a lookup and opposite things to a reader.

    A ``gain_factors`` **moderation variant** (#391 finding 3 decision) is skipped for
    the complementary reason: its posterior *does* contain ``beta_trt``, but by decision
    its interaction-aware marginal is model-dependent (the treatment interactions are
    estimated on all stacked periods, partly post-crossover) and is never presented as
    the causal headline — that lives in the interaction-free primary the variant varies,
    which IS gated. Gating the variant would demand treatment-prior sweep evidence for a
    number the family never releases as causal.

    A ``level_factors`` **pooled** fit (``group_by_time=False``) is the level family's
    analogue of the treated-only companion: its resolved plan records ``focal_term``
    explicitly as null because a pooled ``beta_grp`` mixes post-crossover waves and is
    never a randomised contrast, so there is no causal headline to gate and no focal
    psense row to read — :func:`causal_term_for`'s ``b_grp_time[1]`` fallback would
    name a term the posterior structurally lacks and withhold fail-closed (2026-08-20
    level-factors review, finding 4). The distinction from a stored pre-#552 fit is
    the *presence* of the key: an old plan has no ``focal_term`` at all and keeps the
    fallback, so stored fits still re-decide identically without a refit.
    """
    if config.get("kind") not in GATED_KINDS:
        return False
    plan = config.get("resolved_run_plan") or {}
    if (
        config.get("kind") == "level_factors"
        and "focal_term" in plan
        and plan.get("focal_term") is None
    ):
        return False
    return not (
        config.get("kind") == "gain_factors"
        and (
            bool(plan.get("treated_only", False))
            or bool(plan.get("moderation_variant", False))
        )
    )


def causal_term_for(config: Mapping[str, Any]) -> str:
    """The psense row this fit's release decision turns on.

    ``level_factors`` fits one arm coefficient per timepoint and only the t2
    element is randomised (#389 finding 1), so the gate names that element rather
    than the vector. Reading the bare name instead returns "unavailable" for all
    eleven fits — a gate that withholds every level-factor headline for a diagnosis
    that is present and sitting one row away. Which element it is depends on the
    fitted parameterisation (#552): the persisted plan's ``focal_term`` —
    ``d_grp_time[t2]`` under the t1-referenced default, ``b_grp_time[1]`` under the
    free comparator. A stored fit whose plan predates the field (every pre-#552
    reporting fit) was fitted free, so the fallback is ``b_grp_time[1]``; the
    decision therefore stays reproducible over stored fits without a refit. A
    post-#552 pooled plan records ``focal_term`` as explicitly null and never
    reaches this lookup — :func:`gate_applies` excludes it (2026-08-20 review,
    finding 4).

    The ``did`` dose models have no ``tau_t2`` at all: their focal *coefficient* is
    the dose slope. The choice mirrors ``DiDRunPlan.effect_term`` and is read from the
    persisted plan rather than re-derived from ``spec.extra``, so a fit's release
    decision and its own psense emission cannot disagree about which term matters.

    This returns the **coefficient** power-scaling measures, which for a ``did`` fit
    is not the same object as the estimand it publishes (#576 finding 1) — LRPDID07
    power-scales ``mu_dose`` and publishes a treated-row natural-scale marginal. The
    two are used at different stages and must not be conflated: the psense diagnosis
    reads this name, while the sweep's sign-stability clause reads the published
    estimand's own column via :func:`sweep_sign_column`.
    """
    kind = config.get("kind")
    if kind == "gain_factors":
        return "beta_trt"
    if kind == "level_factors":
        plan = config.get("resolved_run_plan") or {}
        focal = plan.get("focal_term")
        return str(focal) if focal else "b_grp_time[1]"
    if kind == "did":
        plan = config.get("resolved_run_plan") or {}
        if plan.get("period_varying"):
            return "mu_dose"
        return "beta_dose" if plan.get("dose") else "tau_t2"
    return "tau"

_PRIOR_ATTENUATION_NOTE = (
    "The treatment-effect prior is deliberately cautious and pulls this estimate "
    "towards no effect, so the size is best read as a lower bound while the "
    "direction is the more reliable part."
)

_QUALIFY_NOTE = (
    "This estimate leans substantially on the treatment-effect prior rather than on "
    "the data alone, so it is reported as prior-informed and exploratory."
)


@dataclass(frozen=True, slots=True)
class ReleaseDecision:
    """What the robustness evidence permits the key-findings box to say."""

    status: ReleaseStatus
    tau_class: TauSensitivityClass
    #: Why a withhold happened, or why a note is attached. Empty on a clean release.
    reason: str = ""
    #: A sentence to append to the findings when the status is release-with-note or
    #: qualify. Empty when nothing needs saying.
    note: str = ""
    #: What evidence lifted, or would lift, a withhold.
    evidence: str = ""
    tier: str = "primary"
    floor_rule: bool = False
    floor_grid_required: bool = False
    floor_grid_ready: bool = False
    prior_sensitivity: float | None = None
    likelihood_sensitivity: float | None = None
    diagnosis: str | None = None

    @property
    def released(self) -> bool:
        return self.status != "withhold"

    def as_dict(self) -> dict[str, Any]:
        """JSON-ready record for ``key_findings.json``."""
        return {k: v for k, v in asdict(self).items() if v not in ("", None)}


def _tau_row(psense: pd.DataFrame | None, term: str) -> pd.Series | None:
    """The single row for ``term``, or None if absent or ambiguous.

    Duplicated rows are ambiguous rather than "take the first": a release gate that
    silently picks one of two disagreeing diagnoses is worse than one that reports it
    cannot tell.
    """
    if psense is None or psense.empty:
        return None
    wanted = term.strip().casefold()
    mask = pd.Index(psense.index).astype(str).str.strip().str.casefold() == wanted
    rows = psense.loc[mask]
    if len(rows) != 1:
        return None
    return rows.iloc[0]


#: Worst-first, so aggregating a vector-valued term takes the first class present.
_CLASS_SEVERITY: tuple[TauSensitivityClass, ...] = (
    "unavailable",
    "prior_dominant",
    "prior_data_conflict",
    "clear",
)


def _element_rows(psense: pd.DataFrame | None, term: str) -> list[tuple[str, pd.Series]]:
    """Rows for ``term[...]`` — every element of a vector-valued coefficient."""
    if psense is None or psense.empty:
        return []
    wanted = term.strip().casefold()
    out = []
    for label, row in psense.iterrows():
        text = str(label).strip()
        if text.casefold().split("[")[0] == wanted and "[" in text:
            out.append((text, row))
    return out


def _finite(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if pd.notna(number) and abs(number) != float("inf") else None


def classify_tau_sensitivity(
    psense: pd.DataFrame | None, *, term: str = "tau"
) -> tuple[TauSensitivityClass, float | None, float | None, str | None]:
    """Classify a causal term's power-scaling sensitivity.

    Returns the class alongside the prior and likelihood statistics and ArviZ's own
    diagnosis string, so a payload records the numbers a reader would otherwise have
    to open the CSV for.

    The three classes reproduce ``arviz_stats.psense_summary``'s own predicate
    exactly, comparison for comparison, so a fit's release class and the psense table
    printed in its report can never disagree:

    ================================  ==============================================
    ``prior >= t`` and ``lik >= t``   ``prior_data_conflict`` — ArviZ's "potential
                                      prior-data conflict"
    ``prior > t > lik``               ``prior_dominant`` — ArviZ's "potential strong
                                      prior / weak likelihood"
    otherwise                         ``clear`` — ArviZ's ``✓``
    ================================  ==============================================

    The third row is the one worth reading twice, because an intuitive "flag whenever
    either statistic is large" rule gets it backwards. A posterior that is sensitive
    to the *likelihood* and insensitive to the prior is the **ideal** case: the data
    are driving the result and the prior is doing nothing. Kallioinen et al. (2024)
    classify on prior sensitivity, and only ask about the likelihood to separate a
    conflict from a prior-dominated posterior.

    Note also that ArviZ writes a tick (``✓``) for an unflagged parameter, so a reader
    — or a filter — that treats only blank values as clear mis-reads every clean row.
    ``term`` may name a **vector-valued** coefficient, which the ``joint`` family's
    ``tau`` is: one randomised effect per jointly-fitted outcome, and every element a
    causal claim the findings box speaks for. There is no single element to pick, so
    the classification aggregates worst-first over ``term[...]`` and reports the
    element that drove it. A per-fit decision cannot say less than its worst
    constituent without the box overstating what the fit supports.
    """
    row = _tau_row(psense, term)
    if row is None:
        elements = _element_rows(psense, term)
        if elements:
            worst: tuple[TauSensitivityClass, str, Any] | None = None
            for label, element in elements:
                cls, prior, likelihood, diagnosis = classify_tau_sensitivity(
                    psense.loc[[label]].rename(index={label: term}), term=term
                )
                rank = _CLASS_SEVERITY.index(cls)
                if worst is None or rank < _CLASS_SEVERITY.index(worst[0]):
                    worst = (cls, label, (prior, likelihood, diagnosis))
            assert worst is not None
            cls, label, (prior, likelihood, diagnosis) = worst
            suffix = f" (driven by {label})" if cls != "clear" else ""
            return cls, prior, likelihood, (diagnosis or "") + suffix or None
    if row is None or "prior" not in row.index or "likelihood" not in row.index:
        return "unavailable", None, None, None
    prior = _finite(row["prior"])
    likelihood = _finite(row["likelihood"])
    diagnosis = (
        str(row["diagnosis"]).strip() if "diagnosis" in row.index else None
    ) or None
    if prior is None or likelihood is None:
        return "unavailable", prior, likelihood, diagnosis
    if prior >= PSENSE_THRESHOLD and likelihood >= PSENSE_THRESHOLD:
        return "prior_data_conflict", prior, likelihood, diagnosis
    if prior > PSENSE_THRESHOLD > likelihood:
        return "prior_dominant", prior, likelihood, diagnosis
    return "clear", prior, likelihood, diagnosis


def _read_csv(output_dir: str | Path, name: str, **kwargs: Any) -> pd.DataFrame | None:
    path = Path(output_dir) / name
    if not path.exists():
        return None
    try:
        return pd.read_csv(path, **kwargs)
    except (OSError, UnicodeDecodeError, pd.errors.ParserError, ValueError):
        return None


def _config_name(output_dir: str | Path, model_id: str) -> str:
    """The ``-<config>`` suffix of a fit directory (``…-reporting`` -> ``reporting``)."""
    name = Path(output_dir).resolve().name
    prefix = f"{model_id}-"
    return name[len(prefix) :] if model_id and name.startswith(prefix) else ""


def sweep_sign_column(config: Mapping[str, Any] | None) -> tuple[str, str]:
    """Which sweep column's sign must be stable, and what it is called (#576 finding 1).

    The default is ``tau_logit_mean``, the swept coefficient — right for a family
    whose published headline *is* that coefficient's marginal transform. It is wrong
    for a fit whose published headline is a distinct natural-scale quantity. The
    period-varying DiD dose model is the case that forced this: its slopes are
    ``beta_dose_phase[p] = mu_dose + sigma_dose * z[p]``, the gate read ``mu_dose``
    — a hierarchical centre, not the realised mean of two unconstrained slopes —
    while the report published a treated-row marginal that applies each row's own
    realised slope through a nonlinear items transform with unequal per-period row
    counts. The two are not the same estimand, so a fit could clear the robustness
    gate for one quantity and publish another.

    A fit whose resolved plan declares ``focal_estimand_scale == "natural"``
    therefore has its sign-stability clause read the published estimand's column,
    ``items_mean``. A stored fit written before the field existed carries no
    ``focal_estimand_scale`` and keeps the coefficient column, so old decisions stay
    reproducible without a refit.
    """
    plan = (config or {}).get("resolved_run_plan") or {}
    if isinstance(plan, Mapping) and str(plan.get("focal_estimand_scale") or "") == "natural":
        return "items_mean", str(plan.get("focal_estimand") or "the published estimand")
    return "tau_logit_mean", "the swept coefficient"


def _standard_sweep_evidence(
    output_dir: str | Path,
    outcome: str,
    *,
    config: Mapping[str, Any] | None = None,
) -> tuple[bool, str]:
    """Does an attached ``tau`` sweep actually qualify as evidence for this fit?

    Returns ``(ready, reason)``; ``reason`` names the first failure so a withhold can
    say what is wrong with the evidence rather than only that there is none.

    The bar comes from the release policy: a sweep *present in the output directory*,
    *computed from the same trace and commit as the posterior*, and *showing the sign
    of the effect is stable across the grid* — sign stability, not interval width,
    since a conservative prior is expected to move the magnitude. Each clause is
    checked:

    - the file is readable, non-empty, and carries the standard sweep's full column
      set, so a hand-rolled CSV of the same name cannot pass;
    - it has rows for **this fit's outcome**, spanning at least two distinct
      ``tau_sigma`` values (a one-point "sweep" is not a sweep);
    - every such row converged, since an unconverged cell is not evidence;
    - its recorded ``primary_config_sha256`` / ``primary_trace_sha256`` match this
      directory's own ``config.json`` and ``trace.nc``, which is what binds the sweep
      to *this* fit rather than to some earlier one;
    - every fit-locally installed cell trace (a basename ``trace_file``, the
      level/did installers' contract) still exists beside the fit and matches its
      recorded digest (#489 review); sweep-relative ITT paths are validated by the
      sweep-level evaluator instead;
    - the sign of the **published estimand's** column is the same in every cell —
      ``items_mean`` for a fit whose plan declares a natural-scale focal estimand,
      ``tau_logit_mean`` otherwise (:func:`sweep_sign_column`).

    This deliberately does not call ``evaluate_standard_sensitivity``. That evaluator
    measures the sweep against ``_standard_expected_cells()`` — the complete 44-cell
    cross-outcome grid — so its ``ready`` can only be true for the sweep-level artefact
    that lives outside any single fit's directory, and calling it here would withhold
    unconditionally. The checks above are the per-fit subset of the same idea.
    """
    from language_reading_predictors.statistical_models.sensitivity import (
        _STANDARD_REQUIRED_COLUMNS,
        sha256_file,
    )

    output_dir = Path(output_dir)
    path = output_dir / STANDARD_SENSITIVITY_FILENAME
    if not path.is_file():
        return False, "no treatment-prior sweep is attached to this fit"
    frame = _read_csv(output_dir, STANDARD_SENSITIVITY_FILENAME)
    if frame is None or frame.empty:
        return False, (
            f"the attached {STANDARD_SENSITIVITY_FILENAME} is empty or unreadable"
        )
    missing = sorted(set(_STANDARD_REQUIRED_COLUMNS) - set(frame.columns))
    if missing:
        return False, (
            f"the attached {STANDARD_SENSITIVITY_FILENAME} is not a standard "
            f"treatment-prior sweep (missing columns: {', '.join(missing[:4])})"
        )

    rows = frame.loc[frame["outcome"].astype(str) == str(outcome)]
    if rows.empty:
        return False, (
            f"the attached {STANDARD_SENSITIVITY_FILENAME} has no rows for outcome "
            f"{outcome!r}"
        )

    tau_sigmas = pd.to_numeric(rows["tau_sigma"], errors="coerce").dropna().unique()
    if len(tau_sigmas) < 2:
        return False, (
            "the attached treatment-prior sweep varies the prior over fewer than two "
            "scales, so it cannot show the effect is stable across the grid"
        )

    converged = rows["converged"].map(
        lambda value: str(value).strip().casefold() in {"true", "1", "yes"}
    )
    if not bool(converged.all()):
        return False, (
            "one or more cells of the attached treatment-prior sweep did not "
            "converge, so the sweep is not usable evidence"
        )

    for column, artefact in (
        ("primary_config_sha256", "config.json"),
        ("primary_trace_sha256", "trace.nc"),
    ):
        artefact_path = output_dir / artefact
        if not artefact_path.is_file():
            return False, (
                f"this fit has no {artefact}, so the attached treatment-prior sweep "
                "cannot be bound to it"
            )
        recorded = {str(value).strip().lower() for value in rows[column]}
        if recorded != {sha256_file(artefact_path)}:
            return False, (
                "the attached treatment-prior sweep was computed against a different "
                f"{artefact} than this fit's, so it is not this fit's evidence"
            )

    # Trace-backing (#489 review): the level/did installers rewrite
    # ``trace_file`` to the digest-suffixed basename they copy beside the fit,
    # so for those bundles a deleted or swapped cell trace must un-lift the
    # gate rather than leave a manifest that merely *names* evidence. The ITT
    # installer keeps sweep-directory-relative paths (its traces live in the
    # sweep tree and are validated by ``evaluate_standard_sensitivity``'s
    # provenance machinery), so path-bearing entries are not checked here —
    # requiring them fit-locally would withhold every ITT fit.
    for _, row in rows.iterrows():
        name = str(row["trace_file"])
        if "/" in name or "\\" in name:
            continue
        candidate = output_dir / name
        if not candidate.is_file():
            return False, (
                "the attached treatment-prior sweep names an installed cell "
                "trace that is missing, so the bundle is no longer trace-backed"
            )
        if sha256_file(candidate) != str(row["trace_sha256"]).strip().lower():
            return False, (
                "an installed cell trace does not match the attached "
                "treatment-prior sweep's recorded digest"
            )

    if config is None:
        config, _config_error = _read_json(Path(output_dir) / "config.json")
        if not isinstance(config, Mapping):
            config = {}
    # Run-plan binding (#576 finding 6). Checked here as well as at install time so a
    # *stale* attached bundle stops lifting the gate the moment the fit's own plan
    # changes, rather than only when someone re-runs the installer. Opt-in on the
    # plan recording a digest, so a stored fit written before the field existed
    # re-decides exactly as it did.
    plan = config.get("resolved_run_plan") or {}
    recorded_plan_digest = (
        str(plan.get("run_plan_digest") or "").strip().lower()
        if isinstance(plan, Mapping)
        else ""
    )
    if recorded_plan_digest:
        if "primary_run_plan_sha256" not in rows.columns:
            return False, (
                f"the attached {STANDARD_SENSITIVITY_FILENAME} predates run-plan "
                "binding, so it cannot be shown to describe the model this fit "
                "actually fitted"
            )
        recorded = {
            str(value).strip().lower() for value in rows["primary_run_plan_sha256"]
        }
        if recorded != {recorded_plan_digest}:
            return False, (
                "the attached treatment-prior sweep was computed against a "
                "different resolved run plan than this fit's, so it is not this "
                "fit's evidence"
            )
    sign_column, estimand_label = sweep_sign_column(config)
    if sign_column not in rows.columns:
        return False, (
            f"the attached {STANDARD_SENSITIVITY_FILENAME} has no {sign_column!r} "
            f"column, so the sign of {estimand_label} cannot be checked"
        )
    signs = np.sign(
        pd.to_numeric(rows[sign_column], errors="coerce").to_numpy(dtype=float)
    )
    if not np.isfinite(signs).all() or len(set(signs.tolist())) != 1:
        return False, (
            f"{estimand_label} changes sign across the attached treatment-prior "
            "sweep, so its direction is not stable under the prior"
        )
    return True, ""


def _floor_decision(
    output_dir: str | Path,
    config: Mapping[str, Any],
    *,
    tier: str,
    tau_class: TauSensitivityClass,
    prior: float | None,
    likelihood: float | None,
    diagnosis: str | None,
) -> ReleaseDecision:
    """Mirror ``_results_floored.qmd``'s gate, so box and prose cannot disagree.

    The grid is *required* on exactly the condition that partial uses —
    ``tau_psense_status`` in ``{conflict, unavailable}`` — rather than on this module's
    finer class, so the two gates fire together. Both call the same evaluator, which
    recomputes convergence, effects and provenance from the content-addressed traces
    rather than trusting the CSV.
    """
    symbol = str(config.get("outcome_symbol") or "")
    psense = _read_csv(output_dir, "psense_summary.csv", index_col=0)
    grid_required = tau_psense_status(psense) in {"conflict", "unavailable"}

    ready = False
    if grid_required:
        try:
            primary_reference = load_primary_floor_reference(
                Path(output_dir),
                symbol,
                config_name=_config_name(output_dir, str(config.get("model_id") or "")),
            )
        except Exception:  # noqa: BLE001 - an unreadable primary is a gate failure
            primary_reference = None
        status = evaluate_floor_sensitivity(
            _read_csv(output_dir, FLOOR_SENSITIVITY_FILENAME),
            symbol,
            primary_reference=primary_reference,
            trace_root=Path(output_dir),
            require_hash_suffix=True,
        )
        ready = bool(status.get("ready"))

    common = {
        "tau_class": tau_class,
        "tier": tier,
        "floor_rule": True,
        "floor_grid_required": grid_required,
        "floor_grid_ready": ready,
        "prior_sensitivity": prior,
        "likelihood_sensitivity": likelihood,
        "diagnosis": diagnosis,
    }
    if grid_required and not ready and tier in _WITHHOLD_TIERS:
        return ReleaseDecision(
            status="withhold",
            reason=(
                # Ends at the state of the evidence, deliberately: the callout in
                # ``_key_findings.qmd`` supplies the consequence ("... so no
                # cause-and-effect statement is released"), so a reason that also
                # spelled it out rendered the clause twice in the published report.
                f"the {FLOOR_SENSITIVITY_FILENAME} treatment-prior grid this "
                "floor-rule outcome requires is absent, incomplete, or not "
                "provenance-aligned with this fit"
            ),
            evidence=(
                f"a complete, trace-validated {FLOOR_SENSITIVITY_FILENAME} grid in "
                "this fit's output directory"
            ),
            **common,
        )
    # From here the grid is either not required (clean diagnosis) or complete and
    # trace-validated. The per-class treatment mirrors the graded branch below —
    # the grid gates *whether* a floored fit may speak, not *how* its prior
    # dependence is described (2026-08-20 ITT review, finding 2: the floored path
    # released a ``prior_data_conflict`` verdict without the attenuation note the
    # module policy promises).
    if tau_class == "prior_data_conflict":
        return ReleaseDecision(
            status="release",
            note=_PRIOR_ATTENUATION_NOTE,
            reason=(
                "power-scaling flags a prior-data conflict on `tau`, but the "
                "likelihood moves the posterior too, so the conservative prior "
                "attenuates the estimate rather than determining it; the "
                "completed treatment-prior grid bounds how far"
            ),
            **common,
        )
    if tau_class in ("prior_dominant", "unavailable") and grid_required:
        # The graded branch qualifies these classes when a trace-bound sweep
        # exists; the completed six-cell grid is this outcome's estimand-matched
        # sweep, so the same qualification applies rather than a bare release.
        return ReleaseDecision(
            status="qualify",
            note=_QUALIFY_NOTE,
            evidence=(
                f"a complete, trace-validated {FLOOR_SENSITIVITY_FILENAME} grid "
                "showing the off-floor effect across the treatment-prior cells"
            ),
            **common,
        )
    return ReleaseDecision(status="release", **common)


def _gain_offfloor_decision(
    output_dir: str | Path,
    config: Mapping[str, Any],
    *,
    tier: str,
    tau_class: TauSensitivityClass,
    prior: float | None,
    likelihood: float | None,
    diagnosis: str | None,
    causal_term: str,
) -> ReleaseDecision:
    """The ITT floor rule's policy shape for a gain-family off-floor fit (#575 10d).

    The evidence differs — the family treatment-prior sweep, validated per fit by
    :func:`_standard_sweep_evidence` (columns, ≥2 prior cells, per-cell
    convergence, sha256 binding to THIS fit's ``config.json`` and ``trace.nc``,
    installed cell-trace digests, and sign stability of the published
    risk-difference column) — but the *requirement* mirrors ``_floor_decision``:
    any non-clean power-scaling verdict on the off-floor risk difference needs
    that estimand-matched evidence before the findings box may speak, including
    the release-with-note prior-data-conflict class, which the graded route
    releases without a sweep. Before this branch, a clear off-floor gain fit
    would have released on the graded route with no grid at all — the exact gap
    the module docstring recorded.
    """
    sweep_required = tau_class != "clear"
    ready, sweep_reason = (
        _standard_sweep_evidence(
            output_dir, str(config.get("outcome_symbol") or ""), config=config
        )
        if sweep_required
        else (False, "")
    )
    common = {
        "tau_class": tau_class,
        "tier": tier,
        "floor_rule": True,
        "floor_grid_required": sweep_required,
        "floor_grid_ready": bool(ready),
        "prior_sensitivity": prior,
        "likelihood_sensitivity": likelihood,
        "diagnosis": diagnosis,
    }
    if not sweep_required:
        return ReleaseDecision(status="release", **common)
    if not ready and tier in _WITHHOLD_TIERS:
        return ReleaseDecision(
            status="withhold",
            reason=(
                f"power-scaling on `{causal_term}` is not clean for this "
                f"off-floor fit and {sweep_reason}"
            ),
            evidence=(
                f"a trace-bound {STANDARD_SENSITIVITY_FILENAME} covering this "
                "outcome's off-floor risk difference across the treatment-prior "
                "grid"
            ),
            **common,
        )
    if tau_class == "prior_data_conflict":
        return ReleaseDecision(
            status="release",
            note=_PRIOR_ATTENUATION_NOTE,
            reason=(
                f"power-scaling flags a prior-data conflict on `{causal_term}`, "
                "but the likelihood moves the posterior too, so the conservative "
                "prior attenuates the estimate rather than determining it; the "
                "trace-bound treatment-prior sweep bounds how far"
            ),
            **common,
        )
    return ReleaseDecision(
        status="qualify",
        note=_QUALIFY_NOTE,
        # The evidence sentence is asserted only when the sweep actually
        # validated; a non-withhold tier reaches here without one and must not
        # cite evidence it does not have.
        evidence=(
            f"a trace-bound {STANDARD_SENSITIVITY_FILENAME} showing the "
            "off-floor risk difference keeps its sign across the "
            "treatment-prior grid"
        )
        if ready
        else None,
        **common,
    )


def evaluate_release(
    output_dir: str | Path,
    config: Mapping[str, Any] | None = None,
) -> ReleaseDecision:
    """Decide what a randomised-effect fit's robustness evidence permits it to say.

    The family entry point: resolves the term the headline rests on
    (:func:`causal_term_for`) and applies the one policy. Callers that already know
    the term can go straight to :func:`evaluate_itt_release`, which is the shared
    implementation rather than an ITT-specific one — the name predates the extension
    and is kept so existing callers and tests are undisturbed.
    """
    output_dir = Path(output_dir)
    if config is None:
        config = _load_config(output_dir) or {}
    return evaluate_itt_release(
        output_dir, config, causal_term=causal_term_for(config)
    )


def evaluate_itt_release(
    output_dir: str | Path,
    config: Mapping[str, Any] | None = None,
    *,
    causal_term: str = "tau",
) -> ReleaseDecision:
    """Decide what a fit's robustness evidence permits its findings box to say.

    Reads only artefacts already in ``output_dir``, so it re-runs over a stored fit
    without refitting — the same contract ``generate_key_findings`` keeps.

    ``causal_term`` defaults to ITT's ``tau``; :func:`evaluate_release` resolves it
    per family.
    """
    output_dir = Path(output_dir)
    if config is None:
        config = _load_config(output_dir) or {}
    plan = config.get("resolved_run_plan") or {}
    tier = _model_tier(config)

    psense = _read_csv(output_dir, "psense_summary.csv", index_col=0)
    tau_class, prior, likelihood, diagnosis = classify_tau_sensitivity(
        psense, term=causal_term
    )

    # ITT-only: the six-cell grid and its provenance machinery are bound to the
    # registered ITT floor rule. ``gain_factors``' off-floor models carry the
    # same *policy shape* on their own estimand-matched evidence — the
    # trace-bound family treatment-prior sweep — via the branch below (#575
    # finding 10d closed the documented gap: a non-clean off-floor gain verdict
    # now requires that evidence even for the release-with-note conflict class,
    # exactly as the ITT floor rule requires its grid).
    if config.get("kind") == "itt" and bool(plan.get("floor_rule", False)):
        return _floor_decision(
            output_dir,
            config,
            tier=tier,
            tau_class=tau_class,
            prior=prior,
            likelihood=likelihood,
            diagnosis=diagnosis,
        )
    if config.get("kind") == "gain_factors" and bool(plan.get("off_floor", False)):
        return _gain_offfloor_decision(
            output_dir,
            config,
            tier=tier,
            tau_class=tau_class,
            prior=prior,
            likelihood=likelihood,
            diagnosis=diagnosis,
            causal_term=causal_term,
        )

    common = {
        "tau_class": tau_class,
        "tier": tier,
        "prior_sensitivity": prior,
        "likelihood_sensitivity": likelihood,
        "diagnosis": diagnosis,
    }
    if tau_class == "clear":
        return ReleaseDecision(status="release", **common)
    if tau_class == "prior_data_conflict":
        return ReleaseDecision(
            status="release",
            note=_PRIOR_ATTENUATION_NOTE,
            reason=(
                f"power-scaling flags a prior-data conflict on `{causal_term}`, but "
                "the likelihood moves the posterior too, so the conservative prior "
                "attenuates the estimate rather than determining it"
            ),
            **common,
        )
    if tier not in _WITHHOLD_TIERS:
        return ReleaseDecision(status="qualify", note=_QUALIFY_NOTE, **common)
    sweep_ready, sweep_reason = _standard_sweep_evidence(
        output_dir, str(config.get("outcome_symbol") or ""), config=config
    )
    if sweep_ready:
        return ReleaseDecision(
            status="qualify",
            note=_QUALIFY_NOTE,
            evidence=(
                f"a trace-bound {STANDARD_SENSITIVITY_FILENAME} showing the effect "
                "keeps its sign across the treatment-prior grid"
            ),
            **common,
        )
    if tau_class == "prior_dominant":
        reason = (
            f"power-scaling shows `{causal_term}` responds to the prior "
            f"({prior:.3g}) but not to the likelihood ({likelihood:.3g}), and "
            f"{sweep_reason}, so the direction of the effect is not established by "
            "the data alone"
        )
    else:
        reason = (
            f"no unique, interpretable power-scaling diagnosis for `{causal_term}` is "
            "available for this fit, so its prior dependence is unmeasured rather "
            "than measured clean"
        )
    return ReleaseDecision(
        status="withhold",
        reason=reason,
        evidence=(
            f"a {STANDARD_SENSITIVITY_FILENAME} treatment-prior sweep, computed from "
            "this fit's own trace, showing the sign of the effect is stable across "
            "the grid"
        ),
        **common,
    )


def _model_tier(config: Mapping[str, Any]) -> str:
    """Classify a fit for tiering purposes.

    ``adjusted_robustness`` is any ITT fit carrying a robustness adjustment set (the
    SES and general-ability comparators); ``off_grid`` is an outcome the standard
    44-cell sweep does not cover; everything else is ``primary``. Recorded on the
    decision even though the policy is currently uniform, so a graded policy needs no
    new plumbing and so the audit trail says which tier a fit was judged in.

    The ``adjusted_robustness`` test is deliberately ITT-only. Other families put
    confounders in ``adjust_for`` as part of their *primary* specification — the DAG
    adjustment set of a level- or gain-factor primary, not a robustness comparator —
    so keying the tier on the presence of an adjustment set alone labelled eight of
    the eleven level primaries as robustness comparators (#584 lower-severity 7).
    The withhold policy is uniform across tiers, so this corrects the audit label
    rather than any release decision.
    """
    from language_reading_predictors.statistical_models.sensitivity import (
        STANDARD_SENSITIVITY_OUTCOMES,
    )

    plan = config.get("resolved_run_plan") or {}
    if config.get("kind") == "itt" and (
        plan.get("adjust_for") or plan.get("adjustment")
    ):
        return "adjusted_robustness"
    if str(config.get("outcome_symbol") or "") not in STANDARD_SENSITIVITY_OUTCOMES:
        return "off_grid"
    return "primary"


def _load_config(output_dir: Path) -> dict[str, Any] | None:
    path = output_dir / "config.json"
    if not os.path.exists(path):
        return None
    try:
        with open(path, encoding="utf-8") as handle:
            loaded = json.load(handle)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    return loaded if isinstance(loaded, dict) else None


# ---------------------------------------------------------------------------
# The publication decision (#394 design point 3)
# ---------------------------------------------------------------------------

RELEASE_DECISION_FILENAME = "release_decision.json"

PublicationStatus = Literal[
    "ok",
    "not_available",
    "inputs_unresolved",
    "gate_failed",
    "artifacts_incomplete",
    "robustness_unresolved",
]
"""What a fit is permitted to publish, in the vocabulary ``key_findings.json`` uses."""

ReleaseStage = Literal["inputs", "computation", "artifacts", "robustness"]
"""Which stage of the decision settled it.

``inputs`` the fit's own summary files are missing or unreadable, or its recorded
scientific-input contract is unresolved; ``computation`` the sampling-quality gate
failed; ``artifacts`` a required output is not on disk; ``robustness`` required
sensitivity evidence does not preserve the released scientific finding. The order is
the order below: a fit that did not converge is not asked whether its sensitivity
evidence is acceptable.
"""


@dataclass(frozen=True, slots=True)
class ReleaseEvaluation:
    """The whole publication decision for one fit, in the order it is made.

    Before this existed the decision was assembled inline inside
    ``reporting.generate_key_findings`` — four sequential branches over
    ``diagnostics_summary.json``, ``config.json`` and the robustness gate, with no
    object anyone could hold, print, record or test. Report finalisation therefore
    could not *receive* a release decision; it could only call the function that
    happened to make one on its way to writing findings.

    The fields below carry what each stage found, so ``release_decision.json`` can
    state why a fit published what it published — for every family, not only the
    ones the robustness gate covers.
    """

    status: PublicationStatus
    stage: ReleaseStage
    reason: str = ""
    #: Human-readable sampling-quality checks that failed, when ``computation`` decided.
    failing_checks: tuple[str, ...] = ()
    #: Scientific input-validity blockers recorded by the fit-time contract.
    input_failures: tuple[str, ...] = ()
    #: Required artefacts absent from the fit or invalid under a family contract.
    missing_artifacts: tuple[str, ...] = ()
    #: The robustness verdict, when the fit is in scope for that gate.
    robustness: ReleaseDecision | None = None
    #: The fit's ``config.json``, loaded once so callers need not re-read it.
    #: ``None`` when it is unreadable, ``{}`` when it is absent.
    config: Mapping[str, Any] | None = None
    #: Named sampling preset used for the fit, when it can be resolved.
    sampling_preset: str | None = None
    #: True for dev/test or an absent, unknown or inconsistent preset. Such fits may
    #: render local diagnostics but must not be used as publication-grade results.
    development_only: bool = True
    #: Explanation for ``development_only``; kept separate from ``reason`` because a
    #: clean local diagnostic fit still has ``status='ok'``.
    publication_qualification: str = ""
    #: For a bound factorised joint contrast pair, the measured consequence of the
    #: dependence model for the **declared** average-marginal-effect difference
    #: (2026-08-23 joint audit, finding 2). ``None`` when the fit is not a bound
    #: parent; the ``material`` flag says whether it changed the conclusion.
    dependence_contrast: Mapping[str, Any] | None = None

    @property
    def publishable(self) -> bool:
        """May this local fit report render its scientific tables and sentences?"""
        return self.status == "ok"

    @property
    def scientific_publication_eligible(self) -> bool:
        """May this fit be used as a publication-grade scientific result?"""

        return self.publishable and not self.development_only

    @property
    def note(self) -> str:
        """A qualification to attach to released findings, or ``""``."""
        return self.robustness.note if self.robustness is not None else ""

    def as_dict(self) -> dict[str, Any]:
        """JSON-ready record for ``release_decision.json``."""
        record: dict[str, Any] = {
            "status": self.status,
            "stage": self.stage,
            "publishable": self.publishable,
            "scientific_publication_eligible": self.scientific_publication_eligible,
            "development_only": self.development_only,
        }
        if self.sampling_preset is not None:
            record["sampling_preset"] = self.sampling_preset
        if self.publication_qualification:
            record["publication_qualification"] = self.publication_qualification
        if self.reason:
            record["reason"] = self.reason
        if self.input_failures:
            record["input_failures"] = list(self.input_failures)
        if self.failing_checks:
            record["failing_checks"] = list(self.failing_checks)
        if self.missing_artifacts:
            record["missing_artifacts"] = list(self.missing_artifacts)
        if self.robustness is not None:
            record["robustness"] = self.robustness.as_dict()
        if self.dependence_contrast:
            record["dependence_contrast"] = dict(self.dependence_contrast)
        if self.config:
            record["model_id"] = self.config.get("model_id")
            record["kind"] = self.config.get("kind")
        return record

    def summary(self) -> str:
        """One line for the console at finalisation."""
        if self.publishable:
            qualifiers = []
            if self.note:
                qualifiers.append("with note")
            if self.development_only:
                qualifiers.append("development-only")
            return "ok" + (f" ({', '.join(qualifiers)})" if qualifiers else "")
        return f"{self.status} at the {self.stage} stage: {self.reason}"


def _read_json(path: str | Path) -> tuple[Any, str | None]:
    """``(payload, error)`` — ``error`` names why the payload is unusable."""
    if not os.path.exists(path):
        return None, "missing"
    try:
        with open(path, encoding="utf-8") as handle:
            return json.load(handle), None
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None, "unreadable"


#: The stored path's minimum evidence that a directory *is* a completed fit:
#: a posterior and the two tables every registered family writes. Deliberately
#: narrow (2026-08-22 ITT audit, finding 2). The manifest requirement below is
#: what actually closes the hole the audit found; this floor exists so a gutted
#: or legacy directory cannot coast on a manifest that under-declares. Family
#: result tables are *not* listed: the key-findings layer already checks each
#: family's own outputs for presence and internal consistency, and duplicating
#: that here would move those verdicts to a stage that cannot explain them.
_CORE_ARTIFACTS_BASE: tuple[str, ...] = (
    "trace.nc",
    "diagnostics.csv",
    "priors_table.csv",
)


def _core_artifact_failures(output_dir: Path) -> list[str]:
    """Core scientific outputs absent from a stored fit directory.

    The stored-path floor. A fit's own manifest is the authority on what *it*
    wrote, but a manifest cannot vouch for a directory that has no manifest — and
    before this floor existed an otherwise-empty directory carrying only clean
    ``diagnostics_summary.json`` / ``config.json`` was declared publishable
    (2026-08-22 ITT audit, finding 2).
    """
    return [
        name
        for name in _CORE_ARTIFACTS_BASE
        if not os.path.exists(output_dir / name)
    ]


def _recorded_required_artifacts(
    output_dir: Path, artifacts: Any
) -> tuple[str, ...]:
    """Required artefacts the fit recorded but that are not on disk.

    ``artifacts`` is the run's :class:`artifacts.ArtifactLog` during a fit, and
    ``None`` for a post-hoc evaluation over a stored directory — in which case the
    inventory is read back from ``artifact_manifest.json``, so the same decision
    can be reproduced without refitting.

    Only *required* artefacts count. An optional figure that a backend hiccup
    skipped is already recorded with its failure and does not withhold anything;
    that distinction is the whole point of the required/optional split.

    The stored path **fails closed** (2026-08-22 ITT audit, finding 2). A missing,
    unreadable or entry-less ``artifact_manifest.json`` used to return "nothing is
    missing", so a directory holding only a clean gate and config published — and
    that path is not hypothetical: ``_key_findings.qmd`` re-decides publication
    over the stored directory at *render* time, as does
    ``scripts/regenerate_key_findings.py``. An unusable manifest is now itself a
    missing artefact, and :func:`_core_artifact_failures` is applied underneath so
    a manifest that under-declares cannot wave through a directory with no trace
    or no headline table.

    Both paths are seeded with that same floor (#637 stage 1), so the live and
    stored decisions differ **only** by the manifest requirement the live path
    cannot yet meet. The two therefore agree either side of
    ``artifacts.write_manifest``, which is what
    ``test_release_decision`` now asserts directly.
    """
    records = getattr(artifacts, "records", None)
    if records is not None:
        # Fit-time: the live log is the authority on what *this run* declared, and
        # the manifest does not exist yet (finalisation writes it *after* this
        # decision) — so the manifest, and only the manifest, is exempt here.
        #
        # The core floor is not (#637 stage 1). Seeding both paths with it is what
        # makes the two evaluations agree: before this, a directory with clean
        # diagnostics, an empty live log and no ``trace.nc`` published during the
        # fit and came back ``artifacts_incomplete`` the moment the same directory
        # was re-decided at render time. The floor is a property of the directory,
        # not of who is asking.
        missing = _core_artifact_failures(output_dir)
        declared = [
            (rec.filename, rec.status, bool(rec.required)) for rec in records.values()
        ]
        missing.extend(
            filename
            for filename, status, required in declared
            if required
            and status in ("written", "missing")
            and not os.path.exists(output_dir / filename)
        )
        return tuple(sorted(set(missing)))

    missing = _core_artifact_failures(output_dir)
    manifest, error = _read_json(output_dir / "artifact_manifest.json")
    entries = (manifest or {}).get("artifacts") if isinstance(manifest, dict) else None
    if not entries:
        reason = {
            "missing": "is missing",
            "unreadable": "could not be parsed",
        }.get(error or "", "records no artefacts")
        missing.append(f"artifact_manifest.json ({reason})")
        return tuple(sorted(set(missing)))
    declared = [
        (str(e.get("filename")), str(e.get("status")), bool(e.get("required")))
        for e in entries
    ]
    missing.extend(
        filename
        for filename, status, required in declared
        if required
        and status in ("written", "missing")
        and not os.path.exists(output_dir / filename)
    )
    return tuple(sorted(set(missing)))


def _mediation_t3_release_failures(
    output_dir: Path, config: Mapping[str, Any]
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Fail-closed computation and artefact checks for the mediation t3 fit.

    Single-mediator fits without a longitudinal primary estimand always run
    this temporal-ordering sensitivity — the interventional companions included,
    since #585 finding 2 made them run the same fit under their own labels
    (#631 finding 9 closed the release-side exemption that outlived it).  Its
    posterior bypasses the primary ``diagnostics_summary.json`` gate, so release
    requires a checked, converged provenance row, a concordant summary table and
    the persisted sub-fit trace.  Already-longitudinal primaries and the
    period-stacked entry point (whose ``extra`` carries no estimand) do not run
    it and stay exempt.

    The two returned tuples preserve the release-stage contract. A present but
    failed or unchecked convergence verdict is a computation failure; a missing,
    unreadable or internally inconsistent output is an artefact failure. Losing a
    trace after a clean fit must not be reported as evidence that sampling failed.
    """
    if config.get("kind") != "mediation":
        return (), ()
    extra = config.get("extra") or {}
    if not isinstance(extra, Mapping):
        return (), ("config.json (mediation t3 configuration is unreadable)",)
    required = (
        extra.get("estimand") in ("natural", "interventional")
        and extra.get("outcome_time") is None
    )
    if not required:
        return (), ()

    computation_failures: list[str] = []
    artifact_failures: list[str] = []
    summary = _read_csv(output_dir, "mediation_summary_t3.csv")
    if summary is None or summary.empty:
        artifact_failures.append("mediation_summary_t3.csv")
    else:
        if "converged" not in summary.columns:
            artifact_failures.append("mediation_summary_t3.csv (no convergence column)")
        elif not bool(
            summary["converged"]
            .map(lambda value: str(value).strip().casefold() in {"true", "1", "yes"})
            .all()
        ):
            computation_failures.append(
                "mediation t3 sensitivity summary convergence failed or was unchecked"
            )
        trace_files = (
            set(summary["trace_file"].dropna().astype(str))
            if "trace_file" in summary.columns
            else set()
        )
        if trace_files != {MEDIATION_T3_TRACE_FILENAME}:
            artifact_failures.append("mediation_summary_t3.csv (invalid trace binding)")

    provenance = _read_csv(output_dir, "subfit_provenance.csv")
    model_id = str(config.get("model_id") or "")
    if provenance is None or provenance.empty or "label" not in provenance.columns:
        artifact_failures.append("subfit_provenance.csv")
    else:
        rows = provenance.loc[
            provenance["label"].astype(str) == f"{model_id} t3 sensitivity"
        ]
        if len(rows) != 1:
            artifact_failures.append(
                "subfit_provenance.csv (no unique mediation t3 row)"
            )
        else:
            row = rows.iloc[0]
            if str(row.get("role", "")).strip() != "sensitivity":
                artifact_failures.append(
                    "subfit_provenance.csv (invalid mediation t3 role)"
                )
            if "converged" not in provenance.columns:
                artifact_failures.append(
                    "subfit_provenance.csv (no convergence column)"
                )
            elif str(row.get("converged", "")).strip().casefold() not in {
                "true",
                "1",
                "yes",
            }:
                computation_failures.append(
                    "mediation t3 sensitivity provenance failed or was unchecked"
                )
            if str(row.get("trace_file", "")).strip() != MEDIATION_T3_TRACE_FILENAME:
                artifact_failures.append(
                    "subfit_provenance.csv (invalid mediation t3 trace binding)"
                )

    if not (output_dir / MEDIATION_T3_TRACE_FILENAME).is_file():
        artifact_failures.append(MEDIATION_T3_TRACE_FILENAME)
    return tuple(computation_failures), tuple(artifact_failures)


def _joint_mechanism_wave_release_failures(
    output_dir: Path, config: Mapping[str, Any]
) -> tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
    """Fail-closed bundle check for every wave a joint-mechanism levels fit publishes.

    The levels design publishes one posterior per timepoint. Before the 2026-08-23
    follow-up review only the wave hosting the fit-level artefacts passed through the
    full lifecycle, so ``release_decision.json`` could say "ok" while three of the
    four published posteriors had no persisted trace, no informative predictive check
    and no recorded power-scaling result — and the fit-level gate had never seen them.

    Returns ``(computation, artefact, qualification)``:

    * **computation** — a wave whose convergence verdict failed or could not be taken.
      Withholds, exactly as the primary gate does.
    * **artefact** — a missing or internally inconsistent bundle: an absent trace,
      predictive or power-scaling file, a slope table naming waves the diagnostics
      table does not, or a non-hosting wave with no matching sub-fit provenance row.
      Withholds.
    * **qualification** — the predeclared predictive-adequacy rule
      (:data:`JOINT_MECHANISM_MARGINAL_COVERAGE_FLOORS`). Attaches a note; does not
      withhold, because substantive misfit is a finding about the model rather than a
      computational failure.
    """
    if config.get("kind") != "joint_mechanism":
        return (), (), ()
    extra = config.get("extra") or {}
    if not isinstance(extra, Mapping):
        return (), ("config.json (joint-mechanism configuration is unreadable)",), ()
    if str(extra.get("design", "")) != "levels":
        return (), (), ()

    computation: list[str] = []
    artefacts: list[str] = []
    qualifications: list[str] = []

    diagnostics = _read_csv(output_dir, "joint_mechanism_fit_diagnostics.csv")
    required_columns = (
        "wave",
        "role",
        "converged",
        "trace_file",
        "marginal_ppc_file",
        "psense_file",
    )
    if diagnostics is None or diagnostics.empty:
        return (), ("joint_mechanism_fit_diagnostics.csv",), ()
    missing_columns = [c for c in required_columns if c not in diagnostics.columns]
    if missing_columns:
        return (
            (),
            (
                "joint_mechanism_fit_diagnostics.csv (no "
                f"{', '.join(missing_columns)} column)",
            ),
            (),
        )

    if int((diagnostics["role"].astype(str).str.strip() == "anchor").sum()) != 1:
        artefacts.append(
            "joint_mechanism_fit_diagnostics.csv (no unique artefact-hosting wave)"
        )

    provenance = _read_csv(output_dir, "subfit_provenance.csv")
    model_id = str(config.get("model_id") or "")
    for _, row in diagnostics.iterrows():
        wave = str(row["wave"]).strip()
        if _stored_bool(row.get("converged")) is not True:
            computation.append(
                f"joint-mechanism wave {wave} failed or was not convergence-checked"
            )
        for column in ("trace_file", "marginal_ppc_file", "psense_file"):
            filename = str(row.get(column) or "").strip()
            if not filename:
                artefacts.append(
                    f"joint_mechanism_fit_diagnostics.csv (wave {wave} declares no "
                    f"{column})"
                )
            elif not (output_dir / filename).is_file():
                artefacts.append(filename)
        if str(row["role"]).strip() == "anchor":
            continue
        # A non-hosting wave is a sub-fit, and a published sub-fit estimate is only
        # auditable through its provenance row: which rows it was fitted to, at what
        # sampling settings, scanning which parameters, backed by which trace.
        if provenance is None or "label" not in provenance.columns:
            artefacts.append("subfit_provenance.csv")
            continue
        rows = provenance.loc[
            provenance["label"].astype(str) == f"{model_id} wave {wave}"
        ]
        if len(rows) != 1:
            artefacts.append(f"subfit_provenance.csv (no unique {wave} row)")
            continue
        record = rows.iloc[0]
        if str(record.get("role", "")).strip() != "wave":
            artefacts.append(f"subfit_provenance.csv (invalid {wave} role)")
        if str(record.get("trace_file", "")).strip() != str(row["trace_file"]).strip():
            artefacts.append(f"subfit_provenance.csv (invalid {wave} trace binding)")
        if _stored_bool(record.get("converged")) is not True:
            computation.append(
                f"joint-mechanism wave {wave} provenance failed or was unchecked"
            )

    published = set(diagnostics["wave"].astype(str).str.strip())
    slopes = _read_csv(output_dir, "joint_mechanism_slopes.csv")
    if slopes is None or "wave" not in slopes.columns:
        artefacts.append("joint_mechanism_slopes.csv")
    else:
        reported = set(slopes["wave"].astype(str).str.strip())
        if reported != published:
            artefacts.append(
                "joint_mechanism_slopes.csv (waves do not match "
                "joint_mechanism_fit_diagnostics.csv)"
            )

    qualifications.extend(
        _joint_mechanism_coverage_qualifications(output_dir, diagnostics)
    )
    return tuple(computation), tuple(sorted(set(artefacts))), tuple(qualifications)


def _concurrent_published_fit_release_failures(
    output_dir: Path, config: Mapping[str, Any]
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Fail-closed convergence check for every fit a concurrent run publishes.

    The concurrent family publishes one adjusted posterior per wave plus a
    single-skill posterior per wave-by-predictor cell, but only the anchor wave
    passes through the fit-level gate. The pipeline computed
    ``all_published_fits_converged`` and nothing ever read it back (#631
    finding 6), so ``release_decision.json`` could say "ok" while displayed rows
    carried ``converged=False`` — the same class of defect the #591
    joint-mechanism remediation closed for its wave sub-fits.

    A missing or column-incomplete ``concurrent_fit_diagnostics.csv`` is an
    artefact failure; a published row whose verdict failed or was never taken is
    a computation failure, exactly as the primary gate treats the anchor.
    """
    if config.get("kind") != "concurrent":
        return (), ()
    diagnostics = _read_csv(output_dir, "concurrent_fit_diagnostics.csv")
    if diagnostics is None or diagnostics.empty:
        return (), ("concurrent_fit_diagnostics.csv",)
    required_columns = ("timepoint", "fit_kind", "predictor", "converged")
    missing_columns = [c for c in required_columns if c not in diagnostics.columns]
    if missing_columns:
        return (
            (),
            (
                "concurrent_fit_diagnostics.csv (no "
                f"{', '.join(missing_columns)} column)",
            ),
        )
    computation: list[str] = []
    for _, row in diagnostics.iterrows():
        if _stored_bool(row.get("converged")) is not True:
            computation.append(
                f"concurrent published fit t{str(row['timepoint']).strip()} "
                f"{str(row['fit_kind']).strip()} {str(row['predictor']).strip()} "
                "failed or was not convergence-checked"
            )
    return tuple(computation), ()


def _adjusted_ses_release_failures(
    output_dir: Path, config: Mapping[str, Any]
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Fail-closed check for the SES refit an RLI adjusted fit promises.

    The resolved recipe declares the SES complete-case sensitivity as a required
    check, but the pipeline catches every exception in that leg and continues,
    recording the failure only as ``extra["ses_error"]`` — which nothing read
    back, so the fit could publish with the section silently absent (#631
    finding 7). Scope: the ``ses_error`` key is written (null on success)
    exactly by the RLI entry point that promises the refit; the Byrne/RLM
    adjusted fits carry no such key and stay exempt.

    A recorded error or a missing/invalid ``ses_sensitivity.csv`` is an artefact
    failure; a present summary whose convergence failed or was unchecked is a
    computation failure — the same boundary the mediation t3 check draws.
    """
    if config.get("kind") != "adjusted":
        return (), ()
    extra = config.get("extra") or {}
    if not isinstance(extra, Mapping):
        return (), ("config.json (adjusted configuration is unreadable)",)
    if "ses_error" not in extra:
        return (), ()
    ses_error = extra.get("ses_error")
    if ses_error:
        return (
            (),
            (f"ses_sensitivity.csv (SES sensitivity refit failed: {ses_error})",),
        )
    summary = _read_csv(output_dir, "ses_sensitivity.csv")
    if summary is None or summary.empty:
        return (), ("ses_sensitivity.csv",)
    if "converged" not in summary.columns:
        return (), ("ses_sensitivity.csv (no convergence column)",)
    if not all(
        _stored_bool(value) is True for value in summary["converged"].tolist()
    ):
        return (
            ("adjusted SES sensitivity convergence failed or was unchecked",),
            (),
        )
    return (), ()


def _gain_period1_release_failures(
    output_dir: Path, config: Mapping[str, Any]
) -> tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
    """Fail-closed check for the mandatory gain-factors period-1-only refit.

    The recipe declares the period-1-only refit sensitivity mandatory for every
    model of record (`period1_sensitivity_required`), and the pipeline records
    its convergence verdict and headline shift — but release evaluation consumed
    none of it (#631 finding 8), so a fit whose mandatory sensitivity failed or
    never ran still published. Opt-in on the recorded plan flag, so stored fits
    predating the field re-decide unchanged (the growth-influence precedent).

    Returns ``(computation, artefacts, robustness)``:

    * **artefacts** — a missing or malformed ``period1_sensitivity.csv`` (it must
      carry exactly one ``primary_period_stacked`` and one ``period1_only`` row
      with the headline columns) or a missing persisted sub-fit trace.
    * **computation** — a ``period1_only`` convergence verdict that failed or was
      never taken.
    * **robustness** — material disagreement between the stacked primary and the
      period-1-only refit. The documented rule (#631 finding 8) mirrors the
      growth-influence policy: the fit is withheld at the robustness stage when
      the two ``beta_trt`` posterior medians disagree in sign or the two 89%
      intervals fail to overlap; anything milder is left to the report's own
      side-by-side table.
    """
    if config.get("kind") != "gain_factors":
        return (), (), ()
    plan = config.get("resolved_run_plan") or {}
    if not isinstance(plan, Mapping) or not plan.get("period1_sensitivity_required"):
        return (), (), ()

    summary = _read_csv(output_dir, "period1_sensitivity.csv")
    if summary is None or summary.empty:
        return (), ("period1_sensitivity.csv",), ()
    required_columns = (
        "fit",
        "beta_trt_median",
        "beta_trt_lo",
        "beta_trt_hi",
        "converged",
    )
    missing_columns = [c for c in required_columns if c not in summary.columns]
    if missing_columns:
        return (
            (),
            (
                "period1_sensitivity.csv (no "
                f"{', '.join(missing_columns)} column)",
            ),
            (),
        )
    fits = summary["fit"].astype(str).str.strip()
    primary_rows = summary.loc[fits == "primary_period_stacked"]
    refit_rows = summary.loc[fits == "period1_only"]
    if len(primary_rows) != 1 or len(refit_rows) != 1:
        return (
            (),
            (
                "period1_sensitivity.csv (no unique primary_period_stacked / "
                "period1_only row pair)",
            ),
            (),
        )
    artefacts: list[str] = []
    if not (output_dir / "trace_period1_only.nc").is_file():
        artefacts.append("trace_period1_only.nc")

    computation: list[str] = []
    refit = refit_rows.iloc[0]
    if _stored_bool(refit.get("converged")) is not True:
        computation.append(
            "the mandatory gain-factors period-1-only refit sensitivity failed "
            "or was not convergence-checked"
        )

    robustness: list[str] = []
    primary = primary_rows.iloc[0]
    try:
        medians = (
            float(primary["beta_trt_median"]),
            float(refit["beta_trt_median"]),
        )
        intervals = (
            (float(primary["beta_trt_lo"]), float(primary["beta_trt_hi"])),
            (float(refit["beta_trt_lo"]), float(refit["beta_trt_hi"])),
        )
    except (TypeError, ValueError):
        artefacts.append("period1_sensitivity.csv (non-numeric headline columns)")
    else:
        if not all(np.isfinite(v) for v in (*medians, *intervals[0], *intervals[1])):
            artefacts.append("period1_sensitivity.csv (non-finite headline columns)")
        else:
            direction_stable = np.sign(medians[0]) == np.sign(medians[1])
            overlap = max(intervals[0][0], intervals[1][0]) <= min(
                intervals[0][1], intervals[1][1]
            )
            if not (direction_stable and overlap):
                robustness.append(
                    "the period-1-only refit materially disagrees with the "
                    "stacked primary (beta_trt direction or 89% interval overlap)"
                )
    return tuple(computation), tuple(artefacts), tuple(robustness)


def _joint_mechanism_coverage_qualifications(
    output_dir: Path, diagnostics: pd.DataFrame
) -> list[str]:
    """Apply the predeclared new-child coverage floors to every published wave."""
    notes: list[str] = []
    for _, row in diagnostics.iterrows():
        wave = str(row["wave"]).strip()
        filename = str(row.get("marginal_ppc_file") or "").strip()
        if not filename:
            continue
        coverage = _read_csv(output_dir, filename)
        if coverage is None or not {"level_pct", "coverage"} <= set(coverage.columns):
            continue
        for _, entry in coverage.iterrows():
            level = pd.to_numeric(entry.get("level_pct"), errors="coerce")
            value = pd.to_numeric(entry.get("coverage"), errors="coerce")
            floor = JOINT_MECHANISM_MARGINAL_COVERAGE_FLOORS.get(
                int(level) if pd.notna(level) else -1
            )
            if floor is None or pd.isna(value) or float(value) >= floor:
                continue
            stored = entry.get("outcome")
            # The pooled row leaves ``outcome`` null; a per-outcome row names it.
            outcome = "all" if pd.isna(stored) else str(stored).strip() or "all"
            notes.append(
                f"new-child predictive coverage at wave {wave} ({outcome}) is "
                f"{float(value):.2f} at the {int(level)}% level, below the "
                f"predeclared floor of {floor:.2f}"
            )
    return notes


_MISSINGNESS_DIAGNOSTIC_FIELDS = (
    "max_rhat",
    "min_ess",
    "min_bfmi",
    "n_divergences",
)
_MISSINGNESS_RHAT_MAX = 1.01
_MISSINGNESS_ESS_MIN = 400.0
_MISSINGNESS_BFMI_MIN = 0.3
_PUBLICATION_CONFIGS = frozenset({"rep-lite", "reporting"})
_DIAGNOSTIC_CONFIGS = frozenset({"dev", "test"})


def _stored_bool(value: Any) -> bool | None:
    """Parse a persisted Boolean without treating arbitrary truthy text as evidence."""

    normalised = str(value).strip().casefold()
    if normalised in {"true", "1", "yes"}:
        return True
    if normalised in {"false", "0", "no"}:
        return False
    return None


def _missingness_diagnostics(record: Mapping[str, Any]) -> dict[str, float | int] | None:
    """Read the four unrounded missingness-subfit gate signals from one record."""

    values: dict[str, float | int] = {}
    for name in _MISSINGNESS_DIAGNOSTIC_FIELDS:
        value = pd.to_numeric(record.get(name), errors="coerce")
        if pd.isna(value) or not np.isfinite(float(value)):
            return None
        if name == "n_divergences":
            integer = int(value)
            if float(value) != float(integer) or integer < 0:
                return None
            values[name] = integer
        else:
            values[name] = float(value)
    return values


def _missingness_diagnostics_pass(values: Mapping[str, float | int]) -> bool:
    """Apply the same unrounded release thresholds as the primary trace gate."""

    return bool(
        float(values["max_rhat"]) <= _MISSINGNESS_RHAT_MAX
        and float(values["min_ess"]) >= _MISSINGNESS_ESS_MIN
        and float(values["min_bfmi"]) >= _MISSINGNESS_BFMI_MIN
        and int(values["n_divergences"]) == 0
    )


def _growth_influence_release_failures(
    output_dir: Path, config: Mapping[str, Any]
) -> tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
    """Fail closed when a required growth influence refit is absent or unstable."""
    plan = config.get("resolved_run_plan") or {}
    if config.get("kind") != "growth":
        return (), (), ()
    if not isinstance(plan, Mapping):
        return (
            (),
            ("config.json (growth influence configuration is unreadable)",),
            (),
        )
    if not plan.get("observation_influence_sensitivity", False):
        return (), (), ()

    computation_failures: list[str] = []
    artifact_failures: list[str] = []
    robustness_failures: list[str] = []
    pareto = _read_csv(output_dir, "pareto_k.csv")
    pareto_columns = {
        "observation_index",
        "subject_id",
        "wave",
        "outcome",
        "pareto_k",
        "good_k_threshold",
        "loo_reliable",
    }
    if pareto is None or pareto.empty or not pareto_columns.issubset(pareto.columns):
        return (), ("pareto_k.csv (invalid growth observation-cell map)",), ()

    reliable = pareto["loo_reliable"].map(_stored_bool)
    numeric = pareto[
        ["observation_index", "wave", "pareto_k", "good_k_threshold"]
    ].apply(pd.to_numeric, errors="coerce")
    indices = numeric["observation_index"]
    expected_reliable = numeric["pareto_k"] <= numeric["good_k_threshold"]
    numeric_valid = bool(np.isfinite(numeric.to_numpy(dtype=float)).all())
    if numeric_valid:
        integer_indices = indices.to_numpy(dtype=int)
        numeric_valid = bool(
            np.array_equal(indices.to_numpy(dtype=float), integer_indices)
            and set(integer_indices) == set(range(len(indices)))
            and np.array_equal(
                numeric["wave"].to_numpy(dtype=float),
                numeric["wave"].to_numpy(dtype=int),
            )
        )
    if (
        reliable.isna().any()
        or not numeric_valid
        or indices.duplicated().any()
        or pareto[["subject_id", "outcome"]].isna().any().any()
        or not np.array_equal(
            reliable.to_numpy(dtype=bool), expected_reliable.to_numpy()
        )
    ):
        return (), ("pareto_k.csv (internally inconsistent growth diagnostics)",), ()

    flagged = pareto.loc[~reliable.to_numpy(dtype=bool)]
    if flagged.empty:
        return (), (), ()

    summary = _read_csv(output_dir, "growth_influence_sensitivity.csv")
    summary_columns = {
        "coefficient",
        "outcome",
        "n_excluded_cells",
        "n_excluded_children",
        "n_fully_excluded_children",
        "sensitivity_converged",
        "primary_median",
        "primary_lo89",
        "primary_hi89",
        "sensitivity_median",
        "sensitivity_lo89",
        "sensitivity_hi89",
        "median_direction_stable",
        "intervals_overlap",
    }
    if summary is None or summary.empty or not summary_columns.issubset(summary.columns):
        artifact_failures.append("growth_influence_sensitivity.csv")
        summary_verdict: bool | None = None
    else:
        outcomes = set(pareto["outcome"].astype(str))
        expected_rows = {
            (coefficient, outcome)
            for coefficient in ("gamma", "delta")
            for outcome in outcomes
        }
        actual_rows = set(
            summary[["coefficient", "outcome"]]
            .astype(str)
            .itertuples(index=False, name=None)
        )
        if (
            summary.duplicated(subset=["coefficient", "outcome"]).any()
            or actual_rows != expected_rows
        ):
            artifact_failures.append(
                "growth_influence_sensitivity.csv (invalid coefficient rows)"
            )
        counts = pd.to_numeric(summary["n_excluded_cells"], errors="coerce")
        children = pd.to_numeric(summary["n_excluded_children"], errors="coerce")
        fully_excluded = pd.to_numeric(
            summary["n_fully_excluded_children"], errors="coerce"
        )
        expected_children = flagged["subject_id"].astype(str).nunique()
        # A child is *fully* excluded only when every one of its observed cells is
        # unreliable — matching the writer, which keeps a child whose retained-cell
        # count is non-zero and counts the rest under ``all_observed_cells_high_pareto``
        # (``growth._exclude_cells``). Grouping ``~reliable`` with ``.all()`` asks
        # exactly that. The previous form, ``~reliable.groupby(...).all()``, negated
        # *after* the reduction, so it flagged children with **any** unreliable cell —
        # numerically identical to ``expected_children`` above, making the check both
        # redundant and unsatisfiable for any fit with a partially-excluded child.
        none_reliable_by_child = (~reliable).groupby(
            pareto["subject_id"].astype(str)
        ).all()
        expected_fully_excluded = int(none_reliable_by_child.sum())
        if (
            counts.isna().any()
            or not (counts == len(flagged)).all()
            or children.isna().any()
            or not (children == expected_children).all()
            or fully_excluded.isna().any()
            or not (fully_excluded == expected_fully_excluded).all()
        ):
            artifact_failures.append(
                "growth_influence_sensitivity.csv (excluded-cell map mismatch)"
            )

        stability_numeric = summary[
            [
                "primary_median",
                "primary_lo89",
                "primary_hi89",
                "sensitivity_median",
                "sensitivity_lo89",
                "sensitivity_hi89",
            ]
        ].apply(pd.to_numeric, errors="coerce")
        direction_stable = summary["median_direction_stable"].map(_stored_bool)
        intervals_overlap = summary["intervals_overlap"].map(_stored_bool)
        stability_values_valid = bool(
            np.isfinite(stability_numeric.to_numpy(dtype=float)).all()
            and not direction_stable.isna().any()
            and not intervals_overlap.isna().any()
            and (
                stability_numeric["primary_lo89"]
                <= stability_numeric["primary_hi89"]
            ).all()
            and (
                stability_numeric["primary_lo89"]
                <= stability_numeric["primary_median"]
            ).all()
            and (
                stability_numeric["primary_median"]
                <= stability_numeric["primary_hi89"]
            ).all()
            and (
                stability_numeric["sensitivity_lo89"]
                <= stability_numeric["sensitivity_hi89"]
            ).all()
            and (
                stability_numeric["sensitivity_lo89"]
                <= stability_numeric["sensitivity_median"]
            ).all()
            and (
                stability_numeric["sensitivity_median"]
                <= stability_numeric["sensitivity_hi89"]
            ).all()
        )
        if not stability_values_valid:
            artifact_failures.append(
                "growth_influence_sensitivity.csv (invalid coefficient stability values)"
            )
        else:
            expected_direction_stable = (
                np.sign(stability_numeric["primary_median"])
                == np.sign(stability_numeric["sensitivity_median"])
            )
            expected_intervals_overlap = (
                np.maximum(
                    stability_numeric["primary_lo89"],
                    stability_numeric["sensitivity_lo89"],
                )
                <= np.minimum(
                    stability_numeric["primary_hi89"],
                    stability_numeric["sensitivity_hi89"],
                )
            )
            if not (
                np.array_equal(
                    direction_stable.to_numpy(dtype=bool),
                    expected_direction_stable.to_numpy(dtype=bool),
                )
                and np.array_equal(
                    intervals_overlap.to_numpy(dtype=bool),
                    expected_intervals_overlap.to_numpy(dtype=bool),
                )
            ):
                artifact_failures.append(
                    "growth_influence_sensitivity.csv "
                    "(coefficient stability verdict mismatch)"
                )
            elif not (
                direction_stable.to_numpy(dtype=bool).all()
                and intervals_overlap.to_numpy(dtype=bool).all()
            ):
                robustness_failures.append(
                    "growth observation-cell influence sensitivity did not preserve "
                    "every coefficient's median direction with overlapping 89% intervals"
                )
        declared = summary["sensitivity_converged"].map(_stored_bool)
        if declared.isna().any():
            artifact_failures.append(
                "growth_influence_sensitivity.csv (invalid convergence verdict)"
            )
            summary_verdict = None
        elif declared.nunique() != 1:
            artifact_failures.append(
                "growth_influence_sensitivity.csv (inconsistent convergence verdict)"
            )
            summary_verdict = None
        else:
            summary_verdict = bool(declared.iloc[0])
            if not summary_verdict:
                computation_failures.append(
                    "growth observation-cell influence sensitivity failed its "
                    "convergence gate"
                )

    provenance = _read_csv(output_dir, "subfit_provenance.csv")
    model_id = str(config.get("model_id") or "")
    label = f"{model_id} high-Pareto observation-cell exclusion"
    provenance_row: pd.Series | None = None
    provenance_verdict: bool | None = None
    if provenance is None or provenance.empty or "label" not in provenance.columns:
        artifact_failures.append("subfit_provenance.csv")
    else:
        rows = provenance.loc[provenance["label"].astype(str) == label]
        if len(rows) != 1:
            artifact_failures.append(
                "subfit_provenance.csv (no unique growth influence row)"
            )
        else:
            provenance_row = rows.iloc[0]
            if str(provenance_row.get("role", "")).strip() != "sensitivity":
                artifact_failures.append(
                    "subfit_provenance.csv (invalid growth influence role)"
                )
            if (
                str(provenance_row.get("trace_file", "")).strip()
                != GROWTH_INFLUENCE_TRACE_FILENAME
            ):
                artifact_failures.append(
                    "subfit_provenance.csv (invalid growth influence trace binding)"
                )
            values = _missingness_diagnostics(provenance_row)
            declared = _stored_bool(provenance_row.get("converged"))
            if values is None or declared is None:
                artifact_failures.append(
                    "subfit_provenance.csv (invalid growth influence diagnostics)"
                )
            else:
                provenance_verdict = declared
                passed = _missingness_diagnostics_pass(values)
                if declared != passed:
                    artifact_failures.append(
                        "subfit_provenance.csv (growth influence verdict mismatch)"
                    )
                if not passed:
                    computation_failures.append(
                        "growth observation-cell influence sensitivity failed its "
                        "convergence gate"
                    )

    trace_path = output_dir / GROWTH_INFLUENCE_TRACE_FILENAME
    if not trace_path.is_file():
        artifact_failures.append(GROWTH_INFLUENCE_TRACE_FILENAME)
    elif provenance_row is not None:
        from language_reading_predictors.statistical_models.sensitivity import (
            sha256_file,
        )

        recorded = str(provenance_row.get("trace_sha256", "")).strip().lower()
        if len(recorded) != 64 or recorded != sha256_file(trace_path):
            artifact_failures.append(
                "subfit_provenance.csv (growth influence trace hash mismatch)"
            )

    # The growth pipeline records this verdict inside ``config["extra"]``
    # (``pipelines.growth`` builds it as part of the spec's extra payload), so read
    # there as well as at the top level. Looking only at the top level made the
    # verdict unconditionally "missing" for every growth fit that ran the influence
    # sensitivity, withholding a fit whose sensitivity had in fact converged.
    influence_extra = config.get("extra")
    if not isinstance(influence_extra, Mapping):
        influence_extra = {}
    metadata_verdict = _stored_bool(
        config.get(
            "observation_influence_converged",
            influence_extra.get("observation_influence_converged"),
        )
    )
    if metadata_verdict is None:
        artifact_failures.append("config.json (growth influence verdict is missing)")
    elif not metadata_verdict:
        computation_failures.append(
            "growth observation-cell influence sensitivity failed its convergence gate"
        )
    stored_verdicts = {
        verdict
        for verdict in (summary_verdict, provenance_verdict, metadata_verdict)
        if verdict is not None
    }
    if len(stored_verdicts) > 1:
        artifact_failures.append(
            "growth influence convergence verdicts disagree across artifacts"
        )
    return (
        tuple(dict.fromkeys(computation_failures)),
        tuple(dict.fromkeys(artifact_failures)),
        tuple(dict.fromkeys(robustness_failures)),
    )


def _missingness_diagnostics_match(
    left: Mapping[str, float | int], right: Mapping[str, float | int]
) -> bool:
    """Whether two serialisations carry the same unrounded gate evidence."""

    return bool(
        int(left["n_divergences"]) == int(right["n_divergences"])
        and all(
            np.isclose(
                float(left[name]),
                float(right[name]),
                rtol=1e-10,
                atol=1e-12,
            )
            for name in ("max_rhat", "min_ess", "min_bfmi")
        )
    )


def _trailing_size(group: Any, name: str) -> int | None:
    """Size of a variable's last (non chain/draw) dimension, or ``None``."""
    try:
        array = group[name]
    except Exception:  # pragma: no cover - defensive
        return None
    dims = [d for d in getattr(array, "dims", ()) if d not in ("chain", "draw")]
    if not dims:
        return None
    try:
        return int(array.sizes[dims[-1]])
    except Exception:  # pragma: no cover - defensive
        return None


def _missingness_design_dimension_error(
    trace: Any,
    *,
    expected_targets: int | None,
    expected_observations: int | None,
) -> str | None:
    """Check the persisted trace actually carries the registered design."""
    if expected_targets is not None:
        for name in ("p0_target", "p1_target"):
            size = _trailing_size(trace["prior"], name)
            if size is None:
                return f"the /prior group's {name} has no target dimension"
            if size != expected_targets:
                return (
                    f"the /prior group's {name} covers {size} target profiles, "
                    f"not the registered {expected_targets}"
                )
    if expected_observations is not None:
        size = _trailing_size(trace["prior_predictive"], "y_post")
        if size is None:
            return "the /prior_predictive group's y_post has no observation dimension"
        if size != expected_observations:
            return (
                f"the /prior_predictive group's y_post covers {size} observations, "
                f"not the registered {expected_observations}"
            )
    return None


def _missingness_trace_diagnostics(
    trace_path: Path,
    *,
    expected_targets: int | None = None,
    expected_observations: int | None = None,
) -> tuple[dict[str, float | int] | None, str | None]:
    """Recompute the mandatory subfit gate from its persisted NetCDF trace.

    ``expected_targets`` / ``expected_observations`` are the registered design
    dimensions — 57 randomised target profiles and 53 observed word-reading rows.
    Checking them closes the gap the 2026-08-22 ITT audit found (finding 8):
    fresh generation verifies the target count, the likelihood rows and the
    arm / missingness masks, but stored evaluation verified only that the trace
    carried groups and variables *named* ``p0_target`` / ``p1_target`` /
    ``y_post`` — so a trace holding a single target and a single observation
    qualified. Names are not a design.
    """

    if not trace_path.is_file():
        return None, "missing"
    trace = None
    try:
        import arviz as az

        from language_reading_predictors.statistical_models.diagnostics import (
            subfit_convergence,
        )

        trace = az.from_netcdf(trace_path)
        groups = {
            str(group).strip("/")
            for group in getattr(trace, "groups", ())
            if str(group).strip("/")
        }
        required_groups = {"prior", "prior_predictive"}
        if not required_groups.issubset(groups):
            missing = ", ".join(
                f"/{group}" for group in sorted(required_groups - groups)
            )
            return None, f"missing required trace group(s): {missing}"
        prior_vars = set(getattr(trace["prior"], "data_vars", {}))
        prior_predictive_vars = set(
            getattr(trace["prior_predictive"], "data_vars", {})
        )
        if not {"p0_target", "p1_target"}.issubset(prior_vars):
            return None, "the /prior group lacks the registered target probabilities"
        if "y_post" not in prior_predictive_vars:
            return None, "the /prior_predictive group lacks the registered outcome"
        dimension_error = _missingness_design_dimension_error(
            trace,
            expected_targets=expected_targets,
            expected_observations=expected_observations,
        )
        if dimension_error is not None:
            return None, dimension_error
        verdict = subfit_convergence(
            trace,
            label="ITT screening-baseline missingness release check",
            var_names=[
                "alpha",
                "tau",
                "beta_screening_age",
                "beta_screening_word",
                "kappa",
            ],
        )
    except Exception as exc:  # noqa: BLE001 - unreadable trace is an artefact failure
        return None, f"{type(exc).__name__}: {exc}"
    finally:
        with suppress(Exception):
            if trace is not None:
                trace.close()
    diagnostics = _missingness_diagnostics(verdict)
    if diagnostics is None:
        return None, "the trace sampling-quality signals could not be computed"
    return diagnostics, None


def _sampling_preset_qualification(
    output_dir: Path, config: Mapping[str, Any]
) -> tuple[str | None, bool, str]:
    """Resolve whether a clean fit is publication-grade or development-only.

    ``ReleaseEvaluation.publishable`` predates this distinction and means that a local
    report may render its scientific tables.  Keeping that meaning preserves the
    established ``--config dev --render`` diagnostic workflow.  The separate
    ``scientific_publication_eligible`` property fails closed for diagnostic, missing,
    unknown or directory-inconsistent presets.

    Stored fits created before ``config_name`` was added remain decidable from the
    long-standing ``<model-id>-<preset>`` directory convention. New staging
    directories do not have that suffix, but their freshly written config always does.
    """

    explicit = config.get("config_name")
    config_name = str(explicit).strip() if explicit is not None else ""
    inferred = _config_name(output_dir, str(config.get("model_id") or ""))
    if not config_name:
        config_name = inferred
    known = _PUBLICATION_CONFIGS | _DIAGNOSTIC_CONFIGS
    mismatch = bool(config_name and inferred in known and inferred != config_name)
    if config_name in _PUBLICATION_CONFIGS and not mismatch:
        return config_name, False, ""
    if mismatch:
        reason = (
            f"the saved sampling preset {config_name!r} disagrees with the fit "
            f"directory preset {inferred!r}"
        )
    elif config_name in _DIAGNOSTIC_CONFIGS:
        reason = (
            f"the saved sampling preset {config_name!r} is diagnostic-only; only "
            "'rep-lite' and 'reporting' fits are eligible for scientific publication"
        )
    else:
        reason = (
            "the sampling preset is absent or unrecognised, so publication-grade "
            "sampling cannot be verified"
        )
    return config_name or None, True, reason


def _itt_missingness_release_failures(
    output_dir: Path, config: Mapping[str, Any]
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Require the trace-bound all-57 W sensitivity declared by the ITT plan."""

    registered_primary = (
        config.get("model_id") == "lrp-rli-itt-010"
        and config.get("kind") == "itt"
        and config.get("outcome_symbol") == "W"
    )
    if not registered_primary:
        return (), ()
    plan = config.get("resolved_run_plan") or {}
    if not isinstance(plan, Mapping):
        return (), ("config.json (ITT run plan is unreadable)",)
    if not bool(plan.get("missingness_sensitivity_required_for_release")):
        return (), ("config.json (word-reading missingness sensitivity is undeclared)",)

    from language_reading_predictors.statistical_models.itt_missingness import (
        MISSINGNESS_PROVENANCE_FILENAME,
        MISSINGNESS_PPC_FILENAME,
        MISSINGNESS_PRIOR_FILENAME,
        MISSINGNESS_PRIOR_DRAWS,
        MISSINGNESS_SCENARIOS,
        MISSINGNESS_SUBFIT_LABEL,
        MISSINGNESS_SUMMARY_FILENAME,
        MISSINGNESS_TRACE_FILENAME,
        OBSERVED_CONTROL_N,
        OBSERVED_INTERVENTION_N,
        LOST_TO_FOLLOW_UP_N,
        WITHIN_ARCHIVE_W_MISSING_N,
        RLI_ARCHIVE_DOI,
        RLI_ARCHIVE_CSV_SHA256,
        RLI_LOCAL_WIDE_SHA256,
        RLI_RECONCILIATION_DIGEST,
        RANDOMISED_CONTROL_N,
        RANDOMISED_INTERVENTION_N,
        RANDOMISED_N,
        WORD_READING_N,
        DEFAULT_DELTA_ITEMS,
        SCREENING_ALPHA_SIGMA,
        SCREENING_COVARIATES,
        sha256_file,
        validate_missingness_prior_check,
        validate_missingness_summary,
    )

    computation_failures: list[str] = []
    artifact_failures: list[str] = []
    stored_diagnostics: list[
        tuple[str, dict[str, float | int], bool | None]
    ] = []
    saved_missingness_plan = plan.get("missingness_plan") or {}
    expected_plan = {
        "source_csv_sha256": RLI_ARCHIVE_CSV_SHA256,
        "source_doi": RLI_ARCHIVE_DOI,
        "local_wide_sha256": RLI_LOCAL_WIDE_SHA256,
        "reconciliation_digest": RLI_RECONCILIATION_DIGEST,
        "screening_covariates": list(SCREENING_COVARIATES),
        "randomised_n": RANDOMISED_N,
        "randomised_intervention_n": RANDOMISED_INTERVENTION_N,
        "randomised_control_n": RANDOMISED_CONTROL_N,
        "observed_intervention_n": OBSERVED_INTERVENTION_N,
        "observed_control_n": OBSERVED_CONTROL_N,
        "lost_to_follow_up_n": LOST_TO_FOLLOW_UP_N,
        "within_archive_w_missing_n": WITHIN_ARCHIVE_W_MISSING_N,
        "word_reading_n": WORD_READING_N,
        "delta_items": list(DEFAULT_DELTA_ITEMS),
        "scenarios": list(MISSINGNESS_SCENARIOS),
        "common_estimand_class": "common_profile_standardisation",
        "completion_estimand_class": "randomised_arm_factual_completion",
        "intercept_prior_anchor": "mean_all_57_screening_word_reading_logit",
        "intercept_prior_sigma": SCREENING_ALPHA_SIGMA,
        "prior_predictive_draws": MISSINGNESS_PRIOR_DRAWS,
        "trace_filename": MISSINGNESS_TRACE_FILENAME,
        "summary_filename": MISSINGNESS_SUMMARY_FILENAME,
        "ppc_filename": MISSINGNESS_PPC_FILENAME,
        "prior_check_filename": MISSINGNESS_PRIOR_FILENAME,
        "provenance_filename": MISSINGNESS_PROVENANCE_FILENAME,
    }
    if not isinstance(saved_missingness_plan, Mapping) or any(
        saved_missingness_plan.get(key) != value for key, value in expected_plan.items()
    ):
        artifact_failures.append("config.json (invalid word-reading missingness plan)")
    trace_path = output_dir / MISSINGNESS_TRACE_FILENAME
    summary = _read_csv(output_dir, MISSINGNESS_SUMMARY_FILENAME)
    if summary is None or summary.empty:
        artifact_failures.append(MISSINGNESS_SUMMARY_FILENAME)
    else:
        for error in validate_missingness_summary(
            summary,
            trace_path=trace_path,
            require_converged=False,
        ):
            artifact_failures.append(f"{MISSINGNESS_SUMMARY_FILENAME} ({error})")
        required_diagnostic_columns = {
            *_MISSINGNESS_DIAGNOSTIC_FIELDS,
            "converged",
        }
        if not required_diagnostic_columns.issubset(summary.columns):
            artifact_failures.append(
                f"{MISSINGNESS_SUMMARY_FILENAME} (missing raw subfit diagnostics)"
            )
        else:
            summary_values = _missingness_diagnostics(summary.iloc[0])
            rows_agree = summary_values is not None and all(
                (values := _missingness_diagnostics(row)) is not None
                and _missingness_diagnostics_match(values, summary_values)
                for _, row in summary.iterrows()
            )
            declared = {_stored_bool(value) for value in summary["converged"]}
            if not rows_agree or len(declared) != 1 or None in declared:
                artifact_failures.append(
                    f"{MISSINGNESS_SUMMARY_FILENAME} "
                    "(inconsistent or invalid raw subfit diagnostics)"
                )
            else:
                stored_diagnostics.append(
                    (
                        MISSINGNESS_SUMMARY_FILENAME,
                        summary_values,
                        next(iter(declared)),
                    )
                )

    provenance_payload, provenance_error = _read_json(
        output_dir / MISSINGNESS_PROVENANCE_FILENAME
    )
    if provenance_error is not None or not isinstance(provenance_payload, Mapping):
        artifact_failures.append(MISSINGNESS_PROVENANCE_FILENAME)
    else:
        source = provenance_payload.get("source") or {}
        analysis = provenance_payload.get("analysis") or {}
        trace = provenance_payload.get("trace") or {}
        outputs = provenance_payload.get("outputs") or {}
        if (
            not isinstance(source, Mapping)
            or source.get("csv_sha256") != RLI_ARCHIVE_CSV_SHA256
            or source.get("local_wide_sha256") != RLI_LOCAL_WIDE_SHA256
            or source.get("reconciled_included_n") != 54
            or source.get("reconciliation_digest") != RLI_RECONCILIATION_DIGEST
        ):
            artifact_failures.append(
                f"{MISSINGNESS_PROVENANCE_FILENAME} (invalid source binding)"
            )
        if (
            not isinstance(analysis, Mapping)
            or analysis.get("observed_outcome_n") != 53
            or analysis.get("target_profile_n") != RANDOMISED_N
            or analysis.get("randomised_by_arm")
            != {"intervention": RANDOMISED_INTERVENTION_N, "control": RANDOMISED_CONTROL_N}
            or analysis.get("observed_outcome_by_arm")
            != {"intervention": OBSERVED_INTERVENTION_N, "control": OBSERVED_CONTROL_N}
            or analysis.get("lost_to_follow_up_n") != LOST_TO_FOLLOW_UP_N
            or analysis.get("within_archive_word_reading_missing_n")
            != WITHIN_ARCHIVE_W_MISSING_N
            or analysis.get("screening_covariates") != list(SCREENING_COVARIATES)
            or analysis.get("delta_items_grid") != list(DEFAULT_DELTA_ITEMS)
        ):
            artifact_failures.append(
                f"{MISSINGNESS_PROVENANCE_FILENAME} (invalid analysis contract)"
            )
        # The recorded design (2026-08-22 ITT audit, finding 8). Absent on fits
        # written before the block existed, which therefore re-decide exactly as
        # before; present, it must agree with the registered trial contract
        # rather than merely be well-formed. Counts alone cannot establish that
        # two runs completed the same profiles, so the digest must be there too.
        recorded_design = analysis.get("design") if isinstance(analysis, Mapping) else None
        if isinstance(recorded_design, Mapping):
            expected_design = {
                "target_profile_n": RANDOMISED_N,
                "observed_outcome_n": OBSERVED_INTERVENTION_N + OBSERVED_CONTROL_N,
                "target_by_arm": {
                    "intervention": RANDOMISED_INTERVENTION_N,
                    "control": RANDOMISED_CONTROL_N,
                },
                "target_observed_by_arm": {
                    "intervention": OBSERVED_INTERVENTION_N,
                    "control": OBSERVED_CONTROL_N,
                },
                "covariate_names": list(SCREENING_COVARIATES),
            }
            disagreeing = sorted(
                key
                for key, value in expected_design.items()
                if recorded_design.get(key) != value
            )
            if disagreeing:
                artifact_failures.append(
                    f"{MISSINGNESS_PROVENANCE_FILENAME} (recorded design disagrees "
                    f"with the registered trial contract: {', '.join(disagreeing)})"
                )
            if not str(recorded_design.get("target_design_sha256") or ""):
                artifact_failures.append(
                    f"{MISSINGNESS_PROVENANCE_FILENAME} "
                    "(recorded design carries no digest)"
                )
        actual_trace_sha256 = sha256_file(trace_path) if trace_path.is_file() else None
        if (
            not isinstance(trace, Mapping)
            or trace.get("file") != MISSINGNESS_TRACE_FILENAME
            or trace.get("sha256") != actual_trace_sha256
        ):
            artifact_failures.append(
                f"{MISSINGNESS_PROVENANCE_FILENAME} (invalid trace binding)"
            )
        if isinstance(trace, Mapping):
            trace_values = _missingness_diagnostics(trace)
            trace_declared = _stored_bool(trace.get("converged"))
            if trace_values is None or trace_declared is None:
                artifact_failures.append(
                    f"{MISSINGNESS_PROVENANCE_FILENAME} "
                    "(invalid raw subfit diagnostics)"
                )
            else:
                stored_diagnostics.append(
                    (
                        MISSINGNESS_PROVENANCE_FILENAME,
                        trace_values,
                        trace_declared,
                    )
                )
        summary_path = output_dir / MISSINGNESS_SUMMARY_FILENAME
        ppc_path = output_dir / MISSINGNESS_PPC_FILENAME
        prior_path = output_dir / MISSINGNESS_PRIOR_FILENAME
        if (
            not isinstance(outputs, Mapping)
            or outputs.get("summary_file") != MISSINGNESS_SUMMARY_FILENAME
            or outputs.get("summary_sha256")
            != (sha256_file(summary_path) if summary_path.is_file() else None)
            or outputs.get("ppc_file") != MISSINGNESS_PPC_FILENAME
            or outputs.get("ppc_sha256")
            != (sha256_file(ppc_path) if ppc_path.is_file() else None)
            or outputs.get("prior_check_file") != MISSINGNESS_PRIOR_FILENAME
            or outputs.get("prior_check_sha256")
            != (sha256_file(prior_path) if prior_path.is_file() else None)
        ):
            artifact_failures.append(
                f"{MISSINGNESS_PROVENANCE_FILENAME} (invalid output binding)"
            )

    prior_check = _read_csv(output_dir, MISSINGNESS_PRIOR_FILENAME)
    if prior_check is None or prior_check.empty:
        artifact_failures.append(MISSINGNESS_PRIOR_FILENAME)
    else:
        for error in validate_missingness_prior_check(prior_check):
            artifact_failures.append(f"{MISSINGNESS_PRIOR_FILENAME} ({error})")

    bounds = _read_csv(output_dir, "attrition_bounds.csv")
    required_bounds = {
        "outcome",
        "observed_intervention_n",
        "observed_control_n",
        "missing_intervention_n",
        "missing_control_n",
        "n_trials",
    }
    if bounds is None or len(bounds) != 1 or not required_bounds.issubset(bounds.columns):
        artifact_failures.append("attrition_bounds.csv")
    else:
        row = bounds.iloc[0]
        numeric_contract = {
            "observed_intervention_n": OBSERVED_INTERVENTION_N,
            "observed_control_n": OBSERVED_CONTROL_N,
            "missing_intervention_n": 1,
            "missing_control_n": 3,
            "n_trials": WORD_READING_N,
        }
        if str(row.get("outcome")) != "W" or any(
            not np.isclose(
                float(pd.to_numeric(row.get(key), errors="coerce")),
                float(value),
            )
            for key, value in numeric_contract.items()
        ):
            artifact_failures.append("attrition_bounds.csv (invalid W count contract)")

    ppc = _read_csv(output_dir, MISSINGNESS_PPC_FILENAME)
    required_ppc = {
        "arm",
        "n",
        "observed_mean_items",
        "posterior_predictive_mean_items",
        "mean_absolute_prediction_error_items",
        "coverage_50",
        "coverage_89",
    }
    if ppc is None or len(ppc) != 3 or not required_ppc.issubset(ppc.columns):
        artifact_failures.append(MISSINGNESS_PPC_FILENAME)
    else:
        expected_n = {"all": 53, "intervention": 28, "control": 25}
        observed_n = dict(
            zip(
                ppc["arm"].astype(str),
                pd.to_numeric(ppc["n"], errors="coerce"),
                strict=True,
            )
        )
        numeric_ppc = ppc[list(required_ppc - {"arm"})].apply(
            pd.to_numeric, errors="coerce"
        )
        if observed_n != expected_n or not np.isfinite(
            numeric_ppc.to_numpy(dtype=float)
        ).all():
            artifact_failures.append(f"{MISSINGNESS_PPC_FILENAME} (invalid values)")
        elif not (
            numeric_ppc["coverage_50"].between(0.0, 1.0).all()
            and numeric_ppc["coverage_89"].between(0.0, 1.0).all()
        ):
            artifact_failures.append(f"{MISSINGNESS_PPC_FILENAME} (invalid coverage)")

    subfits = _read_csv(output_dir, "subfit_provenance.csv")
    if subfits is None or subfits.empty or "label" not in subfits.columns:
        artifact_failures.append("subfit_provenance.csv")
    else:
        rows = subfits.loc[subfits["label"].astype(str) == MISSINGNESS_SUBFIT_LABEL]
        if len(rows) != 1:
            artifact_failures.append(
                "subfit_provenance.csv (no unique ITT missingness row)"
            )
        else:
            row = rows.iloc[0]
            if str(row.get("role", "")).strip() != "sensitivity":
                artifact_failures.append(
                    "subfit_provenance.csv (invalid ITT missingness role)"
                )
            if str(row.get("trace_file", "")).strip() != MISSINGNESS_TRACE_FILENAME:
                artifact_failures.append(
                    "subfit_provenance.csv (invalid ITT missingness trace binding)"
                )
            subfit_values = _missingness_diagnostics(row)
            subfit_declared = _stored_bool(row.get("converged"))
            if subfit_values is None or subfit_declared is None:
                artifact_failures.append(
                    "subfit_provenance.csv (invalid raw ITT missingness diagnostics)"
                )
            else:
                stored_diagnostics.append(
                    ("subfit_provenance.csv", subfit_values, subfit_declared)
                )
            n_obs = pd.to_numeric(row.get("n_obs"), errors="coerce")
            n_children = pd.to_numeric(row.get("n_children"), errors="coerce")
            if not (
                pd.notna(n_obs)
                and pd.notna(n_children)
                and float(n_obs) == 53.0
                and float(n_children) == 53.0
                and bool(str(row.get("data_digest", "")).strip())
            ):
                artifact_failures.append(
                    "subfit_provenance.csv (invalid ITT missingness data identity)"
                )
    if not trace_path.is_file():
        artifact_failures.append(MISSINGNESS_TRACE_FILENAME)
    trace_diagnostics, trace_diagnostics_error = _missingness_trace_diagnostics(
        trace_path,
        expected_targets=RANDOMISED_N,
        expected_observations=OBSERVED_INTERVENTION_N + OBSERVED_CONTROL_N,
    )
    if trace_diagnostics_error is not None or trace_diagnostics is None:
        if trace_path.is_file():
            artifact_failures.append(
                f"{MISSINGNESS_TRACE_FILENAME} ({trace_diagnostics_error})"
            )
    else:
        trace_passed = _missingness_diagnostics_pass(trace_diagnostics)
        if not trace_passed:
            computation_failures.append(
                "ITT screening-baseline missingness sub-fit failed the raw "
                "sampling-quality thresholds"
            )
        for label, values, declared in stored_diagnostics:
            if not _missingness_diagnostics_match(values, trace_diagnostics):
                artifact_failures.append(
                    f"{label} (raw subfit diagnostics do not match the trace)"
                )
            elif declared != _missingness_diagnostics_pass(values):
                artifact_failures.append(
                    f"{label} (stored convergence verdict contradicts raw diagnostics)"
                )
    return tuple(computation_failures), tuple(artifact_failures)


def _publication_input_failures(config: Mapping[str, Any]) -> tuple[str, ...]:
    """Validate the fit-time scientific-input snapshot, failing closed.

    RLI predates the multi-study input contract and has no unresolved catalogue
    entry.  Every non-RLI fit must carry a stored contract; consulting the current
    catalogue here would let an old fit silently inherit a later sign-off without
    refitting against the now-authoritative inputs.
    """

    study_id = str(config.get("study_id") or "rli")
    if study_id == "rli":
        return ()

    contract = config.get("publication_input_contract")
    if not isinstance(contract, Mapping):
        return (
            f"{study_id}: the fit has no valid publication input contract; "
            "regenerate or refit it under the current fail-closed metadata policy",
        )
    if contract.get("study_id") != study_id:
        return (
            f"{study_id}: the publication input contract names a different study",
        )

    raw_blockers = contract.get("blockers")
    if not isinstance(raw_blockers, list) or any(
        not isinstance(item, str) or not item.strip() for item in raw_blockers
    ):
        return (f"{study_id}: the publication input contract has invalid blockers",)
    blockers = tuple(item.strip() for item in raw_blockers)
    ready = contract.get("publication_ready")
    if ready is True and not blockers:
        return ()
    if ready is False and blockers:
        return blockers
    return (
        f"{study_id}: the publication input contract is internally inconsistent",
    )


def _blending_pair_release_failures(
    output_dir: Path, config: Mapping[str, Any]
) -> tuple[str, ...]:
    """Robustness-stage failures for the mandatory phoneme-blending link pair.

    Three families now carry a version of the policy, dispatched from here: the ITT
    archive-grade pair, the level pair (#584 decision 2) and the DiD pair (#576
    finding 2). They differ in evidence *strength*, never in bindingness.

    The registered policy is that neither ``lrp-rli-itt-008`` nor
    ``lrp-rli-itt-108`` may release without the validated trace-backed paired
    bundle, but until 2026-08-20 that was enforced only in the key-findings
    builder and the copied report partial — ``release_decision.json``, the
    artefact whose stated purpose is to combine exactly these policies, said
    ``publishable: true`` for an unpaired B fit (ITT code review, finding 1,
    ``notes/202608201205-itt-code-review-findings.md``). The requirement is
    derived from the module constant (so a stale stored plan cannot bypass it,
    mirroring the itt-010 missingness gate) *and* from the stored plan's
    ``link_sensitivity_required_for_release`` (so a future B-outcome ITT fit
    outside the registered pair fails closed rather than releasing unpaired).
    """
    kind = str(config.get("kind") or "")
    family_gate = _BLENDING_PAIR_GATES.get(kind)
    if family_gate is not None:
        return family_gate(output_dir, config)
    if kind != "itt":
        # Symbol-keyed fail-closed (#608 decision 1, implemented in #619). Every
        # family that registers a ``B`` model has a gate above. A ``B`` fit in a
        # family that does not is a model whose response-link sensitivity nothing
        # can verify -- so it must not publish, rather than slipping through because
        # its ``kind`` was not remembered. This is the direction the policy always
        # stated and the code did not do: before #619 the dispatch returned early
        # for every unlisted kind, so four families published unpaired ``B`` results
        # for months without anything failing.
        if str(config.get("outcome_symbol") or "") == "B":
            return (
                f"{config.get('model_id')} reports a phoneme-blending (B) outcome, "
                f"but the {kind!r} family has no registered response-link pair gate. "
                "Blending is a three-alternative forced-choice test whose expected "
                "score cannot fall below chance, and the #608 policy requires every "
                "B model to be released beside its guessing-floor twin; add the "
                "family's pairing before releasing this fit",
            )
        return ()
    from language_reading_predictors.statistical_models.blending_sensitivity import (
        BLENDING_LINK_MODELS,
        evaluate_local_blending_link_sensitivity,
    )

    model_id = str(config.get("model_id") or "")
    plan = config.get("resolved_run_plan") or {}
    registered = model_id in dict(BLENDING_LINK_MODELS)
    declared = bool(plan.get("link_sensitivity_required_for_release"))
    if not registered and not declared:
        return ()
    if not registered:
        return (
            f"{model_id} declares a mandatory response-link sensitivity pairing, "
            "but no registered blending-link bundle covers it; register the pair "
            "before releasing",
        )
    try:
        status = evaluate_local_blending_link_sensitivity(output_dir, config=config)
    except Exception as exc:  # noqa: BLE001 - a gate that cannot run must fail closed
        return (f"the B link-sensitivity pair could not be evaluated: {exc}",)
    if status.get("required") and not status.get("ready"):
        reason = str(status.get("reason") or "the paired evidence is stale")
        return (
            "the mandatory trace-backed phoneme-blending link pair "
            "(lrp-rli-itt-008 + lrp-rli-itt-108) is not release-ready: " + reason,
        )
    return ()



def _did_blending_pair_release_failures(
    output_dir: Path, config: Mapping[str, Any]
) -> tuple[str, ...]:
    """The DiD family's phoneme-blending pairing (#576 finding 2).

    Same policy as the ITT and level pairs. It did not exist for ``did``, so
    ``lrp-rli-did-003`` — the ordinary-logit fit of a ten-item, three-alternative
    forced-choice test — could publish an unqualified ``B`` headline with no
    guessing-floor companion anywhere. The ITT companion does not cover it: the
    longitudinal random-intercept likelihood lets t1 and t3 data inform the t2
    posterior, so the two fits' response-link sensitivities are not interchangeable.
    """
    from language_reading_predictors.statistical_models.blending_sensitivity import (
        evaluate_did_blending_link_pair,
    )
    from language_reading_predictors.statistical_models.did import (
        DID_BLENDING_COMPANION_MODEL_ID,
        DID_BLENDING_PRIMARY_MODEL_ID,
    )

    try:
        status = evaluate_did_blending_link_pair(output_dir, config=config)
    except Exception as exc:  # noqa: BLE001 - a gate that cannot run must fail closed
        return (f"the DiD B link pair could not be evaluated: {exc}",)
    if status.get("required") and not status.get("ready"):
        reason = str(status.get("reason") or "the paired evidence is stale")
        return (
            "the mandatory phoneme-blending link pair "
            f"({DID_BLENDING_PRIMARY_MODEL_ID} + {DID_BLENDING_COMPANION_MODEL_ID}) "
            "is not release-ready: " + reason,
        )
    return ()


def _level_blending_pair_release_failures(
    output_dir: Path, config: Mapping[str, Any]
) -> tuple[str, ...]:
    """The level family's phoneme-blending pairing (#584 decision 2).

    Same policy as the ITT pair, one rung down in evidence strength: the level
    check reads both fits' stored artefacts rather than recomputing their estimands
    from trace, because the level family has no content-addressed archive yet. It
    is still binding — a level B fit whose twin is absent, stale, ungated or fitted
    on different rows does not publish — and it fails closed on anything it cannot
    verify, so the weaker apparatus cannot become a weaker *policy*.
    """
    from language_reading_predictors.statistical_models.blending_sensitivity import (
        evaluate_level_blending_link_pair,
    )

    try:
        status = evaluate_level_blending_link_pair(output_dir, config=config)
    except Exception as exc:  # noqa: BLE001 - a gate that cannot run must fail closed
        return (f"the level B link pair could not be evaluated: {exc}",)
    if status.get("required") and not status.get("ready"):
        reason = str(status.get("reason") or "the paired evidence is stale")
        return (
            "the mandatory phoneme-blending link pair "
            "(lrp-rli-lf-006 + lrp-rli-lf-106) is not release-ready: " + reason,
        )
    return ()


def _gain_blending_pair_release_failures(
    output_dir: Path, config: Mapping[str, Any]
) -> tuple[str, ...]:
    """The gain family's phoneme-blending pairing (#596).

    Same policy and the same evidence tier as the level pair: both fits' stored
    artefacts are read and cross-checked rather than recomputed from trace. The
    gain family needed its own instance because neither the ITT nor the level
    companion covers it — it stacks three period transitions under a shared child
    random intercept and conditions on the own baseline, so it is a different
    likelihood over different rows, and its stored ordinary-link posterior puts
    10.7 % of its mass below the three-choice guessing floor.

    Scope is the **model of record**. ``evaluate_gain_blending_link_pair`` reads
    ``link_sensitivity_required_for_release`` from the fit's own resolved plan, and
    the gain resolver sets that only for the interaction-free graded primary — so
    the treated-only ``lrp-rli-gf-106`` and moderation ``lrp-rli-gf-206`` variants
    return "no link pairing" here rather than failing closed. That exemption is
    recorded and dated in
    ``notes/202608251100-gain-blending-guessing-floor-596.md``; it is the same
    boundary :func:`gate_applies` already draws, and it keeps fail-closed from
    demanding floor twins of variants that were never the published headline.
    """
    from language_reading_predictors.statistical_models.blending_sensitivity import (
        evaluate_gain_blending_link_pair,
    )

    try:
        status = evaluate_gain_blending_link_pair(output_dir, config=config)
    except Exception as exc:  # noqa: BLE001 - a gate that cannot run must fail closed
        return (f"the gain B link pair could not be evaluated: {exc}",)
    if status.get("required") and not status.get("ready"):
        reason = str(status.get("reason") or "the paired evidence is stale")
        return (
            "the mandatory phoneme-blending link pair "
            "(lrp-rli-gf-006 + lrp-rli-gf-306) is not release-ready: " + reason,
        )
    return ()


def _aligned_blending_pair_release_failures(
    output_dir: Path, config: Mapping[str, Any]
) -> tuple[str, ...]:
    """The aligned family's phoneme-blending pairing (#619).

    Same policy and the same evidence tier as the level, DiD and gain pairs: both
    fits' stored artefacts are read and cross-checked rather than recomputed from
    trace.

    Nothing in this family is randomised, and that is not an exemption. The #608
    decision binds every ``B`` model whether its published quantity is a contrast or
    an association, because the link determines the mapping from the latent scale to
    the reported one and any natural-scale headline inherits it. LRPAL06's published
    ``cohort_marginal.csv`` is exactly such a headline.

    Scope is the model of record: ``resolve_aligned_run_plan`` sets
    ``link_sensitivity_required_for_release`` only for the non-dose primary, so the
    collider-conditioned dose sensitivity returns "no link pairing" here rather than
    failing closed.
    """
    from language_reading_predictors.statistical_models.blending_sensitivity import (
        evaluate_aligned_blending_link_pair,
    )

    try:
        status = evaluate_aligned_blending_link_pair(output_dir, config=config)
    except Exception as exc:  # noqa: BLE001 - a gate that cannot run must fail closed
        return (f"the aligned B link pair could not be evaluated: {exc}",)
    if status.get("required") and not status.get("ready"):
        reason = str(status.get("reason") or "the paired evidence is stale")
        return (
            "the mandatory phoneme-blending link pair "
            "(lrp-rli-al-006 + lrp-rli-al-306) is not release-ready: " + reason,
        )
    return ()


def _concurrent_blending_pair_release_failures(
    output_dir: Path, config: Mapping[str, Any]
) -> tuple[str, ...]:
    """The concurrent family's phoneme-blending pairing (#619).

    Same policy and evidence tier as the level, DiD, gain and aligned pairs. Two
    features are particular to this family. Its published output is a *table* of
    per-wave marginals rather than a single card, so the pair check verifies the
    identity evidence plus the table's shape rather than comparing one headline
    number. And the link governs blending only as the **outcome**: the six sibling
    models that carry B as a *predictor* take it as a standardised logit covariate,
    not as a score mean, so their plans do not declare the pairing and they return
    "no link pairing" here.
    """
    from language_reading_predictors.statistical_models.blending_sensitivity import (
        evaluate_concurrent_blending_link_pair,
    )

    try:
        status = evaluate_concurrent_blending_link_pair(output_dir, config=config)
    except Exception as exc:  # noqa: BLE001 - a gate that cannot run must fail closed
        return (f"the concurrent B link pair could not be evaluated: {exc}",)
    if status.get("required") and not status.get("ready"):
        reason = str(status.get("reason") or "the paired evidence is stale")
        return (
            "the mandatory phoneme-blending link pair "
            "(lrp-rli-ca-007 + lrp-rli-ca-307) is not release-ready: " + reason,
        )
    return ()


def _dose_blending_pair_release_failures(
    output_dir: Path, config: Mapping[str, Any]
) -> tuple[str, ...]:
    """The dose family's phoneme-blending pairing (#619).

    Same policy and evidence tier as the level, DiD, gain, aligned and concurrent
    pairs. This is the family #608 used to close the observational-exemption
    argument: the declared focal estimand is the natural-scale treated-row dose
    marginal, published in items by ``dose_marginal_summary.csv``, so it inherits
    the link exactly as a randomised contrast does. That no dose slope is causal
    changes what the number means, not what scale it sits on.
    """
    from language_reading_predictors.statistical_models.blending_sensitivity import (
        evaluate_dose_blending_link_pair,
    )

    try:
        status = evaluate_dose_blending_link_pair(output_dir, config=config)
    except Exception as exc:  # noqa: BLE001 - a gate that cannot run must fail closed
        return (f"the dose B link pair could not be evaluated: {exc}",)
    if status.get("required") and not status.get("ready"):
        reason = str(status.get("reason") or "the paired evidence is stale")
        return (
            "the mandatory phoneme-blending link pair "
            "(lrp-rli-dose-084 + lrp-rli-dose-384) is not release-ready: " + reason,
        )
    return ()


def _mediation_blending_pair_release_failures(
    output_dir: Path, config: Mapping[str, Any]
) -> tuple[str, ...]:
    """The mediation family's phoneme-blending pairing (#619).

    Same policy and evidence tier as the other stored-artefact pairs, but the link
    reaches further into this family than any other: every NDE, NIE and total is a
    difference of *simulated outcome means*, so ``score_mean_link`` enters the
    g-formula's counterfactual simulation cell by cell rather than any summary
    afterwards. LRP87's stored posterior also carries the largest below-chance share
    of any registered ``B`` fit (12.1 %).

    Scope is the model of record: ``lrp-rli-med-187`` declares ``companion_of`` and
    reproduces LRP87's numbers under an interventional relabelling, so its plan does
    not declare the pairing and it returns "no link pairing" here.
    """
    from language_reading_predictors.statistical_models.blending_sensitivity import (
        evaluate_mediation_blending_link_pair,
    )

    try:
        status = evaluate_mediation_blending_link_pair(output_dir, config=config)
    except Exception as exc:  # noqa: BLE001 - a gate that cannot run must fail closed
        return (f"the mediation B link pair could not be evaluated: {exc}",)
    if status.get("required") and not status.get("ready"):
        reason = str(status.get("reason") or "the paired evidence is stale")
        return (
            "the mandatory phoneme-blending link pair "
            "(lrp-rli-med-087 + lrp-rli-med-387) is not release-ready: " + reason,
        )
    return ()


#: Per-family phoneme-blending pair gates, keyed by ``ModelSpec.kind`` (#619).
#: ``itt`` is deliberately absent: it takes the trace-backed content-addressed
#: archive path inside :func:`_blending_pair_release_failures` rather than the
#: stored-artefact check these seven share. A ``B`` fit whose kind is in neither
#: place fails closed -- see that function.
_BLENDING_PAIR_GATES: dict[str, Callable[[Path, Mapping[str, Any]], tuple[str, ...]]] = {
    "aligned": _aligned_blending_pair_release_failures,
    "concurrent": _concurrent_blending_pair_release_failures,
    "did": _did_blending_pair_release_failures,
    "dose_response": _dose_blending_pair_release_failures,
    "gain_factors": _gain_blending_pair_release_failures,
    "level_factors": _level_blending_pair_release_failures,
    "mediation": _mediation_blending_pair_release_failures,
}


def _joint_blending_scope_note(output_dir: Path, config: Mapping[str, Any]) -> str:
    """Qualifier when a joint fit carrying ``B`` has no release-ready 008/108 bundle
    beside it (2026-08-23 joint audit, finding 12).

    **The recorded policy scope.** The mandatory phoneme-blending response-link
    pairing (``lrp-rli-itt-008`` + ``lrp-rli-itt-108``) governs the *model of
    record* for ``B``: neither of those fits may release without the validated
    trace-backed bundle. ``lrp-rli-itt-012`` also fits ``B``, on the ordinary logit
    mean, and can publish a row for it — but the gate is keyed to ``kind == "itt"``,
    so nothing verified the condition its own findings box asserts in prose. That
    left an unguarded alternate route to a blending treatment claim.

    The resolution is *scope plus verification*, not extension of the withhold. A
    joint ``B`` row is a **secondary structural cross-check**: it is not
    independently release-qualified and cannot supersede or weaken the paired
    008/108 conclusion. Withholding nine valid outcomes because one row's companion
    is stale would destroy sound information to protect a row that is not the model
    of record — the same reasoning the dependence pairing already uses. So the
    check verifies the sibling bundle and, when it is not ready, attaches a note
    saying the joint ``B`` row must not be read as a blending treatment claim at
    all. Fail-closed: anything unverifiable attaches the note with its reason.
    """
    if str(config.get("kind") or "") != "joint":
        return ""
    if "B" not in [str(o) for o in (_plan(config).get("outcomes") or [])]:
        return ""

    def _note(reason: str) -> str:
        return (
            "This joint fit reports an ordinary-logit phoneme-blending (B) effect, "
            "which is a secondary structural cross-check and is not independently "
            "release-qualified. The mandatory response-link bundle "
            "(lrp-rli-itt-008 + lrp-rli-itt-108) that governs the B model of record "
            f"is not release-ready beside it ({reason}), so the B row here must not "
            "be read as a phoneme-blending treatment claim, and it cannot supersede "
            "or weaken the paired 008/108 conclusion."
        )

    from language_reading_predictors.statistical_models.blending_sensitivity import (
        BLENDING_PRIMARY_MODEL_ID,
        evaluate_local_blending_link_sensitivity,
    )

    try:
        directory = Path(output_dir).resolve()
        config_name = str(config.get("config_name") or "") or _config_name(
            directory, str(config.get("model_id") or "")
        )
        if not config_name:
            return _note("this fit's configuration name could not be resolved")
        primary_dir = directory.parent / f"{BLENDING_PRIMARY_MODEL_ID}-{config_name}"
        primary_config = _load_config(primary_dir)
        if not primary_config:
            return _note(f"{BLENDING_PRIMARY_MODEL_ID} has no readable config.json")
        theirs = str(primary_config.get("data_sha256") or "")
        ours = str(config.get("data_sha256") or "")
        if not theirs or not ours or theirs != ours:
            return _note("the bundle was not fitted on the same input data")
        status = evaluate_local_blending_link_sensitivity(
            primary_dir, config=primary_config
        )
        if status.get("required") and not status.get("ready"):
            return _note(str(status.get("reason") or "the paired evidence is stale"))
    except Exception as exc:  # noqa: BLE001 - a gate that cannot run must fail closed
        return _note(f"the bundle could not be verified: {exc}")
    return ""


def _dependence_identification_note(output_dir: Path) -> str:
    """Qualifier when a fitted dependence block never moved off its prior.

    A companion that switches the LKJ residual block on estimates the within-child
    covariance the parent's factorised interval omits. That covariance is the
    *data's* only if the correlation posterior is distinguishable from the
    correlation prior. For the three registered two-outcome companions
    at n = 53 it is not: posterior-to-prior SD ratios of 1.002, 1.008 and 1.001
    (2026-08-22 ITT audit, finding 3). The interval such a fit publishes is the
    LKJ prior's implied correction, and a reader is entitled to be told so beside
    the number rather than having to reconstruct it.

    A note, never a withhold. The fit is valid and its residual SDs *are*
    informed; what is qualified is the interpretation of the correlation. Silent
    when ``dependence_identification.csv`` is absent (every fit without the block,
    and any stored fit written before the table existed), so old decisions
    re-decide identically.
    """
    frame = _read_csv(output_dir, "dependence_identification.csv")
    if frame is None or frame.empty or "verdict" not in frame.columns:
        return ""
    correlations = frame.loc[frame["role"].astype(str) == "residual correlation"]
    if correlations.empty:
        return ""
    dominated = correlations.loc[
        correlations["verdict"].astype(str) == "prior-dominated"
    ]
    if dominated.empty:
        return ""
    names = ", ".join(str(v) for v in dominated["parameter"])
    return (
        "The within-child residual correlation did not move off its prior "
        f"({names}), so the dependence correction this fit applies to the "
        "contrast's interval is the prior's rather than the data's; read the "
        "interval as a prior-informed sensitivity, not as a measured "
        "within-child covariance."
    )


#: Binding fields a factorised joint contrast parent and its LKJ residual-correlation
#: companion must agree on before the pair counts as evidence about the *same*
#: estimand on the *same* rows (2026-08-23 joint audit, finding 2). Each entry is
#: ``(reader, human-readable description)``; the reader pulls the value out of a
#: stored ``config.json``. Comparing the resolved *plan* rather than the declared
#: settings is deliberate: the plan is what the fit actually ran.
_JOINT_PAIR_BINDING: tuple[tuple[str, Callable[[Mapping[str, Any]], Any]], ...] = (
    ("the ordered outcome list", lambda c: list(_plan(c).get("outcomes") or [])),
    (
        "the contrast direction",
        lambda c: [
            str((_plan(c).get("contrast") or {}).get("left") or ""),
            str((_plan(c).get("contrast") or {}).get("right") or ""),
        ],
    ),
    (
        "the fitted equation's precision terms",
        lambda c: [
            bool(_plan(c).get("use_cross_baselines")),
            bool(_plan(c).get("use_age_linear")),
            bool(_plan(c).get("use_age_gp")),
        ],
    ),
    ("the PSIS-LOO unit", lambda c: str(_plan(c).get("loo_unit") or "")),
    ("the input data checksum", lambda c: str(c.get("data_sha256") or "") or None),
    (
        "the fitted-row identity",
        lambda c: str((c.get("fitted_subject_identity") or {}).get("sha256") or "")
        or None,
    ),
    (
        "the fitted-data digest and observed denominators",
        lambda c: (
            [digest, (c.get("fitted_data_identity") or {}).get("observed")]
            if (digest := str((c.get("fitted_data_identity") or {}).get("digest") or ""))
            else None
        ),
    ),
    ("the sampling configuration", lambda c: c.get("sampling") or None),
    (
        "the source commit",
        lambda c: str(
            ((c.get("provenance") or {}).get("source") or {}).get("commit") or ""
        )
        or None,
    ),
)


def _plan(config: Mapping[str, Any]) -> Mapping[str, Any]:
    """The stored resolved run plan, or an empty mapping."""
    plan = config.get("resolved_run_plan")
    return plan if isinstance(plan, Mapping) else {}


def _required_dependence_companion(config: Mapping[str, Any]) -> str:
    """The companion this fit must be read beside, or ``""``.

    Derived from the **registered** pairing constant first and the stored plan
    second. Deriving it from the stored plan alone left the qualifier dormant on
    every artefact written before ``dependence_companion`` existed — which is all
    three current parent fits (2026-08-23 joint audit, finding 2) — so a stale
    stored plan could bypass a policy the registered module declares. This mirrors
    how the phoneme-blending gate derives its requirement from
    ``BLENDING_LINK_MODELS`` rather than from what a fit happened to record.
    """
    from language_reading_predictors.statistical_models.joint import (
        JOINT_DEPENDENCE_COMPANIONS,
    )

    model_id = str(config.get("model_id") or "")
    registered = JOINT_DEPENDENCE_COMPANIONS.get(model_id, "")
    if registered:
        return registered
    contrast = _plan(config).get("contrast")
    if not isinstance(contrast, Mapping):
        return ""
    return str(contrast.get("dependence_companion") or "")


def _joint_marginal_widths(
    directory: Path, outcomes: tuple[str, str]
) -> dict[str, float] | None:
    """Each contrast outcome's probability-scale AME interval width, or ``None``."""
    frame = _read_csv(directory, "tau_summary.csv")
    if frame is None or frame.empty or "outcome" not in frame.columns:
        return None
    needed = ("ame_prob_lo", "ame_prob_hi")
    if any(column not in frame.columns for column in needed):
        return None
    indexed = frame.set_index(frame["outcome"].astype(str))
    widths: dict[str, float] = {}
    for outcome in outcomes:
        if outcome not in indexed.index:
            return None
        row = indexed.loc[outcome]
        lo, hi = _finite(row["ame_prob_lo"]), _finite(row["ame_prob_hi"])
        if lo is None or hi is None or hi <= lo:
            return None
        widths[outcome] = hi - lo
    return widths


def _joint_width_channels(
    *,
    parent_dir: Path,
    companion_dir: Path,
    outcomes: tuple[str, str],
    parent_width: float,
    companion_width: float,
) -> dict[str, Any]:
    """Split the contrast's width change into marginal and covariance channels.

    2026-08-24 review of the joint audit. Finding 2 asked that the dependence block
    be assessed through its consequence for the declared contrast, which
    :func:`_joint_contrast_consequence` does for the contrast's *location*. But the
    reason three report templates give for running the companion at all is about its
    *width*: that a factorised interval omits within-child cross-outcome covariance,
    so a positive residual correlation leaves it too wide and a negative one too
    narrow. That sign rule describes the covariance term
    ``Var(A - B) = V_A + V_B - 2 Cov(A, B)`` in isolation. It does not describe what
    separates these two fits, because the companion also adds a per-child
    logistic-normal layer whose own parameter uncertainty widens *both* marginals.

    So measure which channel the change came through instead of asserting one. Each
    fit's implied cross-outcome posterior correlation follows from the same identity
    read on equal-tailed interval widths,
    ``r = (W_A^2 + W_B^2 - W_diff^2) / (2 W_A W_B)``; the parent's is structurally
    zero because a factorised fit shares no parameter between outcomes, so its
    measured value is this approximation's own noise floor and is recorded beside
    the companion's for exactly that purpose. ``marginal`` is what the companion's
    wider marginals alone would do at the parent's correlation, and ``covariance``
    is the remainder.

    Returns the record fields, or a ``channel_status`` explaining why the split
    could not be taken. Never raises: this is descriptive provenance attached to a
    release decision, not a gate.
    """
    parent_widths = _joint_marginal_widths(parent_dir, outcomes)
    companion_widths = _joint_marginal_widths(companion_dir, outcomes)
    if parent_widths is None or companion_widths is None:
        return {
            "channel_status": "unavailable",
            "channel_reason": "tau_summary.csv is missing the per-outcome AME interval",
        }
    left, right = outcomes

    def _implied(widths: Mapping[str, float], diff_width: float) -> float | None:
        a, b = widths[left], widths[right]
        value = (a * a + b * b - diff_width * diff_width) / (2 * a * b)
        return value if -1.0 <= value <= 1.0 else None

    parent_r = _implied(parent_widths, parent_width)
    companion_r = _implied(companion_widths, companion_width)
    if parent_r is None or companion_r is None:
        return {
            "channel_status": "unavailable",
            "channel_reason": (
                "the interval widths imply a correlation outside [-1, 1], so the "
                "Gaussian width identity does not describe these posteriors"
            ),
        }
    a, b = companion_widths[left], companion_widths[right]
    marginal_only = float(np.sqrt(max(a * a + b * b - 2 * parent_r * a * b, 0.0)))
    marginal_channel = marginal_only - parent_width
    covariance_channel = companion_width - marginal_only
    moved = abs(marginal_channel) + abs(covariance_channel)
    correlation_change = companion_r - parent_r
    if abs(correlation_change) <= _AME_CORRELATION_NOISE:
        dominant = "marginal_uncertainty"
    elif abs(covariance_channel) > abs(marginal_channel):
        dominant = "cross_outcome_covariance"
    else:
        dominant = "marginal_uncertainty"
    return {
        "channel_status": "measured",
        "parent_marginal_widths": {k: float(v) for k, v in parent_widths.items()},
        "companion_marginal_widths": {
            k: float(v) for k, v in companion_widths.items()
        },
        "parent_implied_ame_correlation": float(parent_r),
        "companion_implied_ame_correlation": float(companion_r),
        "implied_ame_correlation_change": float(correlation_change),
        "marginal_width_channel": float(marginal_channel),
        "covariance_width_channel": float(covariance_channel),
        "covariance_channel_share": (
            float(abs(covariance_channel) / moved) if moved else None
        ),
        "dominant_width_channel": dominant,
    }


def _joint_contrast_consequence(
    parent_dir: Path, companion_dir: Path, *, pair: tuple[str, str] | None = None
) -> tuple[dict[str, Any], str]:
    """Measure what the dependence model does to the **declared contrast**.

    Finding 2's second half. The robustness gate classifies power-scaling rows for
    the conditional-logit ``tau`` vector; clean marginal ``tau`` diagnoses say
    nothing about a nonlinear difference of standardised average marginal effects,
    which is the quantity the findings box actually reports. And requiring every
    nuisance correlation in the LKJ block to be sharply identified is the wrong
    test — at n = 53 it never will be, and it need not be for the contrast to be
    stable. So assess the block *through its consequence for the contrast*: read
    both fits' ``tau_difference.csv`` and compare the declared quantity directly.

    Returns the machine-readable record and a qualifier sentence, empty when the
    dependence model leaves the contrast's conclusion where it was. "Material" is
    a direction flip in the median, or a shift in P(> 0) of at least
    :data:`_CONTRAST_DIRECTION_SHIFT` — deliberately a conclusion-level rule, not
    a threshold on the interval, whose movement *is* the companion's purpose.
    """
    record: dict[str, Any] = {}

    def _unusable(status: str, reason: str) -> tuple[dict[str, Any], str]:
        """The comparison could not be taken, so say so rather than publishing silence.

        Fail-closed, matching the binding checks above: an absent or unreadable
        comparison is not evidence that the dependence model left the contrast
        alone. Without a note the reader sees an unqualified release and has no
        way to tell "checked and unchanged" from "never checked".
        """
        record["status"] = status
        record["reason"] = reason
        return record, (
            "The dependence model's consequence for the declared contrast could "
            f"not be measured ({reason}), so the paired contrast is "
            "dependence-unchecked in substance even though the companion is bound "
            "beside it. Regenerate this decision once both fits carry a readable "
            "contrast summary."
        )

    parent = _read_csv(parent_dir, "tau_difference.csv")
    companion = _read_csv(companion_dir, "tau_difference.csv")
    if parent is None or companion is None or parent.empty or companion.empty:
        return _unusable("unavailable", "one or both fits have no tau_difference.csv")
    p, c = parent.iloc[0], companion.iloc[0]
    if str(p.get("contrast")) != str(c.get("contrast")):
        return _unusable(
            "mismatched",
            f"parent reports {p.get('contrast')!r} and companion "
            f"{c.get('contrast')!r}",
        )
    needed = ("diff_prob_median", "diff_prob_lo", "diff_prob_hi", "prob_diff_pos")
    if any(col not in parent.columns or col not in companion.columns for col in needed):
        return _unusable(
            "unavailable", "tau_difference.csv is missing the contrast columns"
        )
    values = {name: (_finite(p[name]), _finite(c[name])) for name in needed}
    if any(v[0] is None or v[1] is None for v in values.values()):
        return _unusable(
            "unavailable", "tau_difference.csv holds non-finite contrast values"
        )
    p_med, c_med = values["diff_prob_median"]
    p_pos, c_pos = values["prob_diff_pos"]
    p_width = values["diff_prob_hi"][0] - values["diff_prob_lo"][0]
    c_width = values["diff_prob_hi"][1] - values["diff_prob_lo"][1]
    direction_shift = abs(c_pos - p_pos)
    flipped = (p_med > 0) != (c_med > 0)
    record.update(
        {
            "status": "compared",
            "contrast": str(p.get("contrast")),
            "scale": str(p.get("headline_scale") or ""),
            "parent_median": p_med,
            "companion_median": c_med,
            "median_shift": c_med - p_med,
            "parent_prob_positive": p_pos,
            "companion_prob_positive": c_pos,
            "direction_probability_shift": direction_shift,
            "parent_interval_width": p_width,
            "companion_interval_width": c_width,
            "interval_width_ratio": (c_width / p_width) if p_width else None,
            "direction_flipped": bool(flipped),
            "material": bool(flipped or direction_shift >= _CONTRAST_DIRECTION_SHIFT),
        }
    )
    if pair is not None and all(pair):
        record.update(
            _joint_width_channels(
                parent_dir=parent_dir,
                companion_dir=companion_dir,
                outcomes=pair,
                parent_width=p_width,
                companion_width=c_width,
            )
        )
    else:
        record["channel_status"] = "unavailable"
        record["channel_reason"] = "the resolved plan does not name the contrast pair"
    if not record["material"]:
        return record, ""
    cause = (
        "reverses the sign of the contrast median"
        if flipped
        else f"moves P(> 0) by {direction_shift:.2f}"
    )
    return record, (
        "The dependence model materially changes the declared contrast: the LKJ "
        f"companion {cause} (parent P(> 0) = {p_pos:.2f}, companion "
        f"{c_pos:.2f}). Read the paired conclusion from the companion, not from "
        "this fit's factorised interval alone."
    )


#: Binding fields a within-child historical-joint fit and its wider-prior
#: sensitivity companion must agree on before the pair counts as evidence about the
#: *same* model under a *different* prior (#588 finding 5). Everything the fitted
#: equation and the analysis rows depend on, and every prior scale except the one
#: under test.
_HISTORICAL_JOINT_PRIOR_BINDING: tuple[
    tuple[str, Callable[[Mapping[str, Any]], Any]], ...
] = (
    ("the measure list", lambda c: list(_plan(c).get("measures") or [])),
    (
        "the analysis window",
        lambda c: [
            list(_plan(c).get("waves") or []),
            list(_plan(c).get("extension_waves") or []),
        ],
    ),
    ("the likelihood", lambda c: str(_plan(c).get("likelihood") or "")),
    (
        "the priors not under test",
        lambda c: [
            _plan(c).get("eta_prior_sigma"),
            _plan(c).get("sigma_subject_prior_sigma"),
            _plan(c).get("lkj_eta"),
            _plan(c).get("within_lkj_eta"),
        ],
    ),
    ("the input data checksum", lambda c: str(c.get("data_sha256") or "") or None),
    (
        "the fitted-row identity",
        lambda c: str((c.get("fitted_subject_identity") or {}).get("sha256") or "")
        or None,
    ),
)

#: How far a measure's power-scaling prior sensitivity may sit before the fit's own
#: diagnostics say the within-scale prior is doing the deciding. ArviZ's own flag
#: threshold; quoted rather than re-derived so the qualification and the psense
#: table cannot disagree.
_HISTORICAL_JOINT_PRIOR_SENSITIVE = PSENSE_THRESHOLD


def _historical_joint_prior_sensitivity(output_dir: Path) -> str:
    """The measured prior sensitivity of ``sigma_within``, as a phrase or ``""``.

    The qualification below is about a prior whose influence the fit has already
    measured, so quote the measurement rather than asserting that the prior matters.
    """
    frame = _read_csv(output_dir, "psense_summary.csv", index_col=0)
    if frame is None or frame.empty or "prior" not in frame.columns:
        return ""
    rows = frame.loc[frame.index.astype(str).str.startswith("sigma_within")]
    values = pd.to_numeric(rows["prior"], errors="coerce").dropna()
    if values.empty:
        return ""
    top = float(values.max())
    if top < _HISTORICAL_JOINT_PRIOR_SENSITIVE:
        return (
            f" This fit's own power scaling puts the largest sigma_within prior "
            f"sensitivity at {top:.2f}, below ArviZ's "
            f"{_HISTORICAL_JOINT_PRIOR_SENSITIVE:.2f} flag."
        )
    return (
        f" This fit's own power scaling already flags that prior: the largest "
        f"sigma_within prior sensitivity is {top:.2f}, against ArviZ's "
        f"{_HISTORICAL_JOINT_PRIOR_SENSITIVE:.2f} flag threshold."
    )


def _prior_evidence_qualifications(output_dir: Path) -> list[str]:
    """Name the estimands whose prior check could not be computed (#637 stage 1).

    **The policy, stated once.** An ``unavailable`` row in ``prior_pushforward.csv``
    *qualifies* a release; it does not withhold one. The estimand-scale prior check
    is evidence **about the prior**, not a scientific result: the posterior, its
    convergence gate, ``priors_table.csv``, the prior-vs-posterior overlay and
    ``psense_summary.csv`` are all unaffected by its absence, and withholding on it
    would take out every fit whose family legitimately has no contrast to push a
    prior through. What the absence does cost is a reader's ability to judge, on the
    reported scale, how much of the answer the prior supplied — so it must be
    stated, not left to a column nobody reads.

    Before #637 this could not be stated honestly anyway: four families caught every
    exception around the pushforward, so an ``unavailable`` row could mean either an
    honest absence or a ``KeyError``. Now only the first can produce one, and the
    qualification means what it says.

    Reads the stored table, so the same qualification is reproduced when a fit
    directory is re-decided at render time.
    """
    table = _read_csv(output_dir, "prior_pushforward.csv")
    if table is None or "status" not in table.columns:
        return []
    rows = table[table["status"].astype(str) == "unavailable"]
    if rows.empty:
        return []
    estimands = ", ".join(dict.fromkeys(str(value) for value in rows["estimand"]))
    return [
        "the estimand-scale prior check is unavailable for "
        f"{estimands}, so this fit's prior influence on the reported scale is "
        "unquantified"
    ]


def _historical_joint_prior_companion_qualifications(
    output_dir: Path, config: Mapping[str, Any]
) -> list[str]:
    """Qualify a within-child historical-joint fit whose prior sensitivity is absent.

    2026-08-23 joint audit, finding 5, completing what #609 registered. The family
    is descriptive, so :func:`gate_applies` excludes it and no robustness verdict is
    produced for it at all — which left the parent publishing a **prior-dependent
    classification** with nothing machine-readable saying so. Which measures clear
    the 0.05-logit resolvability threshold, and therefore which correlations may be
    interpreted, is decided by ``sigma_within``, whose prior the registered
    companion varies; on the stored fit that parameter is also the most
    power-scaling-sensitive quantity in the model.

    A **qualification, never a withhold**: the fit is valid, its convergence gate
    passes and its tables are correct under the declared prior. What is qualified is
    the robustness of the classification those tables carry. Fail-closed on every
    unreadable or unbound path, and silent for a fit the constant does not pair or
    that has no within-child block (so a stored ``jc-001`` decision is untouched).
    """
    from language_reading_predictors.statistical_models.historical_joint import (
        HISTORICAL_JOINT_PRIOR_COMPANIONS,
    )

    if str(config.get("kind") or "") != "historical_joint":
        return []
    model_id = str(config.get("model_id") or "")
    companion = HISTORICAL_JOINT_PRIOR_COMPANIONS.get(model_id, "")
    if not companion or not bool(_plan(config).get("within_correlation")):
        return []
    measured = _historical_joint_prior_sensitivity(output_dir)

    def _note(reason: str) -> list[str]:
        return [
            f"the registered within-scale prior sensitivity ({companion}) is not "
            f"release-ready beside this fit ({reason}), so which measures clear the "
            "resolvability threshold — and therefore which correlations may be read "
            f"at all — is a conclusion under this fit's prior alone.{measured}"
        ]

    try:
        directory = Path(output_dir).resolve()
        config_name = str(config.get("config_name") or "") or _config_name(
            directory, model_id
        )
        if not config_name:
            return _note("this fit's configuration name could not be resolved")
        companion_dir = directory.parent / f"{companion}-{config_name}"
        decision, decision_error = _read_json(
            companion_dir / RELEASE_DECISION_FILENAME
        )
        if decision_error is not None or not isinstance(decision, Mapping):
            return _note("it has not been fitted, or its release decision is unreadable")
        if not bool(decision.get("publishable")):
            return _note("its own release decision withholds publication")
        companion_config = _load_config(companion_dir)
        if not companion_config:
            return _note("its config.json is missing or unreadable")
        if str(companion_config.get("model_id") or "") != companion:
            return _note("the sibling directory does not identify itself as the companion")
        ours = _plan(config).get("sigma_within_prior_sigma")
        theirs = _plan(companion_config).get("sigma_within_prior_sigma")
        if ours is None or theirs is None:
            return _note("the within-scale prior is not recorded on both fits")
        if ours == theirs:
            return _note(
                "it was fitted under the same within-scale prior, so it varies "
                "nothing"
            )
        for description, reader in _HISTORICAL_JOINT_PRIOR_BINDING:
            mine, yours = reader(config), reader(companion_config)
            if mine is None or yours is None:
                return _note(f"{description} is not recorded on both fits")
            if mine != yours:
                return _note(f"{description} differs between the two fits")
        changed = _historical_joint_resolvability_change(directory, companion_dir)
    except Exception as exc:  # noqa: BLE001 - a check that cannot run must fail closed
        return _note(f"it could not be verified: {exc}")
    if changed:
        return [
            f"the within-scale prior sensitivity ({companion}) changes the "
            f"resolvability classification ({changed}), so the interpretable set of "
            "correlations here depends on that prior rather than on the data."
        ]
    return []


def _historical_joint_resolvability_change(
    parent_dir: Path, companion_dir: Path
) -> str:
    """Which measures the wider prior reclassifies, as a phrase or ``""``.

    The classification *is* the conclusion for this family, so comparing it across
    the two independently sampled fits is the comparison that matters — not pairing
    draws, which are unrelated between chains fitted under different priors.
    """
    parent = _read_csv(parent_dir, "within_scale_summary.csv")
    companion = _read_csv(companion_dir, "within_scale_summary.csv")
    if parent is None or companion is None:
        return "one of the two fits has no within_scale_summary.csv"
    needed = {"measure", "resolvable"}
    if not needed <= set(parent.columns) or not needed <= set(companion.columns):
        return "within_scale_summary.csv does not record the classification"

    def _flags(frame: pd.DataFrame) -> dict[str, bool]:
        return {
            str(row["measure"]): str(row["resolvable"]).strip().lower()
            in {"true", "1"}
            for _, row in frame.iterrows()
        }

    mine, theirs = _flags(parent), _flags(companion)
    if set(mine) != set(theirs):
        return "the two fits classify different measure sets"
    moved = sorted(name for name in mine if mine[name] != theirs[name])
    if not moved:
        return ""
    return ", ".join(
        f"{name}: {'resolvable' if mine[name] else 'unresolvable'} here, "
        f"{'resolvable' if theirs[name] else 'unresolvable'} under the wider prior"
        for name in moved
    )


def _joint_dependence_companion_note(
    output_dir: Path, config: Mapping[str, Any]
) -> tuple[str, dict[str, Any] | None]:
    """Qualifying note when a factorised joint contrast's dependence companion is
    not release-ready **and bound** beside it (2026-08-21 joint review, finding 3;
    binding and contrast consequence added by the 2026-08-23 joint audit, finding 2).

    The three contrast parents' ``dependence_note`` prose has always said the
    contrast is dependence-checked only once the registered LKJ companion
    (lrp-rli-itt-215/315/216, #551) has passed the house gate. Verifying that the
    companion is publishable is necessary but not sufficient: a *different*
    companion fit — other outcomes, the reversed contrast, other rows, other
    sampling settings, another commit — would satisfy it just as well. So the pair
    is now bound field by field through :data:`_JOINT_PAIR_BINDING`, and the
    dependence block is assessed through its consequence for the declared contrast
    rather than through whether every nuisance correlation is sharply identified.

    Deliberately a **qualify-note**, not a withhold: the parent's per-outcome
    marginal effects are fully valid without the companion — only the paired
    contrast's interval is dependence-unchecked — so the failure attaches the
    caveat sentence to the findings box rather than withholding valid marginals.
    During a fresh sweep a parent can finalise before its companion has been
    fitted; the note then attaches and is cleared by regenerating the decision
    (``scripts/regenerate_key_findings.py``) once the companion completes.
    Fail-closed: any error verifying the companion, and any binding field that
    cannot be read on both sides, attaches the note with the reason rather than
    silently releasing an unchecked pairing.

    Returns ``(note, contrast_record)``; the record is the machine-readable
    contrast comparison persisted in ``release_decision.json``.
    """
    if str(config.get("kind") or "") != "joint":
        return "", None
    companion = _required_dependence_companion(config)
    if not companion or bool(_plan(config).get("use_residual_correlation")):
        return "", None

    def _note(reason: str) -> str:
        return (
            f"The declared contrast's dependence-model companion ({companion}) is "
            f"not release-ready beside this fit ({reason}), so the paired contrast "
            "is dependence-unchecked: its interval omits within-child cross-outcome "
            "covariance and is not automatically conservative. Regenerate this "
            "decision once the companion has passed the house gate."
        )

    try:
        directory = Path(output_dir).resolve()
        model_id = str(config.get("model_id") or "")
        config_name = str(config.get("config_name") or "") or _config_name(
            directory, model_id
        )
        if not config_name:
            return _note("this fit's configuration name could not be resolved"), None
        companion_dir = directory.parent / f"{companion}-{config_name}"
        decision, decision_error = _read_json(
            companion_dir / RELEASE_DECISION_FILENAME
        )
        if decision_error is not None or not isinstance(decision, Mapping):
            return _note("its release decision is missing or unreadable"), None
        if not bool(decision.get("publishable")):
            return _note("its own release decision withholds publication"), None
        companion_config = _load_config(companion_dir)
        if not companion_config:
            return _note("its config.json is missing or unreadable"), None
        if str(companion_config.get("model_id") or "") != companion:
            return (
                _note("the sibling directory does not identify itself as the companion"),
                None,
            )
        if not bool(_plan(companion_config).get("use_residual_correlation")):
            return (
                _note("it is not a residual-correlated fit, so it is not a dependence model"),
                None,
            )
        for description, reader in _JOINT_PAIR_BINDING:
            ours, theirs = reader(config), reader(companion_config)
            if ours is None or theirs is None:
                return _note(f"{description} is not recorded on both fits"), None
            if ours != theirs:
                return _note(f"{description} differs between the two fits"), None
    except Exception as exc:  # noqa: BLE001 - a gate that cannot run must fail closed
        return _note(f"the companion could not be verified: {exc}"), None

    declared = _plan(config).get("contrast")
    pair: tuple[str, str] | None = None
    if isinstance(declared, Mapping):
        left, right = str(declared.get("left") or ""), str(declared.get("right") or "")
        pair = (left, right) if left and right else None
    contrast_record, contrast_note = _joint_contrast_consequence(
        directory, companion_dir, pair=pair
    )
    contrast_record["companion"] = companion
    return contrast_note, contrast_record


def evaluate_publication(
    output_dir: str | Path,
    *,
    config: Mapping[str, Any] | None = None,
    artifacts: Any = None,
) -> ReleaseEvaluation:
    """Decide what one fit may publish, as a single structured object.

    The stages run in the order a reader would apply them, and the first to
    object settles it:

    1. **inputs** — ``diagnostics_summary.json`` and ``config.json`` must be
       present and readable, and non-RLI fits must carry a resolved fit-time
       scientific-input contract. The diagnostics file is checked first because
       the sampling-quality gate outranks everything: findings from an unconverged
       fit must not reach a reader even if every other artefact is perfect.
    2. **computation** — the automatic sampling-quality gate must pass cleanly.
    3. **artifacts** — every artefact the fit recorded as *required* must be on
       disk. A required output that vanished between its write and finalisation is
       a withheld release, not a warning (#394 design point 3).
    The joint-mechanism levels design's per-wave bundle is checked alongside these:
    a published wave with no persisted trace, no informative predictive check, no
    recorded power-scaling result, no matching sub-fit provenance row or a failed
    convergence verdict withholds the whole fit, and breaching the predeclared
    new-child coverage floor attaches a qualification (2026-08-23 joint-mechanism
    follow-up review, finding 1).

    4. **robustness** — required influence checks must preserve their named
       scientific quantities; the phoneme-blending fits must carry their current,
       validated trace-backed link pair (``lrp-rli-itt-008`` + ``lrp-rli-itt-108``);
       for the families the treatment-effect gate covers, prior-sensitivity and
       floor-grid evidence must support a causal headline; and a factorised joint
       contrast whose declared LKJ dependence companion is not release-ready
       beside it releases with a dependence-unchecked qualifier attached
       (:func:`_joint_dependence_companion_note`). The saved
       sampling-preset name also distinguishes publication-grade ``rep-lite`` /
       ``reporting`` fits from local ``dev`` / ``test`` diagnostics. An estimand-
       scale prior check the fit could not compute attaches a named qualification
       rather than withholding — see :func:`_prior_evidence_qualifications` for
       why (#637 stage 1).

    Reads only artefacts already in ``output_dir``, so a stored fit can be
    re-decided without refitting — the contract ``evaluate_release`` and
    ``generate_key_findings`` both keep.
    """
    output_dir = Path(output_dir)

    # Loaded before any stage runs but *evaluated* after the sampling-quality gate:
    # the model identity belongs on the record whichever stage objects, while the
    # gate still outranks a missing config in deciding what may be published.
    if config is None:
        loaded = _load_config(output_dir)
        if loaded is None and os.path.exists(output_dir / "config.json"):
            config = None  # present but unreadable
        else:
            config = loaded if loaded is not None else {}

    if isinstance(config, Mapping):
        sampling_preset, development_only, publication_qualification = (
            _sampling_preset_qualification(output_dir, config)
        )
    else:
        sampling_preset, development_only, publication_qualification = (
            None,
            True,
            "config.json is unreadable, so publication-grade sampling cannot be verified",
        )
    qualification = {
        "sampling_preset": sampling_preset,
        "development_only": development_only,
        "publication_qualification": publication_qualification,
    }

    diag, diag_error = _read_json(output_dir / "diagnostics_summary.json")
    if diag_error == "missing":
        return ReleaseEvaluation(
            status="not_available",
            stage="inputs",
            reason=(
                "diagnostics_summary.json is missing, so the convergence gate "
                "cannot be checked"
            ),
            config=config,
            **qualification,
        )
    if diag_error is not None:
        return ReleaseEvaluation(
            status="not_available",
            stage="inputs",
            reason=(
                "diagnostics_summary.json could not be parsed, so the convergence "
                "gate cannot be checked"
            ),
            config=config,
            **qualification,
        )

    # Local import: ``reporting`` reaches this module through its own function-local
    # import of ``release``, and the gate reader lives beside the badge and banner
    # that render the same verdict.
    from language_reading_predictors.statistical_models.reporting import (
        convergence_gate_failures,
    )

    failing = convergence_gate_failures(diag)
    if failing:
        return ReleaseEvaluation(
            status="gate_failed",
            stage="computation",
            reason="the automatic sampling-quality gate failed",
            failing_checks=tuple(failing),
            config=config,
            **qualification,
        )

    if not config:
        return ReleaseEvaluation(
            status="not_available",
            stage="inputs",
            reason=(
                "config.json could not be parsed"
                if config is None
                else "config.json is missing"
            ),
            config=config,
            **qualification,
        )

    input_failures = _publication_input_failures(config)
    if input_failures:
        return ReleaseEvaluation(
            status="inputs_unresolved",
            stage="inputs",
            reason=(
                "publication inputs are unresolved: " + "; ".join(input_failures)
            ),
            input_failures=input_failures,
            config=config,
            **qualification,
        )

    t3_gate_failures, t3_artifact_failures = _mediation_t3_release_failures(
        output_dir, config
    )
    (
        growth_gate_failures,
        growth_artifact_failures,
        growth_robustness_failures,
    ) = (
        _growth_influence_release_failures(output_dir, config)
    )
    itt_missingness_gate_failures, itt_missingness_artifact_failures = (
        _itt_missingness_release_failures(output_dir, config)
    )
    concurrent_gate_failures, concurrent_artifact_failures = (
        _concurrent_published_fit_release_failures(output_dir, config)
    )
    adjusted_ses_gate_failures, adjusted_ses_artifact_failures = (
        _adjusted_ses_release_failures(output_dir, config)
    )
    (
        gain_p1_gate_failures,
        gain_p1_artifact_failures,
        gain_p1_robustness_failures,
    ) = _gain_period1_release_failures(output_dir, config)
    (
        jm_wave_gate_failures,
        jm_wave_artifact_failures,
        jm_wave_qualifications,
    ) = _joint_mechanism_wave_release_failures(output_dir, config)
    # The within-child historical-joint fits are descriptive, so the robustness
    # gate never runs for them and any note computed below would be discarded with
    # it. Their prior-sensitivity qualification therefore attaches here, where a
    # non-gated family's qualifications live (#588 finding 5).
    hj_prior_qualifications = _historical_joint_prior_companion_qualifications(
        output_dir, config
    )
    # Unavailable estimand-scale prior evidence qualifies rather than withholds
    # (#637 stage 1); :func:`_prior_evidence_qualifications` states why.
    prior_evidence_qualifications = _prior_evidence_qualifications(output_dir)
    if jm_wave_qualifications or hj_prior_qualifications or prior_evidence_qualifications:
        qualification["publication_qualification"] = "; ".join(
            part
            for part in (
                qualification["publication_qualification"],
                *jm_wave_qualifications,
                *hj_prior_qualifications,
                *prior_evidence_qualifications,
            )
            if part
        )
    gate_failures = tuple(
        sorted(
            {
                *t3_gate_failures,
                *growth_gate_failures,
                *itt_missingness_gate_failures,
                *jm_wave_gate_failures,
                *concurrent_gate_failures,
                *adjusted_ses_gate_failures,
                *gain_p1_gate_failures,
            }
        )
    )
    if gate_failures:
        return ReleaseEvaluation(
            status="gate_failed",
            stage="computation",
            reason=(
                "a required trace-backed secondary sensitivity did not pass its "
                "sampling-quality gate"
            ),
            failing_checks=gate_failures,
            config=config,
            **qualification,
        )

    missing = tuple(
        sorted(
            {
                *t3_artifact_failures,
                *growth_artifact_failures,
                *itt_missingness_artifact_failures,
                *jm_wave_artifact_failures,
                *concurrent_artifact_failures,
                *adjusted_ses_artifact_failures,
                *gain_p1_artifact_failures,
                *_recorded_required_artifacts(output_dir, artifacts),
            }
        )
    )
    if missing:
        return ReleaseEvaluation(
            status="artifacts_incomplete",
            stage="artifacts",
            reason=(
                "required fit artefacts are missing or invalid: "
                f"{', '.join(missing)}"
            ),
            missing_artifacts=missing,
            config=config,
            **qualification,
        )

    robustness_failures = (
        *growth_robustness_failures,
        *gain_p1_robustness_failures,
        *_blending_pair_release_failures(output_dir, config),
    )
    if robustness_failures:
        return ReleaseEvaluation(
            status="robustness_unresolved",
            stage="robustness",
            reason="; ".join(robustness_failures),
            config=config,
            **qualification,
        )

    robustness = _robustness_decision(output_dir, config)
    if robustness is not None and not robustness.released:
        return ReleaseEvaluation(
            status="robustness_unresolved",
            stage="robustness",
            reason=robustness.reason,
            robustness=robustness,
            config=config,
            **qualification,
        )
    # Joint dependence pairing (2026-08-21 review, finding 3; bound field by field
    # and assessed through the declared contrast by the 2026-08-23 audit, finding
    # 2): a factorised contrast whose registered LKJ companion is not release-ready
    # *and bound* beside it releases with the dependence-unchecked qualifier
    # attached, so the findings box carries the caveat the prose ``dependence_note``
    # has always demanded. When the pair does bind, the measured consequence for the
    # declared contrast is recorded and only qualifies the release if it changes the
    # conclusion.
    companion_note, dependence_contrast = _joint_dependence_companion_note(
        output_dir, config
    )
    # The companion note is for a *parent* whose companion is missing; this one is
    # for the companion itself, whose block may have learned nothing (2026-08-22
    # ITT audit, finding 3). A fit can in principle attract both.
    identification_note = _dependence_identification_note(output_dir)
    # Scope of the phoneme-blending response-link policy in a joint fit (2026-08-23
    # audit, finding 12): the joint B row is a secondary structural cross-check, and
    # the note says so — verified against the sibling bundle — whenever the pairing
    # that governs the B model of record is not release-ready beside it.
    blending_scope_note = _joint_blending_scope_note(output_dir, config)
    attached = " ".join(
        n
        for n in (companion_note, identification_note, blending_scope_note)
        if n
    )
    if attached and robustness is not None:
        robustness = replace(
            robustness, note=(robustness.note + " " + attached).strip()
        )
    return ReleaseEvaluation(
        status="ok",
        stage="robustness",
        robustness=robustness,
        config=config,
        dependence_contrast=dependence_contrast,
        **qualification,
    )


def _robustness_decision(
    output_dir: Path, config: Mapping[str, Any]
) -> ReleaseDecision | None:
    """The treatment-effect robustness verdict, or ``None`` if out of scope.

    A gate that cannot be evaluated **withholds**, matching how an unverifiable
    analysis population is already handled: degrading to "no gating" would silently
    reinstate the defect the gate exists to prevent, and would do so precisely when
    something unexpected is wrong. Withholding is loud, costs no data (every CSV is
    still written), and is repaired by regenerating the decision once the cause is
    fixed. It never raises, so a fit's finalisation is not lost after sampling.
    """
    if not gate_applies(config):
        return None
    try:
        return evaluate_release(output_dir, config)
    except Exception as exc:  # noqa: BLE001 - a gate that cannot run must fail closed
        return ReleaseDecision(
            status="withhold",
            tau_class="unavailable",
            reason=(
                "the robustness release gate could not be evaluated for this fit "
                f"({exc}), so its prior dependence is unverified"
            ),
        )


def write_release_decision(ctx: Any, evaluation: ReleaseEvaluation) -> dict[str, Any]:
    """Persist the decision as ``release_decision.json`` and record the artefact.

    Written before ``key_findings.json`` so the reasoning is on disk whether or not
    the findings that follow from it are. Kept separate from the findings file
    because it answers a different question — *why* this fit published what it did,
    for every family, rather than only for the ones the robustness gate covers.
    """
    from language_reading_predictors.statistical_models.artifacts import record_artifact

    record = evaluation.as_dict()
    path = os.path.join(ctx.output_dir, RELEASE_DECISION_FILENAME)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(record, handle, indent=2)
        handle.write("\n")
    record_artifact(
        ctx,
        "release_decision",
        filename=RELEASE_DECISION_FILENAME,
        kind="json",
    )
    return record
