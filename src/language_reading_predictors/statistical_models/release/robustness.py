# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""The treatment-effect robustness gate.

Whether a causal headline survives its prior-sensitivity and floor-grid
evidence: the tau-sensitivity classification, the standard and floor sweeps,
and the ITT-specific verdict. Only the families ``gate_applies`` covers.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping
import numpy as np
import pandas as pd
from language_reading_predictors.statistical_models.sensitivity import (
    FLOOR_SENSITIVITY_FILENAME,
    STANDARD_SENSITIVITY_FILENAME,
    evaluate_floor_sensitivity,
    load_primary_floor_reference,
    tau_psense_status,
)
from language_reading_predictors.statistical_models.release.base import (
    GATED_KINDS,
    PSENSE_THRESHOLD,
    ReleaseStatus,
    TauSensitivityClass,
    _config_name,
    _finite,
    _load_config,
    _model_tier,
    _read_csv,
    _read_json,
)

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


#: Model tiers to which the withhold applies. Uniform by decision; narrowing this
#: set is how a graded policy would be expressed.
_WITHHOLD_TIERS = frozenset({"primary", "adjusted_robustness", "off_grid"})


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
