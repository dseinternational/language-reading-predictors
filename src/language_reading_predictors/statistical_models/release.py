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
``level_factors`` (``b_grp_time[1]``, the t2 element — the only randomised one).

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

The floor-grid requirement stays ITT-only. It is bound to the registered six-cell
grid and to :func:`sensitivity.evaluate_floor_sensitivity`'s provenance machinery,
neither of which was specified for ``gain_factors``' off-floor models; those are gated
on their ``beta_trt`` prior dependence alone. Both of them (``gf-005``, ``gf-011``)
resolve as prior-dominant qualifies on that route today, so nothing is currently
under-gated — but a *clear* off-floor gain-factor fit would release without the grid
its ITT counterpart needs, which is a gap to close rather than a decision.

**Tiering.** The policy applies uniformly across base ITT models, adjusted-robustness
models and outcomes outside the standard 44-cell sweep. That was the default offered
alongside option A; the graded alternative discussed earlier on #392 (withhold for
primary, qualify-never-withhold for adjusted robustness) is a one-line change to
:data:`_WITHHOLD_TIERS` if it is preferred.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
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
    """
    if config.get("kind") not in GATED_KINDS:
        return False
    plan = config.get("resolved_run_plan") or {}
    return not (
        config.get("kind") == "gain_factors"
        and (
            bool(plan.get("treated_only", False))
            or bool(plan.get("moderation_variant", False))
        )
    )


def causal_term_for(config: Mapping[str, Any]) -> str:
    """The psense row this fit's release decision turns on.

    ``level_factors`` fits one ``b_grp_time`` coefficient per timepoint and only the
    t2 element is randomised (#389 finding 1), so the gate names that element rather
    than the vector. Reading the bare name instead returns "unavailable" for all
    eleven fits — a gate that withholds every level-factor headline for a diagnosis
    that is present and sitting one row away.

    The ``did`` dose models have no ``tau_t2`` at all: their focal quantity is the
    dose slope. The choice mirrors ``DiDRunPlan.effect_term`` and is read from the
    persisted plan rather than re-derived from ``spec.extra``, so a fit's release
    decision and its own psense emission cannot disagree about which term matters.
    """
    kind = config.get("kind")
    if kind == "gain_factors":
        return "beta_trt"
    if kind == "level_factors":
        return "b_grp_time[1]"
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


def _standard_sweep_evidence(output_dir: str | Path, outcome: str) -> tuple[bool, str]:
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
    - the sign of ``tau_logit_mean`` is the same in every cell.

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

    signs = np.sign(
        pd.to_numeric(rows["tau_logit_mean"], errors="coerce").to_numpy(dtype=float)
    )
    if not np.isfinite(signs).all() or len(set(signs.tolist())) != 1:
        return False, (
            "the effect changes sign across the attached treatment-prior sweep, so "
            "its direction is not stable under the prior"
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
    return ReleaseDecision(status="release", **common)


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
    # registered ITT floor rule. ``gain_factors``' off-floor models take the ordinary
    # route on ``beta_trt`` — see this module's docstring for why, and for the gap
    # that leaves if one of them ever comes back clear.
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
        output_dir, str(config.get("outcome_symbol") or "")
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
    """
    from language_reading_predictors.statistical_models.sensitivity import (
        STANDARD_SENSITIVITY_OUTCOMES,
    )

    plan = config.get("resolved_run_plan") or {}
    if plan.get("adjust_for") or plan.get("adjustment"):
        return "adjusted_robustness"
    if str(config.get("outcome_symbol") or "") not in STANDARD_SENSITIVITY_OUTCOMES:
        return "off_grid"
    return "primary"


def _load_config(output_dir: Path) -> dict[str, Any] | None:
    path = output_dir / "config.json"
    if not os.path.exists(path):
        return None
    try:
        with open(path) as handle:
            loaded = json.load(handle)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    return loaded if isinstance(loaded, dict) else None


# ---------------------------------------------------------------------------
# The publication decision (#394 design point 3)
# ---------------------------------------------------------------------------

RELEASE_DECISION_FILENAME = "release_decision.json"

PublicationStatus = Literal[
    "ok", "not_available", "gate_failed", "artifacts_incomplete", "robustness_unresolved"
]
"""What a fit is permitted to publish, in the vocabulary ``key_findings.json`` uses."""

ReleaseStage = Literal["inputs", "computation", "artifacts", "robustness"]
"""Which stage of the decision settled it.

``inputs`` the fit's own summary files are missing or unreadable; ``computation``
the sampling-quality gate failed; ``artifacts`` a required output is not on disk;
``robustness`` the treatment-effect sensitivity evidence does not support a causal
headline. The order is the order below: a fit that did not converge is not asked
whether its prior sensitivity is acceptable.
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
    #: Required artefacts recorded by the fit but absent from its directory.
    missing_artifacts: tuple[str, ...] = ()
    #: The robustness verdict, when the fit is in scope for that gate.
    robustness: ReleaseDecision | None = None
    #: The fit's ``config.json``, loaded once so callers need not re-read it.
    #: ``None`` when it is unreadable, ``{}`` when it is absent.
    config: Mapping[str, Any] | None = None

    @property
    def publishable(self) -> bool:
        """May this fit's key-findings box carry scientific sentences?"""
        return self.status == "ok"

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
        }
        if self.reason:
            record["reason"] = self.reason
        if self.failing_checks:
            record["failing_checks"] = list(self.failing_checks)
        if self.missing_artifacts:
            record["missing_artifacts"] = list(self.missing_artifacts)
        if self.robustness is not None:
            record["robustness"] = self.robustness.as_dict()
        if self.config:
            record["model_id"] = self.config.get("model_id")
            record["kind"] = self.config.get("kind")
        return record

    def summary(self) -> str:
        """One line for the console at finalisation."""
        if self.publishable:
            return "ok" + (" (with note)" if self.note else "")
        return f"{self.status} at the {self.stage} stage: {self.reason}"


def _read_json(path: str | Path) -> tuple[Any, str | None]:
    """``(payload, error)`` — ``error`` names why the payload is unusable."""
    if not os.path.exists(path):
        return None, "missing"
    try:
        with open(path) as handle:
            return json.load(handle), None
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None, "unreadable"


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
    """
    records = getattr(artifacts, "records", None)
    if records is not None:
        declared = [
            (rec.filename, rec.status, bool(rec.required)) for rec in records.values()
        ]
    else:
        manifest, _err = _read_json(output_dir / "artifact_manifest.json")
        entries = (manifest or {}).get("artifacts") if isinstance(manifest, dict) else None
        if not entries:
            return ()
        declared = [
            (str(e.get("filename")), str(e.get("status")), bool(e.get("required")))
            for e in entries
        ]
    missing = [
        filename
        for filename, status, required in declared
        if required
        and status in ("written", "missing")
        and not os.path.exists(output_dir / filename)
    ]
    return tuple(sorted(missing))


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
       present and readable. The diagnostics file is checked first because the
       sampling-quality gate outranks everything: findings from an unconverged fit
       must not reach a reader even if every other artefact is perfect.
    2. **computation** — the automatic sampling-quality gate must pass cleanly.
    3. **artifacts** — every artefact the fit recorded as *required* must be on
       disk. A required output that vanished between its write and finalisation is
       a withheld release, not a warning (#394 design point 3).
    4. **robustness** — for the families the treatment-effect gate covers, the
       prior-sensitivity and floor-grid evidence must support a causal headline.

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
        )

    missing = _recorded_required_artifacts(output_dir, artifacts)
    if missing:
        return ReleaseEvaluation(
            status="artifacts_incomplete",
            stage="artifacts",
            reason=(
                "the fit recorded required artefacts that are not in its output "
                f"directory: {', '.join(missing)}"
            ),
            missing_artifacts=missing,
            config=config,
        )

    robustness = _robustness_decision(output_dir, config)
    if robustness is not None and not robustness.released:
        return ReleaseEvaluation(
            status="robustness_unresolved",
            stage="robustness",
            reason=robustness.reason,
            robustness=robustness,
            config=config,
        )
    return ReleaseEvaluation(
        status="ok", stage="robustness", robustness=robustness, config=config
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
