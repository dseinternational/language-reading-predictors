# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""The ordered publication decision for one fit.

``inputs`` -> ``computation`` -> ``artifacts`` -> ``robustness``: the first
stage to object settles it. This is the only module that reads the checks;
they do not read each other.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Mapping
from language_reading_predictors.statistical_models.convergence import (
    convergence_gate_failures,
)
from language_reading_predictors.statistical_models.release.base import (
    PublicationStatus,
    RELEASE_DECISION_FILENAME,
    ReleaseStage,
    _config_name,
    _load_config,
    _read_csv,
    _read_json,
)
from language_reading_predictors.statistical_models.release.robustness import (
    ReleaseDecision,
    evaluate_release,
    gate_applies,
)
from language_reading_predictors.statistical_models.release.blending import (
    _blending_pair_release_failures,
    _joint_blending_scope_note,
)
from language_reading_predictors.statistical_models.release.family_checks import (
    _adjusted_ses_release_failures,
    _concurrent_published_fit_release_failures,
    _gain_period1_release_failures,
    _growth_influence_release_failures,
    _itt_missingness_release_failures,
    _joint_mechanism_wave_release_failures,
    _mediation_t3_release_failures,
)
from language_reading_predictors.statistical_models.release.dependence import (
    _dependence_identification_note,
    _historical_joint_prior_companion_qualifications,
    _joint_dependence_companion_note,
)

#: Per-wave artefact names for the joint-mechanism levels design. Every published
#: wave carries the same three files, so the release check requires one uniform
#: bundle instead of special-casing the wave hosting the fit-level artefacts
#: (2026-08-23 joint-mechanism follow-up review, finding 1).
JOINT_MECHANISM_WAVE_TRACE = "trace_wave_t{timepoint}.nc"


JOINT_MECHANISM_WAVE_MARGINAL_PPC = "ppc_summary_marginal_t{timepoint}"


JOINT_MECHANISM_WAVE_PSENSE = "psense_wave_t{timepoint}"


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


_PUBLICATION_CONFIGS = frozenset({"rep-lite", "reporting"})


_DIAGNOSTIC_CONFIGS = frozenset({"dev", "test"})


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
