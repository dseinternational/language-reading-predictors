# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Shared fit context for the statistical-model pipelines."""

from __future__ import annotations

import os
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import arviz as az
import pandas as pd
import pymc as pm
import xarray as xr
from rich import print as rprint

import dse_research_utils.statistics.models.reporting as _reporting
import dse_research_utils.statistics.models.sampling as _sampling

from language_reading_predictors import paths as _paths
from language_reading_predictors.statistical_models import environment as _env
from language_reading_predictors.statistical_models.artifacts import ArtifactLog
from language_reading_predictors.statistical_models.subfits import SubfitLog
from language_reading_predictors.statistical_models.preprocessing import (
    LongitudinalPanel,
    PreparedData,
    WavePanel,
)
from language_reading_predictors.statistical_models.run_options import (
    StatisticalRunOptions,
    current_run_options,
)
from language_reading_predictors.statistical_models.output_transaction import (
    OutputTransaction,
)


@dataclass
class ModelSpec:
    """Description of a single model run - lives on the context.

    ``model_id`` is ``"lrp-rli-itt-007"`` etc. ``kind`` is one of the model families in
    ``definitions.KINDS`` — the headline estimands ``"itt"``, ``"joint"``,
    ``"mechanism"``, ``"mediation"``, ``"did"`` (waitlist-crossover),
    ``"gain_factors"`` / ``"level_factors"`` (DAG-focused factor families) and
    ``"aligned"`` (onset-aligned per-protocol single gain), plus the association /
    cross-check / reproduction families ``"adjusted"``, ``"corr_factor"``,
    ``"dose_response"``, ``"lcsm"``, ``"mediation_multi"``, ``"horseshoe"``,
    ``"growth"``, ``"historical_growth"``, ``"historical_joint"``, ``"survival"``,
    ``"block_exposure"``, ``"concurrent"`` and ``"long_corr_factor"``. ``title``
    is the long human-readable title shown on the report. ``model_settings`` is the
    typed family boundary for migrated families; ``extra`` remains the strict legacy
    translation boundary for those families and the migration boundary for the rest.
    """

    model_id: str
    kind: str
    title: str
    outcome_symbol: str | None = None
    """For ITT / mechanism models, the target outcome symbol (``"W"`` etc.)."""
    mechanism_symbol: str | None = None
    """For mechanism models, the mechanism variable symbol."""
    adjustment: list[str] = field(default_factory=list)
    """For mechanism models, the list of adjustment-set symbols."""
    target_accept: float | None = None
    """Model-specific NUTS ``target_accept`` default, or ``None`` for the preset.

    A first-class field because it is a *sampler* knob, not part of any scientific
    recipe, and because the legacy ``extra["target_accept"]`` route is unreachable
    from a typed module: families that have migrated reject any non-empty ``extra``
    beside ``model_settings``, so the documented "model-specific default" tier of
    the precedence could only ever be used by an unmigrated spec (2026-08-20 ITT
    code review finding 5; 2026-08-22 ITT audit, finding 9). Read through
    :func:`spec_target_accept`, which still honours the legacy key.
    """
    model_settings: object | None = None
    """Typed, immutable settings for a family that has completed this migration."""
    extra: dict[str, Any] = field(default_factory=dict)

    # --- Dataset / estimand metadata (#165) -------------------------------
    # Optional and behaviour-preserving: the existing intervention models leave
    # these at their defaults, so their config.json only gains new keys. They let
    # reports state which study a model is fit on and whether it is causal.
    study_id: str = "rli"
    """Dataset / cohort this model is fit on (default the RLI intervention study)."""
    family: str | None = None
    """Model-family grouping (e.g. ``"itt"``, ``"historical_growth"``)."""
    design: str | None = None
    """Study design / estimand identifier for report transparency."""
    estimand_type: str | None = None
    """What is estimated: ``"causal"`` / ``"descriptive"`` / ``"association"`` / ..."""
    causal_status: str | None = None
    """Causal warrant: ``"randomised"`` / ``"adjusted"`` / ``"none"`` / ..."""
    dataset_ref: str | None = None
    """Explicit data reference when multi-source (e.g. ``"rlm:..._long"``)."""
    audit_baseline: str | None = None
    """Reproduction / audit baseline this model checks against, if any."""

    def __post_init__(self) -> None:
        """Fill the shared RLI ITT audit metadata when a spec omits it.

        The trial randomised 57 children and analysed 54 after three losses to
        follow-up. The repository contains those 54, including four children who
        discontinued intervention but were followed. The randomised arm coefficient
        is therefore an available-case modified ITT estimate. A causal interpretation
        for the fitted population requires an ignorable-selection assumption; the
        estimate must not be labelled as a full-57 ITT estimate. Centralising these defaults prevents
        the registered ITT/joint modules from drifting in their saved metadata.
        """

        if self.kind not in {"itt", "joint"}:
            return
        if self.family is None:
            self.family = "itt"
        if self.design is None:
            self.design = "waitlist_randomised_t1_to_t2_available_case_modified_itt"
        if self.estimand_type is None:
            self.estimand_type = "available_case_modified_itt_estimate"
        if self.causal_status is None:
            self.causal_status = "randomised_assignment_conditional_on_observed_analysis_set"
        if self.dataset_ref is None:
            self.dataset_ref = (
                "rli:rli_data_long.csv; 54 analysed after 3 losses to follow-up "
                "from 57 randomised"
            )

    @property
    def banner(self) -> str:
        return f"{self.model_id.upper()}: {self.title}"

    # --- Canonical model-ID scheme (#168) ---------------------------------
    # Since Phase 2 ``model_id`` is the *canonical* id (``lrp-rli-itt-010``); this
    # accessor also still parses a legacy id (``lrpitt10`` + ``kind``/``study_id``)
    # so the canonical/legacy/family metadata is correct whichever form a spec
    # uses. An id the resolver cannot parse yields ``None`` rather than breaking a fit.
    @property
    def _canonical(self):
        from language_reading_predictors import model_ids as _mids

        try:
            if _mids.looks_canonical(self.model_id):
                return _mids.parse_canonical(self.model_id)
            return _mids.parse_legacy(
                self.model_id, kind=self.kind, study=self.study_id
            )
        except _mids.ModelIdError:
            return None

    @property
    def legacy_model_id(self) -> str:
        c = self._canonical
        return c.legacy if c is not None else self.model_id

    @property
    def canonical_model_id(self) -> str | None:
        c = self._canonical
        return c.cli if c is not None else None

    @property
    def project_code(self) -> str | None:
        c = self._canonical
        return c.project.upper() if c is not None else None

    @property
    def study_code(self) -> str | None:
        c = self._canonical
        return c.study.upper() if c is not None else None

    @property
    def family_code(self) -> str | None:
        c = self._canonical
        return c.family.upper() if c is not None else None

    @property
    def variant_role(self) -> str | None:
        c = self._canonical
        return c.variant_role if c is not None else None

    @property
    def parent_model_id(self) -> str | None:
        from language_reading_predictors.model_ids import ModelId

        c = self._canonical
        if c is None or not c.suffix:
            return None
        return ModelId(c.project, c.study, c.family, c.number, None).legacy


@dataclass
class StatisticalFitContext:
    spec: ModelSpec
    reporting: _reporting.ReportingConfiguration
    sampling: _sampling.SamplingConfiguration
    run_options: StatisticalRunOptions = field(default_factory=StatisticalRunOptions)
    prepared: PreparedData | WavePanel | LongitudinalPanel | None = None
    model: pm.Model | None = None
    prior_samples: xr.DataTree | None = None
    trace: xr.DataTree | None = None
    loo: az.ELPDData | None = None
    tables: dict[str, pd.DataFrame] = field(default_factory=dict)
    artifacts: ArtifactLog = field(default_factory=ArtifactLog)
    """Per-fit artefact record consumed by the manifest at finalisation (#394)."""
    subfits: SubfitLog = field(default_factory=SubfitLog)
    """Per-fit record of every secondary / sensitivity sub-fit (#394 design point 5)."""
    resolved_plan: Any | None = None
    """Validated family run plan resolved before data loading."""
    output_transaction: OutputTransaction | None = None
    """Hidden staging directory promoted only after every fit stage succeeds."""
    lifecycle_stages: list[str] = field(default_factory=list)
    """Stages :meth:`SharedFitStages.run_primary_fit` actually ran, in order.

    The lifecycle's own record of itself (#637 stage 4). Before this, the only way
    to check that a family ran power scaling once, in the declared slot, was to
    read its source — and six families ran it *outside* the runner entirely, so
    there was nothing to read but a convention.
    """

    @property
    def output_dir(self) -> str:
        if self.output_transaction is None:
            return self.reporting.output_dir
        return str(self.output_transaction.output_dir)

    @property
    def final_output_dir(self) -> str:
        """Stable publication path, whether or not this run has been promoted."""
        return self.reporting.output_dir

    def ensure_output_dir(self) -> None:
        os.makedirs(self.output_dir, exist_ok=True)

    def reset_output_dir(self) -> None:
        """Start a fresh output transaction while preserving the last publication.

        The working directory is a hidden sibling of ``final_output_dir``. Every
        artefact is regenerated there, which removes the stale-file hazard without
        deleting the previous successful fit before the replacement is ready.
        """
        if self.output_transaction is not None:
            self.output_transaction.abandon()
        self.output_transaction = OutputTransaction.create(
            Path(self.final_output_dir)
        )

    def publish_output_dir(self) -> str:
        """Promote this run with an atomic same-filesystem staging rename."""
        if self.output_transaction is None:
            return self.final_output_dir
        return str(self.output_transaction.publish())

    def abandon_output_dir(self) -> None:
        """Discard this run's unpublished staging data."""
        if self.output_transaction is not None:
            self.output_transaction.abandon()


def spec_target_accept(spec: ModelSpec) -> float | None:
    """Return the validated model-specific sampler default, if declared.

    ``target_accept`` is a cross-family sampling option rather than part of any
    scientific model recipe. Keeping its sole read here makes that distinction
    explicit for fit pipelines and standalone audit runners alike.

    Prefers the first-class :attr:`ModelSpec.target_accept` and falls back to the
    legacy ``extra["target_accept"]``, so unmigrated specs keep working while a
    typed module can finally declare it — until this field existed the documented
    "model-specific default" tier was reachable only from a legacy spec, because
    a migrated family rejects any non-empty ``extra`` beside ``model_settings``.
    Declaring both and disagreeing is a contradiction rather than a precedence
    question, so it is rejected.
    """
    typed = getattr(spec, "target_accept", None)
    legacy = spec.extra.get("target_accept")
    if typed is not None and legacy is not None and float(typed) != float(legacy):
        raise ValueError(
            f"{spec.model_id}: spec.target_accept ({typed!r}) and "
            f"spec.extra['target_accept'] ({legacy!r}) disagree; declare one"
        )
    target_accept = typed if typed is not None else legacy
    if target_accept is None:
        return None
    source = "spec.target_accept" if typed is not None else "spec.extra['target_accept']"
    target_accept = float(target_accept)
    if not 0.0 < target_accept < 1.0:
        raise ValueError(
            f"{source} must be in the open interval (0, 1); got {target_accept!r}"
        )
    return target_accept


def _resolve_target_accept(spec: ModelSpec, sampling, run_options):
    """Resolve NUTS ``target_accept`` with explicit precedence, for **every** family.

    Precedence is **CLI override > model-specific default > config preset**.

    Some models (the horseshoe's global-local funnel, the small-n correlated-factor
    CFA, the HSGP mechanism surfaces) need a higher ``target_accept`` than their tier
    preset gives, and declare it in ``spec.extra``. That must not silently outrank an
    explicit ``--target-accept`` from the command line: an earlier
    ``max(preset_or_cli, spec_value)`` meant a deliberate ``--target-accept 0.95`` was
    replaced by a spec's 0.999, so a diagnostic reproduction or an ablation silently
    did not run at the requested setting.

    This lives in the shared context factory rather than in per-family fit functions.
    It was previously applied by ``pipeline._apply_spec_target_accept``, which only six
    of the family entry points called — so a ``spec.extra["target_accept"]`` added to
    any other family (dose-response, DiD, ITT, …) would have been accepted by the spec
    and then silently ignored at sampling time. Resolving it here means a declaration
    is honoured wherever it is made.
    """
    target_accept = spec_target_accept(spec)
    if run_options.target_accept is not None:
        if target_accept is not None:
            rprint(
                "[yellow]Keeping the CLI --target-accept "
                f"({run_options.target_accept}) over {spec.model_id}'s "
                f"spec default ({target_accept}).[/yellow]"
            )
        return replace(sampling, target_accept=run_options.target_accept)
    if target_accept is not None:
        return replace(sampling, target_accept=target_accept)
    return sampling


def make_context(
    spec: ModelSpec,
    config: str = "dev",
    *,
    ci_prob: float = 0.89,
    random_seed: int = 47,
) -> StatisticalFitContext:
    # Reported credible-interval standard (2026-07-17,
    # ``notes/…-credible-interval-standard.md``): the posterior **median** plus an
    # **inner 50 %** and **outer 89 %** equal-tailed interval, alongside the full
    # posterior. ``ci_prob=0.89`` is the deliberately non-round outer coverage —
    # 95 % is an arbitrary convention imported from frequentist NHST, and at this
    # suite's ESS its 2.5/97.5 % limits are the noisiest quantiles to estimate per
    # effective draw, whereas 89 % (5.5/94.5 %) is markedly more MCMC-stable (Kruschke
    # 2021 BARG, doi:10.1038/s41562-021-01177-7: "For reasonably stable estimates of
    # limits of highest-density intervals (HDIs), I recommend that ESS ≥ 10,000. For
    # stable estimates of limits of equal-tailed intervals, ESS can be lower."). This
    # is a per-effective-draw efficiency point, not low attained ESS — the headline
    # terms reach a Tail-ESS in the low tens of thousands (see the ESS reporting
    # standard, notes/202607181200-ess-reporting-standard.md). The HPDI is kept as a
    # separate per-scale sensitivity companion (dse_research_utils intervals.hdi_1d,
    # #170); ``interval_kind="eti"`` records the equal-tailed convention on the
    # shared ReportingConfiguration so the tables, plots, and diagnostics summary agree.
    # Apply the shared matplotlib house style for every fit path (CLI, notebook,
    # tests, replot) — the CLI also does this via setup.init_script(); idempotent.
    _env.init_plotting()

    reporting = _reporting.ReportingConfiguration(
        model_name=spec.model_id,
        config_name=config,
        output_root_dir=str(_paths.stat_dir()),
        ci_prob=ci_prob,
        interval_kind="eti",
    )
    run_options = current_run_options()
    sampling = _sampling.get_sampling_configuration(config, random_seed=random_seed)
    sampling = _resolve_target_accept(spec, sampling, run_options)
    ctx = StatisticalFitContext(
        spec=spec,
        reporting=reporting,
        sampling=sampling,
        run_options=run_options,
    )
    # Regenerate every artefact in a hidden sibling directory. The shared final
    # stage promotes it only after the fit and report complete successfully.
    ctx.reset_output_dir()
    return ctx
