# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Shared fit context for the statistical-model pipelines."""

from __future__ import annotations

from language_reading_predictors.statistical_models.posteriors import REPORTING_CI_PROB


from language_reading_predictors.statistical_models.run_plans import ResolvedRunPlan

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
    """Scientific specification and report metadata for one model.

    ``model_id`` names the registered model; ``kind`` selects a family from
    ``definitions.KINDS``. ``model_settings`` holds its immutable typed settings.
    ``extra`` supports legacy specifications and must be empty for registered
    typed models.
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

    This sampling option sits outside the scientific settings. Read it through
    :func:`spec_target_accept`, which also accepts the legacy ``extra`` key.
    """
    model_settings: object | None = None
    """Typed, immutable settings for a family that has completed this migration."""
    extra: dict[str, Any] = field(default_factory=dict)

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
            self.dataset_ref = "rli:rli_data_long.csv; 54 analysed after 3 losses to follow-up from 57 randomised"

    @property
    def banner(self) -> str:
        return f"{self.model_id.upper()}: {self.title}"

    @property
    def _canonical(self):
        """Parse canonical or legacy IDs; return None for an unrecognised ID."""
        from language_reading_predictors import model_ids as _mids

        try:
            if _mids.looks_canonical(self.model_id):
                return _mids.parse_canonical(self.model_id)
            return _mids.parse_legacy(self.model_id, kind=self.kind, study=self.study_id)
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
    """Per-fit artefact record consumed by the manifest at finalisation."""
    subfits: SubfitLog = field(default_factory=SubfitLog)
    """Per-fit record of secondary and sensitivity sub-fits."""
    resolved_plan: ResolvedRunPlan | None = None
    """Validated family run plan resolved before data loading."""
    effective_plan: ResolvedRunPlan | None = None
    """Plan after fitted-data restrictions; the declared plan remains unchanged."""
    output_transaction: OutputTransaction | None = None
    """Hidden staging directory promoted only after every fit stage succeeds."""
    lifecycle_stages: list[str] = field(default_factory=list)
    """Stages executed by ``stages.run_primary_fit``, in order."""

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
        self.output_transaction = OutputTransaction.create(Path(self.final_output_dir))

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

    Prefer :attr:`ModelSpec.target_accept`, with the legacy
    ``extra["target_accept"]`` as fallback. Reject conflicting declarations.
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
        raise ValueError(f"{source} must be in the open interval (0, 1); got {target_accept!r}")
    return target_accept


def _resolve_target_accept(
    spec: ModelSpec,
    sampling: _sampling.SamplingConfiguration,
    run_options: StatisticalRunOptions,
) -> _sampling.SamplingConfiguration:
    """Use the CLI override, then the model default, then the sampling preset."""
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


def resolve_sampling_configuration(
    spec: ModelSpec,
    config: str,
    *,
    run_options: StatisticalRunOptions | None = None,
    random_seed: int = 47,
) -> _sampling.SamplingConfiguration:
    """Resolve sampler settings without creating outputs or a fit context.

    Fits and sweep resumption use this same resolver so a requested override,
    model default or preset cannot be ignored when deciding to reuse a fit.
    """
    sampling = _sampling.get_sampling_configuration(config, random_seed=random_seed)
    return _resolve_target_accept(spec, sampling, run_options or current_run_options())


def make_context(
    spec: ModelSpec,
    config: str = "dev",
    *,
    ci_prob: float = REPORTING_CI_PROB,
    random_seed: int = 47,
) -> StatisticalFitContext:
    # Posterior summaries use the project's 89% equal-tailed interval convention.
    _env.init_plotting()

    reporting = _reporting.ReportingConfiguration(
        model_name=spec.model_id,
        config_name=config,
        output_root_dir=str(_paths.stat_dir()),
        ci_prob=ci_prob,
        interval_kind="eti",
    )
    run_options = current_run_options()
    sampling = resolve_sampling_configuration(spec, config, run_options=run_options, random_seed=random_seed)
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
