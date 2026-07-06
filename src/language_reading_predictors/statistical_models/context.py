# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Shared fit context for the statistical-model pipelines."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

import arviz as az
import pandas as pd
import pymc as pm
import xarray as xr

import dse_research_utils.statistics.models.reporting as _reporting
import dse_research_utils.statistics.models.sampling as _sampling

from language_reading_predictors import paths as _paths
from language_reading_predictors.statistical_models.preprocessing import (
    LongitudinalPanel,
    PreparedData,
    WavePanel,
)


@dataclass
class ModelSpec:
    """Description of a single model run - lives on the context.

    ``model_id`` is ``"lrpitt07"`` etc. ``kind`` is one of the model families in
    ``definitions.KINDS`` — the headline estimands ``"itt"``, ``"joint"``,
    ``"mechanism"``, ``"mediation"``, ``"did"`` (waitlist-crossover),
    ``"gain_factors"`` / ``"level_factors"`` (DAG-focused factor families) and
    ``"aligned"`` (onset-aligned per-protocol single gain), plus the association /
    cross-check / reproduction families ``"adjusted"``, ``"corr_factor"``,
    ``"dose_response"``, ``"lcsm"``, ``"mediation_multi"``, ``"horseshoe"``,
    ``"growth"`` and ``"historical_growth"``. ``title`` is the long
    human-readable title shown on the report. ``extra`` is a free-form dict of
    model-specific settings that the pipeline passes to the factory.
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

    @property
    def banner(self) -> str:
        return f"{self.model_id.upper()}: {self.title}"

    # --- Canonical model-ID scheme (#168 Phase 1) -------------------------
    # Non-destructive: derived on read from the legacy ``model_id`` + ``kind`` +
    # ``study_id`` via the shared resolver, so nothing is renamed and an id the
    # resolver cannot parse simply yields ``None`` rather than breaking a fit.
    @property
    def _canonical(self):
        from language_reading_predictors import model_ids as _mids

        try:
            return _mids.parse_legacy(
                self.model_id, kind=self.kind, study=self.study_id
            )
        except _mids.ModelIdError:
            return None

    @property
    def legacy_model_id(self) -> str:
        return self.model_id

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
    prepared: PreparedData | WavePanel | LongitudinalPanel | None = None
    model: pm.Model | None = None
    model_vars: dict[str, Any] | None = None
    prior_samples: xr.DataTree | None = None
    trace: xr.DataTree | None = None
    loo: az.ELPDData | None = None
    tables: dict[str, pd.DataFrame] = field(default_factory=dict)

    @property
    def output_dir(self) -> str:
        return self.reporting.output_dir

    def ensure_output_dir(self) -> None:
        os.makedirs(self.output_dir, exist_ok=True)

    def reset_output_dir(self) -> None:
        """Clear the (per-config) output dir so a re-fit cannot leave stale artefacts.

        The output dir is ``{model_id}-{config}`` (one directory per model×config),
        and every artefact — ``trace.nc``, ``config.json``, all CSVs, plots, and the
        copied ``index.qmd`` / ``_partials`` — is regenerated by the fit. Without
        this, an artefact a re-specified model *stops* emitting (e.g.
        ``rope_summary.csv`` if a δ is withdrawn, ``interaction_summary.csv`` if a
        moderator is dropped) survives from the previous run and the report partials
        render it as if current. Clearing up front removes that hazard.
        """
        import shutil

        if os.path.isdir(self.output_dir):
            for entry in os.scandir(self.output_dir):
                try:
                    if entry.is_dir(follow_symlinks=False):
                        shutil.rmtree(entry.path)
                    else:
                        os.remove(entry.path)
                except OSError:
                    pass
        os.makedirs(self.output_dir, exist_ok=True)


def make_context(
    spec: ModelSpec,
    config: str = "dev",
    *,
    ci_prob: float = 0.95,
    random_seed: int = 47,
) -> StatisticalFitContext:
    # ``ReportingConfiguration.hdi`` (dse_research_utils) stores the interval
    # *coverage* probability, not a highest-density-interval flag: the suite reads
    # it back as ``ci_prob`` and reports equal-tailed intervals, with the HPDI kept
    # as a separate per-scale sensitivity companion (see reporting._hdi_bounds,
    # #170). The external field name is retained for cross-repo compatibility.
    reporting = _reporting.ReportingConfiguration(
        model_name=spec.model_id,
        config_name=config,
        output_root_dir=str(_paths.stat_dir()),
        hdi=ci_prob,
    )
    sampling = _sampling.get_sampling_configuration(config, random_seed=random_seed)
    ctx = StatisticalFitContext(spec=spec, reporting=reporting, sampling=sampling)
    # Clear any artefacts from a previous fit of this model×config so a re-fit that
    # stops emitting a file cannot leave a stale copy for the report to render.
    ctx.reset_output_dir()
    return ctx
