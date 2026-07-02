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

from language_reading_predictors.statistical_models import environment as _env
from language_reading_predictors.statistical_models.preprocessing import (
    PreparedData,
    WavePanel,
)


@dataclass
class ModelSpec:
    """Description of a single model run - lives on the context.

    ``model_id`` is ``"lrpitt07"`` etc. ``kind`` is ``"itt"``, ``"joint"``,
    ``"mechanism"``, ``"mediation"``, ``"did"`` (waitlist-crossover),
    ``"gain_factors"`` / ``"level_factors"`` (DAG-focused factor families) or
    ``"aligned"`` (onset-aligned per-protocol single gain). ``title`` is the long
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

    @property
    def banner(self) -> str:
        return f"{self.model_id.upper()}: {self.title}"


@dataclass
class StatisticalFitContext:
    spec: ModelSpec
    reporting: _reporting.ReportingConfiguration
    sampling: _sampling.SamplingConfiguration
    prepared: PreparedData | WavePanel | None = None
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
        output_root_dir=_env.STAT_OUTPUT_DIR,
        hdi=ci_prob,
    )
    sampling = _sampling.get_sampling_configuration(config, random_seed=random_seed)
    ctx = StatisticalFitContext(spec=spec, reporting=reporting, sampling=sampling)
    ctx.ensure_output_dir()
    return ctx
