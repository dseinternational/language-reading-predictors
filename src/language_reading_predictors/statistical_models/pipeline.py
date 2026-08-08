# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Compatibility facade for the statistical-model fit pipelines.

This module defines no statistical code. Every family's orchestration lives in
its own module under :mod:`pipelines` — one per ``ModelSpec.kind``, each readable
top-to-bottom as a single fit: declaration, preparation, construction, inference,
scientific summaries, finalisation. The entry points are re-exported here so the
179 model modules, the fit script and the tests keep the import path they have
always used, until #394 step 8 migrates those callers and retires the facade.

``MIGRATED_FAMILIES`` in ``tests/statistical_models/test_pipeline_boundaries.py``
is the authoritative list of what lives where, and it is checked against the
contents of the :mod:`pipelines` package rather than maintained by hand.

The shared layer the families are built on sits *below* them and must never
import this module, or the cycle that kept every family in one file returns:

- :mod:`runtime` — the stage binding and spec validation
- :mod:`stages` — the invariant primary-fit lifecycle
- :mod:`artifacts` — the single write-and-register table interface and manifest
- :mod:`publication` — banners, report-template copy, model graph
- :mod:`adjustment` — the record of what a fit actually conditioned on
- :mod:`prior_artifacts`, :mod:`ppc_artifacts`, :mod:`figure_artifacts`
- :mod:`diagnostics` — the samplers, convergence gate and diagnostic figures
- :mod:`reporting` — the posterior summaries, as pure functions
- :mod:`lcf_inference`, :mod:`lcf_summaries` — the correlated-factor algorithms

Every fit, whichever family it belongs to, loads its rows through
:mod:`preprocessing`, builds its PyMC model through :mod:`factories`, runs prior
predictive → sampling → optional LOO → diagnostics → posterior predictive →
convergence gate → trace persistence, and writes ``config.json``,
``diagnostics_summary.json``, ``priors_table.csv``, ``artifact_manifest.json``,
its family tables and the standard diagnostic plots to
``output/statistical_models/models/{model_id}-{config}/``, alongside a copy of
``docs/models/{model_id}/index.qmd`` and the shared partials so the Quarto report
renders in place.
"""

from __future__ import annotations

# Compatibility re-exports. The ``x as x`` form marks them as deliberate
# re-exports rather than unused imports, and keeps
# ``from ...pipeline import fit_itt`` working for every caller until step 8.
from language_reading_predictors.statistical_models.pipelines.adjusted import (
    fit_adjusted as fit_adjusted,
    fit_rlm_adjusted as fit_rlm_adjusted,
)
from language_reading_predictors.statistical_models.pipelines.aligned import (
    fit_aligned as fit_aligned,
)
from language_reading_predictors.statistical_models.pipelines.block_exposure import (
    fit_block_exposure as fit_block_exposure,
)
from language_reading_predictors.statistical_models.pipelines.concurrent import (
    fit_concurrent as fit_concurrent,
)
from language_reading_predictors.statistical_models.pipelines.corr_factor import (
    fit_correlated_factor as fit_correlated_factor,
    fit_rlm_corr_factor as fit_rlm_corr_factor,
)
from language_reading_predictors.statistical_models.pipelines.did import (
    fit_did as fit_did,
)
from language_reading_predictors.statistical_models.pipelines.dose_response import (
    fit_dose_response as fit_dose_response,
)
from language_reading_predictors.statistical_models.pipelines.gain_factors import (
    fit_gain_factors as fit_gain_factors,
)
from language_reading_predictors.statistical_models.pipelines.growth import (
    fit_growth as fit_growth,
)
from language_reading_predictors.statistical_models.pipelines.historical_growth import (
    fit_historical_growth as fit_historical_growth,
)
from language_reading_predictors.statistical_models.pipelines.historical_joint import (
    fit_rlm_joint_growth as fit_rlm_joint_growth,
)
from language_reading_predictors.statistical_models.pipelines.horseshoe import (
    fit_horseshoe as fit_horseshoe,
    fit_rlm_horseshoe as fit_rlm_horseshoe,
)
from language_reading_predictors.statistical_models.pipelines.itt import (
    fit_itt as fit_itt,
)
from language_reading_predictors.statistical_models.pipelines.joint import (
    fit_joint as fit_joint,
)
from language_reading_predictors.statistical_models.pipelines.joint_mechanism import (
    fit_joint_mechanism as fit_joint_mechanism,
)
from language_reading_predictors.statistical_models.pipelines.lcsm import (
    fit_lcsm as fit_lcsm,
)
from language_reading_predictors.statistical_models.pipelines.level_factors import (
    fit_level_factors as fit_level_factors,
)
from language_reading_predictors.statistical_models.pipelines.long_corr_factor import (
    fit_longitudinal_corr_factor as fit_longitudinal_corr_factor,
)
from language_reading_predictors.statistical_models.pipelines.mechanism import (
    fit_mechanism as fit_mechanism,
)
from language_reading_predictors.statistical_models.pipelines.mediation import (
    fit_mediation as fit_mediation,
    fit_mediation_multi as fit_mediation_multi,
    fit_mediation_period_stacked as fit_mediation_period_stacked,
    prepare_mediation_data as prepare_mediation_data,
)
from language_reading_predictors.statistical_models.pipelines.survival import (
    fit_survival as fit_survival,
)
