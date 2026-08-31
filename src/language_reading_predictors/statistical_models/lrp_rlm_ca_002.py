# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""RLMCA02 - concurrent correlates of BPVS receptive vocabulary (#409 C1).

Pooled, per-wave observational regressions over Byrne waves 1-4 using only the
five measures whose denominators and instrument identities are confirmed. Group
membership is nuisance adjustment; wave 4 is an attrition-sensitive extension.
"""

from language_reading_predictors.statistical_models.concurrent import (
    ConcurrentModelSettings,
)
from language_reading_predictors.statistical_models.context import (
    ModelSpec,
    StatisticalFitContext,
)
from language_reading_predictors.statistical_models.pipelines.concurrent import (
    fit_concurrent,
)

SPEC = ModelSpec(
    model_id="lrp-rlm-ca-002",
    kind="concurrent",
    title="Byrne concurrent correlates of BPVS receptive vocabulary, waves 1-4",
    outcome_symbol="bpvs",
    study_id="rlm",
    family="concurrent",
    design="historical_cohort_per_wave_available_case",
    estimand_type="association",
    causal_status="none",
    dataset_ref="rlm:reading_language_memory_data_long",
    model_settings=ConcurrentModelSettings(
        predictor_symbols=("basread", "trog", "basdig", "bassim"),
        waves=(1, 2, 3, 4),
        include_age=True,
        include_group=True,
        predictor_slope_sigma=0.3,
    ),
)


def fit(config: str = "dev") -> StatisticalFitContext:
    return fit_concurrent(SPEC, config=config)
