# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""RLMADJ05 - confirmed-input predictors of verbal-memory gain (#409 D1).

This wider-outcome companion asks how wave-1 BAS word reading (``basread``),
receptive vocabulary (``bpvs``), receptive grammar (``trog``), verbal reasoning
(``bassim``) and age are associated with wave-1 to wave-3 BAS digit-recall gain
after mutual adjustment and conditioning on wave-1 digit recall. All five
measurement inputs have confirmed instrument identities and ceilings.

The pooled complete-case frame contains 71 children: 22 with Down syndrome,
30 average readers and 19 reading-matched children. Reading-matched children
were selected on ``basread`` level, so the word-reading slope and any other
coefficient conditional on it carry a design-selection caveat. Group dummies
are nuisance terms, every coefficient is associational, and neither the
baseline-conditioned model nor its bivariate and prior-width companions support
a causal reading.
"""

from language_reading_predictors.statistical_models.adjusted import (
    AdjustedModelSettings,
)
from language_reading_predictors.statistical_models.context import ModelSpec
from language_reading_predictors.statistical_models.pipelines.adjusted import (
    fit_rlm_adjusted,
)

SPEC = ModelSpec(
    model_id="lrp-rlm-adj-005",
    kind="adjusted",
    title=(
        "Byrne wave-1 predictors of verbal-memory gain, waves 1-3 "
        "(confirmed-input, mutually adjusted)"
    ),
    outcome_symbol="basdig",
    study_id="rlm",
    family="adjusted",
    design="historical_cohort",
    estimand_type="association",
    causal_status="none",
    dataset_ref="rlm:reading_language_memory_data_long",
    audit_baseline="complete_case_summary",
    model_settings=AdjustedModelSettings(
        predictor_measures=("basread", "bpvs", "trog", "bassim"),
        use_age_predictor=True,
        pre_wave=1,
        post_wave=3,
        require_confirmed_inputs=True,
        predictor_slope_sigma=0.3,
        prior_sensitivity_sigmas=(0.5, 0.7),
    ),
)


def fit(config: str = "dev"):
    return fit_rlm_adjusted(SPEC, config=config)
