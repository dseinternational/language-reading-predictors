# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""RLMADJ06 - pooled annual predictors of word-reading progress (#409 D2).

This repeated-child extension of ``lrp-rlm-adj-001`` stacks the four consecutive
Byrne transitions (waves 1->2 through 4->5). BAS word reading at each post wave is
conditioned on its own pre-wave level, transition-specific intercepts, a child
random intercept, reading-group nuisance terms and pooled within-transition-
standardised BPVS, TROG, BAS digit-recall, BAS similarities and age slopes.

All measurement inputs have confirmed identities and ceilings. The final wave is
observed only for the Down syndrome cohort, so the primary four-transition fit is
paired with a common-horizon sensitivity through wave 4 and an explicitly
secondary transition-specific-slope fit. Every slope is an association; neither
the longitudinal ordering nor the random intercept supplies causal identification.
"""

from language_reading_predictors.statistical_models.adjusted import (
    AdjustedModelSettings,
)
from language_reading_predictors.statistical_models.context import (
    ModelSpec,
    StatisticalFitContext,
)
from language_reading_predictors.statistical_models.pipelines.adjusted import (
    fit_rlm_adjusted,
)

SPEC = ModelSpec(
    model_id="lrp-rlm-adj-006",
    kind="adjusted",
    title=(
        "Byrne annual predictors of word-reading progress, waves 1-5 "
        "(pooled transitions, confirmed inputs)"
    ),
    outcome_symbol="basread",
    study_id="rlm",
    family="adjusted",
    design="historical_stacked_transitions",
    estimand_type="association",
    causal_status="none",
    dataset_ref="rlm:reading_language_memory_data_long",
    audit_baseline="complete_case_summary",
    model_settings=AdjustedModelSettings(
        predictor_measures=("bpvs", "trog", "basdig", "bassim"),
        use_age_predictor=True,
        transition_waves=(1, 2, 3, 4, 5),
        common_horizon_last_wave=4,
        per_transition_sensitivity=True,
        require_confirmed_inputs=True,
        predictor_slope_sigma=0.3,
        prior_sensitivity_sigmas=(0.5, 0.7),
    ),
)


def fit(config: str = "dev") -> StatisticalFitContext:
    return fit_rlm_adjusted(SPEC, config=config)
