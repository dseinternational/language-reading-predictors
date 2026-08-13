# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""RLMADJ02 - reduced Down-syndrome-only word-reading-gain companion (#409 D3).

The pre-specified small-sample companion to ``lrp-rlm-adj-001``. It restricts
the Byrne cohort to children with Down syndrome and asks how wave-1 verbal
memory (``basdig``), receptive vocabulary (``bpvs``) and verbal reasoning
(``bassim``) are associated with wave-1 to wave-3 BAS word-reading gain after
mutual adjustment and conditioning on wave-1 word reading. These three measures
have confirmed instrument ceilings and leave 22 complete cases from the 24-child
Down syndrome starting cohort.

The reduced set is deliberate. The pooled parent has five skill predictors plus
age and the own baseline; carrying that full seven-slope specification into about
21 Down syndrome rows would make the result largely prior-determined. Age is not
added as a fourth focal predictor here, so the reported quantities are conditional
associations among the three named skills and own-baseline word reading, not
age-adjusted or causal effects. The bivariate and prior-width sensitivity refits
remain part of the adjusted-family contract.
"""

from language_reading_predictors.statistical_models.adjusted import (
    AdjustedModelSettings,
)
from language_reading_predictors.statistical_models.context import ModelSpec
from language_reading_predictors.statistical_models.pipelines.adjusted import fit_rlm_adjusted

SPEC = ModelSpec(
    model_id="lrp-rlm-adj-002",
    kind="adjusted",
    title=(
        "Byrne Down-syndrome-only predictors of word-reading gain, waves 1-3 "
        "(reduced mutually adjusted set)"
    ),
    outcome_symbol="basread",
    study_id="rlm",
    family="adjusted",
    design="historical_cohort",
    estimand_type="association",
    causal_status="none",
    dataset_ref="rlm:reading_language_memory_data_long",
    audit_baseline="complete_case_summary",
    model_settings=AdjustedModelSettings(
        predictor_measures=("basdig", "bpvs", "bassim"),
        use_age_predictor=False,
        pre_wave=1,
        post_wave=3,
        group_codes=(1,),
        predictor_slope_sigma=0.3,
        prior_sensitivity_sigmas=(0.5, 0.7),
    ),
)


def fit(config: str = "dev"):
    return fit_rlm_adjusted(SPEC, config=config)
