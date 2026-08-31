# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""RLMHS03 - confirmed-input receptive-grammar-gain ranking (#409 D1).

The regularised-horseshoe cross-check for ``lrp-rlm-adj-004`` ranks the same
wave-1 BAS word reading, receptive vocabulary, verbal memory, verbal reasoning
and age predictors of wave-3 TROG after conditioning on wave-1 TROG. It uses
the identical 69-child, pooled complete-case frame and requires confirmed
instrument identities and ceilings for every measurement input.

Reading-matched children were selected on word-reading level, so ``basread`` is
also a selection variable in this pooled frame. The ranking is therefore a
descriptive signal-allocation check against the independently regularised
adjusted model, not a causal or selection-free estimate. There is no Byrne
gradient-boosting comparison.
"""

from language_reading_predictors.statistical_models.context import ModelSpec
from language_reading_predictors.statistical_models.horseshoe import (
    HorseshoeModelSettings,
)
from language_reading_predictors.statistical_models.pipelines.horseshoe import (
    fit_rlm_horseshoe,
)

SPEC = ModelSpec(
    model_id="lrp-rlm-hs-003",
    kind="horseshoe",
    title=(
        "Byrne horseshoe ranking of wave-1 predictors of receptive-grammar "
        "gain, waves 1-3 (confirmed-input)"
    ),
    outcome_symbol="trog",
    study_id="rlm",
    family="horseshoe",
    design="historical_cohort",
    estimand_type="association",
    causal_status="none",
    dataset_ref="rlm:reading_language_memory_data_long",
    audit_baseline="complete_case_summary",
    model_settings=HorseshoeModelSettings(
        predictor_measures=("basread", "bpvs", "basdig", "bassim"),
        use_age_predictor=True,
        pre_wave=1,
        post_wave=3,
        require_confirmed_inputs=True,
        delta=0.1,
        tau0=0.1,
        slab_scale=2.0,
        slab_df=4.0,
    ),
        # The horseshoe's global-local funnel needs smaller steps than the
        # tier defaults. The rep-lite fit diverged at 0.99 and cleared the
        # zero-divergence gate at 0.999.
    target_accept=0.999,
)


def fit(config: str = "dev"):
    return fit_rlm_horseshoe(SPEC, config=config)
