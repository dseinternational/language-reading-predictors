# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""RLMHS02 - confirmed-input receptive-vocabulary-gain ranking (#409 D1).

The regularised-horseshoe cross-check for ``lrp-rlm-adj-003`` ranks the same
wave-1 BAS word reading, receptive grammar, verbal memory, verbal reasoning and
age predictors of wave-3 BPVS after conditioning on wave-1 BPVS. It uses the
identical 71-child, pooled complete-case frame and requires confirmed instrument
identities and ceilings for every measurement input.

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
    model_id="lrp-rlm-hs-002",
    kind="horseshoe",
    title=(
        "Byrne horseshoe ranking of wave-1 predictors of receptive-vocabulary "
        "gain, waves 1-3 (confirmed-input)"
    ),
    outcome_symbol="bpvs",
    study_id="rlm",
    family="horseshoe",
    design="historical_cohort",
    estimand_type="association",
    causal_status="none",
    dataset_ref="rlm:reading_language_memory_data_long",
    audit_baseline="complete_case_summary",
    model_settings=HorseshoeModelSettings(
        predictor_measures=("basread", "trog", "basdig", "bassim"),
        use_age_predictor=True,
        pre_wave=1,
        post_wave=3,
        require_confirmed_inputs=True,
        delta=0.1,
        tau0=0.1,
        slab_scale=2.0,
        slab_df=4.0,
    ),
    extra={
        # The horseshoe's global-local funnel needs smaller steps than the
        # tier defaults. 0.99 cleared the gate under the HalfNormal(50)
        # concentration prior; with the dispersion-scale prior the Byrne
        # factories share since 2026-08-22 the reporting refit produced five
        # divergences in 36 000 draws (R-hat 1.0004, ESS > 11 000, BFMI 0.8 —
        # sporadic funnel divergences, not a boundary pile-up), and the
        # horseshoe ranking is zero-divergence-only, so this model takes the
        # 0.999 its TROG sibling lrp-rlm-hs-003 already uses.
        "target_accept": 0.999,
    },
)


def fit(config: str = "dev"):
    return fit_rlm_horseshoe(SPEC, config=config)
