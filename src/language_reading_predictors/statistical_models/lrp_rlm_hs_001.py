# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""RLMHS01 - Byrne regularised-horseshoe predictor ranking (#338 Phase D).

The Byrne analogue of ``lrp-rli-hs-001``: one regularised-horseshoe regression
of wave-3 BAS word reading on its own wave-1 baseline (``gamma_own``) plus the
standardised wave-1 predictor set (``bpvs``, ``trog``, ``basdig``, ``bassim``,
``basnum``, age) under a shared shrinkage prior, ranked by posterior
``P(|beta| > 0.1)``. The cross-check partner of ``lrp-rlm-adj-001`` over the
identical frame (pooled three-group, group-nuisance dummies outside the
horseshoe, n = 69; 2026-07-16 sign-off). There is no gradient-boosting layer
for the Byrne cohort, so no GB comparison table is produced - the ranking
stands against the adjusted model's mutually-adjusted slopes instead. Not
causal; a which-predictors-carry-signal read only.
"""

from language_reading_predictors.statistical_models.context import ModelSpec
from language_reading_predictors.statistical_models.pipeline import fit_rlm_horseshoe

SPEC = ModelSpec(
    model_id="lrp-rlm-hs-001",
    kind="horseshoe",
    title=(
        "Byrne horseshoe ranking of wave-1 predictors of word-reading gain, "
        "waves 1-3"
    ),
    outcome_symbol="basread",
    study_id="rlm",
    family="horseshoe",
    design="historical_cohort",
    estimand_type="association",
    causal_status="none",
    dataset_ref="rlm:reading_language_memory_data_long",
    extra={
        "study_id": "rlm",
        "predictor_measures": ("bpvs", "trog", "basdig", "bassim", "basnum"),
        "use_age_predictor": True,
        "pre_wave": 1,
        "post_wave": 3,
        "delta": 0.1,
        "tau0": 0.1,
        "slab_scale": 2.0,
        "slab_df": 4.0,
        # The horseshoe's global-local funnel needs smaller steps than the
        # tier defaults, matching the RLI horseshoe fits.
        "target_accept": 0.99,
    },
)


def fit(config: str = "dev"):
    return fit_rlm_horseshoe(SPEC, config=config)
