# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""RLMADJ01 - Byrne wave-1 predictors of subsequent word-reading gain (#338 Phase D).

The Byrne analogue of ``lrp-rli-adj-065``: a between-child Beta-Binomial
regression of BAS word reading at **wave 3** on its own wave-1 baseline
(``gamma_own`` - the gain framing) and the mutually-adjusted standardised
wave-1 skills (receptive vocabulary ``bpvs``, receptive grammar ``trog``,
verbal memory ``basdig``, verbal reasoning ``bassim``, number skills
``basnum``) plus age, pooled over the three reading groups with
non-interpretable group-nuisance dummies (the 2026-07-16 data-owner decisions:
pooled + nuisance; outcome horizon w1 -> w3, the audited core window; n = 69).
Every coefficient is an **adjusted association** - this cohort has no
intervention and nothing here is causal (``causal_status="none"``). The
bivariate (baseline-only-adjusted) comparison fit per predictor and a
slope-prior sensitivity sweep are reported alongside, per the RLI family
convention.

Down-syndrome-only companion fits are deliberately deferred: at n = 21 with
seven slopes the mutually adjusted posteriors would sit at the prior.
``basmat`` is excluded (wave-3+ only - no wave-1 value exists); ``basspel`` and
``woco`` are excluded as reading-route measures too close to the outcome
(their conditional slopes would mostly re-express the baseline).
"""

from language_reading_predictors.statistical_models.context import ModelSpec
from language_reading_predictors.statistical_models.pipeline import fit_rlm_adjusted

SPEC = ModelSpec(
    model_id="lrp-rlm-adj-001",
    kind="adjusted",
    title=(
        "Byrne wave-1 predictors of word-reading gain, waves 1-3 "
        "(between-child, mutually adjusted)"
    ),
    outcome_symbol="basread",
    study_id="rlm",
    family="adjusted",
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
        "predictor_slope_sigma": 0.3,
        "prior_sensitivity_sigmas": (0.5, 0.7),
    },
)


def fit(config: str = "dev"):
    return fit_rlm_adjusted(SPEC, config=config)
