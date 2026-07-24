# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""LRP-CA-010 - Concurrent letter-sound -> word-reading association, minimal adjustment.

#421 Tier 1: the letter-sound -> word-reading review's headline, promoted from a
scratch fit to a registered model. At each timepoint it fits a between-child
Beta-Binomial regression of word-reading *level* on the standardised same-wave logit
of letter-sound knowledge, adjusting **only** for age, hearing and non-verbal ability
(block design) - "at wave t, among children alike on age, hearing and non-verbal
ability, how much higher is word reading per +1 SD of letter sounds".

This is deliberately a *minimal* adjustment. The nearest registered model, ``ca-001``,
mutually adjusts word reading for five other same-wave skills (B, TR, TE, R, E), which
answers a different conditional question; ``ca-010`` isolates the LS -> WR level
association against the trait covariates alone, matching the review that motivated it.

**Estimand and its limits.** Every coefficient is an *adjusted association*, never a
causal effect: latent general ability is not observed, so the LS -> WR slope is
latent-GA-confounded. The only randomisation-licensed effect in the study is the ITT
arm. Report as median + inner 50% + outer 89% credible interval + P(>0), with the
adjusted-association caveat.
"""

from language_reading_predictors.statistical_models.context import ModelSpec
from language_reading_predictors.statistical_models.pipeline import fit_concurrent

SPEC = ModelSpec(
    model_id="lrp-rli-ca-010",
    kind="concurrent",
    title="Concurrent letter sounds -> word reading, minimal adjustment (per wave)",
    outcome_symbol="W",
    family="concurrent",
    design="per-wave cross-sectional conditional association, minimal adjustment",
    estimand_type="association",
    causal_status="none",
    extra={
        # Minimal adjustment: letter sounds is the sole skill predictor (contrast
        # ca-001's six-skill mutual adjustment).
        "predictor_symbols": ["L"],
        # Trait covariates: non-verbal ability (block design) and hearing, entered as
        # t1 baselines broadcast across waves. Age is added via include_age.
        "covariates": ["blocks", "hs"],
        "include_age": True,
        # Group as a flagged, non-interpretable nuisance (absorbs arm composition).
        "include_group": True,
        "predictor_slope_sigma": 0.3,
    },
)


def fit(config: str = "dev"):
    return fit_concurrent(SPEC, config=config)
