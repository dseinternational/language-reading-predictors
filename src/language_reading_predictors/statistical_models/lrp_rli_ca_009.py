# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""LRP-CA-009 - Concurrent conditional associations: skills -> receptive grammar, per wave.

Extends the concurrent family beyond its original reading/vocabulary core set to
Receptive grammar (TROG-2) (T), a scope addition (#371) so the levels panel of the association
matrix covers all non-floored outcomes. At each timepoint it fits a between-child
Beta-Binomial regression of receptive grammar's *level* on the standardised same-wave logits
of the core skill set (word reading W, letter sounds L, phoneme blending B, taught
receptive TR and expressive TE vocabulary, receptive R and expressive E
vocabulary), plus age, a group nuisance term, and the trait covariates aligned with
the gains panel (non-verbal ability, hearing, speech, phonological memory).

Every coefficient is an *adjusted association*, never causal, exactly as in the
sibling models (ca-001..007). Floored measures (P, N) remain excluded as predictors.
T is not floored, so it enters as a focal outcome on the same footing as the
core-set measures; it is not itself added to the other models' predictor sets (an
asymmetric, focal-only scope extension).
"""

from language_reading_predictors.statistical_models.context import ModelSpec
from language_reading_predictors.statistical_models.pipeline import fit_concurrent

SPEC = ModelSpec(
    model_id="lrp-rli-ca-009",
    kind="concurrent",
    title="Concurrent conditional associations: concurrent skills -> receptive grammar (per wave)",
    outcome_symbol="T",
    family="concurrent",
    design="per-wave cross-sectional conditional associations",
    estimand_type="association",
    causal_status="none",
    extra={
        # Full core skill set as predictors (T is outside it, so none removed).
        "predictor_symbols": ["W", "L", "B", "TR", "TE", "R", "E"],
        # Trait covariates aligned with the gains panel (#371).
        "covariates": ["blocks", "hs", "deapp_c", "erbto"],
        "include_age": True,
        "include_group": True,
        "predictor_slope_sigma": 0.3,
    },
)


def fit(config: str = "dev"):
    return fit_concurrent(SPEC, config=config)
