# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""LRP-CA-005 - Concurrent conditional associations: skills -> receptive
vocabulary, per wave.

#336 extends the concurrent conditional-associations family to standardised receptive
vocabulary (R; ROWPVT) as the focal outcome. At each timepoint it fits a between-child
Beta-Binomial regression of R's *level* on the standardised same-wave logits of the
other core skills (word reading W, letter sounds L, blending B, taught receptive TR
and expressive TE vocabulary, and standardised expressive E vocabulary), plus age and
a group nuisance term.

The family's core skill set is {W, L, B, TR, TE, R, E}: each model conditions its
focal outcome on the remaining six, so the models are complementary full conditionals
of the same measure set. Floored measures (P, N) stay excluded. R is a 170-item
standardised transfer measure and does not have the late-wave ceiling compression
flagged for the 24-item taught receptive measure in ``lrp_rli_ca_003``.

**Estimand and its limits.** Every coefficient is an *adjusted association*, never a
causal effect; conditioning on contemporaneous (post-treatment) skill levels is
intentional because nothing here is read causally. Read with the family's three
standing caveats (Table-2 fallacy, qualified measurement-error distortion, and
collinearity plus regularisation) - see ``lrp_rli_ca_001`` and the report for the full
statements.

Design decisions follow ca-001 (issue #312): four separate cross-sectional fits
reported side by side; group as a non-interpretable nuisance; standardised same-wave
logit predictors with regularising ``Normal(0, 0.3)`` slopes.
"""

from language_reading_predictors.statistical_models.context import ModelSpec
from language_reading_predictors.statistical_models.pipeline import fit_concurrent

SPEC = ModelSpec(
    model_id="lrp-rli-ca-005",
    kind="concurrent",
    title=(
        "Concurrent conditional associations: concurrent skills -> receptive "
        "vocabulary (per wave)"
    ),
    outcome_symbol="R",
    family="concurrent",
    design="per-wave cross-sectional conditional associations",
    estimand_type="association",
    causal_status="none",
    extra={
        # The core skill set minus the focal (R). Floored P/N are excluded as
        # predictors (issues #312 and #336).
        "predictor_symbols": ["W", "L", "B", "TR", "TE", "E"],
        # Trait covariates aligned with the gains panel (non-verbal ability, hearing,
        # speech, phonological memory), entered as t1 baselines broadcast across the waves (#371).
        "covariates": ["blocks", "hs", "deapp_c", "erbto"],
        "include_age": True,
        # Group as a flagged, non-interpretable nuisance (absorbs arm composition).
        "include_group": True,
        "predictor_slope_sigma": 0.3,
    },
)


def fit(config: str = "dev"):
    return fit_concurrent(SPEC, config=config)
