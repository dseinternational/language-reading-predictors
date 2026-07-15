# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""LRP-CA-002 - Concurrent conditional associations: skills -> letter sounds, per wave.

#312 extension (descriptive-association workstream #314): letter-sound knowledge as
the focal outcome of the concurrent conditional-associations family. At each timepoint
it fits a between-child Beta-Binomial regression of letter-sound knowledge's *level*
on the standardised same-wave logits of the other core skills (word reading W,
blending B, taught receptive TR and expressive TE vocabulary, receptive R and
expressive E vocabulary), plus age and a group nuisance term - "at wave t, among
children alike on age and the other skills, +n words read is associated with +m
letter sounds".

The family's core skill set is {W, L, B, TR, TE, R, E}: each model conditions its
focal outcome on the remaining six, so together the models describe the conditional
joint distribution of the same measure set (the focal is swapped out of the ca-001
predictor list and word reading swapped in). Floored measures (P, N) stay excluded.

**Estimand and its limits.** Every coefficient is an *adjusted association*, never a
causal effect; conditioning on contemporaneous (post-treatment) skill levels is
intentional and licensed because nothing here is read causally. Read with the
family's three standing caveats (Table-2 fallacy, qualified measurement-error
distortion, and collinearity plus regularisation) - see ``lrp_rli_ca_001`` and the
report for the full statements.

Design decisions follow ca-001 (issue #312): four separate cross-sectional fits
reported side by side; group as a non-interpretable nuisance; standardised same-wave
logit predictors with regularising ``Normal(0, 0.3)`` slopes.
"""

from language_reading_predictors.statistical_models.context import ModelSpec
from language_reading_predictors.statistical_models.pipeline import fit_concurrent

SPEC = ModelSpec(
    model_id="lrp-rli-ca-002",
    kind="concurrent",
    title="Concurrent conditional associations: concurrent skills -> letter sounds (per wave)",
    outcome_symbol="L",
    family="concurrent",
    design="per-wave cross-sectional conditional associations",
    estimand_type="association",
    causal_status="none",
    extra={
        # The core skill set minus the focal (L): word reading enters as a
        # predictor here. Floored P/N are excluded as predictors (issue #312).
        "predictor_symbols": ["W", "B", "TR", "TE", "R", "E"],
        "include_age": True,
        # Group as a flagged, non-interpretable nuisance (absorbs arm composition).
        "include_group": True,
        "predictor_slope_sigma": 0.3,
    },
)


def fit(config: str = "dev"):
    return fit_concurrent(SPEC, config=config)
