# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""LRP-CA-007 - Concurrent conditional associations: skills -> phoneme blending, per wave.

Completes the concurrent family's core skill set {W, L, B, TR, TE, R, E}: word
reading (ca-001), letter sounds (ca-002), taught receptive/expressive vocabulary
(ca-003/004) and standardised receptive/expressive vocabulary (ca-005/006) each had
a focal model; **phoneme blending (B)** was in the predictor core set but had no
focal model of its own. This model adds it, so every core-set measure is described
as a focal outcome and the family's conditional-joint-distribution picture is
symmetric.

At each timepoint it fits a between-child Beta-Binomial regression of blending's
*level* on the standardised same-wave logits of the other core skills (word reading
W, letter sounds L, taught receptive TR and expressive TE vocabulary, receptive R
and expressive E vocabulary), plus age and a group nuisance term - "at wave t, among
children alike on age and the other skills, +1 SD on skill X is associated with +m
blending items". Blending is a small bounded count (10 items) but is not floored (it
is already used as a predictor in the sibling models), so it enters the family on the
same footing as the other core measures.

**Estimand and its limits** (identical to the family, ca-001): every coefficient is
an *adjusted association*, never a causal effect. Conditioning on contemporaneous
(post-treatment) skill levels is intentional and licensed here - the family exists to
describe the conditional joint distribution of the skill levels at each wave. Read
with the family's three standing caveats (Table-2 fallacy; measurement error;
collinearity/regularisation at n ~ 53). Floored measures (P, N) stay excluded as
predictors, exactly as in the siblings.
"""

from language_reading_predictors.statistical_models.context import ModelSpec
from language_reading_predictors.statistical_models.pipeline import fit_concurrent

SPEC = ModelSpec(
    model_id="lrp-rli-ca-007",
    kind="concurrent",
    title="Concurrent conditional associations: concurrent skills -> phoneme blending (per wave)",
    outcome_symbol="B",
    family="concurrent",
    design="per-wave cross-sectional conditional associations",
    estimand_type="association",
    causal_status="none",
    extra={
        # The core skill set minus the focal (B): blending is swapped out and the
        # other six enter as predictors. Floored P/N are excluded as predictors
        # (issue #312), matching the sibling models.
        "predictor_symbols": ["W", "L", "TR", "TE", "R", "E"],
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
