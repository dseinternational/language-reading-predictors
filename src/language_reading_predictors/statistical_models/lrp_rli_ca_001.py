# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""LRP-CA-001 - Concurrent conditional associations: skills -> word reading, per wave.

#312 (descriptive-association workstream #314): the first model of the concurrent
conditional-associations family. At each timepoint it fits a between-child
Beta-Binomial regression of word reading's *level* on the standardised same-wave
logits of a core skill set (letter sounds L, blending B, taught receptive TR and
expressive TE vocabulary, receptive R and expressive E vocabulary), plus age and a
group nuisance term - "at wave t, among children alike on age and the other skills,
+n letter sounds is associated with +m words read".

**Estimand and its limits.** Every coefficient is an *adjusted association*, never a
causal effect. The family exists precisely to describe the conditional joint
distribution of the skill levels at each wave; conditioning on contemporaneous
(post-treatment) skill levels is therefore intentional and licensed here, unlike in
the level-factors family, which excludes cross-skill terms to protect the causal
reading of its group x time contrast. All children are on the intervention from t2,
so these are associations *within a treated system*.

Read with three standing caveats (in the report): (1) the Table-2 fallacy - each
mutually-adjusted coefficient answers a *different* conditional question, and the set
should not be read as competing causal effects; (2) regression dilution - the
predictors are observed scores measured with error, so the associations attenuate
toward zero relative to the latent-skill truth (the longitudinal factor model, #313,
is the follow-on instrument that addresses this); (3) collinearity shrinkage - with
n ~ 53 and a strongly inter-correlated predictor cluster, the regularising priors
pull mutually-adjusted coefficients toward zero, so the adjusted-vs-bivariate gap is
itself informative and both columns are reported.

Design decisions (issue #312, recommendations adopted): four separate
cross-sectional fits reported side by side (not one stacked model with a child random
intercept, which would tilt the coefficients toward a within-child quantity); group
included only as a non-interpretable nuisance term to absorb arm composition;
predictors entered as standardised same-wave logits; floored measures (P, N) excluded
as predictors. Word reading is the focal outcome for this first model; letter sounds
and taught vocabulary are the sibling focal outcomes (``lrp_rli_ca_002``-``004``),
each conditioning on the core skill set {W, L, B, TR, TE, R, E} minus itself.
"""

from language_reading_predictors.statistical_models.context import ModelSpec
from language_reading_predictors.statistical_models.pipeline import fit_concurrent

SPEC = ModelSpec(
    model_id="lrp-rli-ca-001",
    kind="concurrent",
    title="Concurrent conditional associations: concurrent skills -> word reading (per wave)",
    outcome_symbol="W",
    estimand_type="association",
    causal_status="none",
    extra={
        # Contemporaneous predictor skills (standardised same-wave logits). Floored
        # P/N are excluded as predictors (issue #312).
        "predictor_symbols": ["L", "B", "TR", "TE", "R", "E"],
        "include_age": True,
        # Group as a flagged, non-interpretable nuisance (absorbs arm composition).
        "include_group": True,
        "predictor_slope_sigma": 0.3,
    },
)


def fit(config: str = "dev"):
    return fit_concurrent(SPEC, config=config)
