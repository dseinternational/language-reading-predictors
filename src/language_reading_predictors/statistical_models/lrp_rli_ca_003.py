# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""LRP-CA-003 - Concurrent conditional associations: skills -> taught receptive
vocabulary, per wave.

#312 extension (descriptive-association workstream #314): block-1 taught receptive
vocabulary (TR) as the focal outcome of the concurrent conditional-associations
family. At each timepoint it fits a between-child Beta-Binomial regression of TR's
*level* on the standardised same-wave logits of the other core skills (word reading
W, letter sounds L, blending B, taught expressive TE vocabulary, receptive R and
expressive E vocabulary), plus age and a group nuisance term.

The family's core skill set is {W, L, B, TR, TE, R, E}: each model conditions its
focal outcome on the remaining six (see ``lrp_rli_ca_002``). The taught expressive
companion TE stays in TR's predictor set - the two taught tests are strongly
correlated, but the regularising priors handle the collinearity and the
adjusted-vs-bivariate gap is itself informative. Floored measures (P, N) stay
excluded. Note TR approaches its 24-item ceiling at later waves; the Beta-Binomial
respects the bound, but per-wave predictor variance (and hence the associations'
resolution) shrinks as scores compress.

**Estimand and its limits.** Every coefficient is an *adjusted association*, never a
causal effect; conditioning on contemporaneous (post-treatment) skill levels is
intentional and licensed because nothing here is read causally. Read with the
family's three standing caveats (Table-2 fallacy, regression dilution, collinearity
shrinkage) - see ``lrp_rli_ca_001`` and the report for the full statements.

Design decisions follow ca-001 (issue #312): four separate cross-sectional fits
reported side by side; group as a non-interpretable nuisance; standardised same-wave
logit predictors with regularising ``Normal(0, 0.3)`` slopes.
"""

from language_reading_predictors.statistical_models.context import ModelSpec
from language_reading_predictors.statistical_models.pipeline import fit_concurrent

SPEC = ModelSpec(
    model_id="lrp-rli-ca-003",
    kind="concurrent",
    title=(
        "Concurrent conditional associations: concurrent skills -> taught receptive "
        "vocabulary (per wave)"
    ),
    outcome_symbol="TR",
    estimand_type="association",
    causal_status="none",
    extra={
        # The core skill set minus the focal (TR). Floored P/N are excluded as
        # predictors (issue #312).
        "predictor_symbols": ["W", "L", "B", "TE", "R", "E"],
        "include_age": True,
        # Group as a flagged, non-interpretable nuisance (absorbs arm composition).
        "include_group": True,
        "predictor_slope_sigma": 0.3,
    },
)


def fit(config: str = "dev"):
    return fit_concurrent(SPEC, config=config)
