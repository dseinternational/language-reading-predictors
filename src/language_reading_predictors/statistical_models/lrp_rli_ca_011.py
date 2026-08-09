# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""LRP-CA-011 - Letter-sound -> word-reading association, holding nonword decoding fixed.

#421 Tier 1: the "does the letter-sound -> word-reading association survive holding
decoding fixed" decomposition, promoted from a scratch fit. Identical to ``ca-010``
(minimal adjustment for age, hearing and non-verbal ability) but with same-wave
**nonword reading** (N) added as a second predictor: the letter-sound slope here is
the LS -> WR level association *among children alike on nonword decoding*, and
comparing it to ``ca-010``'s slope is the review's attenuation check.

Nonword reading is heavily floored (≈72/64/52/40% at zero across t1-t4); it enters as
a standardised predictor logit (the floor reduces its variance, not its admissibility -
``ca-001`` excluded floored measures by *choice*, not a factory limit). The floored-N
identifiability concern applies to N as an *outcome* / to a nonparametric curve, not to
N as a linear predictor here.

**Estimand and its limits.** As ``ca-010``: every coefficient is a latent-GA-confounded
*adjusted association*, never causal. The single-posterior contrast against ``ca-010``
that the review's "share retained" wants is a #421 Tier-3 joint {W, N} fit (deferred);
this pair of separate fits is the descriptive version. Report median + 50% + 89% + P(>0)
with the adjusted-association caveat.
"""

from language_reading_predictors.statistical_models.concurrent import (
    ConcurrentModelSettings,
)
from language_reading_predictors.statistical_models.context import ModelSpec
from language_reading_predictors.statistical_models.pipeline import fit_concurrent

SPEC = ModelSpec(
    model_id="lrp-rli-ca-011",
    kind="concurrent",
    title="Concurrent letter sounds + nonword decoding -> word reading (per wave)",
    outcome_symbol="W",
    family="concurrent",
    design="per-wave cross-sectional conditional association, decoding held fixed",
    estimand_type="association",
    causal_status="none",
    model_settings=ConcurrentModelSettings(
        # ca-010 plus same-wave nonword reading (N): the decoding-held-fixed check.
        predictor_symbols=("L", "N"),
        covariates=("blocks", "hs", "hs_missing"),
        include_age=True,
        include_group=True,
        predictor_slope_sigma=0.3,
    ),
)


def fit(config: str = "dev"):
    return fit_concurrent(SPEC, config=config)
