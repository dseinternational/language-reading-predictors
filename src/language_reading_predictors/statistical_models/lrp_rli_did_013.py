# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""LRPDID13 - treatment-effect heterogeneity variance component (word reading, W), #230.

A waitlist-crossover DiD on word reading (the anchor outcome with the strongest DiD signal)
with a **per-child random slope on the on-intervention indicator**: the crossover effect
is ``delta_i = delta + sigma_delta * z_i`` (non-centred). ``delta`` is the population-mean
effect (read exactly as the other DiDs); ``sigma_delta`` — the between-child SD of the
crossover effect on the logit scale — is the reported estimand (#230 §2).

This is the honest first rung of the non-responder question (#230 §4a): *does child-level
response variance exist at all?* If ``sigma_delta`` concentrates near zero, apparent
non-response is noise plus modelled covariates, which supports **not** gate-keeping on
early response (consistent with HS-001 selecting no gain predictors and the 7/8 negative
treatment×baseline interactions). Only if variance exists is a one-stage moderation with
shrinkage warranted — never a two-stage classify-then-compare (Senn 2016; see METHODS.md).
The DiD framing keeps the immediate-arm anchoring the §4a caveat calls for; a correlated
intercept-slope (LKJ) form is deliberately deferred as a prior-dominated sensitivity at n≈54.
"""

from language_reading_predictors.statistical_models.context import ModelSpec
from language_reading_predictors.statistical_models.pipeline import fit_did

SPEC = ModelSpec(
    model_id="lrp-rli-did-013",
    kind="did",
    title="Within-person DiD with treatment-effect heterogeneity (word reading, W)",
    outcome_symbol="W",
    family="did",
    design="waitlist-crossover DiD + per-child random on-intervention slope",
    estimand_type="association",
    causal_status="randomised-anchored",
    extra={
        "periods": (0, 1),
        "use_child_re": True,
        "use_age": True,
        "use_varying_delta": True,
    },
)


def fit(config: str = "dev"):
    return fit_did(SPEC, config=config)
