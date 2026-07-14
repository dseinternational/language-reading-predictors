# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""LRPGF04 - gain factors for letter sounds (L).

DAG-focused gain-factors model (#127; adjustment set re-derived against the revised
2026-07-10 DAG, ``dag/dag-language-reading.dagitty``, #247). Associations with how much
children gain in letter sounds (LS) across the three period transitions (ANCOVA,
Beta-Binomial logit, child random intercept).

Under the revised DAG LS has no measured skill parent (its parents are age, general
ability, hearing HS, the intervention, and speech production SP), so ``skill_symbols``
is empty; the non-measure confounder parents hearing and speech enter via ``adjust_for``
(hs, deapp_c and their missing indicators).

Only the randomised on-intervention term is causal — and its **period-1** average
marginal effect (the genuinely randomised, all-untreated-baseline transition) is the
ITT-anchor estimand, not the all-transition pool (#247 P2). ``beta_trt`` itself is the
on-intervention log-odds contrast. Every other coefficient is an *adjusted association*:
the child random intercept is a partial, shrunken stand-in for between-child
heterogeneity — it does **not** control latent general ability, so those slopes remain
descriptive associations. SES excluded (non-DAG / redundant).
"""

from language_reading_predictors.data_variables import Variables as V
from language_reading_predictors.statistical_models.context import ModelSpec
from language_reading_predictors.statistical_models.pipeline import fit_gain_factors

SPEC = ModelSpec(
    model_id="lrp-rli-gf-004",
    kind="gain_factors",
    title="Factors associated with gains in letter sounds (L)",
    outcome_symbol="L",
    extra={
        "skill_symbols": (),
        "ability_covariate": V.BLOCKS,
        "adjust_for": ("hs", "hs_missing", "deapp_c", "deapp_c_missing"),
        "interactions": (("trt", "ability"), ("trt", "own"), ("age", "ability")),
        "treated_only": False,
    },
)


def fit(config: str = "dev"):
    return fit_gain_factors(SPEC, config=config)
