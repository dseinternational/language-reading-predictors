# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""LRPGF02 - gain factors for receptive vocabulary (R).

DAG-focused gain-factors model (#127; adjustment set re-derived against the revised
2026-07-10 DAG, ``dag/dag-language-reading.dagitty``, #247). Associations with how
much children gain in receptive vocabulary (RV) across the three period transitions
(ANCOVA, Beta-Binomial logit, child random intercept).

Under the revised DAG the parents of RV are age, general ability, hearing (HS), taught
receptive vocabulary (TR) and phonological memory (RW). So beyond the own baseline,
age and cognitive ability (blocks), the adjustment set adds the measured upstream skill
TR (``skill_symbols``) and the non-measure confounders hearing and phonological memory
(``adjust_for``: ``hs``/``deapp`` are not RV parents, but ``hs`` and ``erbto`` are).

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
    model_id="lrp-rli-gf-002",
    kind="gain_factors",
    title="Gain factors for receptive vocabulary (R)",
    outcome_symbol="R",
    extra={
        "skill_symbols": ("TR",),
        "ability_covariate": V.BLOCKS,
        "adjust_for": ("hs", "hs_missing", "erbto", "erbto_missing"),
        "interactions": (("trt", "ability"), ("trt", "own"), ("age", "ability")),
        "treated_only": False,
    },
)


def fit(config: str = "dev"):
    return fit_gain_factors(SPEC, config=config)
