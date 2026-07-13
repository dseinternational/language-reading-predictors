# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""LRPGF08 - gain factors for TROG receptive grammar (T).

DAG-focused gain-factors model (#127; adjustment set re-derived against the revised
2026-07-10 DAG, ``dag/dag-language-reading.dagitty``, #247). Associations with how much
children gain in TROG receptive grammar (RG) across the three period transitions
(ANCOVA, Beta-Binomial logit, child random intercept).

Under the revised DAG the measured parents of RG are receptive vocabulary (R) and taught
receptive vocabulary (TR); RG has no non-measure confounder parent, so ``adjust_for`` is
empty. The adjustment set is the own baseline + age + ability (blocks) + ``skill_symbols``
(R, TR).

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
    model_id="lrp-rli-gf-008",
    kind="gain_factors",
    title="Gain factors for TROG receptive grammar (T)",
    outcome_symbol="T",
    extra={
        "skill_symbols": ("R", "TR"),
        "ability_covariate": V.BLOCKS,
        "adjust_for": (),
        "interactions": (("trt", "ability"), ("trt", "own"), ("age", "ability")),
        "treated_only": False,
    },
)


def fit(config: str = "dev"):
    return fit_gain_factors(SPEC, config=config)
