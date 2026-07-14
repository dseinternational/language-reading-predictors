# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""LRPGF10 - gain factors for taught expressive vocabulary (TE).

DAG-focused gain-factors model (#224; adjustment set re-derived against the revised
2026-07-10 DAG, ``dag/dag-language-reading.dagitty``, #247). Associations with how much
children gain in taught expressive vocabulary (block 1, ``b1extau``) across the three
period transitions (ANCOVA gain, Beta-Binomial on the logit scale, child random
intercept).

Under the revised DAG the parents of TE are age, general ability, intervention, hearing
(HS), speech production (SP), phonological memory (RW) and taught receptive vocabulary
(TR). So beyond the own baseline, age and cognitive ability (blocks), the adjustment set
adds the measured upstream skill TR (``skill_symbols``; the standardised transfer
measures RV/EV sit downstream and are not conditioned on) and the non-measure confounders
hearing, speech and phonological memory (``adjust_for``: ``hs``, ``deapp_c`` and
``erbto``).

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
    model_id="lrp-rli-gf-010",
    kind="gain_factors",
    title="Factors associated with gains in taught expressive vocabulary (TE)",
    outcome_symbol="TE",
    extra={
        "skill_symbols": ("TR",),
        "ability_covariate": V.BLOCKS,
        "adjust_for": ("hs", "hs_missing", "deapp_c", "deapp_c_missing", "erbto", "erbto_missing"),
        "interactions": (("trt", "ability"), ("trt", "own"), ("age", "ability")),
        "treated_only": False,
    },
)


def fit(config: str = "dev"):
    return fit_gain_factors(SPEC, config=config)
