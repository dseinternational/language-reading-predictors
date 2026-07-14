# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""LRPGF09 - gain factors for taught receptive vocabulary (TR).

DAG-focused gain-factors model (#224; adjustment set re-derived against the revised
2026-07-10 DAG, ``dag/dag-language-reading.dagitty``, #247). Associations with how much
children gain in taught receptive vocabulary (block 1, ``b1retau``) across the three
period transitions (ANCOVA gain, Beta-Binomial on the logit scale, child random
intercept).

Under the revised DAG the parents of TR are age, general ability, intervention, hearing
(HS) and phonological memory (RW) — no measured upstream skill is a parent (the
standardised transfer measures RV/EV sit *downstream* of taught vocabulary,
``TR -> RV``), so ``skill_symbols`` is empty and beyond the own baseline, age and
cognitive ability (blocks) the adjustment set adds only the non-measure confounders
hearing and phonological memory (``adjust_for``: ``hs`` and ``erbto``; speech/``deapp``
is not a TR parent).

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
    model_id="lrp-rli-gf-009",
    kind="gain_factors",
    title="Factors associated with gains in taught receptive vocabulary (TR)",
    outcome_symbol="TR",
    extra={
        "skill_symbols": (),
        "ability_covariate": V.BLOCKS,
        "adjust_for": ("hs", "hs_missing", "erbto", "erbto_missing"),
        "interactions": (("trt", "ability"), ("trt", "own"), ("age", "ability")),
        "treated_only": False,
    },
)


def fit(config: str = "dev"):
    return fit_gain_factors(SPEC, config=config)
