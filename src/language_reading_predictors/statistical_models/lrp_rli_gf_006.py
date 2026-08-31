# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""LRPGF06 - gain factors for phoneme blending (blending) (B).

DAG-focused gain-factors model (#127; adjustment set re-derived against the revised
2026-07-10 DAG, ``dag/dag-language-reading.dagitty``, #247). Associations with how much
children gain in phoneme blending / phonological awareness (PA) across the three period
transitions (ANCOVA, Beta-Binomial logit, child random intercept).

Under the revised DAG the measured parents of PA are taught expressive vocabulary (TE),
expressive vocabulary (E) and letter sounds (L); its non-measure confounder parents are
hearing (HS), speech production (SP) and phonological memory (RW). So the adjustment set
is the own baseline + age + ability (blocks) + ``skill_symbols`` (L, E, TE) +
``adjust_for`` (hs, deapp_c, erbto and their missing indicators).

Only the randomised on-intervention term is causal — and its **period-1** average
marginal effect (the genuinely randomised, all-untreated-baseline transition) is an
available-case modified ITT estimate, not the all-transition pool (#247 P2). ``beta_trt`` itself is the
on-intervention log-odds contrast. Every other coefficient is an *adjusted association*:
the child random intercept is a partial, shrunken stand-in for between-child
heterogeneity — it does **not** control latent general ability, so those slopes remain
descriptive associations. SES excluded (non-DAG / redundant).

The causal headline is interaction-free (#391 finding 3 decision, 2026-07-22):
the pre-specified trt x ability / trt x own moderation questions live in the
associational variant LRPGF06m, and only the age x ability precision interaction
remains here.
"""

from language_reading_predictors.data_variables import Variables as V
from language_reading_predictors.statistical_models.context import ModelSpec
from language_reading_predictors.statistical_models.gain_factors import (
    GainFactorsModelSettings,
)
from language_reading_predictors.statistical_models.pipelines.gain_factors import fit_gain_factors

SPEC = ModelSpec(
    model_id="lrp-rli-gf-006",
    kind="gain_factors",
    title="Factors associated with gains in phoneme blending (blending) (B)",
    outcome_symbol="B",
    model_settings=GainFactorsModelSettings(
        skill_symbols=("L", "E", "TE"),
        ability_covariate=V.BLOCKS,
        adjust_for=(
            "hs",
            "hs_missing",
            "deapp_c",
            "deapp_c_missing",
            "erbto",
            "erbto_missing",
        ),
        interactions=(("age", "ability"),),
        treated_only=False,
    ),
)


def fit(config: str = "dev"):
    return fit_gain_factors(SPEC, config=config)
