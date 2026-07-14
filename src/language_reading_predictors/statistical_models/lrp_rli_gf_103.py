# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""LRPGF03b - gain factors for expressive vocabulary (E), treated-only.

DAG-focused gain-factors model (#127; adjustment set re-derived against the revised
2026-07-10 DAG, ``dag/dag-language-reading.dagitty``, #247). The LRPGF03 model
restricted to on-intervention period rows (excluding the waitlist arm's untreated
period 1). Associations with how much children gain in expressive vocabulary (EV)
across the period transitions (ANCOVA, Beta-Binomial logit, child random intercept).

Under the revised DAG the parents of EV are age, general ability, hearing (HS), speech
production (SP), phonological memory (RW), receptive vocabulary (R), taught receptive
vocabulary (TR) and taught expressive vocabulary (TE). So beyond the own baseline, age
and cognitive ability (blocks), the adjustment set adds the measured upstream skills
R/TR/TE (``skill_symbols``) and the non-measure confounders hearing, speech and
phonological memory (``adjust_for``: ``hs``, ``deapp_c`` and ``erbto``).

Treated-only companion: every remaining row is on intervention, so in the treated-only
companion the on-intervention term is constant and dropped, along with its
interactions; every coefficient here is an adjusted association, never causal. The
child random intercept is a partial, shrunken stand-in for between-child heterogeneity
— it does **not** control latent general ability, so those slopes remain descriptive
associations. SES excluded (non-DAG / redundant). Compare its factor associations with
LRPGF03.
"""

from language_reading_predictors.data_variables import Variables as V
from language_reading_predictors.statistical_models.context import ModelSpec
from language_reading_predictors.statistical_models.pipeline import fit_gain_factors

SPEC = ModelSpec(
    model_id="lrp-rli-gf-103",
    kind="gain_factors",
    title="Factors associated with gains in expressive vocabulary (E), treated-only (gains while on intervention)",
    outcome_symbol="E",
    extra={
        "skill_symbols": ("R", "TR", "TE"),
        "ability_covariate": V.BLOCKS,
        "adjust_for": ("hs", "hs_missing", "deapp_c", "deapp_c_missing", "erbto", "erbto_missing"),
        "interactions": (("trt", "ability"), ("trt", "own"), ("age", "ability")),
        "treated_only": True,
    },
)


def fit(config: str = "dev"):
    return fit_gain_factors(SPEC, config=config)
