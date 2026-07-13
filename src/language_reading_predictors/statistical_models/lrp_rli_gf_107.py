# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""LRPGF07b - gain factors for CELF basic concepts (F), treated-only.

DAG-focused gain-factors model (#127; adjustment set re-derived against the revised
2026-07-10 DAG, ``dag/dag-language-reading.dagitty``, #247). The LRPGF07 model
restricted to on-intervention period rows (excluding the waitlist arm's untreated
period 1). Associations with how much children gain in language fundamentals / CELF
basic concepts (LF) across the period transitions (ANCOVA, Beta-Binomial logit, child
random intercept).

Under the revised DAG the parents of LF are age, general ability, receptive vocabulary
(R) and taught receptive vocabulary (TR). So beyond the own baseline, age and cognitive
ability (blocks), the adjustment set adds the measured upstream skills R/TR
(``skill_symbols``) and no extra confounders (``adjust_for`` empty: LF has no
hearing/speech/phonological-memory parent).

Treated-only companion: every remaining row is on intervention, so in the treated-only
companion the on-intervention term is constant and dropped, along with its
interactions; every coefficient here is an adjusted association, never causal. The
child random intercept is a partial, shrunken stand-in for between-child heterogeneity
— it does **not** control latent general ability, so those slopes remain descriptive
associations. SES excluded (non-DAG / redundant). Compare its factor associations with
LRPGF07.
"""

from language_reading_predictors.data_variables import Variables as V
from language_reading_predictors.statistical_models.context import ModelSpec
from language_reading_predictors.statistical_models.pipeline import fit_gain_factors

SPEC = ModelSpec(
    model_id="lrp-rli-gf-107",
    kind="gain_factors",
    title="Gain factors for CELF basic concepts (F), treated-only (gains while on intervention)",
    outcome_symbol="F",
    extra={
        "skill_symbols": ("R", "TR"),
        "ability_covariate": V.BLOCKS,
        "adjust_for": (),
        "interactions": (("trt", "ability"), ("trt", "own"), ("age", "ability")),
        "treated_only": True,
    },
)


def fit(config: str = "dev"):
    return fit_gain_factors(SPEC, config=config)
