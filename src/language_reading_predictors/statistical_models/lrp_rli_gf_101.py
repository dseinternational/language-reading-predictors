# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""LRPGF01b - gain factors for word reading (W), treated-only ("gains while on
intervention").

DAG-focused gain-factors model (#127; adjustment set re-derived against the revised
2026-07-10 DAG, ``dag/dag-language-reading.dagitty``, #247). The LRPGF01 model
restricted to on-intervention period rows (excluding the waitlist arm's untreated
period 1). Associations with how much children gain in word reading (WR) across the
period transitions (ANCOVA, Beta-Binomial logit, child random intercept).

Under the revised DAG the measured parents of WR are taught receptive vocabulary (TR),
taught expressive vocabulary (TE), receptive vocabulary (R), expressive vocabulary (E),
letter sounds (L), nonword reading (N) and blending (B); its only non-measure parents
are age, general ability and intervention. So beyond the own baseline, age and
cognitive ability (blocks) the adjustment set adds those upstream skills
(``skill_symbols``) and no extra confounders (``adjust_for`` empty: WR has no
hearing/speech/phonological-memory parent).

Treated-only companion: every remaining row is on intervention, so in the treated-only
companion the on-intervention term is constant and dropped, along with its
interactions; every coefficient here is an adjusted association, never causal. The
child random intercept is a partial, shrunken stand-in for between-child heterogeneity
— it does **not** control latent general ability, so those slopes remain descriptive
associations. SES excluded (non-DAG / redundant). Compare its factor associations with
LRPGF01.
"""

from language_reading_predictors.data_variables import Variables as V
from language_reading_predictors.statistical_models.context import ModelSpec
from language_reading_predictors.statistical_models.pipeline import fit_gain_factors

SPEC = ModelSpec(
    model_id="lrp-rli-gf-101",
    kind="gain_factors",
    title="Gain factors for word reading (W), treated-only (gains while on intervention)",
    outcome_symbol="W",
    extra={
        "skill_symbols": ("TR", "TE", "R", "E", "L", "N", "B"),
        "ability_covariate": V.BLOCKS,
        "adjust_for": (),
        "interactions": (("trt", "ability"), ("trt", "own"), ("age", "ability")),
        "treated_only": True,
    },
)


def fit(config: str = "dev"):
    return fit_gain_factors(SPEC, config=config)
