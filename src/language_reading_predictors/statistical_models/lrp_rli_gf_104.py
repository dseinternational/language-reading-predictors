# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""LRPGF04b - gain factors for letter sounds (L), treated-only.

DAG-focused gain-factors model (#127; adjustment set re-derived against the revised
2026-07-10 DAG, ``dag/dag-language-reading.dagitty``, #247). The LRPGF04 model
restricted to on-intervention period rows (excluding the waitlist arm's untreated
period 1). Associations with how much children gain in letter sounds (LS) across the
period transitions (ANCOVA, Beta-Binomial logit, child random intercept).

Under the revised DAG the parents of LS are age, general ability, intervention, hearing
(HS) and speech production (SP) — no measured upstream skill is a parent, so
``skill_symbols`` is empty and beyond the own baseline, age and cognitive ability
(blocks) the adjustment set adds only the non-measure confounders hearing and speech
(``adjust_for``: ``hs`` and ``deapp_c``; phonological memory/``erbto`` is not an LS
parent).

Treated-only companion: every remaining row is on intervention, so in the treated-only
companion the on-intervention term is constant and dropped, along with its
interactions; every coefficient here is an adjusted association, never causal. The
child random intercept is a partial, shrunken stand-in for between-child heterogeneity
— it does **not** control latent general ability, so those slopes remain descriptive
associations. SES excluded (non-DAG / redundant). Compare its factor associations with
LRPGF04.
"""

from language_reading_predictors.data_variables import Variables as V
from language_reading_predictors.statistical_models.context import ModelSpec
from language_reading_predictors.statistical_models.pipeline import fit_gain_factors

SPEC = ModelSpec(
    model_id="lrp-rli-gf-104",
    kind="gain_factors",
    title="Gain factors for letter sounds (L), treated-only (gains while on intervention)",
    outcome_symbol="L",
    extra={
        "skill_symbols": (),
        "ability_covariate": V.BLOCKS,
        "adjust_for": ("hs", "hs_missing", "deapp_c", "deapp_c_missing"),
        "interactions": (("trt", "ability"), ("trt", "own"), ("age", "ability")),
        "treated_only": True,
    },
)


def fit(config: str = "dev"):
    return fit_gain_factors(SPEC, config=config)
