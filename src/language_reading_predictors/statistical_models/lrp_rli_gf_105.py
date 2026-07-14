# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""LRPGF05b - gain factors for phonetic spelling (P), off-floor, treated-only.

DAG-focused gain-factors model (#127; adjustment set re-derived against the revised
2026-07-10 DAG, ``dag/dag-language-reading.dagitty``, #247). The off-floor LRPGF05
restricted to on-intervention period rows (excluding the waitlist arm's untreated
period 1). Associations with the off-floor status of phonetic spelling (PS) across the
period transitions (Beta-Binomial logit with the suite floor rule, child random
intercept).

Under the revised DAG the parents of PS are age, general ability, intervention, letter
sounds (L), blending (B) and phonological memory (RW). So beyond the own baseline, age
and cognitive ability (blocks), the adjustment set adds the measured upstream skills
L/B (``skill_symbols``) and the non-measure confounder phonological memory
(``adjust_for``: ``erbto``; hearing and speech are not PS parents).

P is heavily floored, so it takes the suite floor rule (``likelihood="bernoulli_offfloor"``):
the outcome is modelled as being off the floor **at post** (post>0) — a Bernoulli on
the off-floor indicator that pools zero→positive movement, persistence above zero and
return-to-floor (not merely "coming off the floor").

Treated-only companion: every remaining row is on intervention, so in the treated-only
companion the on-intervention term is constant and dropped, along with its
interactions; every coefficient here is an adjusted association, never causal. The
child random intercept is a partial, shrunken stand-in for between-child heterogeneity
— it does **not** control latent general ability, so those slopes remain descriptive
associations. SES excluded (non-DAG / redundant). Compare its factor associations with
LRPGF05.
"""

from language_reading_predictors.data_variables import Variables as V
from language_reading_predictors.statistical_models.context import ModelSpec
from language_reading_predictors.statistical_models.pipeline import fit_gain_factors

SPEC = ModelSpec(
    model_id="lrp-rli-gf-105",
    kind="gain_factors",
    title="Factors associated with gains in phonetic spelling (P), off-floor, treated-only",
    outcome_symbol="P",
    extra={
        "skill_symbols": ("L", "B"),
        "ability_covariate": V.BLOCKS,
        "adjust_for": ("erbto", "erbto_missing"),
        "interactions": (("trt", "ability"), ("trt", "own"), ("age", "ability")),
        "treated_only": True,
        "likelihood": "bernoulli_offfloor",
    },
)


def fit(config: str = "dev"):
    return fit_gain_factors(SPEC, config=config)
