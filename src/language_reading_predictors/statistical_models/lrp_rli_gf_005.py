# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""LRPGF05 - gain factors for phonetic spelling (P), off-floor.

Phonetic spelling is heavily floored (most period post-scores are zero), so a
graded Beta-Binomial gain would be leveraged by a few dispersed tail values rather
than the factor contrasts. Following the suite's floor rule, the likelihood is a
**Bernoulli on the binary off-floor indicator** (period post > 0). This models being
**off the floor at post**, not *coming off* the floor: it pools zero→positive movement,
persistence above zero, and return-to-floor across the retained transitions. ``beta_trt``
is the off-floor log-odds, and its **period-1** items-scale marginal collapses to the
off-floor risk difference for the randomised transition (#247 P2).

DAG-focused factors (adjustment set re-derived against the revised 2026-07-10 DAG,
``dag/dag-language-reading.dagitty``, #247): on-intervention (causal), own baseline, age,
ability (blocks), the upstream skills letter sounds (L) + blending (B), and the
non-measure confounder phonological memory (RW) via ``adjust_for`` (erbto). Every
non-causal coefficient is an *adjusted association*: the child random intercept is a
partial, shrunken stand-in for between-child heterogeneity — it does **not** control
latent general ability. SES excluded (non-DAG / redundant). Flagged off-floor in the
report.
"""

from language_reading_predictors.data_variables import Variables as V
from language_reading_predictors.statistical_models.context import ModelSpec
from language_reading_predictors.statistical_models.pipeline import fit_gain_factors

SPEC = ModelSpec(
    model_id="lrp-rli-gf-005",
    kind="gain_factors",
    title="Factors associated with gains in phonetic spelling (P), off-floor",
    outcome_symbol="P",
    extra={
        "skill_symbols": ("L", "B"),
        "ability_covariate": V.BLOCKS,
        "adjust_for": ("erbto", "erbto_missing"),
        "interactions": (("trt", "ability"), ("trt", "own"), ("age", "ability")),
        "treated_only": False,
        "likelihood": "bernoulli_offfloor",
    },
)


def fit(config: str = "dev"):
    return fit_gain_factors(SPEC, config=config)
