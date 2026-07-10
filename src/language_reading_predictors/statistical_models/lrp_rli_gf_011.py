# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""LRPGF11 - gain factors for nonword reading (N), off-floor.

Nonword reading (6 items) is heavily floored (most period post-scores are zero;
the at-floor rate runs 72% -> 40% across t1-t4), so a graded Beta-Binomial gain
would be leveraged by a few dispersed tail values rather than the factor
contrasts. Following the suite's floor rule (as gf-005 phonetic spelling), the
likelihood is a **Bernoulli on the binary off-floor indicator** (period post > 0):
`beta_trt` is the off-floor log-odds and its items-scale marginal collapses to the
off-floor risk difference. DAG-focused factors: on-intervention (causal), own
baseline, age, ability (blocks), and the upstream code skills letter sounds (L) +
blending (B) - the causal-term counterpart of the mech-072/172 L/B->N association.
SES excluded (non-DAG / redundant). Flagged off-floor in the report.
"""

from language_reading_predictors.data_variables import Variables as V
from language_reading_predictors.statistical_models.context import ModelSpec
from language_reading_predictors.statistical_models.pipeline import fit_gain_factors

SPEC = ModelSpec(
    model_id="lrp-rli-gf-011",
    kind="gain_factors",
    title="Gain factors for nonword reading (N), off-floor",
    outcome_symbol="N",
    extra={
        "skill_symbols": ("L", "B"),
        "ability_covariate": V.BLOCKS,
        "interactions": (("trt", "ability"), ("trt", "own"), ("age", "ability")),
        "treated_only": False,
        "likelihood": "bernoulli_offfloor",
    },
)


def fit(config: str = "dev"):
    return fit_gain_factors(SPEC, config=config)
