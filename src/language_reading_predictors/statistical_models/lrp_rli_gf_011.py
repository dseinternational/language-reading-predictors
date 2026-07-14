# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""LRPGF11 - gain factors for nonword reading (N), off-floor.

DAG-focused gain-factors model (#225; adjustment set re-derived against the revised
2026-07-10 DAG, ``dag/dag-language-reading.dagitty``, #247). Associations with being
off the nonword-reading floor **at post** across the three period transitions.

Nonword reading (6 items) is heavily floored (most period post-scores are zero; the
at-floor rate runs 72% -> 40% across t1-t4), so a graded Beta-Binomial gain would be
leveraged by a few dispersed tail values rather than the factor contrasts. Following the
suite's floor rule (``likelihood="bernoulli_offfloor"``, as gf-005 phonetic spelling),
the outcome is modelled as being off the floor **at post** (post>0) — a Bernoulli on the
off-floor indicator that pools zero→positive movement, persistence above zero and
return-to-floor (not merely "coming off the floor"); ``beta_trt`` is the off-floor
log-odds and its items-scale marginal collapses to the off-floor risk difference.

Under the revised DAG the parents of NW are age, general ability, letter sounds (L),
blending (B), speech production (SP) and phonological memory (RW). So beyond the own
baseline, age and cognitive ability (blocks), the adjustment set adds the measured
upstream code skills L/B (``skill_symbols``, the causal-term counterpart of the
mech-072/172 L/B->N association) and the non-measure confounders speech and phonological
memory (``adjust_for``: ``deapp_c`` and ``erbto``; hearing/``hs`` is NOT a NW parent in
the DAG, so it is not conditioned on).

Only the randomised on-intervention term is causal — and its **period-1** average
marginal effect (the genuinely randomised, all-untreated-baseline transition) is the
ITT-anchor estimand, not the all-transition pool (#247 P2). ``beta_trt`` itself is the
on-intervention off-floor log-odds contrast. Every other coefficient is an *adjusted
association*: the child random intercept is a partial, shrunken stand-in for
between-child heterogeneity — it does **not** control latent general ability, so those
slopes remain descriptive associations. SES excluded (non-DAG / redundant). Flagged
off-floor in the report.
"""

from language_reading_predictors.data_variables import Variables as V
from language_reading_predictors.statistical_models.context import ModelSpec
from language_reading_predictors.statistical_models.pipeline import fit_gain_factors

SPEC = ModelSpec(
    model_id="lrp-rli-gf-011",
    kind="gain_factors",
    title="Factors associated with gains in nonword reading (N), off-floor",
    outcome_symbol="N",
    extra={
        "skill_symbols": ("L", "B"),
        "ability_covariate": V.BLOCKS,
        "adjust_for": ("deapp_c", "deapp_c_missing", "erbto", "erbto_missing"),
        "interactions": (("trt", "ability"), ("trt", "own"), ("age", "ability")),
        "treated_only": False,
        "likelihood": "bernoulli_offfloor",
    },
)


def fit(config: str = "dev"):
    return fit_gain_factors(SPEC, config=config)
