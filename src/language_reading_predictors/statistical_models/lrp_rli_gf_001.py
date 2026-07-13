# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""LRPGF01 - gain factors for word reading (W).

DAG-focused factor model (#127; adjustment set re-derived against the revised
2026-07-10 DAG, ``dag/dag-language-reading.dagitty``, #247). What is associated with
how much children gain in word reading (WR) across the three period transitions
(ANCOVA gain, Beta-Binomial on the logit scale, child random intercept).

Under the revised DAG the measured parents of WR are taught receptive vocabulary (TR),
taught expressive vocabulary (TE), receptive vocabulary (R), expressive vocabulary (E),
letter sounds (L), nonword reading (N) and blending (B) — including the new direct
vocab→reading edges (TE→WR, EV→WR). All seven enter as period-baseline ``skill_symbols``
(the full-DAG-parent adjustment agreed for #247). WR has no non-measure confounder
parent (hearing/speech/phonological memory are not WR parents), so ``adjust_for`` is
empty.

Only the randomised on-intervention term is causal — and its **period-1** average
marginal effect (the genuinely randomised, all-untreated-baseline transition) is the
ITT-anchor estimand, not the all-transition pool (#247 P2). ``beta_trt`` itself is the
on-intervention log-odds contrast. The own baseline, age, cognitive ability (blocks) and
every upstream skill are *adjusted associations*: the child random intercept is a
partial, shrunken stand-in for between-child heterogeneity — it does **not** control
latent general ability, so those slopes remain descriptive associations. SES is excluded
(not a DAG node, statistically redundant). Focal interactions: group x ability,
group x own-baseline, age x ability.
"""

from language_reading_predictors.data_variables import Variables as V
from language_reading_predictors.statistical_models.context import ModelSpec
from language_reading_predictors.statistical_models.pipeline import fit_gain_factors

SPEC = ModelSpec(
    model_id="lrp-rli-gf-001",
    kind="gain_factors",
    title="Gain factors for word reading (W)",
    outcome_symbol="W",
    extra={
        "skill_symbols": ("TR", "TE", "R", "E", "L", "N", "B"),
        "ability_covariate": V.BLOCKS,
        "adjust_for": (),
        "interactions": (("trt", "ability"), ("trt", "own"), ("age", "ability")),
        "treated_only": False,
    },
)


def fit(config: str = "dev"):
    return fit_gain_factors(SPEC, config=config)
