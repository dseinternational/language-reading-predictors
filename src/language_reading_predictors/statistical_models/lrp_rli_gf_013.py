# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""LRPGF13 - gain factors for taught expressive vocabulary (TE), with broad vocabulary associates.

#421 Tier 1: extends ``gf-010`` (taught-expressive-vocabulary gains) with the broad
transfer-vocabulary skills. ``gf-010`` carries only ``skill_symbols = ("TR",)``, but the
letter-sound -> word-reading review found both receptive **and** expressive vocabulary
associated with taught-expressive gains (RV ≈ +0.28, EV ≈ +0.31, both P ≈ 0.98). This
model adds ``R`` and ``E`` to the existing ``TR`` term as adjusted associations.

The ``TR``/``R``/``E`` skill terms are **adjusted associations, not DAG-parent
adjustments** - descriptive of the review's finding, latent-GA-confounded. Only the
randomised on-intervention term is causal (its period-1 average marginal effect);
everything else is descriptive. Report median + inner 50% + outer 89% credible interval
+ P(>0) with that caveat.
"""

from language_reading_predictors.data_variables import Variables as V
from language_reading_predictors.statistical_models.context import ModelSpec
from language_reading_predictors.statistical_models.pipeline import fit_gain_factors

SPEC = ModelSpec(
    model_id="lrp-rli-gf-013",
    kind="gain_factors",
    title="Factors associated with gains in taught expressive vocabulary (TE), with broad vocabulary associates",
    outcome_symbol="TE",
    extra={
        # gf-010's TR plus broad receptive/expressive vocabulary (the review's finding).
        "skill_symbols": ("TR", "R", "E"),
        "ability_covariate": V.BLOCKS,
        # TE's non-measure confounders (matches gf-010): hearing, speech, phon. memory.
        "adjust_for": (
            "hs", "hs_missing", "deapp_c", "deapp_c_missing", "erbto", "erbto_missing"
        ),
        "interactions": (("trt", "ability"), ("trt", "own"), ("age", "ability")),
        "treated_only": False,
    },
)


def fit(config: str = "dev"):
    return fit_gain_factors(SPEC, config=config)
