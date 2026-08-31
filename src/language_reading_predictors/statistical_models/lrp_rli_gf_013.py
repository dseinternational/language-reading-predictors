# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""LRPGF13 - gain factors for taught expressive vocabulary (TE), with broad vocabulary associates.

#421 Tier 1: extends ``gf-010`` (taught-expressive-vocabulary gains) with the broad
transfer-vocabulary skills. ``gf-010`` carries only ``skill_symbols = ("TR",)``, but the
letter-sound -> word-reading review found both receptive **and** expressive vocabulary
associated with taught-expressive gains. This model adds ``R`` and ``E`` alongside the
retained ``TR`` term.

The three skill terms have **two different roles**, and should not be read as one block:

* ``TR`` (taught receptive vocabulary) is an **upstream measured DAG parent of TE**,
  inherited unchanged from ``gf-010``'s adjustment set (revised 2026-07-10 DAG, #247) -
  it is here as a confounder adjustment, exactly as in the parent model.
* ``R``/``E`` (standardised receptive/expressive vocabulary) are **downstream descriptive
  associates**, not DAG-parent adjustments: under the revised DAG they sit downstream of
  taught vocabulary (``TR -> RV``), and they are entered only to *describe* the review's
  finding.

Both roles are still *adjusted associations*, latent-GA-confounded - only the randomised
on-intervention term is causal (its period-1 average marginal effect). Report median +
inner 50% + outer 89% credible interval + P(>0) with that caveat, and read the skill
slopes as "associated with", never "drives".

The causal headline is interaction-free (#391 finding 3 decision, 2026-07-22):
the trt x ability / trt x own moderation questions for TE are carried by the
associational variant LRPGF10m (anchored to the per-outcome primary; this alternate
adjustment set declares none), and only the age x ability precision interaction
remains here.
"""

from language_reading_predictors.data_variables import Variables as V
from language_reading_predictors.statistical_models.context import ModelSpec
from language_reading_predictors.statistical_models.gain_factors import (
    GainFactorsModelSettings,
)
from language_reading_predictors.statistical_models.pipelines.gain_factors import fit_gain_factors

SPEC = ModelSpec(
    model_id="lrp-rli-gf-013",
    kind="gain_factors",
    title="Factors associated with gains in taught expressive vocabulary (TE), with broad vocabulary associates",
    outcome_symbol="TE",
    model_settings=GainFactorsModelSettings(
        # gf-010's upstream DAG parent TR (retained as a confounder adjustment) plus the
        # downstream descriptive associates R/E (the review's finding).
        skill_symbols=("TR", "R", "E"),
        # Only R/E carry the descriptive role; TR stays a DAG-parent adjuster
        # exactly as in gf-010 (#575 finding 9).
        descriptive_skills=("R", "E"),
        ability_covariate=V.BLOCKS,
        # TE's non-measure confounders (matches gf-010): hearing, speech, phon. memory.
        adjust_for=(
            "hs", "hs_missing", "deapp_c", "deapp_c_missing", "erbto", "erbto_missing"
        ),
        interactions=(("age", "ability"),),
        treated_only=False,
    ),
)


def fit(config: str = "dev"):
    return fit_gain_factors(SPEC, config=config)
