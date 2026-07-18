# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""LRPITT23 - general-ability-adjusted ITT for letter-sound knowledge (YARC-LSK) (L).

Ability-robustness companion to LRPITT07, part of the LRPITT17-24 ability-adjusted
family (parallel to the SES-adjusted LRPITT13/13b). Adds the baseline nonverbal
cognitive ability measure - block design (``blocks``), recorded at t1 only - as a
linear precision covariate, on top of the own baseline and linear age the uniform
LRPITT spec already carries.

Block design is a child trait measured *before* randomisation, so - like SES - it
cannot confound the randomised effect; randomisation balances it across arms in
expectation. This adjustment is a precision / chance-imbalance robustness check, not
confounding control (the immediate-intervention arm started ~0.27 SD higher in block
design, and ability is prognostic of the outcomes - most strongly for vocabulary).
Holding age in makes the block-design term read as ability *for age* - the general
latent ability construct (raw block design conflates ability with maturation).

Block design is complete for all 54 children, so no rows drop: LRPITT23 vs LRPITT07
is a same-sample adjusted-vs-unadjusted contrast and no matched comparator is needed.

Sign convention: positive tau => intervention helps.
"""

from language_reading_predictors.data_variables import Variables as V
from language_reading_predictors.statistical_models.context import ModelSpec
from language_reading_predictors.statistical_models.itt import IttModelSettings
from language_reading_predictors.statistical_models.pipeline import fit_itt

# Baseline nonverbal cognitive ability (block design), measured at t1 only.
ABILITY_ADJUSTER = (V.BLOCKS,)

SPEC = ModelSpec(
    model_id="lrp-rli-itt-023",
    kind="itt",
    title="Ability-adjusted ITT effect of group assignment on letter-sound knowledge (YARC-LSK) (L)",
    outcome_symbol="L",
    adjustment=list(ABILITY_ADJUSTER),
    model_settings=IttModelSettings(adjust_for=ABILITY_ADJUSTER),
)


def fit(config: str = "dev"):
    return fit_itt(SPEC, config=config)
