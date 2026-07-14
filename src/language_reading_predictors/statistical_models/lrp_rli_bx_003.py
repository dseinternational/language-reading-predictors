# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""LRPBX03 - block-2 NOT-taught EXPRESSIVE vocabulary, block-active exposure (UE2).

Block-2 not-taught expressive vocabulary (``b2exnt``): the expressive
generalisation / specificity comparator to the taught expressive model
(``lrp_rli_bx_001``). Same staggered block-active exposure structure and the same
parallel-trends caveat (see ``build_block_exposure_model``); ``delta`` is an
ASSOCIATION.

Placebo logic: because these words were **not** taught in block 2, a block-2
exposure effect on them should be ≈ 0. A positive ``delta`` on the taught (TE2)
outcome with a null ``delta`` here is the within-family intervention-fidelity /
specificity check (taught > not-taught). Adjusters match the taught expressive model
so the only difference is the outcome.
"""

from language_reading_predictors.data_variables import Variables as V
from language_reading_predictors.statistical_models.context import ModelSpec
from language_reading_predictors.statistical_models.pipeline import fit_block_exposure

SPEC = ModelSpec(
    model_id="lrp-rli-bx-003",
    kind="block_exposure",
    title="Block-2 not-taught expressive vocabulary, block-active exposure (UE2)",
    outcome_symbol="UE2",
    extra={
        "ability_covariate": V.BLOCKS,
        "adjust_for": ("hs", "hs_missing", "deapp_c", "deapp_c_missing", "erbto", "erbto_missing"),
        "use_child_re": True,
    },
)


def fit(config: str = "dev"):
    return fit_block_exposure(SPEC, config=config)
