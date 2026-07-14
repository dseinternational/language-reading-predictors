# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""LRPBX04 - block-2 NOT-taught RECEPTIVE vocabulary, block-active exposure (UR2).

Block-2 not-taught receptive vocabulary (``b2rent``): the receptive
generalisation / specificity comparator to the taught receptive model
(``lrp_rli_bx_002``). Same staggered block-active exposure structure and
parallel-trends caveat (see ``build_block_exposure_model``); ``delta`` is an
ASSOCIATION and, as with TR2, is expected to be weak (receptive near-ceiling).

**Corrupt source cell.** ``b2rent`` has one impossible value (subject
``ID_0D60E282E4368506`` at t4: ``b2rent=31`` against the 12-item ceiling, with
``b2reto=10 < b2retau=21`` — total below taught, so the whole receptive-not-taught
triple for that row is unrecoverable). ``drop_ceiling_violations=("UR2",)`` sets that
single cell to NaN (dropped as missing, logged) instead of raising the ceiling guard;
the denominator stays 12. Flagged to the data owner for a source fix.
"""

from language_reading_predictors.data_variables import Variables as V
from language_reading_predictors.statistical_models.context import ModelSpec
from language_reading_predictors.statistical_models.pipeline import fit_block_exposure

SPEC = ModelSpec(
    model_id="lrp-rli-bx-004",
    kind="block_exposure",
    title="Block-2 not-taught receptive vocabulary, block-active exposure (UR2)",
    outcome_symbol="UR2",
    extra={
        "ability_covariate": V.BLOCKS,
        "adjust_for": ("hs", "hs_missing", "erbto", "erbto_missing"),
        "use_child_re": True,
        "drop_ceiling_violations": ("UR2",),
    },
)


def fit(config: str = "dev"):
    return fit_block_exposure(SPEC, config=config)
