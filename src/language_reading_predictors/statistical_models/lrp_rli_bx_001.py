# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""LRPBX01 - block-2 taught EXPRESSIVE vocabulary, block-active exposure (TE2).

Block 2 of the taught-vocabulary programme (``b2extau``) has no t1 baseline and no
randomised contrast, so this fits a staggered block-active exposure association
(#228 item 5): the immediate arm is taught block 2 in phase 2 (measured at t3) while
the wait-list arm is still on block 1; the wait-list arm reaches block 2 in phase 3
(t4). The focal ``delta`` is the shift in the block-2 taught-word level attributable
to block 2 being actively taught — an ASSOCIATION (parallel-trends), never a
randomised effect, and the contrast is "block-2-active vs block-1-active", not
treated-vs-untreated. A child random intercept and per-timepoint intercepts absorb
stable between-child differences and the secular trend; age-at-block-2 and
arm-specific slopes remain confounded (see ``build_block_exposure_model``).

TE2 (expressive) is the informative outcome: the block-2 divergence appears at t3
(immediate 9.89 vs wait-list 8.46 items, equal at t2). Adjusters mirror the block-1
taught expressive model (lf-010): hearing (HS), speech production (SP, ``deapp_c``)
and phonological memory (RW, ``erbto``) enter as linear precision/confounder terms;
SP/RW are read at the pre-randomisation baseline (#247, A1).
"""

from language_reading_predictors.data_variables import Variables as V
from language_reading_predictors.statistical_models.context import ModelSpec
from language_reading_predictors.statistical_models.pipeline import fit_block_exposure

SPEC = ModelSpec(
    model_id="lrp-rli-bx-001",
    kind="block_exposure",
    title="Block-2 taught expressive vocabulary, block-active exposure (TE2)",
    outcome_symbol="TE2",
    extra={
        "ability_covariate": V.BLOCKS,
        "adjust_for": ("hs", "hs_missing", "deapp_c", "deapp_c_missing", "erbto", "erbto_missing"),
        "use_child_re": True,
    },
)


def fit(config: str = "dev"):
    return fit_block_exposure(SPEC, config=config)
