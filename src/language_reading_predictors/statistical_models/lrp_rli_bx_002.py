# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""LRPBX02 - block-2 taught RECEPTIVE vocabulary, block-active exposure (TR2).

Block-2 taught receptive vocabulary (``b2retau``), fitted as the same staggered
block-active exposure association as ``lrp_rli_bx_001`` (see that module and
``build_block_exposure_model`` for the estimand and the parallel-trends caveat).
The focal ``delta`` is an ASSOCIATION, not a randomised effect.

Receptive taught vocabulary is near-ceiling and flat across the arms (t3: 16.68 vs
16.62 of 24), so ``delta`` here is expected to be weak/null — the block-2 exposure
effect is expressive-specific, mirroring the block-1 taught pattern (TE >> TR).
Adjusters mirror the block-1 taught receptive model (lf-009): hearing (HS) and
phonological memory (RW, ``erbto``) only; speech production (``deapp_c``) is a
production confounder and is not carried for a receptive outcome.
"""

from language_reading_predictors.data_variables import Variables as V
from language_reading_predictors.statistical_models.context import ModelSpec
from language_reading_predictors.statistical_models.pipeline import fit_block_exposure

SPEC = ModelSpec(
    model_id="lrp-rli-bx-002",
    kind="block_exposure",
    title="Block-2 taught receptive vocabulary, block-active exposure (TR2)",
    outcome_symbol="TR2",
    extra={
        "ability_covariate": V.BLOCKS,
        "adjust_for": ("hs", "hs_missing", "erbto", "erbto_missing"),
        "use_child_re": True,
    },
)


def fit(config: str = "dev"):
    return fit_block_exposure(SPEC, config=config)
