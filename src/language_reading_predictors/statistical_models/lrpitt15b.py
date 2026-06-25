# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""LRPITT15b - generalisation contrast (receptive): taught vs not-taught.

Receptive companion to LRPITT15: the two Block 1 receptive outcomes — directly
taught (``TR`` = b1retau) and not-taught (``UR`` = b1rent) — modelled jointly over
the randomised window, with headline contrast ``tau[TR] - tau[UR]`` (benefit on
taught minus benefit on not-taught; positive => limited generalisation). Sign
convention: positive tau => intervention helps.
"""

from language_reading_predictors.statistical_models.context import ModelSpec
from language_reading_predictors.statistical_models.pipeline import fit_joint

SPEC = ModelSpec(
    model_id="lrpitt15b",
    kind="joint",
    title="Generalisation contrast (receptive): taught vs not-taught vocabulary, block 1",
    # LKJ residual correlation is OFF (same spec as LRPITT15): the within-child
    # residual SD sits at ~0 once baselines are conditioned on, so the LKJ block
    # only destabilised the sigma_outcome variance component (R-hat ~1.01) for no
    # gain. Dropping it is the documented conservative fallback and converges
    # cleanly.
    extra={
        "outcomes": ("TR", "UR"),
        "use_residual_correlation": False,
        "difference": ("TR", "UR"),
    },
)


def fit(config: str = "dev"):
    return fit_joint(SPEC, config=config)
