# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""LRPITT15 - generalisation contrast (expressive): taught vs not-taught (migrates LRP76).

A within-data test of the trial's "gains are largest in directly-taught skills,
with little generalisation" finding. The two Block 1 expressive outcomes — the
directly-taught target words (``TE`` = b1extau) and the not-taught comparison words
(``UE`` = b1exnt) — are modelled jointly over the randomised window, and the
headline quantity is the difference in intervention benefit between them.

Sign convention: ``tau`` is the coefficient on the intervention indicator
(positive => the intervention raised that outcome). The reported contrast is
``tau[TE] - tau[UE]`` (benefit on taught minus benefit on not-taught): a positive
value means the programme moved the words it taught *more* than untaught words —
i.e. limited transfer/generalisation. The receptive companion is LRPITT15b.
"""

from language_reading_predictors.statistical_models.context import ModelSpec
from language_reading_predictors.statistical_models.pipeline import fit_joint

SPEC = ModelSpec(
    model_id="lrp-rli-itt-015",
    kind="joint",
    title="Generalisation contrast (expressive): taught vs not-taught vocabulary, block 1",
    # Two-outcome joint Beta-Binomial over the randomised window.
    # ``difference=("TE","UE")`` asks the pipeline to summarise tau[TE] - tau[UE].
    # LKJ residual correlation is OFF: once the baselines are conditioned on, the
    # within-child taught/not-taught residual SD sits at ~0 (the boundary), which
    # left sigma_outcome poorly mixed (R-hat ~1.01) for no gain in the difference.
    # Dropping it is the documented conservative fallback (the difference is then
    # marginally wider, ignoring a near-zero correlation) and converges cleanly;
    # the receptive companion LRPITT15b uses the same specification.
    extra={
        "outcomes": ("TE", "UE"),
        # DAG-faithful spec, mirroring the single-outcome suite (own baseline +
        # linear age, no cross-baselines).
        "use_cross_baselines": False,
        "use_age_linear": True,
        "use_residual_correlation": False,
        "difference": ("TE", "UE"),
    },
)


def fit(config: str = "dev"):
    return fit_joint(SPEC, config=config)
