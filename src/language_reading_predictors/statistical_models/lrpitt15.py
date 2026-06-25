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
    model_id="lrpitt15",
    kind="joint",
    title="Generalisation contrast (expressive): taught vs not-taught vocabulary, block 1",
    # Two-outcome joint Beta-Binomial over the randomised window; the 2x2 LKJ
    # residual correlation models the within-child taught/not-taught dependence
    # (identifiable at K=2, unlike the prior-dominated larger joints) so the
    # difference interval is not needlessly inflated. ``difference=("TE","UE")``
    # asks the pipeline to summarise tau[TE] - tau[UE].
    extra={
        "outcomes": ("TE", "UE"),
        "use_residual_correlation": True,
        "difference": ("TE", "UE"),
    },
)


def fit(config: str = "dev"):
    return fit_joint(SPEC, config=config)
