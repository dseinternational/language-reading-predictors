# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""LRP76 - taught vs not-taught expressive vocabulary (block 1): generalisation.

A within-data test of the trial's "gains are largest in directly-taught skills,
with little generalisation" finding. The two Block 1 expressive outcomes - the
directly-taught target words (``TE`` = b1extau) and the not-taught comparison
words (``UE`` = b1exnt) - are modelled jointly over the randomised window, and
the headline quantity is the difference in intervention benefit between them.

Sign note: ``tau`` is the coefficient on the intervention indicator
(``G = 2 - group``; group 1 receives the intervention from t1), so a *positive*
``tau`` means the intervention raised that outcome. The reported contrast is
therefore ``tau[TE] - tau[UE]`` (= benefit on taught minus benefit on not-taught):
a *positive* value means the programme moved the words it taught *more* than
untaught words - i.e. limited transfer/generalisation.
"""

from language_reading_predictors.statistical_models.context import ModelSpec
from language_reading_predictors.statistical_models.pipeline import fit_joint

SPEC = ModelSpec(
    model_id="lrp76",
    kind="joint",
    title=(
        "Taught vs not-taught expressive vocabulary (block 1): generalisation "
        "contrast"
    ),
    # Two-outcome joint Beta-Binomial over the randomised window. A 2x2 LKJ
    # residual correlation models the within-child taught/not-taught dependence
    # (identifiable here, unlike the prior-dominated 8-outcome LRP55 block) so the
    # difference interval is not needlessly inflated; toggled off in the report's
    # sensitivity note if it destabilises sampling. ``difference=("TE","UE")``
    # asks the pipeline to summarise tau[TE] - tau[UE] (benefit on taught minus
    # benefit on not-taught; positive => limited generalisation).
    extra={
        "outcomes": ("TE", "UE"),
        "use_residual_correlation": True,
        "difference": ("TE", "UE"),
    },
)


def fit(config: str = "dev"):
    return fit_joint(SPEC, config=config)
