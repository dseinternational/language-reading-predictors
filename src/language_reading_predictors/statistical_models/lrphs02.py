# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""LRPHS02 - regularized-horseshoe ranking cross-check: word-reading LEVEL (#116 Phase E).

The level companion to :mod:`lrphs01`: an independent Bayesian sensitivity check on
the gradient-boosting predictor ranking for word-reading **level** (GB
``lrpgbl12``). A regularized ("Finnish") horseshoe sparse regression regresses the
concurrent word-reading count on the standardised **same-wave** levels of the other
constructs (plus age), with a subject random intercept for the repeated timepoints;
all coefficients share a global-local shrinkage prior. Predictors are ranked by
posterior ``P(|beta| > delta)`` and compared to the GB **cluster** ranking (both at
construct level) in a dated ``notes/`` entry.

Concurrent-level framing (``phase_mode="levels"``, one row per child x timepoint,
n ~ 215 over 54 children): a high naive association is partly same-wave
same-construct correlation, so read the ranking as exploratory. This is a
**ranking cross-check, not a causal model**.

Construct predictor set: the non-floored concurrent measure levels (``L``, ``R``,
``E``, ``B``, ``F``, ``T``) plus age. Unlike the gain model this omits the RLI-blocks
/ behaviour covariates: they are wave-sparse (missing at baseline), and requiring
them collapses the levels frame from ~215 rows to 54 (one per child), destroying
the repeated-measures structure the subject random intercept needs. Params are
weakly-informative priors, not tuned.
"""

from language_reading_predictors.statistical_models.context import ModelSpec
from language_reading_predictors.statistical_models.pipeline import fit_horseshoe

_PREDICTORS = ["L", "R", "E", "B", "F", "T", "age"]

SPEC = ModelSpec(
    model_id="lrphs02",
    kind="horseshoe",
    title="Regularized-horseshoe ranking cross-check - word-reading level",
    outcome_symbol="W",
    extra={
        "gain": False,
        "predictors": _PREDICTORS,
        "covariates": [],
        "delta": 0.1,
        "target_accept": 0.99,
        "gb_reference": "lrpgbl12",
    },
)


def fit(config: str = "dev"):
    return fit_horseshoe(SPEC, config=config)
