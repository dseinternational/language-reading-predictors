# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""LRPHS04 - regularized-horseshoe ranking cross-check: letter-sound LEVEL (#228 item 3).

The level companion to :mod:`lrp-rli-hs-003`, and the letter-sound counterpart to
:mod:`lrp-rli-hs-002` (word-reading level): an independent Bayesian sensitivity check on
the gradient-boosting predictor ranking for letter-sound **level** (GB
``lrp-rli-gbl-009``). A regularized ("Finnish") horseshoe sparse regression regresses
the concurrent letter-sound count on the standardised **same-wave** levels of the
other constructs (plus age), with a subject random intercept for the repeated
timepoints; all coefficients share a global-local shrinkage prior. Predictors are
ranked by posterior ``P(|beta| > delta)`` and compared to the GB **cluster** ranking.

Concurrent-level framing (``phase_mode="levels"``, one row per child x timepoint): a
high naive association is partly same-wave same-construct correlation, so read the
ranking as exploratory. This is a **ranking cross-check, not a causal model**.

Construct predictor set: the non-floored concurrent measure levels OTHER than the
outcome (``W``, ``R``, ``E``, ``B``, ``F``, ``T``) plus age. Like hs-002 this omits the
RLI-blocks / behaviour covariates (wave-sparse; requiring them would collapse the
levels frame to one row per child and destroy the repeated-measures structure the
subject random intercept needs). Weakly-informative priors, not tuned.
"""

from language_reading_predictors.statistical_models.context import ModelSpec
from language_reading_predictors.statistical_models.pipeline import fit_horseshoe

_PREDICTORS = ["W", "R", "E", "B", "F", "T", "age"]

SPEC = ModelSpec(
    model_id="lrp-rli-hs-004",
    kind="horseshoe",
    title="Regularized-horseshoe ranking cross-check - letter-sound level",
    outcome_symbol="L",
    extra={
        "gain": False,
        "predictors": _PREDICTORS,
        "covariates": [],
        "delta": 0.1,
        "target_accept": 0.99,
        "gb_reference": "lrp-rli-gbl-009",
    },
)


def fit(config: str = "dev"):
    return fit_horseshoe(SPEC, config=config)
