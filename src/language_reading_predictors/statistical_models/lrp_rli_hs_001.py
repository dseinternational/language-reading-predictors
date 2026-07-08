# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""LRPHS01 - regularized-horseshoe ranking cross-check: word-reading GAIN (#116 Phase E).

An **independent Bayesian sensitivity check** on the gradient-boosting predictor
ranking for word-reading gain (GB ``lrp-rli-gbg-012``). A regularized ("Finnish")
horseshoe sparse regression (Piironen & Vehtari 2017) regresses the word-reading
post-count on its own T1 baseline (``gamma_own``) plus the standardised T1
baselines of the other constructs, all sharing a global-local shrinkage prior.
Predictors are ranked by posterior ``P(|beta| > delta)`` and compared to the GB
**cluster** ranking (both at construct level) in a dated ``notes/`` entry.

Between-child framing (``phase_mode="span"``, one row per child), mirroring LRP65:
the coefficients are between-child associations, not within-child change. This is a
**ranking cross-check, not a causal model** - if the horseshoe broadly agrees with
the GB order, that is reassurance the ranking is not a tree-method artefact; the
randomised causal claim continues to live in the ITT suite.

Construct predictor set: the non-floored measure baselines (letter sounds ``L``,
receptive/expressive vocabulary ``R``/``E``, blending ``B``, basic concepts ``F``,
receptive grammar ``T``) plus age and the RLI-blocks / behaviour covariates.
Floored / post-only measures (``P`` spelling, ``N`` nonword) and the block-1
taught/not-taught vocabulary (unconfirmed denominators, #144) are left out of the
pilot. Params are weakly-informative priors, not tuned. n ~ 54; intervals are wide.
"""

from language_reading_predictors.statistical_models.context import ModelSpec
from language_reading_predictors.statistical_models.pipeline import fit_horseshoe

_PREDICTORS = ["L", "R", "E", "B", "F", "T", "age", "blocks", "behav"]

SPEC = ModelSpec(
    model_id="lrp-rli-hs-001",
    kind="horseshoe",
    title="Regularized-horseshoe ranking cross-check - word-reading gain",
    outcome_symbol="W",
    extra={
        "gain": True,
        "predictors": _PREDICTORS,
        "covariates": ["blocks", "behav"],
        "delta": 0.1,
        "target_accept": 0.99,
        "gb_reference": "lrp-rli-gbg-012",
    },
)


def fit(config: str = "dev"):
    return fit_horseshoe(SPEC, config=config)
