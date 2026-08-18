# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""LRP258 - ability-adjusted counterpart of the LRP58 letter-sound -> word-reading curve.

LRP58 fits the nonparametric HSGP *shape* of the letter-sound -> word-reading
relationship and is the mechanism family's headline figure. LRP201 adds the measured
general-ability proxy ``blocks`` to the *linear* Tier-1 anchor (LRP101). Neither on its
own supports the question "what does the curve look like once measured ability is
partialled out?", because comparing LRP58 with LRP201 would confound the shape
assumption with the adjustment.

This is the missing cell: LRP58 with ``ability_covariate="blocks"`` and nothing else
changed - same exposure, outcome, own baseline, conditioning set, HSGP basis
(``mech_hsgp_m=6``, tight lengthscale) and ``target_accept``. Curve against curve, so
the only difference between the two fitted shapes is the ability term.

``blocks`` is complete, recorded once pre-randomisation and constant within child, so it
enters via ``baseline_covariates`` (broadcast from t1) rather than a per-row pull. It is a
single noisy subtest and not the latent ``GA`` node the design note calls structurally
unblockable, so ``f_mech`` remains an **adjusted association** either way; this model
bounds how much of the curve the measured proxy accounts for, not how much the latent
construct does.
"""

from language_reading_predictors.statistical_models.context import ModelSpec
from language_reading_predictors.statistical_models.pipelines.mechanism import fit_mechanism

SPEC = ModelSpec(
    model_id="lrp-rli-mech-258",
    kind="mechanism",
    title="Ability-adjusted mechanism curve: letter sounds (L) -> word reading (W)",
    outcome_symbol="W",
    mechanism_symbol="L",
    adjustment=["G", "A", "W_pre"],
    extra={
        "outcomes": ("W", "L"),
        "adjust_baseline_symbol": "W",
        "adjust_for": ("hs", "hs_missing", "attend", "deapp_c", "deapp_c_missing"),
        "ability_covariate": "blocks",
        "use_age_gp": False,
        "phase_specific_mechanism": False,
        "use_subject_random_intercept": True,
        # Matched to LRP58 exactly so the curves are comparable.
        "target_accept": 0.999,
        "mech_hsgp_m": 6,
        "mech_lengthscale_tight": True,
    },
)


def fit(config: str = "dev"):
    return fit_mechanism(SPEC, config=config)
