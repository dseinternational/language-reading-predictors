# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""LRP205 - no-interaction baseline for LRP105 (L -> W moderated by NW).

#421 Tier 2 companion. Identical to ``lrp-rli-mech-105`` except the L x NW interaction
term is dropped (``include_interaction=False``): it keeps the ``f_mech`` letter-sound
curve and the ``gamma_mod`` nonword-decoding main effect. Differing by exactly the
interaction, a PSIS-LOO comparison of LRP105 against this baseline is a clean nested test
of whether the decoding moderation of the letter-sound -> word-reading effect helps
out-of-sample prediction (see
``compare_statistical_models.nw_moderation_loo_compare``). As in LRP105, NW is a mediator
and heavily floored, so any signal here is a descriptive, floor-limited **adjusted
association**, never causal.
"""

from language_reading_predictors.statistical_models.context import ModelSpec
from language_reading_predictors.statistical_models.pipeline import fit_mechanism

SPEC = ModelSpec(
    model_id="lrp-rli-mech-205",
    kind="mechanism",
    title=(
        "Mechanism model: letter-sound (L) -> word reading (W), "
        "NW main effect only (no-interaction baseline for LRP105)"
    ),
    outcome_symbol="W",
    mechanism_symbol="L",
    adjustment=["G", "A", "W_pre"],
    extra={
        "outcomes": ("W", "L", "N"),
        "adjust_baseline_symbol": "W",
        "adjust_for": ("hs", "hs_missing", "attend", "deapp_c", "deapp_c_missing"),
        "moderator_symbol": "N",
        "include_interaction": False,
        "use_age_gp": False,
        "phase_specific_mechanism": False,
        "use_subject_random_intercept": True,
    },
)


def fit(config: str = "dev"):
    return fit_mechanism(SPEC, config=config)
