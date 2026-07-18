# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""LRP95 - Joint-readiness interaction: letter sounds (L) -> word reading (W),
moderated by TAUGHT-EXPRESSIVE vocabulary (TE).

Completes the L x vocabulary interaction set - LRP71 (E), LRP93 (R), LRP94 (TR),
LRP95 (TE) - so the joint-readiness question ("do letter sounds and vocabulary have to
be high together for larger word-reading gains?") is answered for all four vocabulary
measures. Letter sounds enter as the HSGP ``f_mech``; taught-expressive vocabulary
enters as ``gamma_mod * z(TE)`` + the interaction ``gamma_int * z(logit L) * z(TE)``.

``gamma_int > 0`` = synergy (both high needed); ``~0`` = additive; ``< 0`` =
substitutive. L and TE are positively correlated in this cohort, so the discordant
corners are sparse and the interaction is weakly powered - exploratory only.

Adjustment set = the LRP58 L -> W set {G, A, W_pre, HS, IS, SP} + the TE main effect and
interaction (as LRP71). Conditioning on the vocabulary moderator can open a collider
path for the L main association, so the L main slope is read only with the moderation.
Latent-GA-confounded ADJUSTED ASSOCIATION, never causal. target_accept 0.999.
"""

from language_reading_predictors.statistical_models.context import ModelSpec
from language_reading_predictors.statistical_models.pipeline import fit_mechanism

SPEC = ModelSpec(
    model_id="lrp-rli-mech-095",
    kind="mechanism",
    title=(
        "Joint-readiness interaction: letter sounds (L) -> word reading (W), "
        "moderated by taught-expressive vocabulary (TE)"
    ),
    outcome_symbol="W",
    mechanism_symbol="L",
    adjustment=["G", "A", "W_pre"],
    extra={
        "outcomes": ("W", "L", "TE"),
        "adjust_baseline_symbol": "W",
        "adjust_for": ("hs", "hs_missing", "attend", "deapp_c", "deapp_c_missing"),
        "use_age_gp": False,
        "phase_specific_mechanism": False,
        "use_subject_random_intercept": True,
        "moderator_symbol": "TE",
        "target_accept": 0.999,
    },
)


def fit(config: str = "dev"):
    return fit_mechanism(SPEC, config=config)
