# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""LRP94 - Joint-readiness interaction: letter sounds (L) -> word reading (W),
moderated by TAUGHT-RECEPTIVE vocabulary (TR).

Companion to LRP71 (L x E) and LRP93 (L x R), pairing the letter-sound -> word-reading
HSGP curve with the taught-receptive vocabulary measure - the one vocabulary exposure
whose own dose-response curve (LRP188) was the closest to a rising "knee". The question
is the joint-readiness one: is word reading largest when letter sounds and taught
vocabulary are *both* high (synergy), or does either help on its own?

Letter sounds enter as the HSGP ``f_mech``; taught-receptive vocabulary enters as
``gamma_mod * z(TR)`` plus the interaction ``gamma_int * z(logit L) * z(TR)``.
``gamma_int > 0`` = synergy (both need to be high); ``~0`` = additive; ``< 0`` =
substitutive. L and TR are positively correlated (r ~ 0.63), so the discordant corners
are sparse and the interaction is weakly powered - exploratory only.

Adjustment set = the LRP58 L -> W set {G, A, W_pre, HS, IS, SP} + the TR main effect and
interaction (same construction as LRP71). Conditioning on the vocabulary moderator can
open a collider path for the L main association, so the L main slope is read only with
the moderation. Latent-GA-confounded ADJUSTED ASSOCIATION, never causal. target_accept
0.999.
"""

from language_reading_predictors.statistical_models.context import ModelSpec
from language_reading_predictors.statistical_models.pipeline import fit_mechanism

SPEC = ModelSpec(
    model_id="lrp-rli-mech-094",
    kind="mechanism",
    title=(
        "Joint-readiness interaction: letter sounds (L) -> word reading (W), "
        "moderated by taught-receptive vocabulary (TR)"
    ),
    outcome_symbol="W",
    mechanism_symbol="L",
    adjustment=["G", "A", "W_pre"],
    extra={
        "outcomes": ("W", "L", "TR"),
        "adjust_baseline_symbol": "W",
        "adjust_for": ("hs", "hs_missing", "attend", "deapp_c", "deapp_c_missing"),
        "use_age_gp": False,
        "phase_specific_mechanism": False,
        "use_subject_random_intercept": True,
        "moderator_symbol": "TR",
        "target_accept": 0.999,
    },
)


def fit(config: str = "dev"):
    return fit_mechanism(SPEC, config=config)
