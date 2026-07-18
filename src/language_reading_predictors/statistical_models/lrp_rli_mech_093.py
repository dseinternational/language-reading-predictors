# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""LRP93 - Joint-readiness interaction: letter sounds (L) -> word reading (W),
moderated by RECEPTIVE vocabulary (R).

Companion to LRP71 (which moderates the same L -> W curve by EXPRESSIVE vocabulary E),
built to probe the "joint readiness" hypothesis: do letter-sound knowledge and
vocabulary have to be high *together* for larger word-reading gains, or does either
one help on its own? The letter-sound mechanism enters as the HSGP curve ``f_mech`` (as
in LRP58); receptive vocabulary enters as a standardised linear main effect
``gamma_mod * z(R)`` plus the interaction ``gamma_int * z(logit L) * z(R)``.

**Reading ``gamma_int``.** It is the change in the letter-sound -> word-reading slope per
+1 SD of receptive vocabulary (equivalently, by symmetry of the product, the change in
the vocabulary slope per +1 SD of letter sounds). ``gamma_int > 0`` = **synergy**: the
two skills reinforce each other, so word reading is highest when *both* are high (the
"both above a threshold" pattern). ``gamma_int ~ 0`` = the two contribute roughly
independently (additive; either being high helps). ``gamma_int < 0`` = they partly
**substitute** (diminishing returns to having both). Note L and R are positively
correlated in this cohort (r ~ 0.55), so the discordant high-one/low-other children are
relatively few and the interaction is correspondingly weakly powered - read it as an
exploratory description.

Adjustment set = the LRP58/LRP71 L -> W set {G, A, W_pre, HS, IS, SP}; the moderator R
enters additionally via its main effect + interaction. Conditioning on a vocabulary
measure can open a collider path for the L -> W main association (the reason the plain
LRP58 curve omits vocabulary adjusters), so the L main slope here is read only jointly
with the moderation, exactly as in LRP71.

Everything is a latent-GA-confounded ADJUSTED ASSOCIATION, never causal. target_accept
0.999 per LRP58/LRP71.
"""

from language_reading_predictors.statistical_models.context import ModelSpec
from language_reading_predictors.statistical_models.pipeline import fit_mechanism

SPEC = ModelSpec(
    model_id="lrp-rli-mech-093",
    kind="mechanism",
    title=(
        "Joint-readiness interaction: letter sounds (L) -> word reading (W), "
        "moderated by receptive vocabulary (R)"
    ),
    outcome_symbol="W",
    mechanism_symbol="L",
    adjustment=["G", "A", "W_pre"],
    extra={
        "outcomes": ("W", "L", "R"),
        "adjust_baseline_symbol": "W",
        "adjust_for": ("hs", "hs_missing", "attend", "deapp_c", "deapp_c_missing"),
        "use_age_gp": False,
        "phase_specific_mechanism": False,
        "use_subject_random_intercept": True,
        "moderator_symbol": "R",
        "target_accept": 0.999,
    },
)


def fit(config: str = "dev"):
    return fit_mechanism(SPEC, config=config)
