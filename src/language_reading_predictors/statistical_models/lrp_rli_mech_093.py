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

**Reading ``gamma_int``.** It changes the letter-sound log-odds association
per +1 SD of receptive vocabulary. A positive value makes that association
larger as vocabulary rises; a negative value makes it smaller. A value near
zero gives little evidence for the added product term. Its sign alone does not
establish that either skill helps, that both must be high, or that a threshold
exists. L and R are positively correlated in this cohort (r ~ 0.55), so few
children have discordant skill levels and the interaction is imprecise. Read it
as an exploratory adjusted association.

Adjustment set = the LRP58/LRP71 L -> W set {G, A, W_pre, HS, IS, SP}; the moderator R
enters additionally via its main effect + interaction. Conditioning on a vocabulary
measure can open a collider path for the L -> W main association (the reason the plain
LRP58 curve omits vocabulary adjusters), so the L main slope here is read only jointly
with the moderation, exactly as in LRP71.

Everything is a latent-GA-confounded ADJUSTED ASSOCIATION, never causal. target_accept
0.999 per LRP58/LRP71.
"""

from language_reading_predictors.statistical_models.context import (
    ModelSpec,
    StatisticalFitContext,
)
from language_reading_predictors.statistical_models.mechanism import (
    MechanismModelSettings,
)
from language_reading_predictors.statistical_models.pipelines.mechanism import fit_mechanism

SPEC = ModelSpec(
    model_id="lrp-rli-mech-093",
    kind="mechanism",
    title=("Joint-readiness interaction: letter sounds (L) -> word reading (W), moderated by receptive vocabulary (R)"),
    outcome_symbol="W",
    mechanism_symbol="L",
    adjustment=["G", "A", "W_pre"],
    target_accept=0.999,
    model_settings=MechanismModelSettings(
        outcomes=("W", "L", "R"),
        adjust_baseline_symbol="W",
        adjust_for=("hs", "hs_missing", "attend", "deapp_c", "deapp_c_missing"),
        use_age_gp=False,
        phase_specific_mechanism=False,
        use_subject_random_intercept=True,
        moderator_symbol="R",
        # Thin-support HSGP reparameterisation (#438 / notes/202607251500-mech-hsgp-
        # reparameterisation.md): basis count 6 (from the shared default 10) and the
        # tighter InverseGamma(8, 8) lengthscale prior. Adopted here because this fit
        # holds 1 divergence(s) at target_accept 0.999, and a nonlinear knee/shape is
        # zero-divergence-only under notes/202608021625-divergence-qualification-policy.md
        # — the geometry has to be fixed, not waived. Per-model opt-in, not a default:
        # the same lever regressed mech-173 from 0 to 10 divergences.
        mech_hsgp_m=6,
        mech_lengthscale_tight=True,
    ),
)


def fit(config: str = "dev") -> StatisticalFitContext:
    return fit_mechanism(SPEC, config=config)
