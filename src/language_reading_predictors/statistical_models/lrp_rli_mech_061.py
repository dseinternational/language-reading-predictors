# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""LRP61 - Joint-readiness interaction: letter sounds (L) -> word reading (W),
moderated by PHONEME BLENDING (B).

Companion to the vocabulary-moderator joint-readiness family (LRP71 x E, LRP93 x R,
LRP94 x TR, LRP95 x TE), extending it to a CODE-route moderator. The mediation suite
already shows the intervention's word-reading gain does not measurably run *through*
blending (med-066 NIE_B ~ -0.03 words, P=42%; med-075 L->B->W ~ 0, P=42%); this model
asks the different question of whether the letter-sound -> word-reading conversion is
*gated by* blending - i.e. whether letter sounds only translate into word reading once
blending is in place (synergy), or convert independently.

The letter-sound mechanism enters as the HSGP curve ``f_mech`` (as in LRP58); phoneme
blending enters as a standardised linear main effect ``gamma_mod * z(B)`` plus the
interaction ``gamma_int * z(logit L) * z(B)``. Its nearest sibling LRP72 asks the same
L x B question on *decoding* (nonword) and came out sub-additive (gamma_int = -0.33,
89% CrI -0.57 to -0.09); LRP61 moves the outcome to word reading, the cell the suite
otherwise leaves open.

**Reading ``gamma_int``.** The change in the letter-sound -> word-reading slope per +1 SD
of blending. ``gamma_int > 0`` = **synergy** (word reading highest when both are high);
``gamma_int ~ 0`` = additive (either helps on its own); ``gamma_int < 0`` = substitutive.

**Estimand caveat (as on LRP72).** Blending B = PA is a DAG-DESCENDANT of the exposure L
(``LS -> PA``), so conditioning on it gives ``beta_mech`` the SHAPE of a
controlled-direct effect - letter sounds at fixed blending - without its identification:
that would need the exposure randomised or its backdoors closed (#586 finding 10).
``gamma_int`` is DESCRIPTIVE moderation by a downstream skill, not causal effect
modification and not a symmetric "need both" prerequisite test. Everything is a
latent-GA-confounded ADJUSTED ASSOCIATION, never causal. B has a 10-item ceiling.

Adjustment set = the LRP58 L -> W set {G, A, W_pre, HS, IS, SP}; B enters additionally via
its main effect + interaction. target_accept 0.999 per LRP58/LRP93.
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
    model_id="lrp-rli-mech-061",
    kind="mechanism",
    title=(
        "Joint-readiness interaction: letter sounds (L) -> word reading (W), "
        "moderated by phoneme blending (B)"
    ),
    outcome_symbol="W",
    mechanism_symbol="L",
    adjustment=["G", "A", "W_pre"],
    target_accept=0.999,
    model_settings=MechanismModelSettings(
        outcomes=("W", "L", "B"),
        adjust_baseline_symbol="W",
        adjust_for=("hs", "hs_missing", "attend", "deapp_c", "deapp_c_missing"),
        use_age_gp=False,
        phase_specific_mechanism=False,
        use_subject_random_intercept=True,
        moderator_symbol="B",
        # Thin-support HSGP reparameterisation (#430/#434 lever, adopted #438). The
        # default basis (m=10) with ell ~ InverseGamma(5, 5) leaves this curve's
        # geometry marginal at n = 54: the primary fit is borderline and, more to the
        # point, its leave-one-out REFITS diverge, so the influential points that
        # PSIS-LOO cannot handle cannot be repaired by exact refit either and the
        # nested comparison against its pair is unavailable. m=6 + the tighter
        # ell ~ InverseGamma(8, 8) clears both. Measured on adoption: the fitted curve
        # is NOT flattened (amplitude +1.5 to +2.9%, max pointwise change < 0.02 on a
        # ~0.9 span) and the moderation coefficient is unmoved, so this buys geometry
        # without changing the readout. It is deliberately per-model and not a change
        # to the shared default: the same lever regressed the age-moderation baseline
        # from 0 to 10 divergences (that pair was retired in #438), so the shared
        # defaults stay as they are. See
        # notes/202607251500-mech-hsgp-reparameterisation.md.
        mech_hsgp_m=6,
        mech_lengthscale_tight=True,
    ),
)


def fit(config: str = "dev") -> StatisticalFitContext:
    return fit_mechanism(SPEC, config=config)
