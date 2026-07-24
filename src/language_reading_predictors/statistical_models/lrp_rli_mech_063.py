# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""LRP63 - Joint-readiness interaction: letter sounds (L) -> word reading (W),
moderated by NONWORD DECODING (N).

Companion to LRP61 (letter sounds x phoneme blending -> word reading) and to the
vocabulary-moderator family (LRP71/93/94/95), extending the joint-readiness question to
genuine nonword decoding. The mediation suite shows the word-reading gain does not
measurably run *through* decoding (med-074 NIE ~ +0.0 words, 89% CrI -0.4 to +0.5); this
model asks instead whether the letter-sound -> word-reading conversion is *gated by*
decoding - whether letter sounds only translate into word reading once nonword decoding
is in place (synergy), or convert independently.

The letter-sound mechanism enters as the HSGP curve ``f_mech`` (as in LRP58); nonword
decoding enters as a standardised linear main effect ``gamma_mod * z(N)`` plus the
interaction ``gamma_int * z(logit L) * z(N)``.

**Reading ``gamma_int``.** The change in the letter-sound -> word-reading slope per +1 SD
of decoding. ``gamma_int > 0`` = **synergy** (word reading highest when both are high);
``gamma_int ~ 0`` = additive; ``gamma_int < 0`` = substitutive.

**Estimand + power caveats.** Decoding N is a DAG-DESCENDANT of the exposure L
(``LS -> ... -> NW``), so conditioning on it makes ``beta_mech`` a **controlled-direct**
effect of letter sounds at fixed decoding, and ``gamma_int`` is **effect-modification by
a downstream skill**, not a symmetric "need both" prerequisite test. Everything is a
latent-GA-confounded ADJUSTED ASSOCIATION, never causal. N is ~57% floored (median 0), so
``z(N)`` carries little variation and this interaction is **weakly powered - suggestive
at best**; read alongside the descriptive finding that a small sight-word subgroup reads
with little decoding while fluent reading (40+ words) always co-occurs with decoding.

Adjustment set = the LRP58 L -> W set {G, A, W_pre, HS, IS, SP}; N enters additionally via
its main effect + interaction. target_accept 0.999 per LRP58/LRP93.
"""

from language_reading_predictors.statistical_models.context import ModelSpec
from language_reading_predictors.statistical_models.pipeline import fit_mechanism

SPEC = ModelSpec(
    model_id="lrp-rli-mech-063",
    kind="mechanism",
    title=(
        "Joint-readiness interaction: letter sounds (L) -> word reading (W), "
        "moderated by nonword decoding (N)"
    ),
    outcome_symbol="W",
    mechanism_symbol="L",
    adjustment=["G", "A", "W_pre"],
    extra={
        "outcomes": ("W", "L", "N"),
        "adjust_baseline_symbol": "W",
        "adjust_for": ("hs", "hs_missing", "attend", "deapp_c", "deapp_c_missing"),
        "use_age_gp": False,
        "phase_specific_mechanism": False,
        "use_subject_random_intercept": True,
        "moderator_symbol": "N",
        "target_accept": 0.999,
    },
)


def fit(config: str = "dev"):
    return fit_mechanism(SPEC, config=config)
