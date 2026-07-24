# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""LRP105 - is the letter-sound -> word-reading effect carried by children who decode?

#421 Tier 2 (letter-sound -> word-reading review note, #424): the decoding-route
companion to ``mech-104``. Q2 of the review showed a majority of the letter-sound ->
word-reading association does not run through measured nonword decoding; this model asks
the moderation form of the same question -

    eta = ... + f_mech(logit L) + gamma_mod·z(NW) + gamma_int·z(logit L)·z(NW)

- does letter-sound knowledge convert to word reading **more** strongly for children who
can already decode nonwords (NW)? ``gamma_int > 0`` would say the alphabetic principle
pays off once a child can apply it.

**Measure moderator.** NW is a ``MEASURES`` symbol (the 6-item nonword count), so this is
the plain ``mech-071`` pattern - the moderator enters as its standardised same-wave logit
(``moderator_is_covariate`` is left False). Confounder set is the ``mech-058``/``mech-073``
letter-sound -> word-reading set {G, A, HS, IS(attend), SP} + ``W_pre``. W is not floored,
so the ``f_mech`` HSGP is kept.

**Two sharp caveats (from #424).** NW is a **mediator** of the letter-sound -> word-reading
path (``LS -> NW -> WR`` under the revised DAG), so this moderation is **descriptive
only** - conditioning on a mediator's level can also open a collider path; read it
alongside Q2, never as a controlled effect. And NW is **heavily floored** (zero for
72/64/52/40% of children at t1-t4), so the interaction is floor-limited early and its
power is weak; the off-floor-indicator variant Frank flags may be better-powered and is
left for a follow-up. Every coefficient is an **adjusted association**, never causal.
``lrp-rli-mech-205`` is the no-interaction companion for the nested PSIS-LOO test.
"""

from language_reading_predictors.statistical_models.context import ModelSpec
from language_reading_predictors.statistical_models.pipeline import fit_mechanism

SPEC = ModelSpec(
    model_id="lrp-rli-mech-105",
    kind="mechanism",
    title=(
        "Mechanism model: letter-sound (L) -> word reading (W), "
        "moderated by nonword decoding (NW)"
    ),
    outcome_symbol="W",
    mechanism_symbol="L",
    adjustment=["G", "A", "W_pre"],
    extra={
        "outcomes": ("W", "L", "N"),
        "adjust_baseline_symbol": "W",
        "adjust_for": ("hs", "hs_missing", "attend", "deapp_c", "deapp_c_missing"),
        "moderator_symbol": "N",
        "include_interaction": True,
        "use_age_gp": False,
        "phase_specific_mechanism": False,
        "use_subject_random_intercept": True,
    },
)


def fit(config: str = "dev"):
    return fit_mechanism(SPEC, config=config)
