# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""LRP71 - Mechanism model: letter-sound (L) -> word reading (W), with a LINEAR
moderation by expressive vocabulary (E = eowpvt).

This is the first interaction model in the suite. It extends LRP58 (L -> W,
adjustment {G, A, E, R, W_pre}) with

    ... + gamma_mod * z(E_post) + gamma_int * z(logit L_post) * z(E_post)

so the letter-sound -> reading effect can scale with a child's vocabulary level
(the lexical-quality / dual-route hypothesis). ``gamma_int > 0`` would mean
the code-based route converts to reading *more* strongly for higher-vocabulary children.

Adjustment set is kept identical to LRP58. E is already a DAG-required adjuster
there, so conditioning on it as a moderator is clean (unlike celf/F, a
descendant of L — that model, LRP70, is deferred pending a DAG review). The
pipeline removes E from the plain confounder loop because the standardised
moderator main effect ``gamma_mod`` already adjusts for E; this avoids a
collinear duplicate term.

The nonparametric ``f_mech(L)`` HSGP is unchanged (it stays a function of the
raw ``logit L_post``); the GP-varying-slope refinement is deferred. With the
interaction present, the ``f_mech`` slope is the L -> W effect *at the mean of
E*.

See the companion note and ``docs/models/lrp71/index.qmd``.
"""

from language_reading_predictors.statistical_models.context import ModelSpec
from language_reading_predictors.statistical_models.pipeline import fit_mechanism

SPEC = ModelSpec(
    model_id="lrp71",
    kind="mechanism",
    title=(
        "Mechanism model: letter-sound (L) -> word reading (W), "
        "linear moderation by expressive vocabulary (E)"
    ),
    outcome_symbol="W",
    mechanism_symbol="L",
    adjustment=["G", "A", "E", "R", "W_pre"],
    # Same flags as LRP58 plus the moderator. Age GP off, subject random
    # intercept on — see notes/202604181700-lrp55-age-gp-drop.md and
    # notes/202604181800-mechanism-random-intercepts.md.
    extra={
        "adjust_baseline_symbol": "W",
        "use_age_gp": False,
        "phase_specific_mechanism": False,
        "use_subject_random_intercept": True,
        "moderator_symbol": "E",
    },
)


def fit(config: str = "dev"):
    return fit_mechanism(SPEC, config=config)
