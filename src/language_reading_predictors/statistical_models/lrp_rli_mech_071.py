# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""LRP71 - Mechanism model: letter-sound (L) -> word reading (W), with a LINEAR
moderation by expressive vocabulary (E = eowpvt).

This is the first interaction model in the suite. It extends LRP58 (L -> W, revised
adjustment {G, A, HS, IS(attend), SP, W_pre}; #245) with

    ... + gamma_mod * z(E_post) + gamma_int * z(logit L_post) * z(E_post)

so the letter-sound -> reading effect can scale with a child's vocabulary level
(the lexical-quality / dual-route hypothesis). ``gamma_int > 0`` would mean
the code-based route converts to reading *more* strongly for higher-vocabulary children.

Confounder set is the revised LRP58 set {G, A, HS, IS(attend), SP} + W_pre. Under
the revised DAG (2026-07-10, #245) expressive vocabulary is no longer a
*confounder* of L -> W (the EV -> LS edge is dropped), so E enters here purely as
the tested *moderator* — its standardised main effect ``gamma_mod`` plus the
``gamma_int`` interaction. celf/F, a descendant of L, is still excluded (LRP70
deferred pending a DAG review).

The nonparametric ``f_mech(L)`` HSGP is unchanged (it stays a function of the
raw ``logit L_post``); the GP-varying-slope refinement is deferred. With the
interaction present, the ``f_mech`` slope is the L -> W effect *at the mean of
E*.

See the companion note and ``docs/models/lrp-rli-mech-071/index.qmd``.
"""

from language_reading_predictors.statistical_models.context import ModelSpec
from language_reading_predictors.statistical_models.pipeline import fit_mechanism

SPEC = ModelSpec(
    model_id="lrp-rli-mech-071",
    kind="mechanism",
    title=(
        "Mechanism model: letter-sound (L) -> word reading (W), "
        "linear moderation by expressive vocabulary (E)"
    ),
    outcome_symbol="W",
    mechanism_symbol="L",
    adjustment=["G", "A", "W_pre"],
    # Same flags as LRP58 plus the moderator. Age GP off (age enters linearly),
    # subject random intercept on.
    extra={
        "outcomes": ("W", "L", "E"),
        "adjust_baseline_symbol": "W",
        "adjust_for": ("hs", "hs_missing", "attend", "deapp_c", "deapp_c_missing"),
        "use_age_gp": False,
        "phase_specific_mechanism": False,
        "use_subject_random_intercept": True,
        "moderator_symbol": "E",
    },
)


def fit(config: str = "dev"):
    return fit_mechanism(SPEC, config=config)
