# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""LRP204 - no-interaction baseline for LRP104 (L -> W moderated by RW).

#421 Tier 2 companion. Identical to ``lrp-rli-mech-104`` except the L x RW interaction
term is dropped: it keeps the ``f_mech`` letter-sound curve and the ``gamma_mod`` RW main
effect but sets ``include_interaction=False``. Because the two models then differ by
exactly the interaction, a PSIS-LOO comparison of LRP104 against this baseline is a clean
nested test of whether the phonological-memory moderation of the letter-sound ->
word-reading effect improves out-of-sample prediction at all (see
``compare_statistical_models.rw_moderation_loo_compare``). Every coefficient remains an
**adjusted association**, never causal.
"""

from language_reading_predictors.statistical_models.context import ModelSpec
from language_reading_predictors.statistical_models.mechanism import (
    MechanismModelSettings,
)
from language_reading_predictors.statistical_models.pipelines.mechanism import fit_mechanism

SPEC = ModelSpec(
    model_id="lrp-rli-mech-204",
    kind="mechanism",
    title=(
        "Mechanism model: letter-sound (L) -> word reading (W), "
        "RW main effect only (no-interaction baseline for LRP104)"
    ),
    outcome_symbol="W",
    mechanism_symbol="L",
    adjustment=["G", "A", "W_pre"],
    model_settings=MechanismModelSettings(
        outcomes=("W", "L"),
        adjust_baseline_symbol="W",
        adjust_for=("hs", "hs_missing", "attend", "deapp_c", "deapp_c_missing"),
        moderator_symbol="erbto",
        moderator_is_covariate=True,
        # Match mech-104: drop mean-imputed erbto rows rather than moderate the
        # main effect by an average-filled modifier (loads erbto_missing).
        require_observed=("erbto",),
        include_interaction=False,
        use_age_gp=False,
        phase_specific_mechanism=False,
        use_subject_random_intercept=True,
    ),
)


def fit(config: str = "dev"):
    return fit_mechanism(SPEC, config=config)
