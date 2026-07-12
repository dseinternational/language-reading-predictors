# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""LRP73base - no-interaction baseline for LRP73 (L mechanism + age main effect).

Same outcome / mechanism (`f_mech` HSGP) / adjustment / age main effect as LRP73,
but with `include_interaction=False` so there is **no** `gamma_int` (L × age)
term. Because the two models then differ by exactly the interaction, a PSIS-LOO
comparison of LRP73 against this baseline is a clean nested test of whether the
age moderation improves predictive fit. See `lrp-rli-mech-073.py`.
"""

from language_reading_predictors.statistical_models.context import ModelSpec
from language_reading_predictors.statistical_models.pipeline import fit_mechanism

SPEC = ModelSpec(
    model_id="lrp-rli-mech-173",
    kind="mechanism",
    title=(
        "Mechanism baseline (no interaction): letter-sound (L) -> word reading (W) "
        "+ age main effect"
    ),
    outcome_symbol="W",
    mechanism_symbol="L",
    adjustment=["G", "A", "W_pre"],
    extra={
        "outcomes": ("W", "L"),
        "adjust_baseline_symbol": "W",
        "adjust_for": ("hs", "hs_missing", "attend", "deapp_c", "deapp_c_missing"),
        "moderator_symbol": "A",
        "moderator_is_covariate": True,
        "include_interaction": False,
        "use_age_gp": False,
        "phase_specific_mechanism": False,
        "use_subject_random_intercept": True,
    },
)


def fit(config: str = "dev"):
    return fit_mechanism(SPEC, config=config)
