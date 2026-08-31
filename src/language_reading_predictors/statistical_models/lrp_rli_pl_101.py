# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""LRPPL101 - wave-pooled level association: letter-sound knowledge (L) -> word reading (W).

Comparator for LRPPL01 with a single intercept: beta_mech then also carries the secular co-movement of both measures across waves. Reported to bound that contribution, never as the headline.

One Beta-Binomial likelihood over every child-wave row, with letter-sound knowledge at the same
wave as the standardised exposure and a child random intercept carrying the repeated
measures. There is deliberately **no own-baseline term**: its absence is what makes
this a levels estimand rather than the transition estimand the ``mechanism`` family
reports.

Adjusters match the ``concurrent`` family so the pooled and per-wave views are
comparable, and include the measured general-ability proxy ``blocks``.

Association only, and weaker than the mechanism family on temporal grounds:
exposure and outcome are measured at the same wave, so nothing here orders them.
"""

from language_reading_predictors.statistical_models.context import (
    ModelSpec,
    StatisticalFitContext,
)
from language_reading_predictors.statistical_models.pooled_levels import (
    PooledLevelsModelSettings,
)
from language_reading_predictors.statistical_models.pipelines.pooled_levels import (
    fit_pooled_levels,
)

SPEC = ModelSpec(
    model_id="lrp-rli-pl-101",
    kind="pooled_levels",
    title="Wave-pooled level association: letter-sound knowledge (L) -> word reading (W)",
    outcome_symbol="W",
    mechanism_symbol="L",
    model_settings=PooledLevelsModelSettings(
        adjust_for=("hs", "hs_missing", "deapp_c", "deapp_c_missing"),
        ability_covariate="blocks",
        use_wave_intercepts=False,
        use_subject_random_intercept=True,
    ),
)


def fit(config: str = "dev") -> StatisticalFitContext:
    return fit_pooled_levels(SPEC, config=config)
