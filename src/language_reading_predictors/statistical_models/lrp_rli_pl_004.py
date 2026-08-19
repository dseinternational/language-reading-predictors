# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""LRPPL004 - wave-pooled level association: receptive vocabulary (R) -> word reading (W).

Primary (#553). The between/within split extended to receptive vocabulary: one
Beta-Binomial likelihood over every child-wave row, receptive vocabulary at the
same wave as the standardised-logit exposure split into ``beta_between`` /
``beta_within``, per-wave intercepts, a child random intercept, and **no
own-baseline term** (the levels estimand; the transition estimand is
``lrp-rli-mech-056``, R -> W, +0.06 log-odds per SD, P = 0.80).

Adjustment set mirrors ``lrp-rli-mech-056`` minus the own baseline: hearing
(``hs``) and phonological memory (``erbto``) with their missing-indicators, the
same-wave standardised logit of taught receptive vocabulary (``TR``) as a skill
adjuster, the t1 block-design ability proxy, linear age at the wave. No
``attend`` (a transition covariate; omitted as in ``pl-001``).

What the split settles: whether receptive vocabulary's level association with
word reading (concurrent per-wave +0.1, +1.4, +3.8, +2.2 words/SD) is trait-level
covariation between children or tracks within-child change; the expectation is
between-child dominated with a small within-child term.

Association only: exposure and outcome are measured at the same wave, and the
skill adjuster is a contemporaneous, possibly post-treatment, level.
"""

from language_reading_predictors.statistical_models.context import ModelSpec
from language_reading_predictors.statistical_models.pipelines.pooled_levels import (
    fit_pooled_levels,
)
from language_reading_predictors.statistical_models.pooled_levels import (
    PooledLevelsModelSettings,
)

SPEC = ModelSpec(
    model_id="lrp-rli-pl-004",
    kind="pooled_levels",
    title="Wave-pooled level association: receptive vocabulary (R) -> word reading (W)",
    outcome_symbol="W",
    mechanism_symbol="R",
    model_settings=PooledLevelsModelSettings(
        adjust_for=("hs", "hs_missing", "erbto", "erbto_missing"),
        skill_symbols=("TR",),
        ability_covariate="blocks",
        use_wave_intercepts=True,
        decompose_between_within=True,
        use_subject_random_intercept=True,
    ),
)


def fit(config: str = "dev"):
    return fit_pooled_levels(SPEC, config=config)
