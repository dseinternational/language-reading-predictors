# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""LRPPL003 - wave-pooled level association: expressive vocabulary (E) -> word reading (W).

Primary (#553). The between/within split the family gives letter sounds
(``lrp-rli-pl-001``), extended to expressive vocabulary: one Beta-Binomial
likelihood over every child-wave row, with expressive vocabulary at the same wave
as the standardised-logit exposure, split into the child study-average
(``beta_between``) and the within-child wave deviation (``beta_within``), per-wave
intercepts, and a child random intercept carrying the repeated measures. There is
deliberately **no own-baseline term**: its absence is what makes this a levels
estimand rather than the transition estimand ``lrp-rli-mech-057`` reports (E -> W,
+0.12 log-odds per SD, P = 0.93).

Adjustment set mirrors ``lrp-rli-mech-057`` minus the own baseline: the revised-DAG
non-measure confounders hearing (``hs``), phonological memory (``erbto``) and
speech production (``deapp_c``), each with its missing-indicator; the same-wave
standardised logits of taught receptive (``TR``), taught expressive (``TE``) and
receptive (``R``) vocabulary as skill adjusters (``skill_symbols``, #553); the
t1 block-design ability proxy broadcast across waves; linear age at the wave. No
``attend``: the interval session dose is a transition covariate and is omitted
here as in ``pl-001``.

What the split settles: whether expressive vocabulary's level association with
word reading (concurrent per-wave +1.0, -0.2, +3.1, +2.4 words/SD; ``hs-002``
rank 2) is trait-level covariation between children or tracks within-child
change. The expectation (from the letter-sound result) is between-child
dominated with a small within-child term.

Association only: exposure and outcome are measured at the same wave, so nothing
here orders them, and the same-wave skill adjusters are contemporaneous,
possibly post-treatment, levels (Table-2 fallacy applies).
"""

from language_reading_predictors.statistical_models.context import (
    ModelSpec,
    StatisticalFitContext,
)
from language_reading_predictors.statistical_models.pipelines.pooled_levels import (
    fit_pooled_levels,
)
from language_reading_predictors.statistical_models.pooled_levels import (
    PooledLevelsModelSettings,
)

SPEC = ModelSpec(
    model_id="lrp-rli-pl-003",
    kind="pooled_levels",
    title="Wave-pooled level association: expressive vocabulary (E) -> word reading (W)",
    outcome_symbol="W",
    mechanism_symbol="E",
    model_settings=PooledLevelsModelSettings(
        adjust_for=(
            "hs", "hs_missing", "erbto", "erbto_missing", "deapp_c", "deapp_c_missing",
        ),
        skill_symbols=("TR", "TE", "R"),
        ability_covariate="blocks",
        use_wave_intercepts=True,
        decompose_between_within=True,
        use_subject_random_intercept=True,
    ),
)


def fit(config: str = "dev") -> StatisticalFitContext:
    return fit_pooled_levels(SPEC, config=config)
