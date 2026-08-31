# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""LRP77base - Pooled-dose comparator for LRP77 (no period variation).

The no-period-variation companion to :mod:`lrp-rli-dose-077`: a single pooled slope
``beta_dose`` on the within-child attendance deviation instead of partial-pooled
per-period slopes. The two models are identical otherwise — same on-intervention
indicator, post-crossover arm term, own baseline, linear age, child random
intercept and between-child attendance split, and neither fits a cumulative-dose
control — so a **leave-one-child-out** PSIS comparison
(``compare_statistical_models.py``) answers the headline question directly:
**does letting the attendance slope vary by period improve prediction for a child
the model has not seen?**

The unit is a whole child, not a row (#587 finding 4). A transition row's own
baseline IS the previous transition's fitted outcome — for every period-2 row and
all but one period-3 row — so a row-level score would leave the held-out outcome
in the next row's design matrix and would not be out-of-sample at all.

Given the weak Phase-1 dose structure, the period-varying model is expected to
shrink toward this one; the comparison is interpreted cautiously at this n because
the dynamic companions were not estimable.

The causal reading is LRP77's and is inherited whole: the revised DAG has
``A -> IS``, ``GA -> IS`` and ``IG -> IS``, latent general ability ``GA`` is not
closed by any measured baseline, and the pooled slope here is an adjusted
association like every other slope in the family. Winning this comparison would say
a pooled slope predicts a held-out child better — nothing about causation.
"""

from language_reading_predictors.statistical_models.context import (
    ModelSpec,
    StatisticalFitContext,
)
from language_reading_predictors.statistical_models.dose_response import (
    DoseResponseModelSettings,
)
from language_reading_predictors.statistical_models.pipelines.dose_response import fit_dose_response

SPEC = ModelSpec(
    model_id="lrp-rli-dose-277",
    kind="dose_response",
    title="Dose-response (pooled dose slope) - no-period-variation comparator to LRP77",
    outcome_symbol="W",
    adjustment=["G", "A", "W_pre"],
    model_settings=DoseResponseModelSettings(
        adjust_baseline_symbol="W",
        dose_covariate="attend",
        # No cumulative-dose (attend_cumul) control — IS collider (#269). Matching
        # dose-077 keeps the period-varying-vs-pooled nested LOO test clean.
        period_varying_dose=False,
        use_subject_random_intercept=True,
        outcomes=("W",),
    ),
)


def fit(config: str = "dev") -> StatisticalFitContext:
    return fit_dose_response(SPEC, config=config)
