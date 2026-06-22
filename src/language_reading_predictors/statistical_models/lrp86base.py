# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""LRP86base - Pooled-dose comparator for LRP86 (letter sounds, no period variation).

The no-period-variation companion to :mod:`lrp86`: a single pooled dose slope on
letter-sound knowledge (L, n_trials = 32) instead of partial-pooled per-period
slopes. Identical otherwise. Two uses: the nested PSIS-LOO test of whether the
letter-sound dose-gain slope varies by period (it almost certainly does not, as
for word reading), and the credible pooled dose-slope estimate (the period-
varying model's overall slope has an inflated CI from the partial-pooling
hyper-prior). See LRP86 for the estimand, adjustment set and expectations.
"""

from language_reading_predictors.statistical_models.context import ModelSpec
from language_reading_predictors.statistical_models.pipeline import fit_dose_response

SPEC = ModelSpec(
    model_id="lrp86base",
    kind="dose_response",
    title="Dose-response (pooled dose slope) - letter-sounds comparator to LRP86",
    outcome_symbol="L",
    adjustment=["G", "A", "L_pre"],
    extra={
        "adjust_baseline_symbol": "L",
        "dose_covariate": "attend",
        "dose_stage_covariate": "attend_cumul",
        "period_varying_dose": False,
        "use_subject_random_intercept": True,
        "outcomes": ("L",),
    },
)


def fit(config: str = "dev"):
    return fit_dose_response(SPEC, config=config)
