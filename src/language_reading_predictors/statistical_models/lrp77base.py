# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""LRP77base - Pooled-dose comparator for LRP77 (no period variation).

The no-period-variation companion to :mod:`lrp77`: a single pooled dose slope
``beta_dose`` instead of partial-pooled per-period slopes. The two models are
identical otherwise, so a PSIS-LOO comparison (``compare_statistical_models.py``)
answers the headline question directly - **does letting the dose-gain slope vary
by period improve predictive fit?** Given the weak Phase-1 dose structure, the
period-varying model is expected to shrink toward this one; LOO is interpreted
cautiously at this n because the dynamic companions were not estimable.
"""

from language_reading_predictors.statistical_models.context import ModelSpec
from language_reading_predictors.statistical_models.pipeline import fit_dose_response

SPEC = ModelSpec(
    model_id="lrp77base",
    kind="dose_response",
    title="Dose-response (pooled dose slope) - no-period-variation comparator to LRP77",
    outcome_symbol="W",
    adjustment=["G", "A", "W_pre"],
    extra={
        "adjust_baseline_symbol": "W",
        "dose_covariate": "attend",
        "dose_stage_covariate": "attend_cumul",
        "period_varying_dose": False,
        "use_subject_random_intercept": True,
        "outcomes": ("W",),
    },
)


def fit(config: str = "dev"):
    return fit_dose_response(SPEC, config=config)
