# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""LRP86 - Period-resolved dose-response: intervention dose -> letter sounds (L).

The #104 Phase-3 generalisation of LRP77 from word reading to **letter-sound
knowledge** (L, YARC-LSK, n_trials = 32) - does the small dose effect found for
word reading extend to a decoding skill? Identical machinery to LRP77
(``build_dose_response_model``): per-period dose (``attend``) with partial-pooled
period slopes, ``attend_cumul`` dose-stage control, ``{G, A, L_pre}`` adjustment
(DAG v5: G the sole dose confounder, L_pre the RTM baseline, age a precision
covariate), subject random intercept; conditional change, all gain rows incl. the
zero-dose anchor.

Expectations (carried into the report): the Phase-1 GB diagnostic showed dose
importance for letter-sound *gains* sat near the noise floor (much lower than for
word reading), so this fit is a generalisation **check** likely to return a weak
or null dose slope - a useful "dose helps word reading specifically, not decoding
broadly" result rather than an expected positive finding. Read the credible
pooled slope from the comparator ``lrp86base``; the LRP86-vs-LRP86base PSIS-LOO
answers whether the L dose-gain slope varies by period.
"""

from language_reading_predictors.statistical_models.context import ModelSpec
from language_reading_predictors.statistical_models.pipeline import fit_dose_response

SPEC = ModelSpec(
    model_id="lrp86",
    kind="dose_response",
    title="Period-resolved dose-response: intervention dose -> letter sounds (L)",
    outcome_symbol="L",
    adjustment=["G", "A", "L_pre"],
    extra={
        "adjust_baseline_symbol": "L",
        "dose_covariate": "attend",
        "dose_stage_covariate": "attend_cumul",
        "period_varying_dose": True,
        "use_subject_random_intercept": True,
        "outcomes": ("L",),
    },
)


def fit(config: str = "dev"):
    return fit_dose_response(SPEC, config=config)
