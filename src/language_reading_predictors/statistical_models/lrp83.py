# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""LRP83 - Within-control crossover: word reading (W), waitlist as own control.

The #104 Phase-3 design-based triangulation of the randomised period-1 ITT
(LRP52). Restricts to the **waitlist control arm**, which is off intervention in
period 1 and on from period 2 after crossover, and estimates the within-child
on-vs-off effect on word-reading conditional change (Beta-Binomial on W_post |
W_pre, n_trials = 79). The design follows Snowling / Gertz / Bayliss et al.

Estimand: ``beta_on``, the crossover effect, on the same logit scale as the ITT
``tau`` (so the two are directly comparable), plus a probability-scale average
marginal effect. ``beta_on > 0`` means children score higher when on
intervention. A subject random intercept makes each control its own baseline; a
linear age term is the smooth maturation control; there are **no period
intercepts** (for controls ``on`` equals "period >= 2", so a per-period intercept
would absorb the effect).

Caveat (carried into the report): because ``on`` and period coincide for
controls, ``beta_on`` is the intervention effect *only* under a
no-strong-(differential)-maturation assumption. It is less rigorous than the
two-arm ITT - a triangulation point, not a clean randomised contrast. Read it
alongside the ITT (LRP52) and the dose-response (LRP77): convergent signs across
all three are the strength of the conclusion.
"""

from language_reading_predictors.statistical_models.context import ModelSpec
from language_reading_predictors.statistical_models.pipeline import fit_crossover

SPEC = ModelSpec(
    model_id="lrp83",
    kind="crossover",
    title="Within-control crossover: word reading (W), waitlist as own control",
    outcome_symbol="W",
    extra={
        "adjust_baseline_symbol": "W",
        "adjust_age": True,
        "use_subject_random_intercept": True,
        "outcomes": ("W",),
    },
)


def fit(config: str = "dev"):
    return fit_crossover(SPEC, config=config)
