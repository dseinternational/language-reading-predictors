# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""LRP84 - period-resolved dose-response: intervention dose -> phoneme blending (B).

The phoneme-blending companion to ``lrp-rli-dose-077`` (word reading), completing the
dose-response family's coverage of the two largest ITT effects (L and B) (#228 item
2). Same observational estimand and causal structure as dose-077 -- see that module's
docstring for the full treatment; only the outcome changes.

Estimand: how blending **conditional change** relates to the per-period intervention
**dose** (``attend``), with partial pooling across the three periods and a test of
whether the dose-gain slope varies by period. The outcome is the Beta-Binomial
post-count of B conditional on its own baseline logit (``adjust_baseline_symbol =
"B"``, ``n_trials = 10``) -- conditional change, never raw change.

Causal structure (revised DAG): the focal edge is ``sessions (dose) -> outcome``; the
back-door adjustment set is {G (arm, the sole confounder), B_pre (autoregression
control), A (maturation precision)}, with no ``ability -> dose`` edge assumed. The
cumulative prior dose (``attend_cumul``) is a descendant of the ``IS`` collider and is
**not** conditioned on (#269). An **adjusted association, not "dose drives gains"**:
dose is not randomised, and the omitted ability->dose edge (Frank 2012) is a live
confounder. G is coded ``G = 2 - group`` (positive = benefit).
"""

from language_reading_predictors.statistical_models.context import ModelSpec
from language_reading_predictors.statistical_models.pipeline import fit_dose_response

SPEC = ModelSpec(
    model_id="lrp-rli-dose-084",
    kind="dose_response",
    title="Period-resolved dose-response: intervention dose -> phoneme blending (B)",
    outcome_symbol="B",
    adjustment=["G", "A", "B_pre"],
    extra={
        "adjust_baseline_symbol": "B",
        "dose_covariate": "attend",
        "period_varying_dose": True,
        "use_subject_random_intercept": True,
        "outcomes": ("B",),
    },
)


def fit(config: str = "dev"):
    return fit_dose_response(SPEC, config=config)
