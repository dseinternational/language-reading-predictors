# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""LRP83 - period-resolved dose-response: intervention dose -> letter sounds (L).

The letter-sound companion to ``lrp-rli-dose-077`` (word reading), added to give the
dose-response family coverage of the two *largest* ITT effects (L and B), which the
W-only family lacked (#228 item 2). Same observational estimand and causal structure
as dose-077 -- see that module's docstring for the full treatment; only the outcome
changes.

Estimand: how letter-sound **conditional change** relates to the per-period
intervention **dose** (``attend``), with partial pooling across the three periods and
a test of whether the dose-gain slope varies by period. The outcome is the
Beta-Binomial post-count of L conditional on its own baseline logit
(``adjust_baseline_symbol = "L"``, ``n_trials = 32``) -- conditional change, never raw
change (Lord's paradox / regression to the mean).

Causal structure (revised DAG): the focal edge is ``sessions (dose) -> outcome``; the
back-door adjustment set is {G (arm, the sole confounder), L_pre (autoregression
control), A (maturation precision)}, with no ``ability -> dose`` edge assumed. The
cumulative prior dose (``attend_cumul``) is a descendant of the ``IS`` collider and is
**not** conditioned on (#269). This is an **adjusted association, not "dose drives
gains"**: dose is not randomised (only ``intervention`` is), and Frank's 2012 caveat
(poorest attenders were the least able to learn) is exactly the omitted
ability->dose edge -- read the slope with that in mind. G is coded ``G = 2 - group``
(G=1 = immediate-intervention, G=0 = waitlist; positive = benefit).
"""

from language_reading_predictors.statistical_models.context import ModelSpec
from language_reading_predictors.statistical_models.pipeline import fit_dose_response

SPEC = ModelSpec(
    model_id="lrp-rli-dose-083",
    kind="dose_response",
    title="Period-resolved dose-response: intervention dose -> letter sounds (L)",
    outcome_symbol="L",
    adjustment=["G", "A", "L_pre"],
    extra={
        "adjust_baseline_symbol": "L",
        "dose_covariate": "attend",
        "period_varying_dose": True,
        "use_subject_random_intercept": True,
        "outcomes": ("L",),
    },
)


def fit(config: str = "dev"):
    return fit_dose_response(SPEC, config=config)
