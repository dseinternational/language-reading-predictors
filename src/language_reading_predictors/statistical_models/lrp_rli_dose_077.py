# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""LRP77 - Period-resolved dose-response: intervention dose -> word reading (W).

The gated #104 Phase-2 follow-up to the period-resolved GB diagnostic
. Phase 1 found the
structure in the near-noise gain models sits on the **dose / intervention-status
axis**, with a weak positive dose signal concentrated in period 1 for word
reading. This model quantifies that one signal with honest uncertainty.

Estimand
--------
How word-reading **conditional change** relates to the per-period intervention
**dose** (``attend``), with partial pooling across the three periods, and whether
that dose-gain slope **varies by period**. The outcome is the Beta-Binomial
post-count of W conditional on its own baseline logit (``adjust_baseline_symbol``
= ``W``, ``n_trials`` = 79) - conditional change, never raw change scores
(Lord's paradox / regression to the mean).

Causal structure (revised DAG; dag/dag-language-reading.dagitty)
---------------------------------------------------------------
The focal edge is ``sessions (dose) -> outcome``. Sessions is the revised DAG's
``IS`` node; this is the ID-3 **observational** dose-response (adjust ``{IG, A}``,
labelled an adjusted association). The per-period ``attend`` is the model's
*exposure* — regressing the outcome on it is the estimand, not a "conditioning on
IS" violation of the ITT rule. What the ID-3 rule forbids is conditioning on
*other* functions of ``IS`` that open a back door: in particular the **cumulative
prior dose** ``attend_cumul`` (a running sum of earlier-period ``IS``), which an
earlier version adjusted via ``dose_stage_covariate`` — reopening the latent-GA
back door. That term is **dropped** from the headline here (#269); so nothing
downstream of, or aggregating, the focal dose is conditioned on. It remains
available only as a **flagged sensitivity option** (set
``extra["dose_stage_covariate"] = "attend_cumul"``, the dose-response analogue of
the aligned family's cumulative-session collider sensitivity) — read any movement
of the slope under it as a back-door sensitivity, not a better estimate. Dose has
one parent,
``intervention``, so the back-door adjustment set is ``{G, W_pre, A}``
but as: **G (arm) is the sole confounder** (blocks ``dose <- intervention ->
outcome``); **W_pre** is the regression-to-the-mean / autoregression control
(parameterisation, not a back-door); **age** is a precision / maturation
covariate (v5 has ``age -> outcome`` but no ``age -> dose``). The analysis sample
keeps **all** gain rows including the waitlist controls' period-1 ``attend == 0``
rows - the zero-dose anchor that identifies the slope.

Load-bearing assumption (no ``g -> dose``)
------------------------------------------
v5 deliberately omits ``ability -> dose``, but Frank's 2012 caveat ("the children
least able to learn tended to show the poorest attendance") is exactly that edge.
Dose is **not** randomised (only ``intervention`` is), so randomisation does not
rescue this. The subject random intercept absorbs *stable* child differences;
the ability-adjusted sensitivity fit ``lrp-rli-dose-177`` conditions on the baseline-skill
cluster (L/E/B) to test whether the dose slope survives ability adjustment. If it
collapses there, the dose signal is substantially ability-confounded.

Caveats (carried into the report)
---------------------------------
Adjusted association, not "dose drives gains". Phase-1 best strata reached only
R^2 0.1-0.3; the deliverable is a calibrated dose slope with credible intervals,
not a strong predictor. Group is coded ``G = 2 - group`` (G=1 = immediate-intervention,
G=0 = waitlist control; positive = benefit), per the #117 sign convention.
"""

from language_reading_predictors.statistical_models.context import ModelSpec
from language_reading_predictors.statistical_models.pipeline import fit_dose_response

SPEC = ModelSpec(
    model_id="lrp-rli-dose-077",
    kind="dose_response",
    title="Period-resolved dose-response: intervention dose -> word reading (W)",
    outcome_symbol="W",
    adjustment=["G", "A", "W_pre"],
    extra={
        "adjust_baseline_symbol": "W",
        "dose_covariate": "attend",
        # No cumulative-dose (attend_cumul) control: it conditions on the IS collider
        # and reopens the latent-GA backdoor (#269). It is available only as a flagged
        # sensitivity option (set dose_stage_covariate="attend_cumul").
        "period_varying_dose": True,
        "use_subject_random_intercept": True,
        "outcomes": ("W",),
    },
)


def fit(config: str = "dev"):
    return fit_dose_response(SPEC, config=config)
