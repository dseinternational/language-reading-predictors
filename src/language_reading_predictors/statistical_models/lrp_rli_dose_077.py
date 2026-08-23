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
Among rows **on the intervention**, how word-reading **conditional change**
relates to how many sessions were attended — the *intensive* margin — with
partial pooling across the three periods, and whether that slope **varies by
period**. The outcome is the Beta-Binomial post-count of W conditional on its own
baseline logit (``adjust_baseline_symbol`` = ``W``, ``n_trials`` = 79) -
conditional change, never raw change scores (Lord's paradox / regression to the
mean).

Two margins, deliberately separated (#587 finding 2)
----------------------------------------------------
In period 1 every immediate-arm child attended 45-91 sessions and every waitlist
child attended **zero**: arm and session count correlate at 0.970. A single dose
coefficient spanning both groups is therefore not an attendance effect at all —
arithmetically it is the randomised treatment contrast divided by the mean dose.
(On the pre-repair stored fit that identity was exact: the period-1 slope times
the treated mean dose, plus the arm term, reproduced ``lrp-rli-itt-010``'s
randomised tau to three decimal places.)

So sessions enter **centred and standardised over the fitted on-intervention rows
only**, zero elsewhere, and a separate ``theta_treated`` indicator carries the
extensive margin. Assigned arm enters only from period 2 (``beta_arm_late``),
where both arms are treated and arm reads as intervention *order*; in period 1 it
would be exactly collinear with ``treated``. The exposure is additionally split
Mundlak-style into each child's study-average attendance (``beta_dose_between``)
and their within-child deviation from it, because a lone slope over a child random
intercept returns a precision-weighted blend of the two.

Causal structure (revised DAG; dag/dag-language-reading.dagitty)
---------------------------------------------------------------
The focal edge is ``sessions -> outcome``. Sessions is the revised DAG's ``IS``
node, and the per-period ``attend`` is the model's *exposure* — regressing the
outcome on it is the estimand, not a "conditioning on IS" violation of the ITT
rule. What that rule forbids is conditioning on *other* functions of ``IS`` that
open a back door: in particular the **cumulative prior dose** ``attend_cumul`` (a
running sum of earlier-period ``IS``), which an earlier version adjusted via
``dose_stage_covariate`` — reopening the latent-GA back door. That term is
**dropped** from the headline here (#269) and is not fitted; nothing downstream of,
or aggregating, the focal dose is conditioned on. It remains available only as a
**flagged sensitivity option** (set
``DoseResponseModelSettings(dose_stage_covariate="attend_cumul")``, the
dose-response analogue of the aligned family's cumulative-session collider
sensitivity) — read any movement of the slope under it as a back-door sensitivity,
not a better estimate.

**There is no clean back-door set.** ``IS`` has three parents in the revised DAG —
``A -> IS``, ``GA -> IS`` and ``IG -> IS`` — and all three also point into the
outcomes. Age and assigned group are measured and adjusted. **Latent general
ability GA is not**, and conditioning on measured baselines does not close it.
``W_pre`` is the regression-to-the-mean / autoregression control (parameterisation,
not a back-door). An earlier version of this docstring claimed v5 has no
``age -> dose`` edge and deliberately omits ``ability -> dose``; both claims are
contradicted by the DAG file (#587 finding 7).

The unblocked path (``GA -> IS``)
---------------------------------
Frank's 2012 caveat ("the children least able to learn tended to show the poorest
attendance") is exactly the ``GA -> IS`` edge. Attendance is **not** randomised
(only ``intervention`` is), so randomisation does not rescue this. The subject
random intercept absorbs *stable* child differences and the between/within split
names them; the baseline-ability sensitivity fit ``lrp-rli-dose-177`` conditions on
the t1 skill cluster (L/E/B) to probe whether the slope survives. If it collapses
there, the signal is substantially ability-confounded — but surviving is
reassurance, not proof, because those are noisy proxies for ``GA``, not ``GA``.

Caveats (carried into the report)
---------------------------------
Adjusted association, not "dose drives gains". The one randomised quantity in this
fit is ``theta_treated`` read in period 1. Phase-1 best strata reached only
R^2 0.1-0.3; the deliverable is a calibrated attendance slope with credible
intervals, not a strong predictor. Group is coded ``G = 2 - group``
(G=1 = immediate-intervention, G=0 = waitlist control; positive = benefit), per the
#117 sign convention.
"""

from language_reading_predictors.statistical_models.context import ModelSpec
from language_reading_predictors.statistical_models.dose_response import (
    DoseResponseModelSettings,
)
from language_reading_predictors.statistical_models.pipelines.dose_response import fit_dose_response

SPEC = ModelSpec(
    model_id="lrp-rli-dose-077",
    kind="dose_response",
    title="Period-resolved dose-response: intervention dose -> word reading (W)",
    outcome_symbol="W",
    adjustment=["G", "A", "W_pre"],
    model_settings=DoseResponseModelSettings(
        adjust_baseline_symbol="W",
        dose_covariate="attend",
        # No cumulative-dose (attend_cumul) control: it conditions on the IS collider
        # and reopens the latent-GA backdoor (#269). It is available only as a flagged
        # sensitivity option (set dose_stage_covariate="attend_cumul").
        period_varying_dose=True,
        use_subject_random_intercept=True,
        outcomes=("W",),
    ),
    extra={
        # The between-period dose-slope scale is a funnel: only three phase slopes
        # inform ``sigma_dose``, so NUTS occasionally shoots into its upper tail.
        # ``beta_dose_phase`` is already non-centred, so the remaining lever is the
        # step size. 0.99 cleared the pre-#587 geometry but leaves 2 divergences in
        # the repaired one (both at sigma_dose ~ 1.4 against a posterior median of
        # 0.14); the strict gate requires zero and the divergence-qualification policy
        # will not waive them. Recorded so the stored reporting fit is reproducible
        # from the registry rather than only from a CLI override —
        # notes/202608050649-reporting-refit-predictive-checks.md and
        # notes/202608232100-dose-response-587-remediation.md.
        "target_accept": 0.995,
    },
)


def fit(config: str = "dev"):
    return fit_dose_response(SPEC, config=config)
