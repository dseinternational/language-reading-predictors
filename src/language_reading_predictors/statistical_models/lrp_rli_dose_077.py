# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
# Documentation edited by a LLM-based AI tool (Codex/GPT-6).

"""Period-resolved attendance associations with word reading (lrp-rli-dose-077).

The Beta-Binomial model predicts the word-reading post-score conditional on its
own baseline logit over all three transitions. It estimates associations with
attendance, not a raw gain score or a causal attendance effect.

Sessions are centred and standardised over fitted rows with positive attendance.
The model separates presence (``theta_treated``) from between-child attendance
and period-specific within-child attendance deviations. Each child's mean uses
their fitted positive-attendance rows only; untreated rows contribute zero to
both dose components. Assigned arm enters from period 2 because it is collinear
with presence in period 1. Phase intercepts allow different mean changes across
the unequal assessment intervals, without defining a common per-month slope.

Age and arm are adjusted, while latent general ability can still confound
attendance and outcome. A child random intercept does not remove that
confounding. The baseline-skill companion ``lrp-rli-dose-177`` checks sensitivity
to measured proxies; a change in slope does not quantify confounding removed.
Cumulative prior dose is omitted from the headline and retained only as a
flagged collider sensitivity (issue #269).

Every coefficient remains an adjusted association, including presence in period
1: conditioning on treatment-induced attendance prevents interpreting it as the
assigned-arm effect. Use the separate available-case modified ITT analysis for
that contrast, subject to its selection, missing-data and timing assumptions.
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
    target_accept=0.995,
)


def fit(config: str = "dev") -> StatisticalFitContext:
    return fit_dose_response(SPEC, config=config)
