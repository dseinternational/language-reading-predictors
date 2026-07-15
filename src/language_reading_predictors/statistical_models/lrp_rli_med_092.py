# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""MED-092 - period-stacked g-formula: reading via letter-sounds, on the gain-factor scaffold.

The #229 recommendation-2 companion to the Phase-0 mediation suite: LRP59
(``med-059``) decomposes the randomised t1->t2 intervention effect on word
reading (W) into the part flowing through letter-sound knowledge (L) and the
rest, but pins itself to the one randomised window (n ~ 53 children, one row
each). MED-092 refits the same joint mediator + outcome design over **all
stacked period transitions** (``phase_mode="all"``, ~3 rows per child), with
the exposure being the per-period **on-intervention** indicator — the term the
gain-factor models already treat as ignorable — rather than the Phase-0
randomised group.

Design (see :func:`factories.build_period_stacked_mediation_model` and
:func:`mediation.decompose_period_stacked`):

- Both legs carry the gain-factor machinery: per-phase intercepts and a
  per-leg child random intercept (a partial, shrunken stand-in for stable
  between-child differences — not a control for latent general ability).
- The on-intervention indicator varies between arms **only in period 1** (after
  the waitlist crossover both arms are on the programme), so the exposure
  contrast is still anchored on the randomised window. What the stacking buys
  is that the mediator -> outcome leg (``b_M``, the coefficient carrying the
  indirect effect) and every covariate term are informed by **all** periods.
- Confounders mirror LRP59's set: expressive/receptive vocabulary at the
  **period start**, hearing and speech production as raw covariates on the
  gain-factor timing split (hearing contemporaneous; speech at the t1
  baseline, the A1 timing decision 2026-07-13).
- ``mediation_summary.csv`` is the all-period decomposition;
  ``mediation_summary_p1.csv`` restricts the averaging to the period-1
  (randomised, all-untreated-baseline) rows — the LRP59-comparable readout,
  mirroring the gain-factor family's period-1 treatment marginal (#247 P2).

**The stated, reviewable trade (#229):** every quantity here leans on the
gain-factor family's ignorability assumption — on-intervention exchangeable
given period-start state, phase, age and the child intercept — instead of pure
randomisation. Post-crossover period-start baselines are descendants of
earlier periods' exposure, which is admissible for this **per-period**
estimand but is exactly why no cumulative (multi-period) effect is decomposed.
All of LRP59's non-identification caveats carry over unchanged (latent-GA
mediator-outcome confounding; treatment-induced dose ``IS``; mediator and
outcome contemporaneous within each period). Read this as a triangulation
companion to ``med-059``/``064`` — direction and rough-share agreement is the
deliverable — never a replacement.
"""

from language_reading_predictors.statistical_models.context import ModelSpec
from language_reading_predictors.statistical_models.pipeline import (
    fit_mediation_period_stacked,
)

SPEC = ModelSpec(
    model_id="lrp-rli-med-092",
    kind="mediation",
    title=(
        "Period-stacked mediation: does being on the programme raise word "
        "reading (W) via letter-sound knowledge (L)? (gain-factor scaffold)"
    ),
    outcome_symbol="W",
    mechanism_symbol="L",  # the mediator
    adjustment=[
        # T = per-period on-intervention exposure; A = age; own baselines are
        # structural. Confounders mirror med-059: E/R at the period start,
        # hearing + speech production as raw covariates (gain-factor timing).
        "T", "A", "E", "R", "L_pre", "W_pre",
        "hs", "hs_missing", "deapp_c", "deapp_c_missing",
    ],
    extra={},
)


def fit(config: str = "dev"):
    return fit_mediation_period_stacked(SPEC, config=config)
