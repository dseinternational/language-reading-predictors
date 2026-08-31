# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""LRP302 - phase-stability sensitivity for the L -> W linear slope (#604).

Companion to **LRP101**, identical in rows, outcome, baseline, adjustment set and
priors, differing in exactly one thing: the exposure slope is allowed to vary by
period transition, partially pooled toward a shared mean
(``beta_mech_phase = mu_mech + sigma_mech_phase * z_phase``).

**Why.** Every active mechanism fit stacks three period transitions - t1->t2
(randomised), t2->t3 and t3->t4 (both post-crossover) - and gives each phase its own
intercept but **one common exposure slope** and **one common arm coefficient**. That
is an assumption, not a finding: it says the exposure-outcome relationship is stable
across three substantively different treatment histories, and nothing in the
published output tested it. If the assumption fails, omitted group-by-period
structure has nowhere to go but the exposure term.

``phase_specific_mechanism`` looks like the answer but is not - it builds per-period
*curves* that the family's curve, items and steepest-interval artefacts cannot
report, and #599 rejects it during run-plan resolution. This is the linear
alternative, and it is deliberately not a per-period curve.

**Overlap.** The audit made this conditional on the periods occupying comparable
parts of the exposure range. ``exposure_support.csv`` (#599) now publishes fitted
exposure support by period and arm, and across the regenerated mechanism fits every
one has overlapping per-period interquartile ranges - 17-26.5 of 32 items for the
letter-sound models. The report renders that table beside the per-period slopes so a
reader can see the overlap the comparison rests on.

**Partial pooling, not three free slopes.** At roughly 52 rows per period,
independent slopes are noisy; the deviations are shrunk toward ``mu_mech`` unless
the data show real period variation, matching how ``dose_response`` handles its
period-varying dose. ``lrp-rli-mech-101`` is the pooled comparator for the nested
PSIS-LOO test, so "is the slope stable across periods?" gets a predictive answer
rather than an eyeballed forest.

**Read.** A period difference is evidence against pooling, **not** evidence about
mechanism change over time: a child's third transition differs from their first in
age, treatment history and measurement position at once. Only the t1->t2 transition
is randomised-arm-clean, and even there the *exposure* is not randomised - so all
three per-period slopes are adjusted associations.
"""

from language_reading_predictors.statistical_models.context import (
    ModelSpec,
    StatisticalFitContext,
)
from language_reading_predictors.statistical_models.mechanism import (
    MechanismModelSettings,
)
from language_reading_predictors.statistical_models.pipelines.mechanism import fit_mechanism

SPEC = ModelSpec(
    model_id="lrp-rli-mech-302",
    kind="mechanism",
    title="Phase-varying slope: letter sounds (L) -> word reading (W)",
    outcome_symbol="W",
    mechanism_symbol="L",
    # The non-centred per-period slope vector is a small hierarchical funnel:
    # ``sigma_mech_phase`` is informed by three periods, so its lower tail pinches the
    # geometry and the preset 0.95 leaves a handful of divergences. 0.995 clears them
    # (verified at rep-lite: 8 divergences at 0.90, 2 at 0.95, 0 at 0.995). This is a
    # *raise* over the preset for a specific, diagnosed geometry — never a blanket
    # escalation, which can lower an in-module default and manufacture a false pass.
    target_accept=0.995,
    adjustment=["G", "A", "W_pre"],
    model_settings=MechanismModelSettings(
        adjust_baseline_symbol="W",
        outcomes=("W", "L"),
        adjust_for=("hs", "hs_missing", "attend", "deapp_c", "deapp_c_missing"),
        linear_mechanism=True,
        phase_varying_slope=True,
        use_age_gp=False,
        use_subject_random_intercept=True,
    ),
)


def fit(config: str = "dev") -> StatisticalFitContext:
    return fit_mechanism(SPEC, config=config)
