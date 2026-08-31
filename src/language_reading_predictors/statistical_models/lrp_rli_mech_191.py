# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""LRP191 - GP knee-test: intervention sessions delivered (IS) -> word reading (W).

A NEW dose-response knee-test: does the amount of intervention delivered show a
"knee" - a number of sessions beyond which more sessions go with a more marked
difference in word reading - the way LRP58 found for letter sounds? The
dose_response family (LRP77 etc.) fits sessions as a straight-line / period-varying
slope and has no curve to find a knee in; this model fits the session exposure as a
nonparametric HSGP curve instead, so a knee can be looked for. It is the first
continuous-covariate GP mechanism in the suite: the readiness-threshold knee is
reported in the exposure's own raw units (sessions attended in a period), not a
bounded count. target_accept is 0.999 (per LRP58); a persisting divergence is itself
an honest result and would leave the knee untestable.

**Exposure.** Per-period sessions attended (``attend``), the same exposure LRP77 uses
- NOT cumulative dose. Cumulative attendance is a running sum of the IS collider and,
if *conditioned on*, reopens the latent-GA backdoor (see LRP77/#269); here sessions is
the exposure, not an adjuster, so that caveat is about a different quantity, but
per-period sessions keeps this model consistent with the established dose model and the
DAG's IS node. ``mechanism_is_covariate`` enters it as a standardised continuous
covariate.

**Population: on-intervention periods only** (``exposure_positive_only``, #586
finding 2, decided 2026-08-23). ``attend`` is an *interval* covariate read from a
transition's **pre** row, and the loader records an absent session count as a zero
rather than as missing. An earlier version of this model claimed its 55 missing
``attend`` values left "exactly the on-intervention rows". They did not: 54 of those 55
are t4 cells, which are never a transition's pre row and so could not have entered the
three-transition frame at all, and only one was a relevant t2 cell. The frame that
actually fitted held **156 rows from 53 children, 28 of them at zero sessions** - and
those zeros were structured, not scattered. In period 1 *all 25* fitted waitlist rows
sat at zero and *no* immediate-arm row did, with only seven fitted rows anywhere in
1-30 sessions. The bottom of the "0 to 94 sessions" range was therefore an arm and
period contrast wearing a dose label, and an arm covariate cannot manufacture overlap
where the exposure support is structurally disjoint.

The zero-session rows are now excluded. What remains is an association **among treated
periods** - the intensive margin only. It does not use the randomised zero-dose anchor
and says nothing about intervention versus none; that contrast belongs to the ITT
suite. (No ``require_observed`` flag: that path needs an ``attend_missing`` indicator
the loader does not build for ``attend``, and adding one would silently change the
fitted sample of every model that already adjusts for ``attend``, e.g. LRP58.)

**Adjustment set (revised DAG, 2026-07-10).** Re-derived by a backdoor d-separation
search with the latent GA held (the criterion that reproduces LRP58 and dose-077
exactly): IS's parents are {A, GA, IG} and nothing else, so the minimal observed set
blocking every backdoor to WR is {A, IG}. Group G(=IG) is the always-in term, A the
linear age precision covariate, and W_pre the autoregressive baseline. No measure or
hearing/speech adjusters are needed (they are not parents of IS). Identical adjustment
to dose-077 - only the functional form (HSGP curve vs period-varying slope) changes.

**Strictly observational.** Session dose was not randomised (how much a child attended
reflects ability, attendance and availability) and IS is a partial collider, so the
curve is an ADJUSTED ASSOCIATION / sensitivity view, never a treatment effect. Residual
confounding by latent general ability (GA) remains. The corresponding randomised-arm
result is an available-case modified ITT estimate in the ITT suite; read this as
"among children receiving the intervention, those who attended more scored higher, and
here is whether that relationship bends", not "more sessions cause faster progress" and
not "this many sessions is the threshold". Any located steepest interval is a feature
of the fitted curve over the observed range, and the report only calls it a knee when
the curve genuinely bends and the location is interior to the observed support.

**Awaiting refit.** The stored ``reporting`` artefacts predate this population change
and were fitted on the zero-anchored frame; they must not be read against the text
above until the model is refitted.
"""

from language_reading_predictors.statistical_models.context import ModelSpec
from language_reading_predictors.statistical_models.mechanism import (
    MechanismModelSettings,
)
from language_reading_predictors.statistical_models.pipelines.mechanism import fit_mechanism

SPEC = ModelSpec(
    model_id="lrp-rli-mech-191",
    kind="mechanism",
    title="GP knee-test: intervention sessions (IS) -> word reading (W)",
    outcome_symbol="W",
    mechanism_symbol="attend",
    adjustment=["G", "A", "W_pre"],
        # target_accept 0.999 per LRP58 (the setting that stabilises the L->W curve).
    target_accept=0.999,
    model_settings=MechanismModelSettings(
        # Only the outcome (W) is a bounded-count measure; the exposure is the
        # attend covariate, so the measure complete-case mask is W alone.
        outcomes=("W",),
        adjust_baseline_symbol="W",
        # IS's parents are {A, GA, IG}; A + G(=IG) block every observable backdoor,
        # so no hearing/speech/measure adjusters are required (they are not parents
        # of IS). Matches dose-077.
        adjust_for=(),
        # Continuous-covariate exposure with the HSGP curve ON: the steepest
        # interval is reported in raw sessions, not a bounded count.
        mechanism_is_covariate=True,
        # On-intervention periods only (#586 finding 2). Without it the frame kept
        # 28 zero-session rows, 25 of them the entire period-1 waitlist arm, so the
        # low end of the curve was an arm/period contrast rather than a dose one.
        exposure_positive_only=True,
        use_age_gp=False,
        phase_specific_mechanism=False,
        use_subject_random_intercept=True,
        # Thin-support HSGP reparameterisation (#438 / notes/202607251500-mech-hsgp-
        # reparameterisation.md): basis count 6 (from the shared default 10) and the
        # tighter InverseGamma(8, 8) lengthscale prior. Adopted here because this fit
        # holds 28 divergences at target_accept 0.999, and a nonlinear knee/shape is
        # zero-divergence-only under notes/202608021625-divergence-qualification-policy.md
        # — the geometry has to be fixed, not waived. Per-model opt-in, not a default:
        # the same lever regressed mech-173 from 0 to 10 divergences.
        mech_hsgp_m=6,
        mech_lengthscale_tight=True,
    ),
)


def fit(config: str = "dev"):
    return fit_mechanism(SPEC, config=config)
