# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""LRP92 - GP knee-test: intervention sessions delivered (IS) -> word reading (W).

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
covariate. ``attend`` is not in the mean-fill covariate set, so its 55 missing values
(the pre-/no-intervention rows, where a per-period session count is undefined) survive
as NaN and are dropped by the factory's exposure keep-mask - the on-intervention rows
that actually carry a dose are exactly the ones a dose-response should use. (No
``require_observed`` flag: that path needs an ``attend_missing`` indicator the loader
does not build for ``attend``, and adding one would silently change the fitted sample
of every model that already adjusts for ``attend``, e.g. LRP58.)

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
confounding by latent general ability (GA) remains. The randomised causal claim lives
in the ITT suite; read this as "children who attended more scored higher, and here is
whether that relationship bends", not "more sessions cause faster progress".
"""

from language_reading_predictors.statistical_models.context import ModelSpec
from language_reading_predictors.statistical_models.pipeline import fit_mechanism

SPEC = ModelSpec(
    model_id="lrp-rli-mech-191",
    kind="mechanism",
    title="GP knee-test: intervention sessions (IS) -> word reading (W)",
    outcome_symbol="W",
    mechanism_symbol="attend",
    adjustment=["G", "A", "W_pre"],
    extra={
        # Only the outcome (W) is a bounded-count measure; the exposure is the
        # attend covariate, so the measure complete-case mask is W alone.
        "outcomes": ("W",),
        "adjust_baseline_symbol": "W",
        # IS's parents are {A, GA, IG}; A + G(=IG) block every observable backdoor,
        # so no hearing/speech/measure adjusters are required (they are not parents
        # of IS). Matches dose-077.
        "adjust_for": (),
        # Continuous-covariate exposure with the HSGP curve ON (knee-test): the
        # readiness knee is reported in raw sessions, not a bounded count.
        "mechanism_is_covariate": True,
        "use_age_gp": False,
        "phase_specific_mechanism": False,
        "use_subject_random_intercept": True,
        # target_accept 0.999 per LRP58 (the setting that stabilises the L->W curve).
        "target_accept": 0.999,
    },
)


def fit(config: str = "dev"):
    return fit_mechanism(SPEC, config=config)
