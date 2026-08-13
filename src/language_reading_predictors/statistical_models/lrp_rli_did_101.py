# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""LRPDID101 - independent-prior intercept sensitivity companion to LRPDID01 (W).

The arm-by-wave models centre their intercept prior on the pooled observed t1
logit — an **empirical-Bayes anchor**: the same t1 observations then enter the
fitted likelihood (#390 P1). Frank's 2026-07-24 option-B ruling retained the
anchor with three conditions; conditions 2 and 3 (the explicit label and the
stated prior-predictive limitation, applied across the anchored families)
landed in #481. This companion is **condition 1**: a sensitivity fit whose
intercept prior is *genuinely independent* of the outcomes — not a wider sigma
around the same anchor, whose mean would remain data-dependent, but the
ordinary free zero-centred ``alpha ~ Normal(0, 1.5)`` the dose variants already
use, via the typed ``use_intercept_anchor=False`` setting.

LRPDID101 is identical to LRPDID01 in every other respect: same data, waves,
child random intercept, age term, likelihood, tau tier and sampler settings.
How to read the comparison: the anchor moves the *level*; ``tau_t2`` is a
within-t2 arm contrast a full wave away from the t1 anchor, so if the anchor is
doing no inferential work beyond locating the intercept, ``tau_t2`` (and the
other arm gaps) should match LRPDID01 to within Monte-Carlo error. A material
shift would mean the anchor is leaking into the randomised contrast and the
family's intercept policy needs revisiting. The prior-predictive check here is
a genuine location check — this fit's intercept was *not* told where the data
are, which is exactly the limitation #481's note records for the anchored fits.

Same reading rules as LRPDID01: ``tau_t2`` is the clean randomised t2 contrast;
``arm_gap_t1`` is a baseline-balance quantity; ``arm_gap_t3`` and the crossover
contrast are post-crossover associations.
"""

from language_reading_predictors.statistical_models.context import ModelSpec
from language_reading_predictors.statistical_models.pipelines.did import fit_did

SPEC = ModelSpec(
    model_id="lrp-rli-did-101",
    kind="did",
    title=(
        "Independent-prior intercept sensitivity for the word-reading "
        "arm-by-wave contrasts (EWRSWR) (W)"
    ),
    outcome_symbol="W",
    family="did",
    design="waitlist-crossover arm-by-wave levels",
    estimand_type="mixed",
    causal_status="t2 randomised; post-crossover contrasts associational",
    extra={
        # Identical to LRPDID01 in every respect except the intercept prior.
        "outcomes": ("W",),
        "waves": (0, 1, 2),
        "use_child_re": True,
        "use_age": True,
        "dose": False,
        # #390 P1 condition 1: replace the empirical-Bayes pooled-t1 anchor
        # with the free zero-centred tier-scale intercept.
        "use_intercept_anchor": False,
    },
)


def fit(config: str = "dev"):
    return fit_did(SPEC, config=config)
