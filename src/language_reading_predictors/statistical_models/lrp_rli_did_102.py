# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""LRPDID102 - wide-tau_t2 prior-sensitivity companion to LRPDID02 (letter sounds).

The prior-critical review (`notes/202607211500-prior-critical-review.md`, #382
recommendation 3) found the letter-sound arm-by-wave fit the clearest case of a
posterior in its prior's right tail: under the tier default ``tau_t2 ~
Normal(0, 0.5)`` LRPDID02 reports a posterior median of about 0.60 logits, 89%
[0.20, 0.98], P(>0) = 0.99 — the data want a larger positive effect than the
zero-centred half-unit scale comfortably allows, and power-scaling flags a
prior-data conflict. The review reads this as mild *attenuation, not a false
result*; this companion tests exactly that.

LRPDID102 is **identical to LRPDID02 except** ``tau_t2``, which takes
``Normal(0, 1)`` — the wide end of the review's defensible 0.75-1.0 range — via
the typed ``tau_t2_prior_sigma`` setting. ``arm_gap_t3``, ``beta_period`` and
every other term keep the tier scale, so the single causal coefficient's prior
is the only difference and any posterior shift is attributable to it alone.
(``delta_crossover = tau_t2 - arm_gap_t3`` is therefore *not* a comparison
target here: its two legs are deliberately priced differently in this fit.)

How to read the comparison: if LRPDID02's ``tau_t2`` is genuinely
data-dominated, the posterior here should match it to within Monte-Carlo error;
a materially larger median under the wider prior confirms the reference
estimate is prior-attenuated and should be read as conservative. Per
recommendation 3's scope note, the dose model LRPDID07 has **no** ``tau_t2``
(its estimand is the observational ``mu_dose ~ Normal(0, 1)``, already on the
wide scale, assigned no causal status), so no dose companion exists.

Same reading rules as LRPDID02: ``tau_t2`` is the clean randomised t2 contrast;
``arm_gap_t1`` is a baseline-balance quantity; ``arm_gap_t3`` is the randomised
early-start-versus-delayed-start schedule contrast and ``delta_crossover`` the
change between the two randomised regime contrasts, neither of them
mechanism-identified.
"""

from language_reading_predictors.statistical_models.context import ModelSpec
from language_reading_predictors.statistical_models.pipelines.did import fit_did

SPEC = ModelSpec(
    model_id="lrp-rli-did-102",
    kind="did",
    title=(
        "Wide-tau_t2 prior sensitivity for the letter-sound arm-by-wave contrasts "
        "(YARC-LSK) (L)"
    ),
    outcome_symbol="L",
    family="did",
    design="waitlist-crossover arm-by-wave levels",
    estimand_type="mixed",
    causal_status="t2 randomised; t3 a randomised treatment-schedule contrast",
    extra={
        # Identical to LRPDID02 in every respect except tau_t2_prior_sigma.
        "outcomes": ("L",),
        "waves": (0, 1, 2),
        "use_child_re": True,
        "use_age": True,
        "dose": False,
        # The single free variable (#382 rec 3): the causal contrast's prior
        # widens from the proximal-tier Normal(0, 0.5) to Normal(0, 1).
        "tau_t2_prior_sigma": 1.0,
    },
)


def fit(config: str = "dev"):
    return fit_did(SPEC, config=config)
