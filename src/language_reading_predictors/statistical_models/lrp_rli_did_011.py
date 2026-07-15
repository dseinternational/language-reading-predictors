# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""LRPDID11 - arm-by-wave off-floor prevalence contrasts for phonetic spelling (P).

The Bernoulli outcome is whether each child is off the floor at t1, t2 and t3.
This is prevalence, ``Pr(score > 0)``, not the ITT floor-exit estimand
``Pr(post > 0 | pre = 0)``. ``tau_t2`` is the clean randomised t2 arm contrast in
off-floor log-odds; its standardised marginal is a risk difference. The t3 arm gap
and the derived waitlist catch-up contrast are post-crossover associations.

The model deliberately does not condition on a period-start score: the immediate
arm's t2 score is treatment-affected before P2. Modelling the t1 prevalence directly
retains baseline balance without post-treatment adjustment. LRPITT09 remains the
primary randomised floor-exit analysis.
"""

from language_reading_predictors.statistical_models.context import ModelSpec
from language_reading_predictors.statistical_models.pipeline import fit_did

SPEC = ModelSpec(
    model_id="lrp-rli-did-011",
    kind="did",
    title="Waitlist-crossover arm-by-wave off-floor prevalence contrasts for phonetic spelling (SPPHON) (P)",
    outcome_symbol="P",
    family="did",
    design="waitlist-crossover arm-by-wave prevalence levels",
    estimand_type="mixed",
    causal_status="t2 randomised; post-crossover contrasts associational",
    extra={
        "outcomes": ("P",),
        "waves": (0, 1, 2),
        "use_child_re": True,
        "use_age": True,
        "dose": False,
        "likelihood": "bernoulli_offfloor",
    },
)


def fit(config: str = "dev"):
    return fit_did(SPEC, config=config)
