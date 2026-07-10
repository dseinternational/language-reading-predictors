# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""LRPDID11 - waitlist-crossover / difference-in-differences ITT effect on phonetic spelling (SPPHON) (P), off-floor.

A within-person crossover (difference-in-differences) read on phonetic spelling -
a within-person triangulation of the randomised floor-rule ITT (LRPITT09). Stacks
the two early periods (P1 = t1->t2, P2 = t2->t3) for both arms, with each child as
their own control (a child random intercept) and the immediate arm anchoring the
time/maturation trend.

Phonetic spelling is heavily floored, so - like its ITT sibling - this model takes
the suite's **floor rule** (#119): the observation is a Bernoulli on the binary
off-floor indicator (period post > 0), not the graded Beta-Binomial count. The DiD
contrast ``delta`` is therefore the within-person effect on the LOG-ODDS of coming
off the floor, and its marginal (``delta_items``, n_trials = 1) is an off-floor
RISK DIFFERENCE - the change in the probability of coming off the floor - not an
item count. Same waitlist-crossover design as the graded family; only the
likelihood differs (no dispersion kappa).

The parallel-trends assumption now applies to the off-floor PROBABILITY scale
rather than the graded count - flagged for review sign-off (#226). Sign convention:
positive => intervention helps (raises Pr(off-floor)).
"""

from language_reading_predictors.statistical_models.context import ModelSpec
from language_reading_predictors.statistical_models.pipeline import fit_did

SPEC = ModelSpec(
    model_id="lrp-rli-did-011",
    kind="did",
    title="Waitlist-crossover (DiD) ITT effect of the intervention on phonetic spelling (SPPHON) (P), off-floor",
    outcome_symbol="P",
    extra={
        "outcomes": ("P",),
        "periods": (0, 1),
        "use_child_re": True,
        "use_age": True,
        "dose": False,
        "likelihood": "bernoulli_offfloor",
    },
)


def fit(config: str = "dev"):
    return fit_did(SPEC, config=config)
