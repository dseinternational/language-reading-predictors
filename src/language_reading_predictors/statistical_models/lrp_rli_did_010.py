# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""LRPDID10 - waitlist-crossover / difference-in-differences effect on basic concept knowledge (CELF) (F).

A within-person crossover (difference-in-differences) read on basic concept
knowledge. Stacks the two early periods (P1 = t1->t2, P2 = t2->t3) for both arms,
with each child as their own control (a child random intercept) and the immediate
arm anchoring the time/maturation trend.

Unlike the other DiD models, basic concepts (CELF) has **no** standalone randomised
ITT model in the suite (F sits outside the eight standardised ITT outcomes), so this
is the within-person estimate of whether the intervention lifts concept knowledge -
read it as triangulation under the parallel-trends assumption, not a replication of
a randomised result. Same waitlist-crossover structure and Beta-Binomial-on-logit
convention as the rest of the family.

Sign convention: positive => intervention helps.
"""

from language_reading_predictors.statistical_models.context import ModelSpec
from language_reading_predictors.statistical_models.pipeline import fit_did

SPEC = ModelSpec(
    model_id="lrp-rli-did-010",
    kind="did",
    title="Waitlist-crossover (DiD) effect of the intervention on basic concept knowledge (CELF) (F)",
    outcome_symbol="F",
    extra={
        "outcomes": ("F",),
        "periods": (0, 1),
        "use_child_re": True,
        "use_age": True,
        "dose": False,
    },
)


def fit(config: str = "dev"):
    return fit_did(SPEC, config=config)
