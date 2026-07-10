# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""LRPDID10 - waitlist-crossover / difference-in-differences effect on basic concept knowledge (CELF) (F).

A within-person crossover (difference-in-differences) read on basic concept
knowledge. Stacks the two early periods (P1 = t1->t2, P2 = t2->t3) for both arms,
with each child as their own control (a child random intercept) and the immediate
arm anchoring the time/maturation trend.

Basic concepts (CELF) is one of the eight standardised ITT outcomes; its standalone
randomised ITT is LRPITT25 (added under #228). Read this DiD as the within-person
triangulation of that randomised estimate under the parallel-trends assumption, not
an independent replication. Same waitlist-crossover structure and Beta-Binomial-on-logit
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
