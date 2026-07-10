# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""LRPDID09 - waitlist-crossover / difference-in-differences ITT effect on standardised expressive vocabulary (EOWPVT) (E).

A within-person replication of the randomised ITT effect (LRPITT06). Stacks the two
early periods (P1 = t1->t2, P2 = t2->t3) for both arms, with each child as their
own control (a child random intercept) and the immediate arm - treated in both
periods - anchoring the time/maturation trend. The treatment coefficient is a
difference-in-differences estimate of the ITT effect.

Standardised expressive vocabulary is a distal broad-transfer measure, so - as with
the standardised-receptive DiD (LRPDID05) - a near-null within-person effect would
triangulate the ITT vocabulary result rather than contradict it. Beta-Binomial on
the logit scale (the suite convention), so the ceiling is respected.

Sign convention: positive => intervention helps.
"""

from language_reading_predictors.statistical_models.context import ModelSpec
from language_reading_predictors.statistical_models.pipeline import fit_did

SPEC = ModelSpec(
    model_id="lrp-rli-did-009",
    kind="did",
    title="Waitlist-crossover (DiD) ITT effect of the intervention on standardised expressive vocabulary (EOWPVT) (E)",
    outcome_symbol="E",
    extra={
        "outcomes": ("E",),
        "periods": (0, 1),
        "use_child_re": True,
        "use_age": True,
        "dose": False,
    },
)


def fit(config: str = "dev"):
    return fit_did(SPEC, config=config)
