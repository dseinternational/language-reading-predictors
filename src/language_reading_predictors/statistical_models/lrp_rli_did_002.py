# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""LRPDID02 - waitlist-crossover / difference-in-differences ITT effect on letter-sound knowledge (YARC-LSK) (L).

A within-person replication of the randomised ITT effect (LRPITT07). Stacks the two
early periods (P1 = t1->t2, P2 = t2->t3) for both arms, with each child as their
own control (a child random intercept) and the immediate arm - treated in both
periods - anchoring the time/maturation trend. The waitlist arm is the only
untreated cell (P1) and crosses to the intervention in P2, so the treatment
coefficient is a difference-in-differences estimate of the ITT effect, identified
jointly by the period-1 between-arm contrast and the waitlist's own P1->P2 jump.

Beta-Binomial on the logit scale (the suite convention), so the ceiling is
respected - an immediate-arm child near the top of a bounded test makes a small
P2 gain *expected*, not a spurious negative trend (the failure mode of a raw-gain
DiD). Compare delta against the single-outcome RCT tau (LRPITT07).

Sign convention: positive => intervention helps.
"""

from language_reading_predictors.statistical_models.context import ModelSpec
from language_reading_predictors.statistical_models.pipeline import fit_did

SPEC = ModelSpec(
    model_id="lrp-rli-did-002",
    kind="did",
    title="Waitlist-crossover (DiD) ITT effect of the intervention on letter-sound knowledge (YARC-LSK) (L)",
    outcome_symbol="L",
    extra={
        "outcomes": ("L",),
        "periods": (0, 1),
        "use_child_re": True,
        "use_age": True,
        "dose": False,
    },
)


def fit(config: str = "dev"):
    return fit_did(SPEC, config=config)
