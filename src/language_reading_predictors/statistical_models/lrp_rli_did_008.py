# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""LRPDID08 - waitlist-crossover / difference-in-differences ITT effect on taught receptive vocabulary, block 1 (b1retau) (TR).

A within-person replication of the randomised ITT effect (LRPITT01). Stacks the two
early periods (P1 = t1->t2, P2 = t2->t3) for both arms, with each child as their
own control (a child random intercept) and the immediate arm - treated in both
periods - anchoring the time/maturation trend. The waitlist arm is the only
untreated cell (P1) and crosses to the intervention in P2, so the treatment
coefficient is a difference-in-differences estimate of the ITT effect, identified
jointly by the period-1 between-arm contrast and the waitlist's own P1->P2 jump.

Beta-Binomial on the logit scale (the suite convention), so the ceiling is
respected. Compare delta against the single-outcome RCT tau (LRPITT01).

Sign convention: positive => intervention helps.
"""

from language_reading_predictors.statistical_models.context import ModelSpec
from language_reading_predictors.statistical_models.pipeline import fit_did

SPEC = ModelSpec(
    model_id="lrp-rli-did-008",
    kind="did",
    title="Waitlist-crossover (DiD) ITT effect of the intervention on taught receptive vocabulary, block 1 (b1retau) (TR)",
    outcome_symbol="TR",
    extra={
        "outcomes": ("TR",),
        "periods": (0, 1),
        "use_child_re": True,
        "use_age": True,
        "dose": False,
    },
)


def fit(config: str = "dev"):
    return fit_did(SPEC, config=config)
