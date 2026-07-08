# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""LRPDID06 - waitlist-crossover / difference-in-differences ITT effect on word reading (EWRSWR) - dose-response (W).

A within-person replication of the randomised ITT effect (LRPITT10). Stacks the two
early periods (P1 = t1->t2, P2 = t2->t3) for both arms, with each child as their
own control (a child random intercept) and the immediate arm - treated in both
periods - anchoring the time/maturation trend. The waitlist arm is the only
untreated cell (P1) and crosses to the intervention in P2, so the treatment
coefficient is a difference-in-differences estimate of the ITT effect, identified
jointly by the period-1 between-arm contrast and the waitlist's own P1->P2 jump.

Beta-Binomial on the logit scale (the suite convention), so the ceiling is
respected - an immediate-arm child near the top of a bounded test makes a small
P2 gain *expected*, not a spurious negative trend (the failure mode of a raw-gain
DiD). Compare beta_dose against the single-outcome RCT tau (LRPITT10).

This is the **dose-response** variant: the binary treated indicator is replaced by
the standardised per-period intervention-session count, so ``beta_dose`` is the
effect per 1 SD of sessions attended (a variant of LRPDID01).

Sign convention: positive => intervention helps.
"""

from language_reading_predictors.statistical_models.context import ModelSpec
from language_reading_predictors.statistical_models.pipeline import fit_did

SPEC = ModelSpec(
    model_id="lrp-rli-did-006",
    kind="did",
    title="Waitlist-crossover (DiD) ITT effect of the intervention on word reading (EWRSWR) - dose-response (W)",
    outcome_symbol="W",
    extra={
        "outcomes": ("W",),
        "periods": (0, 1),
        "use_child_re": True,
        "use_age": True,
        "dose": True,
    },
)


def fit(config: str = "dev"):
    return fit_did(SPEC, config=config)
