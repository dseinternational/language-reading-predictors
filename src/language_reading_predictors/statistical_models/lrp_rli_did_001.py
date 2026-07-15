# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""LRPDID01 - waitlist-crossover arm-by-wave contrasts for word reading (W).

The t1--t3 outcome levels are modelled with a saturated arm-by-wave structure.
``tau_t2`` is the clean randomised immediate-minus-waitlist contrast at t2 and is
compared with LRPITT10. ``arm_gap_t3`` is the post-crossover 40-week-versus-20-week
association; ``delta_crossover = tau_t2 - arm_gap_t3`` describes waitlist catch-up
and is not a second randomised effect. Modelling t1 as an outcome avoids conditioning
on the treatment-affected t2 period-start score while retaining baseline balance.
"""

from language_reading_predictors.statistical_models.context import ModelSpec
from language_reading_predictors.statistical_models.pipeline import fit_did

SPEC = ModelSpec(
    model_id="lrp-rli-did-001",
    kind="did",
    title="Waitlist-crossover arm-by-wave contrasts for word reading (EWRSWR) (W)",
    outcome_symbol="W",
    family="did",
    design="waitlist-crossover arm-by-wave levels",
    estimand_type="mixed",
    causal_status="t2 randomised; post-crossover contrasts associational",
    extra={
        "outcomes": ("W",),
        "waves": (0, 1, 2),
        "use_child_re": True,
        "use_age": True,
        "dose": False,
    },
)


def fit(config: str = "dev"):
    return fit_did(SPEC, config=config)
