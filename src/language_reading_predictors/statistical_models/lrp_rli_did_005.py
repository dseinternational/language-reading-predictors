# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""LRPDID05 - arm-by-wave contrasts for receptive vocabulary (R).

The t1--t3 outcome levels are modelled with a saturated arm-by-wave structure.
``tau_t2`` is the clean randomised t2 contrast and is compared with LRPITT05.
``arm_gap_t3`` is also identified by the original randomisation, but of a
different exposure -- assignment to the early-start versus delayed-start
treatment schedule (both arms treated by t3) -- and ``delta_crossover = tau_t2 -
arm_gap_t3`` is the change between those two randomised regime contrasts, never
an identified catch-up. The design does not condition on the treatment-affected
t2 score.
"""

from language_reading_predictors.statistical_models.context import ModelSpec
from language_reading_predictors.statistical_models.pipelines.did import fit_did

SPEC = ModelSpec(
    model_id="lrp-rli-did-005",
    kind="did",
    title="Waitlist-crossover arm-by-wave contrasts for receptive vocabulary (ROWPVT) (R)",
    outcome_symbol="R",
    family="did",
    design="waitlist-crossover arm-by-wave levels",
    estimand_type="mixed",
    causal_status="t2 randomised; t3 a randomised treatment-schedule contrast",
    extra={
        "outcomes": ("R",),
        "waves": (0, 1, 2),
        "use_child_re": True,
        "use_age": True,
        "dose": False,
    },
)


def fit(config: str = "dev"):
    return fit_did(SPEC, config=config)
