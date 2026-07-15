# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""LRPDID10 - arm-by-wave contrasts for basic concept knowledge (F).

The t1--t3 outcome levels are modelled with a saturated arm-by-wave structure.
``tau_t2`` is the clean randomised t2 contrast and is compared with LRPITT25.
``arm_gap_t3`` and the derived waitlist catch-up contrast are post-crossover
associations. The design does not condition on the treatment-affected t2 score.
"""

from language_reading_predictors.statistical_models.context import ModelSpec
from language_reading_predictors.statistical_models.pipeline import fit_did

SPEC = ModelSpec(
    model_id="lrp-rli-did-010",
    kind="did",
    title="Waitlist-crossover arm-by-wave contrasts for basic concept knowledge (CELF) (F)",
    outcome_symbol="F",
    family="did",
    design="waitlist-crossover arm-by-wave levels",
    estimand_type="mixed",
    causal_status="t2 randomised; post-crossover contrasts associational",
    extra={
        "outcomes": ("F",),
        "waves": (0, 1, 2),
        "use_child_re": True,
        "use_age": True,
        "dose": False,
    },
)


def fit(config: str = "dev"):
    return fit_did(SPEC, config=config)
