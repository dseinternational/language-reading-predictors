# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""LRPDID13 - exploratory heterogeneity in waitlist catch-up (word reading, W).

This arm-by-wave model adds a waitlist-child random deviation to the t3 catch-up
association. ``delta_crossover_i = delta_crossover + v_delta_i`` varies only the
waitlist arm's post-crossover t3 level; it is not a random treatment-effect slope.
``sigma_delta`` therefore describes heterogeneity in observed waitlist catch-up,
which may combine response, maturation, history and measurement variation.

The fixed ``tau_t2`` remains the clean randomised t2 arm contrast. The heterogeneity
component is exploratory and cannot identify individual causal responders.
"""

from language_reading_predictors.statistical_models.context import ModelSpec
from language_reading_predictors.statistical_models.pipeline import fit_did

SPEC = ModelSpec(
    model_id="lrp-rli-did-013",
    kind="did",
    title="Waitlist-crossover arm-by-wave model with exploratory catch-up heterogeneity (W)",
    outcome_symbol="W",
    family="did",
    design="waitlist-crossover arm-by-wave levels + waitlist t3 random deviation",
    estimand_type="association",
    causal_status="t2 fixed contrast randomised; heterogeneity associational",
    extra={
        "outcomes": ("W",),
        "waves": (0, 1, 2),
        "use_child_re": True,
        "use_age": True,
        "use_varying_delta": True,
    },
)


def fit(config: str = "dev"):
    return fit_did(SPEC, config=config)
