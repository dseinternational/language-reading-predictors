# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""LRPDID06 - pooled session-dose association for word reading (W).

This transition-based sensitivity model keeps P1/P2 because sessions are interval
exposures. It separates randomised arm/history, current treatment presence and
session intensity, and adjusts both periods for the shared pre-randomisation t1
outcome and t1 age. Sessions are centred and scaled only among treated rows;
untreated rows have zero intensity. ``beta_dose`` is therefore an observational
intensive-margin association per treated-row SD, not an ITT or causal effect.
"""

from language_reading_predictors.statistical_models.context import ModelSpec
from language_reading_predictors.statistical_models.pipeline import fit_did

SPEC = ModelSpec(
    model_id="lrp-rli-did-006",
    kind="did",
    title="Waitlist-crossover pooled session-dose association for word reading (EWRSWR) (W)",
    outcome_symbol="W",
    family="did",
    design="waitlist-crossover transition dose intensive margin",
    estimand_type="association",
    causal_status="none for session-dose coefficient",
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
