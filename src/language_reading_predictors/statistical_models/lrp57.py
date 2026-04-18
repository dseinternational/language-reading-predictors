# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""LRP57 - Mechanism model: expressive vocabulary (E) -> word reading (W).

Adjustment set (DAG-derived):

- G, A: standard confounders.
- R: confounds E and W (R -> E and R -> W). Must be conditioned on.
- W_pre: baseline word reading.

F, L, P, B are either descendants of E or lie on post-E mediation paths and
are therefore excluded from the adjustment set.
"""

from language_reading_predictors.statistical_models.context import ModelSpec
from language_reading_predictors.statistical_models.pipeline import fit_mechanism

SPEC = ModelSpec(
    model_id="lrp57",
    kind="mechanism",
    title="Mechanism model: expressive vocabulary (E) -> word reading (W)",
    outcome_symbol="W",
    mechanism_symbol="E",
    adjustment=["G", "A", "R", "W_pre"],
    # Age GP off, subject random intercept on — see
    # notes/202604181700-lrp55-age-gp-drop.md and
    # notes/202604181800-mechanism-random-intercepts.md.
    extra={
        "adjust_baseline_symbol": "W",
        "use_age_gp": False,
        "phase_specific_mechanism": False,
        "use_subject_random_intercept": True,
    },
)


def fit(config: str = "dev"):
    return fit_mechanism(SPEC, config=config)
