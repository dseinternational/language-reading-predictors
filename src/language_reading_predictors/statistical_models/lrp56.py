# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""LRP56 - Mechanism model: receptive vocabulary (R) -> word reading (W).

DAG-derived adjustment set (blocks all backdoor paths from R_post to W_post):

- G: treatment arm (confounds R and W through the intervention).
- A: age (confounds R and W developmentally).
- W_pre: baseline word reading (temporal precedence; not a collider).

E is a *descendant* of R (R -> E) that also affects W. Conditioning on E
would block the legitimate indirect path R -> E -> W and bias f^R toward
zero. E is therefore deliberately NOT in the adjustment set.
"""

from language_reading_predictors.statistical_models.context import ModelSpec
from language_reading_predictors.statistical_models.pipeline import fit_mechanism

SPEC = ModelSpec(
    model_id="lrp56",
    kind="mechanism",
    title="Mechanism model: receptive vocabulary (R) -> word reading (W)",
    outcome_symbol="W",
    mechanism_symbol="R",
    adjustment=["G", "A", "W_pre"],
    # Age GP is off by default — see notes/202604181700-lrp55-age-gp-drop.md.
    # Subject random intercept is on by default: 157 rows per fit = up to 3
    # phases × 53 children, so rows are not independent. See
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
