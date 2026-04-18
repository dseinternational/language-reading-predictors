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
    # Re-enable via extra={"use_age_gp": True} for a sensitivity fit.
    extra={
        "adjust_baseline_symbol": "W",
        "use_age_gp": False,
        "phase_specific_mechanism": False,
    },
)


def fit(config: str = "dev"):
    return fit_mechanism(SPEC, config=config)
