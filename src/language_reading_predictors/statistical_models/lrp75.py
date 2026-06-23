# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""LRP75 - ITT analysis of taught receptive vocabulary (block 1).

The receptive companion to LRP74. The trial reported gains on directly-taught
*expressive* vocabulary "though not on the equivalent receptive vocabulary
measure" (Burgoyne et al., 2012) - an expressive/receptive asymmetry the authors
attribute to the teaching emphasis on *using* the words and to differing task
demands. This model estimates the randomised-window effect on taught receptive
vocabulary (``b1retau``); the expectation is a weaker / near-null effect than
LRP74, and the pair is reported together rather than as a single "vocabulary"
result.
"""

from language_reading_predictors.statistical_models.context import ModelSpec
from language_reading_predictors.statistical_models.pipeline import fit_itt

SPEC = ModelSpec(
    model_id="lrp75",
    kind="itt",
    title="ITT effect of group assignment on taught receptive vocabulary (block 1)",
    outcome_symbol="TR",
    # As LRP74, but conditioned on the standardised *receptive*-vocabulary
    # baseline ``R``. See lrp74.py for the cross-baseline rationale.
    extra={
        "outcomes": ("TR", "R"),
        "cross_symbols": ("R",),
        "use_age_gp": False,
        "use_own_baseline_gp": False,
        "use_varying_tau": False,
    },
)


def fit(config: str = "dev"):
    return fit_itt(SPEC, config=config)
