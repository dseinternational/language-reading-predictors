# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""LRPITT08b - chance-floor link sensitivity for phoneme blending (B).

Robustness companion to LRPITT08. It keeps the same randomised ITT design,
Beta-Binomial observation family, adjustment set, and priors, but maps the
inverse-logit mean onto [1/3, 1] to represent chance performance on each
three-alternative item. Sign convention: positive ``tau`` means the intervention
raises the outcome.
"""

from language_reading_predictors.statistical_models.context import ModelSpec
from language_reading_predictors.statistical_models.itt import IttModelSettings
from language_reading_predictors.statistical_models.pipeline import fit_itt

SPEC = ModelSpec(
    model_id="lrp-rli-itt-108",
    kind="itt",
    title=(
        "ITT effect of group assignment on phoneme blending (B): "
        "three-choice guessing-floor link sensitivity"
    ),
    outcome_symbol="B",
    model_settings=IttModelSettings(
        score_mean_link="three_choice_guessing_floor",
    ),
)


def fit(config: str = "dev"):
    return fit_itt(SPEC, config=config)
