# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""LRP60 - SES-adjusted ITT robustness check for word reading (W)."""

from language_reading_predictors.data_variables import Variables as V
from language_reading_predictors.statistical_models.context import ModelSpec
from language_reading_predictors.statistical_models.pipeline import fit_itt


SES_ADJUSTERS = (
    V.MUMEDUPOST16,
    V.DADEDUPOST16,
    V.AGEBOOKS,
)


SPEC = ModelSpec(
    model_id="lrp60",
    kind="itt",
    title="SES-adjusted ITT effect of group assignment on word reading (W)",
    outcome_symbol="W",
    adjustment=list(SES_ADJUSTERS),
    extra={
        "use_age_gp": False,
        "use_own_baseline_gp": False,
        "use_varying_tau": False,
        "adjust_for": SES_ADJUSTERS,
    },
)


def fit(config: str = "dev"):
    return fit_itt(SPEC, config=config)
