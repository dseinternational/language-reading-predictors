# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""LRPITT13 - SES-adjusted available-case modified ITT estimate for word reading.

The LRPITT10 specification (own baseline, linear age and no cross-baselines)
adds parental education and age first exposed to books as precision covariates.
Their availability restricts the fit to the SES-complete subset. LRPITT14 uses
the same rows without SES adjustment, so their comparison holds the sample
fixed. Positive ``tau`` favours the immediate-intervention arm.

These covariates precede randomisation and are balanced across arms in
expectation before selection. Complete-case selection and outcome availability
still require the available-case assumptions in ``METHODS.md``. Adjustment
need not improve precision. The prior-table role remains ``precision``.
"""

from language_reading_predictors.data_variables import Variables as V
from language_reading_predictors.statistical_models.context import (
    ModelSpec,
    StatisticalFitContext,
)
from language_reading_predictors.statistical_models.itt import IttModelSettings
from language_reading_predictors.statistical_models.pipelines.itt import fit_itt

SES_ADJUSTERS = (
    V.MUMEDUPOST16,
    V.DADEDUPOST16,
    V.AGEBOOKS,
)

SPEC = ModelSpec(
    model_id="lrp-rli-itt-013",
    kind="itt",
    title=("SES-adjusted available-case modified ITT estimate of the assigned-arm contrast in word reading (W)"),
    outcome_symbol="W",
    adjustment=list(SES_ADJUSTERS),
    model_settings=IttModelSettings(adjust_for=SES_ADJUSTERS),
)


def fit(config: str = "dev") -> StatisticalFitContext:
    return fit_itt(SPEC, config=config)
