# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""LRPITT13b - SES-adjusted available-case modified ITT estimate for letter sounds.

The letter-sound companion to LRPITT13 uses the same SES precision covariates.
LRPITT14b is the unadjusted comparator on the same SES-complete rows. Positive
``tau`` favours the immediate-intervention arm.

The covariates precede randomisation, but selection into the SES-complete,
observed-outcome subset still requires the available-case assumptions in
``METHODS.md``. Adjustment need not improve precision. Their prior-table role
remains ``precision``.
"""

from language_reading_predictors.statistical_models.context import (
    ModelSpec,
    StatisticalFitContext,
)
from language_reading_predictors.statistical_models.itt import IttModelSettings
from language_reading_predictors.statistical_models.lrp_rli_itt_013 import SES_ADJUSTERS
from language_reading_predictors.statistical_models.pipelines.itt import fit_itt

SPEC = ModelSpec(
    model_id="lrp-rli-itt-113",
    kind="itt",
    title=(
        "SES-adjusted available-case modified ITT estimate of the assigned-arm contrast in letter-sound knowledge (L)"
    ),
    outcome_symbol="L",
    adjustment=list(SES_ADJUSTERS),
    model_settings=IttModelSettings(adjust_for=SES_ADJUSTERS),
)


def fit(config: str = "dev") -> StatisticalFitContext:
    return fit_itt(SPEC, config=config)
