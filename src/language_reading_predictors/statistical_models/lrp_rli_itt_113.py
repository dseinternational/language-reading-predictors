# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""LRPITT13b - SES-adjusted ITT for letter-sound knowledge (L).

Letter-sound companion to LRPITT13: the same SES-adjusted uniform spec, re-targeted
to the other headline outcome with a credible effect. LRPITT14b is its matched
unadjusted comparator. Sign convention: positive tau => intervention helps.
"""

from language_reading_predictors.statistical_models.context import ModelSpec
from language_reading_predictors.statistical_models.itt import IttModelSettings
from language_reading_predictors.statistical_models.lrp_rli_itt_013 import SES_ADJUSTERS
from language_reading_predictors.statistical_models.pipeline import fit_itt

SPEC = ModelSpec(
    model_id="lrp-rli-itt-113",
    kind="itt",
    title="SES-adjusted ITT effect of group assignment on letter-sound knowledge (L)",
    outcome_symbol="L",
    adjustment=list(SES_ADJUSTERS),
    model_settings=IttModelSettings(adjust_for=SES_ADJUSTERS),
)


def fit(config: str = "dev"):
    return fit_itt(SPEC, config=config)
