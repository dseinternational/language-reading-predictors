# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""LRPITT14b - matched complete-case comparator to LRPITT13b (letter-sound).

Letter-sound analogue of LRPITT14: the unadjusted ITT on the SES complete-case
subset, the matched comparator that isolates the SES adjustment in LRPITT13b
(sample held fixed). Sign convention: positive tau => intervention helps.
"""

from language_reading_predictors.statistical_models.context import ModelSpec
from language_reading_predictors.statistical_models.itt import IttModelSettings
from language_reading_predictors.statistical_models.lrp_rli_itt_013 import SES_ADJUSTERS
from language_reading_predictors.statistical_models.pipeline import fit_itt

SPEC = ModelSpec(
    model_id="lrp-rli-itt-114",
    kind="itt",
    title="Unadjusted ITT on the SES complete-case subset, letter-sound knowledge (matched comparator to LRPITT13b)",
    outcome_symbol="L",
    adjustment=[],
    model_settings=IttModelSettings(restrict_complete=SES_ADJUSTERS),
)


def fit(config: str = "dev"):
    return fit_itt(SPEC, config=config)
