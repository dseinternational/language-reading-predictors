# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""LRPITT14b - matched complete-case comparator to LRPITT13b (letter-sound).

Letter-sound analogue of LRPITT14: the unadjusted ITT on the SES complete-case
subset, the matched comparator that isolates the SES adjustment in LRPITT13b
(sample held fixed). Sign convention: positive tau => intervention helps.
"""

from language_reading_predictors.statistical_models.context import ModelSpec
from language_reading_predictors.statistical_models.lrpitt13 import SES_ADJUSTERS
from language_reading_predictors.statistical_models.pipeline import fit_itt

SPEC = ModelSpec(
    model_id="lrpitt14b",
    kind="itt",
    title="Unadjusted ITT on the SES complete-case subset, letter-sound knowledge (matched comparator to LRPITT13b)",
    outcome_symbol="L",
    adjustment=[],
    extra={
        "outcomes": ("L",),
        "cross_symbols": (),
        "use_age_gp": False,
        "use_own_baseline_gp": False,
        "use_age_linear": True,
        "use_own_baseline": True,
        "adjust_for": (),
        "restrict_complete": SES_ADJUSTERS,
        "variant_of": "lrpitt13b",
    },
)


def fit(config: str = "dev"):
    return fit_itt(SPEC, config=config)
