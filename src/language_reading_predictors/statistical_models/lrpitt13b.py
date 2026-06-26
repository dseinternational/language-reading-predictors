# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""LRPITT13b - SES-adjusted ITT for letter-sound knowledge (L).

Letter-sound companion to LRPITT13: the same SES-adjusted uniform spec, re-targeted
to the other headline outcome with a credible effect. LRPITT14b is its matched
unadjusted comparator. Sign convention: positive tau => intervention helps.
"""

from language_reading_predictors.statistical_models.context import ModelSpec
from language_reading_predictors.statistical_models.lrpitt13 import SES_ADJUSTERS
from language_reading_predictors.statistical_models.pipeline import fit_itt

SPEC = ModelSpec(
    model_id="lrpitt13b",
    kind="itt",
    title="SES-adjusted ITT effect of group assignment on letter-sound knowledge (L)",
    outcome_symbol="L",
    adjustment=list(SES_ADJUSTERS),
    extra={
        "outcomes": ("L",),
        "cross_symbols": (),
        "use_age_gp": False,
        "use_own_baseline_gp": False,
        "use_age_linear": True,
        "use_own_baseline": True,
        "adjust_for": SES_ADJUSTERS,
    },
)


def fit(config: str = "dev"):
    return fit_itt(SPEC, config=config)
