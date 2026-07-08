# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""LRPITT13 - SES-adjusted ITT for word reading (W) (migrates LRP60).

SES-robustness companion to LRPITT10: the uniform LRPITT spec (own baseline +
linear age, no cross-baselines) plus linear adjustment for parental education and
age first exposed to books. Requesting these covariates triggers complete-case
dropping, so the fit runs on the SES-complete subset; LRPITT14 is the matched
unadjusted comparator on the same rows (so LRPITT13 vs LRPITT14 isolates the SES
adjustment, sample held fixed). Sign convention: positive tau => intervention
helps.
"""

from language_reading_predictors.data_variables import Variables as V
from language_reading_predictors.statistical_models.context import ModelSpec
from language_reading_predictors.statistical_models.pipeline import fit_itt

SES_ADJUSTERS = (
    V.MUMEDUPOST16,
    V.DADEDUPOST16,
    V.AGEBOOKS,
)

SPEC = ModelSpec(
    model_id="lrp-rli-itt-013",
    kind="itt",
    title="SES-adjusted ITT effect of group assignment on word reading (W)",
    outcome_symbol="W",
    adjustment=list(SES_ADJUSTERS),
    extra={
        "outcomes": ("W",),
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
