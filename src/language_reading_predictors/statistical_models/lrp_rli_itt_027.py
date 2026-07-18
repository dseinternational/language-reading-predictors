# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""LRPITT27 - site-adjusted ITT for word reading (W) (#228 Tier-1).

Site-robustness companion to LRPITT10: the uniform LRPITT spec (own baseline +
linear age, no cross-baselines) plus a linear adjustment for study site (``area``:
1 = North, 2 = South; two sites, 30/24 children). ``area`` is fully observed, so -
unlike the SES-adjusted LRPITT13 - there is no complete-case dropping and the fit
stays on all 54 children, directly comparable to the unadjusted LRPITT10 tau (no
matched comparator needed). ``area`` enters as a precision / adjustment term only;
randomisation still identifies tau by the empty adjustment set. Sign convention:
positive tau => intervention helps.
"""

from language_reading_predictors.data_variables import Variables as V
from language_reading_predictors.statistical_models.context import ModelSpec
from language_reading_predictors.statistical_models.itt import IttModelSettings
from language_reading_predictors.statistical_models.pipeline import fit_itt

SITE_ADJUSTERS = (V.AREA,)

SPEC = ModelSpec(
    model_id="lrp-rli-itt-027",
    kind="itt",
    title="Site-adjusted ITT effect of group assignment on word reading (W)",
    outcome_symbol="W",
    adjustment=list(SITE_ADJUSTERS),
    model_settings=IttModelSettings(adjust_for=SITE_ADJUSTERS),
)


def fit(config: str = "dev"):
    return fit_itt(SPEC, config=config)
