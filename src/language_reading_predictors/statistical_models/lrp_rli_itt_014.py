# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""LRPITT14 - matched complete-case comparator to LRPITT13 (word reading).

LRPITT13 adjusts the word-reading ITT for SES, but requesting those covariates
triggers complete-case dropping, so it runs on only the SES-complete subset. A
naive LRPITT10-vs-LRPITT13 comparison therefore confounds two things: the SES
adjustment and the sample change (the dropped children).

LRPITT14 is the **unadjusted** word-reading ITT on the **exact same complete-case
subset** (`restrict_complete=SES_ADJUSTERS`, `adjust_for=()`), so:

- LRPITT14 vs LRPITT10 isolates the dropped-children / selection effect.
- LRPITT13 vs LRPITT14 isolates the SES adjustment, sample held fixed.

All other settings match LRPITT13. Sign convention: positive tau => intervention
helps.
"""

from language_reading_predictors.statistical_models.context import ModelSpec
from language_reading_predictors.statistical_models.itt import IttModelSettings
from language_reading_predictors.statistical_models.lrp_rli_itt_013 import SES_ADJUSTERS
from language_reading_predictors.statistical_models.pipeline import fit_itt

SPEC = ModelSpec(
    model_id="lrp-rli-itt-014",
    kind="itt",
    title="Unadjusted ITT on the SES complete-case subset, word reading (matched comparator to LRPITT13)",
    outcome_symbol="W",
    adjustment=[],
    model_settings=IttModelSettings(restrict_complete=SES_ADJUSTERS),
)


def fit(config: str = "dev"):
    return fit_itt(SPEC, config=config)
