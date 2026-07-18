# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""LRPITT09 - modified-ITT phonetic-spelling floor-rule analysis.

Phonetic spelling is heavily floored: 42/54 scores (~78%) are zero at baseline
and 34/53 observed scores (~64%) are zero at t2. This reanalysis chose its floor
rule after examining the outcome distribution and earlier results, so the rule
is post-hoc and data-adaptive even though its gate is applied arm-blind. It:

- drops the degenerate own baseline and uses an age-only predictor; and
- reports ``Pr(post > 0 | observed pre == 0)`` among 41 eligible children as the
  exploratory headline, retaining the graded Beta-Binomial ``tau`` only as a
  flagged, detection-limited secondary.

Sign convention: positive ``tau`` means the intervention raises the outcome
(here, raises the probability of coming off the floor).
"""

from language_reading_predictors.statistical_models.context import ModelSpec
from language_reading_predictors.statistical_models.itt import IttModelSettings
from language_reading_predictors.statistical_models.pipeline import fit_itt

SPEC = ModelSpec(
    model_id="lrp-rli-itt-009",
    kind="itt",
    title=(
        "Modified-ITT effect of group assignment on phonetic spelling (P) - "
        "post-hoc floor-rule branch"
    ),
    outcome_symbol="P",
    design="waitlist_randomised_t1_to_t2_observed_baseline_floor_subgroup",
    estimand_type="post_hoc_exploratory_available_case_subgroup_risk_difference",
    causal_status=(
        "randomised_assignment_within_observed_baseline_floor_subgroup_"
        "conditional_on_eligibility_and_outcome_missingness"
    ),
    model_settings=IttModelSettings.for_floor_outcome(),
)


def fit(config: str = "dev"):
    return fit_itt(SPEC, config=config)
