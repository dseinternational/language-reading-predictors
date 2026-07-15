# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""LRPITT11 - modified-ITT nonword-reading floor-transition analysis.

Nonword reading is heavily floored: 36/50 observed scores (72%) are zero at
baseline and 34/53 observed scores (~64%) are zero at t2. This reanalysis chose
its floor rule after examining the outcome distribution and earlier results, so
the rule is post-hoc and data-adaptive even though its gate is applied arm-blind.
It:

- uses an age-only predictor (no baseline regression term);
- keeps missing t1 scores during loading so their eligibility is counted rather
  than silently dropped, but excludes the three children who have an observed t2
  score and unknown baseline-floor status from the transition analysis; and
- reports ``Pr(post > 0 | observed pre == 0)`` among 36 eligible children as the
  exploratory headline, with the graded Beta-Binomial ``tau`` as a flagged,
  detection-limited secondary over all 53 observed t2 scores.

Sign convention: positive ``tau`` means the intervention raises the outcome
(here, raises the probability of coming off the floor).
"""

from language_reading_predictors.statistical_models.context import ModelSpec
from language_reading_predictors.statistical_models.pipeline import fit_itt

SPEC = ModelSpec(
    model_id="lrp-rli-itt-011",
    kind="itt",
    title=(
        "Modified-ITT effect of group assignment on nonword reading (N) - "
        "post-hoc floor-rule branch"
    ),
    outcome_symbol="N",
    design="waitlist_randomised_t1_to_t2_observed_baseline_floor_subgroup",
    estimand_type="post_hoc_exploratory_available_case_subgroup_risk_difference",
    causal_status=(
        "randomised_assignment_within_observed_baseline_floor_subgroup_"
        "conditional_on_eligibility_and_outcome_missingness"
    ),
    extra={
        "floor_rule": True,
        "floor_rule_provenance": "post_hoc_data_adaptive_t2_zero_rate",
        "floor_estimand_role": "exploratory_headline",
        "outcomes": ("N",),
        "pre_required": (),
        "cross_symbols": (),
        "use_age_gp": False,
        "use_own_baseline_gp": False,
        "use_age_linear": True,
        "use_own_baseline": False,
    },
)


def fit(config: str = "dev"):
    return fit_itt(SPEC, config=config)
