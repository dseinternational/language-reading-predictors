# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""LRPITT11 - ITT effect on nonword reading (N), floor-rule branch (post-only).

Nonword reading is the repo's "post-only" measure and is heavily floored at t2
(~72% of post-scores at zero). Under the pre-specified, arm-blind floor rule
(issue #119; >= 40% at zero at t2) this model:

- uses an age-only predictor (no own baseline); ``pre_required=()`` keeps the
  four children with a missing t1 nonword baseline, which the age-only model
  never uses (the baseline is degenerate, not the outcome); and
- reports the binary "off-floor at t2" effect (``Pr(post > 0)``) as the PRIMARY
  estimand, with the graded Beta-Binomial ``tau`` as a flagged, detection-limited
  SECONDARY.

Sign convention: positive ``tau`` means the intervention raises the outcome
(here, raises the probability of coming off the floor).
"""

from language_reading_predictors.statistical_models.context import ModelSpec
from language_reading_predictors.statistical_models.pipeline import fit_itt

SPEC = ModelSpec(
    model_id="lrpitt11",
    kind="itt",
    title="ITT effect of group assignment on nonword reading (N) - floor-rule branch",
    outcome_symbol="N",
    extra={
        "floor_rule": True,
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
