# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""LRPITT09 - ITT effect on phonetic spelling (P), floor-rule branch.

Phonetic spelling is heavily floored at t2 (~78% of post-scores at zero), so a
graded Beta-Binomial ``tau`` is leveraged by a handful of dispersed tail values
rather than by the arm contrast. Under the pre-specified, arm-blind floor rule
(issue #119; >= 40% at zero at t2) this model:

- drops the degenerate own baseline and uses an age-only predictor; and
- reports the binary "off-floor at t2" effect (``Pr(post > 0)``) as the PRIMARY
  estimand, retaining the graded Beta-Binomial ``tau`` only as a flagged,
  detection-limited SECONDARY.

Sign convention: positive ``tau`` means the intervention raises the outcome
(here, raises the probability of coming off the floor).
"""

from language_reading_predictors.statistical_models.context import ModelSpec
from language_reading_predictors.statistical_models.pipeline import fit_itt

SPEC = ModelSpec(
    model_id="lrpitt09",
    kind="itt",
    title="ITT effect of group assignment on phonetic spelling (P) - floor-rule branch",
    outcome_symbol="P",
    extra={
        "floor_rule": True,
        "outcomes": ("P",),
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
