# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""LRPITT26 - available-case modified ITT estimate for receptive grammar (T, TROG-2).

Standalone available-case modified ITT estimate for receptive grammar (#228
suite-gap Tier-1): T previously had only gain-factor / level-factor / aligned models.
The uniform DAG-faithful model (issue #119) requires no adjustment set for tau;
the own baseline and linear age are precision terms only and no cross-baselines
enter. T has carried an education-lead-agreed ROPE delta of 1.0 items since
2026-07-20 (ratified 2026-08-19; ``measures.ROPE_DELTA``), so the report includes
the P(benefit >= delta) meaningful-benefit table alongside the estimate — an
earlier version of this docstring predated the delta and wrongly said the table
was omitted (corrected in the 2026-08-20 ITT code review). The test ceiling is
the trial's short TROG-2 administration, 8 constructs x 4 items = 32, item-scored
(Burgoyne et al. 2012, Methods; see ``measures.py``). Sign convention: positive
tau means the intervention raises the outcome.
"""

from language_reading_predictors.statistical_models.context import ModelSpec
from language_reading_predictors.statistical_models.itt import IttModelSettings
from language_reading_predictors.statistical_models.pipelines.itt import fit_itt

SPEC = ModelSpec(
    model_id="lrp-rli-itt-026",
    kind="itt",
    title=(
        "Available-case modified ITT estimate of the assigned-arm contrast in "
        "receptive grammar (T)"
    ),
    outcome_symbol="T",
    model_settings=IttModelSettings(),
)


def fit(config: str = "dev"):
    return fit_itt(SPEC, config=config)
