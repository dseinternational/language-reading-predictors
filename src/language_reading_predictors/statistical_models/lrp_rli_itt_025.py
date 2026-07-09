# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""LRPITT25 - ITT effect on basic concept knowledge (F, CELF).

Standalone ITT for basic concepts (#228 suite-gap Tier-1): F previously had only
gain-factor / level-factor / aligned models and no standalone randomised ITT.
Uniform DAG-faithful ITT (issue #119): randomisation identifies tau by the empty
adjustment set, so the own baseline and linear age are precision terms only and no
cross-baselines enter. Basic concepts sits outside the eight standardised ITT
outcomes and has no education-lead-agreed ROPE delta, so the report gives the
effect tau (size / direction / probability) but omits the P(benefit >= delta)
meaningful-benefit table the other ITT outcomes carry. Sign convention: positive
tau means the intervention raises the outcome.
"""

from language_reading_predictors.statistical_models.context import ModelSpec
from language_reading_predictors.statistical_models.pipeline import fit_itt

SPEC = ModelSpec(
    model_id="lrp-rli-itt-025",
    kind="itt",
    title="ITT effect of group assignment on basic concept knowledge (F)",
    outcome_symbol="F",
    extra={
        "outcomes": ("F",),
        "cross_symbols": (),
        "use_age_gp": False,
        "use_own_baseline_gp": False,
        "use_age_linear": True,
        "use_own_baseline": True,
    },
)


def fit(config: str = "dev"):
    return fit_itt(SPEC, config=config)
