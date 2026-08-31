# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""LRPITT129 - available-case modified ITT estimate for APT expressive information, whole marks (EI40).

The Action Picture Test (Renfrew 1997) is an expressive-language measure the DAG
already designates a direct outcome of randomised assignment (``IG -> {EI, EG}``), but
which had no confirmed ceiling until the manual maxima were supplied on 2026-08-18:
Grammar 37, Information 40. It was listed under "Deferred / follow-ups" in
``notes/202606251321-lrpitt-suite-design.md`` for exactly that reason.

This is the **denominator-sensitivity comparator** for LRPITT029, not a separate
construct. It rounds Information to whole marks and models it out of 40, which is
honest about the trial count but perturbs 44% of observations by up to half a mark.
Read the pair together: if ``tau`` differs materially between them, that is a finding
about the half-mark encoding rather than about the intervention.

Uniform DAG-faithful available-case modified ITT model: under the locked DAG the
assigned-arm coefficient needs no adjustment set, so the own baseline and linear age
are PRECISION terms only. Sign convention: positive ``tau`` means the intervention
raises the outcome.
"""

from language_reading_predictors.statistical_models.context import (
    ModelSpec,
    StatisticalFitContext,
)
from language_reading_predictors.statistical_models.itt import IttModelSettings
from language_reading_predictors.statistical_models.pipelines.itt import fit_itt

SPEC = ModelSpec(
    model_id="lrp-rli-itt-129",
    kind="itt",
    title=(
        "Available-case modified ITT estimate of the assigned-arm contrast in "
        "APT expressive information, whole marks (EI40)"
    ),
    outcome_symbol="EI40",
    model_settings=IttModelSettings(),
)


def fit(config: str = "dev") -> StatisticalFitContext:
    return fit_itt(SPEC, config=config)
