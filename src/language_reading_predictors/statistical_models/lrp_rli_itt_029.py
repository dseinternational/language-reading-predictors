# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""LRPITT029 - available-case modified ITT estimate for APT expressive information (EI).

The Action Picture Test (Renfrew 1997) is an expressive-language measure the DAG
already designates a direct outcome of randomised assignment (``IG -> {EI, EG}``), but
which had no confirmed ceiling until the manual maxima were supplied on 2026-08-18:
Grammar 37, Information 40. It was listed under "Deferred / follow-ups" in
``notes/202606251321-lrpitt-suite-design.md`` for exactly that reason.

Information awards half marks on some items, so the raw 0-40 score is not an integer
count. The modelled outcome is therefore the **doubled half-mark scale, out of 80**
(``aptinfo_x2``), which is exact — every observed fractional part is 0.5 — and
preserves the proportion and hence the logit mean structure. The cost is that 80
exchangeable trials are asserted where there are 40 partial-credit items, which
overstates per-child precision; the Beta-Binomial concentration absorbs part of it
and LRPITT129 is the registered out-of-40 comparator.

Uniform DAG-faithful available-case modified ITT model: under the locked DAG the
assigned-arm coefficient needs no adjustment set, so the own baseline and linear age
are PRECISION terms only. Sign convention: positive ``tau`` means the intervention
raises the outcome.
"""

from language_reading_predictors.statistical_models.context import ModelSpec
from language_reading_predictors.statistical_models.itt import IttModelSettings
from language_reading_predictors.statistical_models.pipelines.itt import fit_itt

SPEC = ModelSpec(
    model_id="lrp-rli-itt-029",
    kind="itt",
    title=(
        "Available-case modified ITT estimate of the assigned-arm contrast in "
        "APT expressive information (EI)"
    ),
    outcome_symbol="EI",
    model_settings=IttModelSettings(),
)


def fit(config: str = "dev"):
    return fit_itt(SPEC, config=config)
