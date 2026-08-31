# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""LRPDID014 - arm-by-wave contrasts for APT expressive information (EI).

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

The t1--t3 outcome levels are modelled with a saturated arm-by-wave structure.
``tau_t2`` is the clean randomised t2 contrast and is the quantity to compare with
the matching ITT fit. ``arm_gap_t3`` is also identified by the original
randomisation, but of a different exposure -- assignment to the early-start versus
delayed-start treatment schedule (both arms treated by t3) -- and ``delta_crossover
= tau_t2 - arm_gap_t3`` is the change between those two randomised regime
contrasts, never an identified catch-up. The design does not condition on the
treatment-affected t2 score.
"""

from language_reading_predictors.statistical_models.context import (
    ModelSpec,
    StatisticalFitContext,
)
from language_reading_predictors.statistical_models.did import (
    DiDModelSettings,
)
from language_reading_predictors.statistical_models.pipelines.did import fit_did

SPEC = ModelSpec(
    model_id="lrp-rli-did-014",
    kind="did",
    title="Waitlist-crossover arm-by-wave contrasts for APT expressive information (EI)",
    outcome_symbol="EI",
    family="did",
    design="waitlist-crossover arm-by-wave levels",
    estimand_type="mixed",
    causal_status="t2 randomised; t3 a randomised treatment-schedule contrast",
    model_settings=DiDModelSettings(
        outcomes=("EI",),
        waves=(0, 1, 2),
        use_child_re=True,
        use_age=True,
        dose=False,
    ),
)


def fit(config: str = "dev") -> StatisticalFitContext:
    return fit_did(SPEC, config=config)
