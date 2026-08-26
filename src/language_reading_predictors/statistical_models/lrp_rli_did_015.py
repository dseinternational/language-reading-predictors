# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""LRPDID015 - arm-by-wave contrasts for APT expressive grammar (EG).

The Action Picture Test (Renfrew 1997) is an expressive-language measure the DAG
already designates a direct outcome of randomised assignment (``IG -> {EI, EG}``), but
which had no confirmed ceiling until the manual maxima were supplied on 2026-08-18:
Grammar 37, Information 40. It was listed under "Deferred / follow-ups" in
``notes/202606251321-lrpitt-suite-design.md`` for exactly that reason.

Grammar is scored in whole marks out of 37 and needs no rescaling, so this fit
carries none of the half-mark encoding caveats that apply to Information.

The t1--t3 outcome levels are modelled with a saturated arm-by-wave structure.
``tau_t2`` is the clean randomised t2 contrast and is the quantity to compare with
the matching ITT fit. ``arm_gap_t3`` is also identified by the original
randomisation, but of a different exposure -- assignment to the early-start versus
delayed-start treatment schedule (both arms treated by t3) -- and ``delta_crossover
= tau_t2 - arm_gap_t3`` is the change between those two randomised regime
contrasts, never an identified catch-up. The design does not condition on the
treatment-affected t2 score.
"""

from language_reading_predictors.statistical_models.context import ModelSpec
from language_reading_predictors.statistical_models.pipelines.did import fit_did

SPEC = ModelSpec(
    model_id="lrp-rli-did-015",
    kind="did",
    title="Waitlist-crossover arm-by-wave contrasts for APT expressive grammar (EG)",
    outcome_symbol="EG",
    family="did",
    design="waitlist-crossover arm-by-wave levels",
    estimand_type="mixed",
    causal_status="t2 randomised; t3 a randomised treatment-schedule contrast",
    extra={
        "outcomes": ("EG",),
        "waves": (0, 1, 2),
        "use_child_re": True,
        "use_age": True,
        "dose": False,
    },
)


def fit(config: str = "dev"):
    return fit_did(SPEC, config=config)
