# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""LRPSURV11 - time-to-off-floor survival for nonword reading (N), #230 §5.

The four-wave generalisation of the LRPITT11 floor rule: instead of the single
t1->t2 off-floor transition, a discrete-time survival model for *when* a child at
the nonword floor at baseline first comes off it. The hazard carries a per-interval
baseline plus baseline (prognostic) letter-sound knowledge, word reading and age,
and a treatment hazard contrast fitted in the randomised first interval only
(``treatment_window="randomised"``, 2026-08-21 survival review, finding 1): every
person-period row after the wait-list crossover is treatment-on, so the later
intervals carry no arm contrast and fit their own both-arms-treated baseline
hazards.

Prognostic, not causal: tau is anchored on the randomised first interval among
children at the floor at t1 and is reported as a prognostic association, not a
randomised effect of record (see METHODS.md and the descriptive note).
"""

from language_reading_predictors.statistical_models.context import ModelSpec
from language_reading_predictors.statistical_models.pipelines.survival import fit_survival
from language_reading_predictors.statistical_models.survival import (
    SurvivalModelSettings,
)

SPEC = ModelSpec(
    model_id="lrp-rli-surv-011",
    kind="survival",
    title="Time-to-off-floor survival for nonword reading (N)",
    outcome_symbol="N",
    family="survival",
    design="discrete-time off-floor hazard (person-period)",
    estimand_type="descriptive",
    causal_status="none",
    model_settings=SurvivalModelSettings(
        hazard_link="cloglog",
        use_treatment=True,
        treatment_window="randomised",
    ),
)


def fit(config: str = "dev"):
    return fit_survival(SPEC, config=config)
