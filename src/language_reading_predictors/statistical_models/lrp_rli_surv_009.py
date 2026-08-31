# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""LRPSURV09 - time-to-off-floor survival for phonetic spelling (P), #230 §5.

The four-wave generalisation of the LRPITT09 floor rule: instead of the single
t1->t2 off-floor transition, a discrete-time survival model for *when* a child at
the phonetic-spelling floor at baseline first comes off it. The hazard carries a
per-interval baseline plus baseline (prognostic) letter-sound knowledge, word
reading and age, and a treatment hazard contrast fitted in the randomised first
interval only (``treatment_window="randomised"``, 2026-08-21 survival review,
finding 1): every person-period row after the wait-list crossover is
treatment-on, so the later intervals carry no arm contrast and fit their own
both-arms-treated baseline hazards.

What tau is (#631 finding 11): a model-based, available-case modified-ITT
randomised-window assignment contrast — the covariate-adjusted
immediate-versus-waitlist off-floor hazard contrast in the randomised first
interval among children at the floor at t1. The baseline-subgroup restriction,
the observed-wave-2 availability requirement, mean-imputed covariates and the
hazard-model form qualify it, and this family releases no causal headline (see
METHODS.md and the descriptive note).
"""

from language_reading_predictors.statistical_models.context import (
    ModelSpec,
    StatisticalFitContext,
)
from language_reading_predictors.statistical_models.pipelines.survival import fit_survival
from language_reading_predictors.statistical_models.survival import (
    SurvivalModelSettings,
)

SPEC = ModelSpec(
    model_id="lrp-rli-surv-009",
    kind="survival",
    title="Time-to-off-floor survival for phonetic spelling (P)",
    outcome_symbol="P",
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


def fit(config: str = "dev") -> StatisticalFitContext:
    return fit_survival(SPEC, config=config)
