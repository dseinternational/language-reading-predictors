# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""LRPSURV09 - time-to-off-floor survival for phonetic spelling (P), #230 §5.

The four-wave generalisation of the LRPITT09 floor rule: instead of the single
t1->t2 off-floor transition, a discrete-time survival model for *when* a child at
the phonetic-spelling floor at baseline first comes off it. The hazard carries a
per-interval baseline plus baseline (prognostic) letter-sound knowledge and word
reading, and an intervention-aligned treatment hazard shift.

Prognostic, not causal: by t4 both arms have been treated, so a positive treatment
hazard shift is an association anchored on the immediate arm's randomised first
interval, not a randomised effect of record (see METHODS.md and the descriptive note).
"""

from language_reading_predictors.statistical_models.context import ModelSpec
from language_reading_predictors.statistical_models.pipeline import fit_survival

SPEC = ModelSpec(
    model_id="lrp-rli-surv-009",
    kind="survival",
    title="Time-to-off-floor survival for phonetic spelling (P)",
    outcome_symbol="P",
    family="survival",
    design="discrete-time off-floor hazard (person-period)",
    estimand_type="descriptive",
    causal_status="none",
    extra={
        "hazard_link": "cloglog",
        "use_treatment": True,
    },
)


def fit(config: str = "dev"):
    return fit_survival(SPEC, config=config)
