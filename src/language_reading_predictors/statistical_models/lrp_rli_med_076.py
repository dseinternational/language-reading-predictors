# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""LRP76 - longitudinal-ordering mediation: letter sounds (t2) -> word reading (t4).

Every contemporaneous mediation model (LRP59/62/64/66/68/74) measures the mediator
and the outcome at the *same* wave (t2), so it cannot establish temporal precedence
- the binding weakness the DAG's own §12 flags and scopes to a wave-unrolled
workstream (issue #250). Those models already carry a t3 *sensitivity* (mediator at
t2, outcome at t3). LRP76 promotes that ordering to the **primary** estimand and
pushes the separation to the maximum the panel allows: the mediator (letter-sound
knowledge L) stays at t2, the outcome (word reading W) is taken at **t4**, so the
mediator strictly precedes the outcome by two waves.

Design (see `factories.build_mediation_model` + `mediation.decompose`, with the
outcome lagged via `load_and_prepare_lagged_outcome`):

- Mediator M = L_t2 (Beta-Binomial, conditioned on L_t1); outcome Y = W_t4
  (conditioned on W_t1). Adjustment {G, A, E, R, W_pre, L_t1} — identical to LRP59.
- Because the outcome is lagged, the t3-sensitivity sub-fit is skipped (it would
  double-lag).

STRONG CAVEAT: the **t2 -> t4 increment is NOT randomised** — by t4 both arms have
been treated (the wait-list has crossed over), so this is a triangulation design
for the temporal-precedence question, read under stated assumptions, **not** a
cleaner causal estimate than the contemporaneous LRP59. It is a within-current-design
longitudinal check, and is explicitly *not* the full wave-unrolled cross-lagged
model (issue #250), which additionally handles time-varying confounding. ID-2
applies throughout (GA-confounded + treatment-induced dose confounder); the NIE is
an adjusted association under a stated forward direction.
"""

from language_reading_predictors.statistical_models.context import ModelSpec
from language_reading_predictors.statistical_models.pipeline import fit_mediation

SPEC = ModelSpec(
    model_id="lrp-rli-med-076",
    kind="mediation",
    title=(
        "Longitudinal-ordering mediation: does letter-sound knowledge at t2 carry "
        "the intervention's effect on word reading at t4?"
    ),
    outcome_symbol="W",
    mechanism_symbol="L",  # the mediator
    adjustment=["G", "A", "E", "R", "W_pre", "L_t1"],
    # Primary outcome taken at t4 (mediator stays at t2): mediator precedes outcome.
    extra={"outcome_time": 4},
)


def fit(config: str = "dev"):
    return fit_mediation(SPEC, config=config)
