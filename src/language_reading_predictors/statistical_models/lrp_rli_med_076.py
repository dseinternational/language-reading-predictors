# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
# Documentation edited by a LLM-based AI tool (Codex/GPT-6).

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
  (conditioned on W_t1). Adjustment {G, A, E, R, W_pre, L_t1, HS, SP} — now genuinely
  identical to LRP59: both carry hearing HS (`hs`) and speech SP (`deapp_c`), the
  exogenous letter-sound → reading confounders under the revised DAG. (`hs`/`deapp_c`
  added 2026-07-17 — the pre-#259 set was carried forward here by mistake; see
  `notes/202607172000-adjustment-set-review-full-suite.md`.) Baseline E/R stay in as
  admissible pre-treatment precision terms (#264).
- Because the outcome is lagged, the t3-sensitivity sub-fit is skipped (it would
  double-lag).

Causal interpretation
---------------------
Assignment still randomises early-start versus delayed-start treatment schedules after
crossover. It does not randomise letter-sound knowledge or identify the mediator–outcome
relation. Measuring the mediator at t2 and the outcome at t4 establishes their
measurement order, while latent ability and treatment-induced dose confounding still
prevent a causal natural-effect interpretation. Read the decomposition under its stated
forward model and compare it with the contemporaneous and t3-outcome analyses as a
sensitivity to timing.
"""

from language_reading_predictors.statistical_models.context import (
    ModelSpec,
    StatisticalFitContext,
)
from language_reading_predictors.statistical_models.mediation_settings import (
    MediationModelSettings,
)
from language_reading_predictors.statistical_models.pipelines.mediation import fit_mediation

SPEC = ModelSpec(
    model_id="lrp-rli-med-076",
    kind="mediation",
    title=(
        "Longitudinal-ordering mediation: does letter-sound knowledge at t2 carry "
        "the intervention's effect on word reading at t4?"
    ),
    outcome_symbol="W",
    mechanism_symbol="L",  # the mediator
    adjustment=[
        "G",
        "A",
        "E",
        "R",
        "W_pre",
        "L_t1",
        "hs",
        "hs_missing",
        "deapp_c",
        "deapp_c_missing",
    ],
    # Primary outcome taken at t4 (mediator stays at t2): mediator precedes outcome.
    model_settings=MediationModelSettings(outcome_time=4),
)


def fit(config: str = "dev") -> StatisticalFitContext:
    return fit_mediation(SPEC, config=config)
