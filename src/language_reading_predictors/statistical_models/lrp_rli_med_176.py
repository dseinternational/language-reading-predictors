# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
# Documentation edited by a LLM-based AI tool (Codex/GPT-6).

"""LRP176 - reverse longitudinal-ordering mediation: word reading (t2) -> letter sounds (t4).

The companion to ``lrp-rli-med-076`` swaps the mediator and outcome: word
reading at t2 predicts letter-sound knowledge at t4 under an assumed reverse
direction. It uses the ``WR_t -> LS_t1`` edge added to the lagged DAG in July.
Roch and Jarrold (2012), DOI 10.1016/j.jcomdis.2011.11.001, reported word reading
as a longitudinal predictor of later nonword reading; that finding does not
identify an effect on letter-sound knowledge. The design record is
``notes/202607172100-reverse-mediation-wr-ls-direction-spec.md``.

Design (see `factories.build_mediation_model` + `mediation.decompose`, with the
outcome lagged via `load_and_prepare_lagged_outcome`):

- Mediator M = W_t2 (Beta-Binomial, conditioned on W_t1); outcome Y = L_t4
  (conditioned on L_t1), so the mediator strictly precedes the outcome by two waves.
- **Adjustment {G, A, W_pre, W_t1, HS, SP}**: the exogenous common causes of the
  mediator WR and the outcome LS — hearing HS (`hs`) and speech SP (`deapp_c`) —
  entered at baseline (`W_pre` is the generic outcome-own-baseline marker, here LS's;
  `W_t1` is the mediator WR's baseline). Note the asymmetry vs LRP76: phonological
  memory RW (`erbto`) is **not** a DAG parent of LS, and baseline vocabulary E/R cause
  WR but not LS, so none of them is a WR<->LS mediator-outcome confounder here and all
  are correctly omitted. Because the outcome is lagged, the t3-sensitivity sub-fit is
  skipped (it would double-lag).

The letter-sound outcome is bounded at 32. Check its fitted ceiling frequency
and the t3-outcome companion ``lrp-rli-med-276``. Their comparison changes both
ceiling support and follow-up time, so it cannot isolate a ceiling effect or
rule out the reverse causal hypothesis.

Causal interpretation
---------------------
Assignment still randomises treatment schedules after crossover. Word reading is not
randomised as a mediator, and latent ability and treatment-induced dose can confound its
association with later letter-sound knowledge. This decomposition assumes a reverse
direction; it does not establish that direction. The intervention can affect several
skills, so assignment is not an isolated experimental change in either word reading or
letter sounds. Check ceiling support and compare with the t3-outcome companion without
treating that comparison as an isolated ceiling correction.
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
    model_id="lrp-rli-med-176",
    kind="mediation",
    title=(
        "Reverse longitudinal-ordering mediation: does word reading at t2 carry "
        "the intervention's effect on letter-sound knowledge at t4?"
    ),
    outcome_symbol="L",
    mechanism_symbol="W",  # the mediator (word reading)
    adjustment=[
        "G",
        "A",
        "W_pre",
        "W_t1",
        "hs",
        "hs_missing",
        "deapp_c",
        "deapp_c_missing",
    ],
    # Primary outcome taken at t4 (mediator W stays at t2): mediator precedes outcome.
    model_settings=MediationModelSettings(outcome_time=4),
)


def fit(config: str = "dev") -> StatisticalFitContext:
    return fit_mediation(SPEC, config=config)
