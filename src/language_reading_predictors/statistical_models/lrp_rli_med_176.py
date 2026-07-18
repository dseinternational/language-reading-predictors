# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""LRP176 - reverse longitudinal-ordering mediation: word reading (t2) -> letter sounds (t4).

The mirror of [LRP76](lrp_rli_med_076) (`letter sounds t2 -> word reading t4`).
Every model in the suite otherwise *assumes* the direction letter sounds -> word
reading (`LS -> WR`): the contemporaneous base DAG hard-codes it, and until
2026-07-17 the lagged DAG (`dag/dag-language-reading-lagged.dagitty`) carried the
reverse edges `WR_t -> {TE, TR, PA, RW}_{t+1}` but pointedly **not** `WR_t -> LS_t1`.
This model exercises the newly-added `WR_t -> LS_t1` edge (the reading-feeds-back-
into-the-code hypothesis, motivated by the DS-specific `WR -> NW` finding of Roch &
Jarrold 2012, doi:10.1016/j.jcomdis.2011.11.001): a single-mediator g-formula decomposition of the
intervention's effect on **letter-sound knowledge at t4** through **word reading at
t2**. Read side by side with LRP76, the pair is the direction contrast `LS -> WR` vs
`WR -> LS`. Full design rationale:
`notes/202607172100-reverse-mediation-wr-ls-direction-spec.md`.

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

CEILING CAVEAT (decisive for interpretation): the outcome LS is bounded 0-32 and
rises across waves - at t4 ~12% of children are at the ceiling (32) and ~25% are >=30
(t1 mean 14.3 -> t4 mean 23.7). So a near-zero reverse NIE has two explanations that
must be separated - a genuinely absent `WR -> LS` effect, or a ceiling artefact. Read
the estimate alongside the LS-ceiling share, and if it is near zero fit the cleaner-
ceiling t3 read (`W_t2 -> L_t3`; ~6% at ceiling) before concluding.

STRONG CAVEAT (as LRP76): the t2 -> t4 increment is **not randomised** (both arms are
treated by t4), so this is a triangulation design read under stated assumptions, not a
clean causal estimate. ID-2 applies throughout (GA-confounded + treatment-induced dose
confounder `IG -> IS -> {WR, LS}`), so the NIE is an adjusted association under a
stated *reverse* direction, never proof that reading drives the code. The design also
randomises an early LS shock but no early WR shock (WR moves later), so this reverse
direction is intrinsically less well-powered than the forward LRP76 - a weak reverse
NIE is *not* strong evidence against `WR -> LS`.
"""

from language_reading_predictors.statistical_models.context import ModelSpec
from language_reading_predictors.statistical_models.pipeline import fit_mediation

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
        "G", "A", "W_pre", "W_t1",
        "hs", "hs_missing", "deapp_c", "deapp_c_missing",
    ],
    # Primary outcome taken at t4 (mediator W stays at t2): mediator precedes outcome.
    extra={"outcome_time": 4},
)


def fit(config: str = "dev"):
    return fit_mediation(SPEC, config=config)
