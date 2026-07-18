# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""LRP276 - reverse WR->LS, t3 outcome: less-ceilinged sensitivity to LRP176.

The letter-sound outcome LS is bounded 0-32 and mildly ceilinged at t4 (~12% of
children at the maximum, ~25% >= 30), which can attenuate the reverse indirect
effect in [LRP176](lrp_rli_med_176) (mediator word reading at t2, outcome LS at t4).
LRP276 is the ceiling-sensitivity companion: identical design, but the outcome is
taken at **t3** (only ~6% of children at the LS ceiling), so the reverse
`WR -> LS` indirect is read with more room in the outcome to move. Compare its NIE
to LRP176's: close agreement means the small reverse signal is not a ceiling
artefact; a larger t3 NIE would mean t4 was ceiling-attenuated. Design rationale
and the full direction contrast: `notes/202607172100-reverse-mediation-wr-ls-
direction-spec.md` and the mediation findings note.

Design mirrors LRP176 exactly, only ``outcome_time`` changes (4 -> 3):

- Mediator M = W_t2 (Beta-Binomial on W_t1); outcome Y = L_t3 (on L_t1).
- **Adjustment {G, A, W_pre, W_t1, HS, SP}** — the exogenous common causes of the
  mediator WR and the outcome LS (hearing `hs`, speech `deapp_c`) at baseline; RW
  and vocabulary are not WR<->LS confounders and are omitted (see LRP176).

STRONG CAVEAT (as LRP176/LRP76): the t2 -> t3 increment is **not randomised** (the
wait-list crosses over after t2), and the whole decomposition is ID-2
(`GA`-confounded + treatment-induced dose confounder), so the NIE is an adjusted
association under a stated *reverse* direction, never proof that reading drives the
code. The design randomises no early WR shock, so the reverse direction is
intrinsically under-powered.
"""

from language_reading_predictors.statistical_models.context import ModelSpec
from language_reading_predictors.statistical_models.pipeline import fit_mediation

SPEC = ModelSpec(
    model_id="lrp-rli-med-276",
    kind="mediation",
    title=(
        "Reverse WR->LS mediation, t3 outcome (less-ceilinged sensitivity to "
        "LRP176): word reading at t2 -> letter-sound knowledge at t3?"
    ),
    outcome_symbol="L",
    mechanism_symbol="W",  # the mediator (word reading)
    adjustment=[
        "G", "A", "W_pre", "W_t1",
        "hs", "hs_missing", "deapp_c", "deapp_c_missing",
    ],
    # Outcome at t3 (mediator W stays at t2): the less-ceilinged ceiling-sensitivity.
    extra={"outcome_time": 3},
)


def fit(config: str = "dev"):
    return fit_mediation(SPEC, config=config)
