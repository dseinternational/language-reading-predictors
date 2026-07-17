# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""LRP80 - does word reading route through taught-*receptive* vocabulary (TR)?

The receptive companion to [LRP68](lrp_rli_med_068) (taught *expressive* TE). The DAG
gives both taught nodes a direct edge to word reading (`TR -> WR` predates decision 5;
`TE -> WR` was added with it), so a `TR -> WR` mediation is graph-warranted. Running TE
and TR through the identical machinery asks whether the intervention's word-reading gain
routes through taught *comprehension* as well as through taught *production* — a symmetric
check on the lexical route.

Design mirrors LRP68 exactly, swapping the mediator TE -> TR:

- **Phase 0 only** (`phase_mode="itt"`, t1 -> t2), ``n ~ 53``, one row per child.
- **Mediator** M = TR_t2 (Beta-Binomial on TR_t1); **outcome** Y = W_t2 with a G x M
  interaction. NDE/NIE by counterfactual simulation.
- **Adjustment {G, A, L, E, W_pre, TR_t1, HS, RW}**: the revised-DAG exogenous common
  causes of the mediator TR and word reading — hearing status HS (`hs`) and phonological
  memory RW (`erbto`) — are the measured mediator-outcome confounders, entered at
  baseline. (Speech SP is not a DAG parent of TR, so `deapp_c` is correctly omitted.)
  Added 2026-07-17 for parity with the flagship route models; see
  `notes/202607172000-adjustment-set-review-full-suite.md`. Baseline letter-sound
  knowledge L and standardised *expressive* vocab E stay in as admissible pre-treatment
  precision terms (#264), not as the exogenous confounders. Standardised *receptive*
  vocab R is a **descendant** of TR (transfer edge `TR -> RV`), so it is deliberately
  **not** adjusted — mirroring LRP68's exclusion of E for the expressive mediator, and
  keeping the two taught-vocab mediations structurally comparable (each drops its own
  same-modality standardised measure, keeps the cross-modality one plus L).

Honesty: an `IG -> TR -> WR` decomposition is ID-2 — the `TR -> WR` leg is `GA`-confounded
and dose is a treatment-induced mediator-outcome confounder — so the NDE/NIE are **adjusted
associations, not point-identified**. Contemporaneous measurement; ``n ~ 53`` -> wide
intervals.
"""

from language_reading_predictors.statistical_models.context import ModelSpec
from language_reading_predictors.statistical_models.pipeline import fit_mediation

SPEC = ModelSpec(
    model_id="lrp-rli-med-080",
    kind="mediation",
    title=(
        "Mediation: does the intervention raise word reading (W) via "
        "taught-receptive vocabulary (TR)?"
    ),
    outcome_symbol="W",
    mechanism_symbol="TR",  # the mediator
    adjustment=[
        "G", "A", "L", "E", "W_pre", "TR_t1",
        "hs", "hs_missing", "erbto", "erbto_missing",
    ],
    # TR is outside ITT_OUTCOMES, so name the symbols the model loads (outcome +
    # mediator + confounders); restricts the complete-case mask to them.
    extra={"outcomes": ("W", "TR", "L", "E")},
)


def fit(config: str = "dev"):
    return fit_mediation(SPEC, config=config)
