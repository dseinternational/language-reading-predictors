# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""LRP59 - does the intervention raise word reading *because* it raises letter-sounds?

The causal capstone the original brief reserved for LRP59: an ITT-phase
mediation decomposition. All three legs are estimated credibly elsewhere —
intervention -> letter-sound (strong; LRP55 tau_L), letter-sound -> word reading
(LRP58), intervention -> word reading (LRP52, tau_W). LRP59 stitches them into the
active-ingredient claim: **how much of the intervention's effect on word reading
flows through its effect on letter-sound knowledge** (natural indirect effect,
NIE) versus other paths (natural direct effect, NDE)?

Design (see `factories.build_mediation_model` + `mediation.decompose`):

- **Phase 0 only** (`phase_mode="itt"`, t1 -> t2): the only window in which group
  is a randomised treatment (group 1 on intervention, group 2 wait-list control;
  after t2 both arms are treated). One row per child, n ~ 53.
- **Treatment** G (randomised), **mediator** M = L_t2 (letter-sound, conditioned
  on L_t1), **outcome** Y = W_t2 (word reading, conditioned on W_t1).
- Joint mediator + outcome Beta-Binomial model with a G×M interaction; NDE/NIE
  computed by counterfactual simulation from the posterior (the g-formula), NOT
  as a coefficient product (invalid on the logit scale).

**Adjustment set {G, A, HS, SP, L_t1, W_pre}** (revised DAG, #246) — randomisation
handles G->Y and G->M confounding; the binding unverifiable assumption is no
unmeasured mediator-outcome (L->W) confounding. Under the revised DAG the L->W
confounders are age (A), hearing (HS; hs/hs_missing) and speech production
(SP; deapp_c), all pre-treatment, plus the baselines L_t1, W_t1. The old E/R
adjusters are **dropped**: EV and RV are now descendants of the intervention
(IG->TE->EV, IG->TR->RV), so conditioning on them would adjust a **treatment-affected**
confounder and violate the cross-world assumption (the recanting-witness problem).
Confounders are taken at **baseline (t1)**, not post (t2). The report states the
assumptions prominently and names residual confounding as the limit.

Expect **wide** posteriors (n ~ 53). The headline is the proportion mediated with
its full uncertainty; a wide interval pointing at "mostly via letter-sounds" is a
legitimate result, and so is "inconclusive". The Total should reconcile in sign
and rough magnitude with LRP52's tau_W.
"""

from language_reading_predictors.statistical_models.context import ModelSpec
from language_reading_predictors.statistical_models.pipeline import fit_mediation

SPEC = ModelSpec(
    model_id="lrp-rli-med-059",
    kind="mediation",
    title=(
        "Mediation: does the intervention raise word reading (W) via "
        "letter-sound knowledge (L)?"
    ),
    outcome_symbol="W",
    mechanism_symbol="L",  # the mediator
    adjustment=[
        "G", "A", "L_t1", "W_pre",
        "hs", "hs_missing", "deapp_c", "deapp_c_missing",
    ],
    extra={},
)


def fit(config: str = "dev"):
    return fit_mediation(SPEC, config=config)
