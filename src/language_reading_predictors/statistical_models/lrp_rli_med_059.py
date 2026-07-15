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

**Adjustment set {G, A, E, R, HS, SP, L_t1, W_pre}** (revised DAG, #246) —
randomisation handles G->Y and G->M confounding; the binding unverifiable
assumption is no unmeasured mediator-outcome (L->W) confounding. Under the revised
DAG the L->W confounders are age (A), hearing (HS; hs/hs_missing) and speech
production (SP; deapp_c), plus the baselines L_t1, W_pre. **Baseline expressive /
receptive vocabulary (E, R) are retained — settled by the time-indexed
d-separation (#264;** ``notes/202607142340-lrp264-mediation-adjustment-dsep.md``):
on the wave-unrolled graph they are not descendants of the t1->t2 intervention
(they precede it), dropping them changes no backdoor-blocking status, and they
open no backdoor — admissible precision terms here. All
confounders enter at **baseline (t1)**, not post (t2). The report states the
assumptions prominently and names residual confounding as the limit.

**Not an identified natural effect.** Beyond the unmeasured L->W confounding, a
second, structural obstacle survives even randomisation: dose IS (sessions) is a
treatment-induced (exposure-induced) mediator-outcome confounder (IG -> IS; IS ->
L, W), so NDE/NIE are not identified and are *not* repaired by adjusting IS,
which is itself a descendant of the exposure (VanderWeele, Vansteelandt & Robins
2014, doi:10.1097/EDE.0000000000000034). Read the output as a model-based
g-formula decomposition under stated (cross-world) assumptions. An interventional
(rather than natural) estimand — fitted for this route as LRP78 — escapes *this*
obstacle (no cross-world quantity is invoked) but is not thereby identified: it
still assumes no unmeasured mediator-outcome confounding (Hejazi, Rudolph, van der
Laan & Diaz 2022, A5, doi:10.1093/biostatistics/kxac002), which latent GA violates
here. A weaker-assumption target, not a defensible one.

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
        # E/R retained — settled by the time-indexed d-separation (#264;
        # notes/202607142340-lrp264-mediation-adjustment-dsep.md); the revised-DAG
        # common causes HS/SP added by the missing-indicator method (#246).
        "G", "A", "E", "R", "L_t1", "W_pre",
        "hs", "hs_missing", "deapp_c", "deapp_c_missing",
    ],
    extra={},
)


def fit(config: str = "dev"):
    return fit_mediation(SPEC, config=config)
