# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""LRP89 - Mechanism model: taught expressive vocabulary (TE) -> word reading (W).

#311 (descriptive-association workstream #314): the taught expressive-vocabulary
dose-response into word reading, as an adjusted association across all three phase
transitions - the expressive companion to LRP88 (TR -> W).

Revised-DAG (2026-07-10) adjustment set for the TE_post -> W_post backdoor. The
parents of TE are {A, GA, HS, IG, IS, RW, SP, TR}, so:

- A (age): linear ``gamma_A`` term. G (group) is always controlled via beta_G;
  W_pre is the autoregressive baseline.
- TR: taught receptive vocabulary (``TR -> TE`` and ``TR -> WR``) - the one
  measured skill confounder; enters at its *post* score, concurrent with the
  exposure, exactly as in LRP56/57.
- HS: hearing status (hs / hs_missing) - covariate adjuster.
- SP: speech production (deapp_c / deapp_c_missing) - covariate adjuster;
  blocks TE <- SP -> {EV LS PA NW} -> W.
- RW: phonological memory (erbto / erbto_missing) - covariate adjuster; blocks
  TE <- RW -> {TR RV EV PA NW} -> W.
- IS: intervention sessions (attend) - covariate adjuster (see the note below);
  blocks the TE <- IS -> WR backdoor.

**Intervention sessions (IS) are now adjusted (revised 2026-07-17, reversing #309).**
``IS -> TE`` and ``IS -> WR`` make session dose a genuine common cause, so omitting it
leaves the ``TE <- IS -> WR`` backdoor open. IS is also treatment-affected (``IG ->
IS``) and a collider (``IG -> IS <- GA``); an earlier handling (#309) left it unadjusted
for fear of opening the ``IG -> IS <- GA -> W`` collider path. A formal d-separation
check (``notes/202607171700-mech-intervention-sessions-adjustment-collider-review.md``)
shows that concern does not bite here: the collider path runs through the randomised arm
``IG`` as a *fork*, and the model always conditions on the arm (``beta_G``), which closes
the path at ``IG``. So conditioning on IS removes real confounding and opens no new
observable bias - with GA held hypothetically, {G, A, HS, IS, RW, SP, TR} d-separates TE
from WR in the backdoor graph, whereas without IS it does not. This aligns LRP89 with
LRP58 (L -> W), which already adjusts IS. The #269/#309 collider rule still governs the
dose-response and ITT/DiD families, where the arm or dose is the *exposure* (not a
conditioned covariate).

EV, EG and EI are *descendants* of TE (``TE -> EV``, ``TE -> EG``, ``TE -> EI``);
conditioning on them would block legitimate indirect paths (e.g.
``TE -> EV -> W``) and bias the slope toward the direct-only component, so they are
deliberately NOT in the adjustment set. GA (general ability) is latent and
unadjustable - the child random intercept proxies its time-invariant part, so the
slope stays an adjusted association, never a causal effect. As with LRP88,
taught-vocabulary variation is largely intervention-generated, so the slope
describes covariation within a treated system.

Linear mechanism per the vocabulary-exposure precedent (LRP56/57); the estimand is
the LINEAR TE -> W adjusted association (a single slope, not a shape).
"""

from language_reading_predictors.statistical_models.context import ModelSpec
from language_reading_predictors.statistical_models.pipeline import fit_mechanism

SPEC = ModelSpec(
    model_id="lrp-rli-mech-089",
    kind="mechanism",
    title="Mechanism model: taught expressive vocabulary (TE) -> word reading (W)",
    outcome_symbol="W",
    mechanism_symbol="TE",
    adjustment=["G", "A", "TR", "W_pre"],
    extra={
        # Load the exposure (TE), outcome (W) and the TR measure confounder - TR is
        # a concurrent parent of TE_post and W_post, so it enters at its *post*
        # score (not baseline); the complete-case mask is W + TE + TR.
        "outcomes": ("W", "TE", "TR"),
        "adjust_baseline_symbol": "W",
        # attend (IS) added 2026-07-17 (reversing #309): IS -> TE and IS -> WR make it
        # a genuine confounder; the IG -> IS <- GA collider path it could open is
        # closed at the always-conditioned arm G. See the review note in notes/.
        "adjust_for": (
            "hs", "hs_missing", "attend", "erbto", "erbto_missing",
            "deapp_c", "deapp_c_missing",
        ),
        "use_age_gp": False,
        "phase_specific_mechanism": False,
        "use_subject_random_intercept": True,
        # LINEAR mechanism, not the HSGP curve - the vocabulary-predictor precedent
        # (see LRP56/57). The estimand is the LINEAR TE -> W adjusted association.
        "linear_mechanism": True,
    },
)


def fit(config: str = "dev"):
    return fit_mechanism(SPEC, config=config)
