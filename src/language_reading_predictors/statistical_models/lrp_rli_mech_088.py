# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""LRP88 - Mechanism model: taught receptive vocabulary (TR) -> word reading (W).

#311 (descriptive-association workstream #314): the taught-vocabulary dose-response
into word reading - "an additional p taught spoken words is associated with o
additional words read" - as an adjusted association across all three phase
transitions.

Revised-DAG (2026-07-10) adjustment set for the TR_post -> W_post backdoor. The
parents of TR are {A, GA, HS, IG, IS, RW}, so:

- A (age): linear ``gamma_A`` term. G (group) is always controlled via beta_G;
  W_pre is the autoregressive baseline.
- HS: hearing status (hs / hs_missing) - covariate adjuster; blocks every
  TR <- HS -> {RV TE EV SP RW PA LS} -> W route at its root.
- RW: phonological memory (erbto / erbto_missing) - covariate adjuster; blocks
  TR <- RW -> {TE RV EV PA NW} -> W.
- IS: intervention sessions (attend) - covariate adjuster (see the note below);
  blocks the TR <- IS -> WR backdoor.
- TR has no measured skill parent under the revised DAG, so no concurrent measure
  confounder enters (unlike LRP57/LRP89, where TR itself is one).

**Intervention sessions (IS) are now adjusted (revised 2026-07-17, reversing #309).**
``IS -> TR`` and ``IS -> WR`` make session dose a genuine common cause, so omitting it
leaves the ``TR <- IS -> WR`` backdoor open. IS is also treatment-affected (``IG ->
IS``) and a collider (``IG -> IS <- GA``); an earlier handling (#309) left it unadjusted
for fear of opening the ``IG -> IS <- GA -> W`` collider path. A formal d-separation
check (``notes/202607171700-mech-intervention-sessions-adjustment-collider-review.md``)
shows that concern does not bite here: the collider path runs through the randomised arm
``IG`` as a *fork*, and the model always conditions on the arm (``beta_G``), which closes
the path at ``IG``. So conditioning on IS removes real confounding and opens no new
observable bias - with GA held hypothetically, {G, A, HS, IS, RW} d-separates TR from WR
in the backdoor graph, whereas without IS it does not. This aligns LRP88 with LRP58
(L -> W), which already adjusts IS. The #269/#309 collider rule still governs the
dose-response and ITT/DiD families, where the arm or dose is the *exposure* (not a
conditioned covariate).

TE and RV (and their descendants) are *descendants* of TR (``TR -> TE``,
``TR -> RV``) that also affect W: conditioning on them would block legitimate
indirect paths (e.g. ``TR -> TE -> W``) and bias the slope toward the direct-only
component, so they are deliberately NOT in the adjustment set. GA (general ability)
is latent and unadjustable - the child random intercept proxies its time-invariant
part, so the slope stays an adjusted association, never a causal effect. One
further interpretive caveat specific to this exposure: taught-vocabulary variation
is largely intervention-generated, so the slope describes covariation within a
treated system.

Linear mechanism per the vocabulary-exposure precedent (LRP56/57): the HSGP curve
showed sampler-geometry pathology on the vocabulary predictors, and TR is a
(taught) vocabulary measure; the estimand is the LINEAR TR -> W adjusted
association (a single slope, not a shape).
"""

from language_reading_predictors.statistical_models.context import ModelSpec
from language_reading_predictors.statistical_models.pipeline import fit_mechanism

SPEC = ModelSpec(
    model_id="lrp-rli-mech-088",
    kind="mechanism",
    title="Mechanism model: taught receptive vocabulary (TR) -> word reading (W)",
    outcome_symbol="W",
    mechanism_symbol="TR",
    adjustment=["G", "A", "W_pre"],
    # Age enters as a linear gamma_A term; the subject random intercept handles the
    # non-independent rows (up to 3 phases x 53 children) and proxies the
    # time-invariant part of latent ability.
    extra={
        # Load the exposure (TR) and outcome (W); TR has no measure confounder, so
        # the complete-case mask is W + TR only.
        "outcomes": ("W", "TR"),
        "adjust_baseline_symbol": "W",
        # attend (IS) added 2026-07-17 (reversing #309): IS -> TR and IS -> WR make it
        # a genuine confounder; the IG -> IS <- GA collider path it could open is
        # closed at the always-conditioned arm G. See the review note in notes/.
        "adjust_for": ("hs", "hs_missing", "attend", "erbto", "erbto_missing"),
        "use_age_gp": False,
        "phase_specific_mechanism": False,
        "use_subject_random_intercept": True,
        # LINEAR mechanism, not the HSGP curve - the vocabulary-predictor precedent
        # (see LRP56/57): the nonparametric curve does not converge on vocabulary
        # exposures at reporting tier, and DAG-required adjusters are not dropped to
        # buy convergence. The estimand is the LINEAR TR -> W adjusted association.
        "linear_mechanism": True,
    },
)


def fit(config: str = "dev"):
    return fit_mechanism(SPEC, config=config)
