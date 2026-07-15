# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""LRP87 - does the intervention raise phoneme blending *because* it raises letter-sounds?

The downstream link in the phonological chain (#228 item 12): an ITT-phase mediation
decomposition of the intervention's effect on phoneme blending (B) through
letter-sound knowledge (L). The natural companion to LRP59 (W via L) and LRP86
(N via L) — together they trace the code route through its intermediate (blending)
and terminal (nonword decoding) outcomes, both currently visible only as
associations (mech-072/172).

Blending is graded (~4% floored, 10 items), so this is a standard graded
Beta-Binomial mediation, identical in structure to LRP59 with the outcome swapped:
phase 0 only (`phase_mode="itt"`, t1 -> t2, the randomised window); treatment G,
mediator M = L_t2 (conditioned on L_t1), outcome Y = B_t2 (conditioned on B_t1);
NDE/NIE on the items scale by counterfactual simulation from the posterior.

**Adjustment set (mediation criterion; signed off by @frankbuckley 2026-07-14).**
The L -> PA (blending) confounders under the revised DAG are the common parents of
LS and PA: age (A), hearing (HS = `hs`) and speech production (SP = `deapp_c`), via
the missing-indicator method, plus the baselines L_t1 and the outcome own-baseline
(the "W_pre" marker, resolved to B's pre-score in the factory).

**Phonological memory (RW = `erbto`) considered and excluded — precision-only.** RW
is a parent of PA but NOT of LS, so it is not a common cause of the mediator and
outcome and sits on no L -> PA backdoor; the one route it touches, `LS <- HS -> RW ->
PA`, is already blocked at HS (which is adjusted). Its exclusion is therefore
identification-clean (recorded here so the #264 re-derivation sees the decision).

**Intervention sessions (IS = `attend`) confirmed NOT adjusted.** IS is a common cause
of LS and PA but is treatment-affected (IG -> IS): `LS <- IS -> PA` is a recanting
witness, and no adjustment set rescues natural effects from that structure — while
adjusting IS would change the estimand and open the collider `IG -> IS <- GA -> B`
(GA latent), strictly worse. The constructive/quantitative answers are MED-187
(the interventional-effects companion, #323) and #324 (IS on the #289 sensitivity
surface). The report names shared session dose as a plausible inflator of the NIE
via L.

The binding unverifiable assumption is no unmeasured L -> B confounding (latent
general ability violates it), as for LRP59 — a model-based g-formula decomposition
under stated (cross-world) assumptions, wide at n ~ 53, not an identified natural
effect. **Confirmed by the time-indexed re-derivation (#264;**
``notes/202607142340-lrp264-mediation-adjustment-dsep.md``): on the wave-unrolled
graph EV/RV do not enter the mediator's parent set (the E/R absence stands), and
the recanting witness appears explicitly as ``LS_2 <- IS_1 -> PA_2``, blockable
only at treatment descendants — MED-187 plus #324 remain the right response.
"""

from language_reading_predictors.statistical_models.context import ModelSpec
from language_reading_predictors.statistical_models.pipeline import fit_mediation

SPEC = ModelSpec(
    model_id="lrp-rli-med-087",
    kind="mediation",
    title=(
        "Mediation: does the intervention raise phoneme blending (B) via "
        "letter-sound knowledge (L)?"
    ),
    outcome_symbol="B",
    mechanism_symbol="L",  # the mediator
    adjustment=[
        # L->PA confounders under the mediation criterion (signed off 2026-07-14): HS
        # (hs), SP (deapp_c) via missing indicators. RW considered and excluded
        # (not an LS parent -> precision-only; LS<-HS->RW->PA blocked at HS). IS/attend
        # confirmed NOT adjusted (treatment-affected recanting witness; MED-187/#324).
        # "W_pre" is the outcome-own-baseline marker (stripped by fit_mediation; the
        # factory uses B's pre-score).
        "G", "A", "W_pre", "L_t1",
        "hs", "hs_missing", "deapp_c", "deapp_c_missing",
    ],
    extra={
        # Restrict the complete-case mask to B + L (both are in ITT_OUTCOMES).
        "outcomes": ("B", "L"),
    },
)


def fit(config: str = "dev"):
    return fit_mediation(SPEC, config=config)
