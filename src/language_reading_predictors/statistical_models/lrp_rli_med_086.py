# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""LRP86 - does the intervention raise nonword reading *because* it raises letter-sounds?

The code route on the **purest decoding outcome** (#228 item 12). An ITT-phase
mediation decomposition of the intervention's effect on nonword reading (N) through
letter-sound knowledge (L), the nonword companion to LRP59 (W via L). Nonword
reading isolates alphabetic decoding with no lexical/whole-word support, so if the
intervention's decoding benefit runs through letter sounds it should show here.

**Off-floor outcome.** Nonword reading is ~57% floored, so — as everywhere in the
suite (LRPITT11, LRPDID12, gain-factor LRPGF11) — the outcome is the **off-floor
indicator** (post > 0), fitted as a Bernoulli. The decomposition (see
`factories.build_mediation_model(outcome_kind="bernoulli_offfloor")` +
`mediation.decompose`) therefore reports the natural direct/indirect effects on the
**off-floor risk-difference** scale: the change in P(reads >= 1 nonword) attributable
to the intervention directly (NDE) vs through the letter-sound gain (NIE). Off-floor
outcomes drop the own-baseline term (the Normal(1,.) autoregressive prior does not
transfer to a binary indicator, and a floored baseline logit is degenerate), matching
the off-floor ITT/DiD/gain-factor convention.

Design: phase 0 only (`phase_mode="itt"`, t1 -> t2, the randomised window); treatment
G (randomised), mediator M = L_t2 (conditioned on L_t1), outcome Y = 1[N_t2 > 0].
NDE/NIE by counterfactual simulation from the posterior, not a coefficient product.

**A test of the DAG's own structure.** The revised DAG (`dag/dag-language-reading.dagitty`)
has **no direct IG -> NW edge**: the intervention is drawn to reach nonword reading
*only* through the code skills. So a decomposition finding the effect runs through L
(NIE carries it, NDE ~ 0) also corroborates that exclusion restriction; a large direct
effect would instead flag a missing edge.

**Adjustment set (mediation criterion; signed off by @frankbuckley 2026-07-14).**
The set must block common causes of the mediator L and the outcome NW **including
indirect routes through unadjusted intermediates** — arrows *into the mediator*
matter, not just parents of the outcome (the mediation criterion, not the
single-outcome "gf-011 rule"). Under the revised DAG the arrows into LS are
{A, GA, HS, SP, IG, IS}. Adjusted: age (A); speech production (SP = `deapp_c`);
phonological memory (RW = `erbto`), a common cause via `LS <- RW -> NW`; the mediator
baseline L_t1; and — the identification fix over the first proposal — **hearing (HS =
`hs`/`hs_missing`) is INCLUDED**: HS is a parent of the mediator with a route to NW
that RW does not block, `LS <- HS -> PA -> NW`, so it confounds L -> N *through
blending* even though HS is not a nonword *parent* (true and irrelevant — the earlier
single-outcome rule missed this indirect route). **Baseline blending (B, at t1) is
added** as proximal reinforcement on the PA-mediated backdoors plus precision (the
med-066/075 precedent; B enters at its t1 logit as a covariate confounder, *not* the
outcome own-baseline, so it does not conflict with the off-floor no-`b_W`
convention). **No outcome own-baseline** (off-floor). The binding unverifiable
assumption is no unmeasured L -> NW confounding — latent general ability (GA) violates
it, as for LRP59 — so read this as a model-based g-formula decomposition under stated
(cross-world) assumptions, wide at n ~ 50, not an identified natural effect.

**Shared session dose (IS).** `IS -> PA -> NW` and `IS -> LS` make intervention dose a
treatment-induced common cause of the mediator and outcome; per the family precedent
it is not conditioned on (conditioning on a treatment-affected variable is
inadmissible for natural effects), and the report names shared session dose as a
plausible inflator of the NIE via L.

**Confirmed by the time-indexed re-derivation (#264;**
``notes/202607142340-lrp264-mediation-adjustment-dsep.md``): on the wave-unrolled
graph EV/RV do not enter the mediator's parent set, so the absence of E/R stands;
the shared-dose structure appears explicitly as the witness backdoor
``LS_2 <- IS_1 -> PA_2 -> NW_2``, blockable only at treatment descendants — the
interventional MED-186 companion (#323) plus #324's observed/fitted IS calibration
on the NIE sensitivity surface is the right response, not adjustment.
"""

from language_reading_predictors.statistical_models.context import ModelSpec
from language_reading_predictors.statistical_models.pipeline import fit_mediation

SPEC = ModelSpec(
    model_id="lrp-rli-med-086",
    kind="mediation",
    title=(
        "Mediation: does the intervention raise nonword reading (N, off-floor) "
        "via letter-sound knowledge (L)?"
    ),
    outcome_symbol="N",
    mechanism_symbol="L",  # the mediator
    adjustment=[
        # L->NW confounders under the mediation criterion (signed off 2026-07-14):
        # HS (hs/hs_missing) blocks LS<-HS->PA->NW; SP (deapp_c) and RW (erbto) the
        # other mediator-parent routes; baseline blending B (bare symbol -> taken at
        # t1 by the factory) reinforces the PA-mediated backdoors. No outcome
        # own-baseline (off-floor drops b_W).
        "G", "A", "L_t1", "B",
        "hs", "hs_missing", "deapp_c", "deapp_c_missing", "erbto", "erbto_missing",
    ],
    extra={
        # Off-floor (Bernoulli) outcome: NIE/NDE on the off-floor risk-difference scale.
        "outcome_kind": "bernoulli_offfloor",
        # N is not in ITT_OUTCOMES (floored), and B must be loaded so its t1 baseline
        # is available as a confounder; request them explicitly. This also restricts
        # the complete-case mask to N + L + B.
        "outcomes": ("N", "L", "B"),
    },
)


def fit(config: str = "dev"):
    return fit_mediation(SPEC, config=config)
