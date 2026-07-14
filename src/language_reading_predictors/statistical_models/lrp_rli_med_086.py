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

**Adjustment set (proposed; for @frankbuckley's DAG review — the #246/#258/#259
pattern).** The L -> NW confounders under the revised DAG are the common parents of
LS and NW: age (A), speech production (SP = `deapp_c`) and phonological memory
(RW = `erbto`), via the missing-indicator method, plus the mediator baseline L_t1.
**Hearing (HS) is excluded** — HS is a parent of LS but NOT of NW in the revised DAG
(the gf-011 rule), so it does not confound the L -> NW path. **No outcome
own-baseline** (off-floor). The binding unverifiable assumption is no unmeasured
L -> NW confounding (latent general ability violates it), as for LRP59 — read as a
model-based g-formula decomposition under stated (cross-world) assumptions, wide at
n ~ 50, not an identified natural effect.

Blending (B) is deliberately NOT added here: the refit found blending is downstream
of letter sounds and adds no independent code route (LRP66/75), so this starts with L
alone; an L + B decomposition on N is a deferred follow-up.
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
        # L->NW confounders (revised DAG, proposed — Frank to confirm): SP (deapp_c),
        # RW (erbto) via missing indicators; HS EXCLUDED (not a nonword parent); no
        # outcome own-baseline (off-floor outcome drops b_W).
        "G", "A", "L_t1",
        "deapp_c", "deapp_c_missing", "erbto", "erbto_missing",
    ],
    extra={
        # Off-floor (Bernoulli) outcome: NIE/NDE on the off-floor risk-difference scale.
        "outcome_kind": "bernoulli_offfloor",
        # N is not in ITT_OUTCOMES (floored), so request it explicitly; this also
        # restricts the complete-case mask to N + L.
        "outcomes": ("N", "L"),
    },
)


def fit(config: str = "dev"):
    return fit_mediation(SPEC, config=config)
