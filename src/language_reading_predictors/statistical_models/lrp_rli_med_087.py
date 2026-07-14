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

**Adjustment set (proposed; for @frankbuckley's DAG review — the #246/#258/#259
pattern).** The L -> PA (blending) confounders under the revised DAG are the common
parents of LS and PA: age (A), hearing (HS = `hs`) and speech production
(SP = `deapp_c`), via the missing-indicator method, plus the baselines L_t1 and the
outcome own-baseline (the "W_pre" marker, resolved to B's pre-score in the factory).
**Phonological memory (RW = `erbto`) is excluded** — RW is a parent of PA but NOT of
LS, so it does not confound the L -> PA path. **Flag for Frank:** intervention
sessions IS (`attend`) are a common cause of LS and PA but are treatment-affected
(IG -> IS), so conditioning on them is a recanting-witness / cross-world risk (the
reason E/R were dropped, #246/#264) — proposed **not** adjusted; Frank to confirm.

The binding unverifiable assumption is no unmeasured L -> B confounding (latent
general ability violates it), as for LRP59 — a model-based g-formula decomposition
under stated (cross-world) assumptions, wide at n ~ 53, not an identified natural
effect.
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
        # L->PA confounders (revised DAG, proposed — Frank to confirm): HS (hs), SP
        # (deapp_c) via missing indicators; RW EXCLUDED (not an L parent); IS/attend
        # flagged for Frank (treatment-affected). "W_pre" is the outcome-own-baseline
        # marker (stripped by fit_mediation; the factory uses B's pre-score).
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
