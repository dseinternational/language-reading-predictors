# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""LRP68 - does word reading route through taught-expressive vocabulary (TE)?

The 2026-07-10 DAG revision added the direct edges ``TE -> WR`` and ``EV -> WR``
(decision 5) precisely so that estimands conditioning on the phonics gateway no
longer force the non-phonemic *expressive* route to zero by construction. This
model exercises that newly-opened edge: a single-mediator g-formula decomposition
of the intervention's word-reading (W) effect through **taught-expressive
vocabulary (TE)** — the whole-word / paired-associate route, motivated within DS by
Mengoni, Nash & Hulme (2014), where familiarising a spoken form improved how
accurately individuals with DS then read the printed word.

`TE` (the bespoke taught target words) is the more proximal product of this
whole-word intervention than the standardised expressive measure `E` used in the
LRP64 L-vs-E split, so it is the better test of the lexical route.

Design (see `factories.build_mediation_model` + `mediation.decompose`):

- **Phase 0 only** (`phase_mode="itt"`, t1 -> t2), ``n ~ 53``, one row per child.
- **Mediator** M = TE_t2 (Beta-Binomial, conditioned on TE_t1); **outcome**
  Y = W_t2 (conditioned on W_t1), with a G x M interaction. NDE/NIE by
  counterfactual simulation.
- **Adjustment {G, A, L, R, W_pre, TE_t1, HS, SP, RW}**: the revised-DAG exogenous
  common causes of the mediator TE and word reading — hearing status HS (``hs``),
  speech production SP (``deapp_c``) and phonological memory RW (``erbto``) — are the
  measured mediator-outcome confounders, entered at baseline (cross-world
  assumption). They were added 2026-07-17 for parity with the flagship route models
  (MED-059/062/064), which have carried them since the #259 revision; see
  ``notes/202607172000-adjustment-set-review-full-suite.md``. Baseline letter-sound
  knowledge L and receptive vocabulary R stay in as admissible pre-treatment
  precision terms (#264), not as the exogenous confounders. Standardised expressive
  vocabulary E is a *descendant* of TE (transfer edge ``TE -> EV``), so it is
  deliberately **not** adjusted for (that would be over-control).

Honesty: an `IG -> TE -> WR` decomposition is ID-2 (§11) — **not point-identified**,
because `TE -> WR` is `GA`-confounded and dose is a treatment-induced
mediator-outcome confounder. Read as an adjusted association under stated
assumptions, never as the effect "running through" taught vocabulary. Mediator and
outcome are the same wave (no temporal precedence); ``n ~ 53`` means wide intervals.
"""

from language_reading_predictors.statistical_models.context import ModelSpec
from language_reading_predictors.statistical_models.pipeline import fit_mediation

SPEC = ModelSpec(
    model_id="lrp-rli-med-068",
    kind="mediation",
    title=(
        "Mediation: does the intervention raise word reading (W) via "
        "taught-expressive vocabulary (TE)?"
    ),
    outcome_symbol="W",
    mechanism_symbol="TE",  # the mediator
    adjustment=[
        "G", "A", "L", "R", "W_pre", "TE_t1",
        "hs", "hs_missing", "deapp_c", "deapp_c_missing", "erbto", "erbto_missing",
    ],
    # TE is outside ITT_OUTCOMES, so name the symbols the model loads (outcome +
    # mediator + confounders); restricts the complete-case mask to them.
    extra={"outcomes": ("W", "TE", "L", "R")},
)


def fit(config: str = "dev"):
    return fit_mediation(SPEC, config=config)
