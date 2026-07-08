# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""LRP64 - do letter-sounds and vocabulary act as *independent* paths to reading?

The single-mediator LRP59 (letter-sound only) and the route-composite LRP62
(letter-sound + blending) each collapse the question to one pathway. LRP64 instead
asks the two-mediator question directly: of the intervention's effect on word
reading (W), how much runs through **letter-sound knowledge (L)**, how much through
**expressive vocabulary (E)**, and how much is direct/residual?

Design (see `factories.build_two_mediator_model` + `mediation.decompose_two_mediator`):

- **Phase 0 only** (`phase_mode="itt"`, t1 -> t2): the single randomised contrast,
  ``n ~ 53``. One row per child.
- **Two count mediators** L and E, each a Beta-Binomial leg conditioned on its own
  baseline; the **outcome** leg adds both standardised post-mediators and their
  treatment interactions. NDE/NIE are computed by counterfactual simulation (the
  g-formula), not from coefficients.
- **Adjustment {G, A, R, W_pre, L_t1, E_t1}**: receptive vocabulary R is the
  remaining mediator-outcome confounder (E is now a *mediator*, not a confounder),
  taken at baseline (cross-world assumption).

Framing (the headline is robust; the split is exploratory):

- **Joint indirect effect** through the ``{L, E}`` block — the assumption-light
  headline, in the LRP62 spirit ("how much flows through this block at all").
- **Path-specific** ``NIE_L`` / ``NIE_E`` — reported as a clearly-flagged
  **exploratory** split under a stated mediator ordering (L before E) and a
  conditional-independence assumption between the mediators. They sum to the joint
  effect but the attribution is ordering-dependent.

Honesty front-and-centre: this is a **decomposition under assumptions, not proof of
a causal route.** Mediators and outcome are measured at the same wave (no temporal
precedence); the binding, unverifiable assumption is no unmeasured
mediator-outcome confounding; children can sight-read words without decoding, so W
does not reveal *how* they read; ``n ~ 53`` means **wide** intervals. All language
stays associational / under stated assumptions.
"""

from language_reading_predictors.statistical_models.context import ModelSpec
from language_reading_predictors.statistical_models.pipeline import fit_mediation_multi

SPEC = ModelSpec(
    model_id="lrp-rli-med-064",
    kind="mediation_multi",
    title=(
        "Two-mediator decomposition: how much of the word-reading (W) gain runs "
        "via letter-sound knowledge (L) vs expressive vocabulary (E)?"
    ),
    outcome_symbol="W",
    mechanism_symbol=None,  # two mediators; named in extra["mediators"]
    adjustment=["G", "A", "R", "W_pre", "L_t1", "E_t1"],
    extra={
        "mediators": ("L", "E"),
        # Path-specific split ordering for the exploratory NIE_L / NIE_E (L first).
        "order": ("L", "E"),
    },
)


def fit(config: str = "dev"):
    return fit_mediation_multi(SPEC, config=config)
