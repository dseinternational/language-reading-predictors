# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""LRP66 - do letter-sounds and blending act as *independent* phonics paths to reading?

The route-composite LRP62 bundles letter-sound knowledge (L) and phoneme blending
(B) into a single "code-based route" and measures how much of the word-reading (W)
gain flows through the block as a whole. It cannot say how much runs through
blending *over and above* letter sounds. LRP66 is the phonics analogue of LRP64
(which split L vs expressive vocabulary E): it asks the two-mediator question
directly — of the intervention's effect on word reading, how much runs through
**letter-sound knowledge (L)**, how much through **phoneme blending (B)**, and how
much is direct/residual.

Design (see `factories.build_two_mediator_model` + `mediation.decompose_two_mediator`):

- **Phase 0 only** (`phase_mode="itt"`, t1 -> t2): the single randomised contrast,
  ``n ~ 53``. One row per child.
- **Two count mediators** L and B, each a Beta-Binomial leg conditioned on its own
  baseline; the **outcome** leg adds both standardised post-mediators and their
  treatment interactions. NDE/NIE are computed by counterfactual simulation (the
  g-formula), not from coefficients.
- **Adjustment {G, A, E, R, W_pre, L_t1, B_t1, HS, SP, RW}**: the revised-DAG
  exogenous common causes of the ``{L, B}`` code-route mediators and reading —
  hearing HS (``hs``), speech SP (``deapp_c``) and phonological memory RW
  (``erbto``) — are the measured mediator-outcome confounders, now genuinely the same
  set LRP62 adjusts for, taken at baseline (cross-world assumption). (``hs`` /
  ``deapp_c`` / ``erbto`` added 2026-07-17 for parity with LRP62; see
  ``notes/202607172000-adjustment-set-review-full-suite.md``.) The two vocabulary
  measures E and R stay in as admissible pre-treatment terms: both mediators are
  phonics skills, so baseline vocabulary is a common cause off the code path (a
  proxy / precision term), not a mediator here.

Framing (the headline is robust; the split is exploratory) — identical to LRP64:

- **Joint indirect effect** through the ``{L, B}`` block — the assumption-light
  headline ("how much flows through the phonics block at all").
- **Path-specific** ``NIE_L`` / ``NIE_B`` — a clearly-flagged **exploratory** split
  under a stated mediator ordering (L before B) and a conditional-independence
  assumption between the mediators. They sum to the joint effect but the attribution
  is ordering-dependent, and L and B are strongly correlated skills, so read the
  per-path split with particular caution.

Honesty front-and-centre: this is a **decomposition under assumptions, not proof of
a causal route.** Mediators and outcome are measured at the same wave (no temporal
precedence); the binding, unverifiable assumption is no unmeasured
mediator-outcome confounding; ``n ~ 53`` means **wide** intervals. All language
stays associational / under stated assumptions.
"""

from language_reading_predictors.statistical_models.context import ModelSpec
from language_reading_predictors.statistical_models.pipeline import fit_mediation_multi

SPEC = ModelSpec(
    model_id="lrp-rli-med-066",
    kind="mediation_multi",
    title=(
        "Two-mediator decomposition: how much of the word-reading (W) gain runs "
        "via letter-sound knowledge (L) vs phoneme blending (B)?"
    ),
    outcome_symbol="W",
    mechanism_symbol=None,  # two mediators; named in extra["mediators"]
    adjustment=[
        "G", "A", "E", "R", "W_pre", "L_t1", "B_t1",
        "hs", "hs_missing", "deapp_c", "deapp_c_missing", "erbto", "erbto_missing",
    ],
    extra={
        "mediators": ("L", "B"),
        # Path-specific split ordering for the exploratory NIE_L / NIE_B (L first).
        "order": ("L", "B"),
    },
)


def fit(config: str = "dev"):
    return fit_mediation_multi(SPEC, config=config)
