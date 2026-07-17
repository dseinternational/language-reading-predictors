# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""LRP74 - does word reading route through nonword decoding (N)?

The revised DAG labels nonword reading ``NW`` the **"code route (mediator)"** — the
decoding step between the phonics skills (letter sounds `LS`, blending `PA`) and word
reading `WR`. Yet every existing mediation model routes reading through the phonics
*knowledge* directly, skipping the decoding *behaviour* the graph designates as the
carrier. LRP74 fills that gap: a single-mediator g-formula decomposition of the
intervention's word-reading (W) effect through nonword decoding (N). Because
nonwords cannot be sight-read, a route through N isolates genuine decoding in a way
a route to W (which can be recognised as a whole word) does not.

Design (see `factories.build_mediation_model` + `mediation.decompose`):

- **Phase 0 only** (`phase_mode="itt"`, t1 -> t2), ``n ~ 53``, one row per child.
- **Mediator** M = N_t2 (Beta-Binomial, conditioned on N_t1); **outcome** Y = W_t2.
- **Adjustment {G, A, E, R, W_pre, N_t1, SP, RW}**: the revised-DAG exogenous common
  causes of the mediator N and word reading — speech production SP (``deapp_c``) and
  phonological memory RW (``erbto``) — are the measured mediator-outcome confounders,
  entered at baseline. (Hearing HS is not a DAG parent of NW, so ``hs`` is correctly
  omitted.) Added 2026-07-17 for parity with the flagship route models; see
  ``notes/202607172000-adjustment-set-review-full-suite.md``. The vocabulary / lexical
  measures E and R stay in as admissible baseline precision terms (a parallel
  whole-word route to reading), not as the exogenous confounders. Letter sounds L and
  blending B are *upstream* of N (they cause decoding), so they are on the
  ``IG -> ... -> N`` path and are **not** adjusted for (that would block part of the
  indirect effect).

STRONG CAVEATS specific to this model:

- **N is heavily floored** in this cohort (the DAG documents the `NW` floor and a
  corrected DS decoding deficit, g ~ -0.89). The Beta-Binomial mediator leg fits,
  but if the intervention barely moves N off the floor the indirect effect is
  ~0 with wide intervals — read that as *floor-limited*, not as evidence of no route.
- **Direction is contested in DS.** Roch & Jarrold (2012) found the *longitudinal*
  arrow within reading running `WR -> NW` (and level-gated), so the forward
  `NW -> WR` mediation imported from TD is only weakly corroborated in DS. This is
  an ID-2 adjusted association under a *stated, DS-uncertain* forward direction, not
  a causal route.
"""

from language_reading_predictors.statistical_models.context import ModelSpec
from language_reading_predictors.statistical_models.pipeline import fit_mediation

SPEC = ModelSpec(
    model_id="lrp-rli-med-074",
    kind="mediation",
    title=(
        "Mediation: does the intervention raise word reading (W) via "
        "nonword decoding (N)? (floor-limited, direction DS-uncertain)"
    ),
    outcome_symbol="W",
    mechanism_symbol="N",  # the mediator
    adjustment=[
        "G", "A", "E", "R", "W_pre", "N_t1",
        "deapp_c", "deapp_c_missing", "erbto", "erbto_missing",
    ],
    # N is outside ITT_OUTCOMES, so name the symbols the model loads (outcome +
    # mediator + confounders); restricts the complete-case mask to them.
    extra={"outcomes": ("W", "N", "E", "R")},
)


def fit(config: str = "dev"):
    return fit_mediation(SPEC, config=config)
