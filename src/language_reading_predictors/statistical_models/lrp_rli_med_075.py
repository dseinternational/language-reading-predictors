# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""LRP75 - sequential code route: letter sounds -> blending -> word reading.

The parallel two-mediator split LRP66 found phoneme blending (B) adds essentially
nothing to word reading (W) *over and above* letter-sound knowledge (L). The DAG's
code skeleton explains why: it is a **chain** (`LS`/`PA` -> decoding -> `WR`, with L
feeding B's downstream products), so B is largely *downstream* of L, not a competing
parallel route — a parallel model cannot see that. LRP75 fits the sequential
version: the same two count mediators, but the blending leg now regresses on
post-letter-sounds (the ``L -> B`` edge, coefficient ``aB_L``), and the g-formula
draws B **conditional on the simulated L**, so the joint indirect effect carries the
``L -> B -> W`` sub-path.

Design (see `factories.build_two_mediator_model(chain=True)` +
`mediation.decompose_two_mediator`):

- **Phase 0 only** (`phase_mode="itt"`, t1 -> t2), ``n ~ 53``, one row per child.
- Mediators L and B (Beta-Binomial legs); the B leg adds ``aB_L * z(L_t2)``.
  Adjustment {G, A, E, R, W_pre, L_t1, B_t1, HS, SP, RW} — as in LRP66, the exogenous
  code-route mediator-outcome confounders are hearing HS (``hs``), speech SP
  (``deapp_c``) and phonological memory RW (``erbto``); baseline vocabulary E, R stay
  in as admissible pre-treatment proxy/precision terms. (``hs`` / ``deapp_c`` /
  ``erbto`` added 2026-07-17 for parity with LRP62/66; see
  ``notes/202607172000-adjustment-set-review-full-suite.md``.)

What to read (and what not to):

- **Robust headline** - the **joint indirect effect through the ``{L, B}`` block**
  (``NIE_joint``): both mediators respond to treatment *including the ``L -> B``
  coupling*, so this is the total code-route mediation. Compare it to LRP66's joint
  ``{L, B}`` NIE - close agreement confirms adding the chain edge does not change how
  much flows through the block, only how it is apportioned.
- **The ``L -> B`` coupling ``aB_L``** (a fitted coefficient, an *association*):
  a clearly-positive value is the direct evidence that B is downstream of L - the
  point of the sequential reframing.
- **Per-path ``NIE_L`` / ``NIE_B``**: reported but **exploratory and
  convention-dependent**. In the chained draw the second mediator moves with its own
  exposure *and* with the L it is conditioned on, so the L-vs-B apportionment depends
  on the mediator ordering and cross-world assumptions and must NOT be read as a
  clean path split. The joint block and the coupling are the interpretable outputs.

Honesty: ID-2 throughout (`GA`-confounded + treatment-induced dose confounder), so
every quantity is an adjusted association / decomposition under stated assumptions,
never a causal route; ``n ~ 53`` -> wide intervals.
"""

from language_reading_predictors.statistical_models.context import ModelSpec
from language_reading_predictors.statistical_models.pipeline import fit_mediation_multi

SPEC = ModelSpec(
    model_id="lrp-rli-med-075",
    kind="mediation_multi",
    title=(
        "Sequential code route: does the word-reading (W) gain run "
        "letter sounds (L) -> phoneme blending (B) -> reading?"
    ),
    outcome_symbol="W",
    mechanism_symbol=None,  # two mediators; named in extra["mediators"]
    adjustment=[
        "G", "A", "E", "R", "W_pre", "L_t1", "B_t1",
        "hs", "hs_missing", "deapp_c", "deapp_c_missing", "erbto", "erbto_missing",
    ],
    extra={
        "mediators": ("L", "B"),
        "order": ("L", "B"),
        "chain": True,  # add the L -> B edge; draw B conditional on simulated L
    },
)


def fit(config: str = "dev"):
    return fit_mediation_multi(SPEC, config=config)
