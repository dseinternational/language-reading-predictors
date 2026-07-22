# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""LRP102 - Mechanism model: taught receptive vocabulary (TR) -> letter sounds (L),
concurrent readout.

#405 (companion to #404): does taught vocabulary predict letter-sound knowledge?
A descriptive pass found TR is a strong *concurrent* correlate of letter sounds
(pooled r ~ 0.62, rising across waves) but does **not** predict letter-sound
*growth* (baseline TR ~ -0.1 with the subsequent L gain). This model turns the
concurrent half of that read into an adjusted association: the same-wave TR_post
-> L_post slope, conditioning on the child's own letter-sound baseline L_pre.

**Exploratory edge-check, never causal.** The revised DAG
(``dag/dag-language-reading.dagitty``) posits **no vocabulary -> letter-sound
edge**: TR -> {TE, RV, EV, LF, RG, WR} and none of those reach LS, so there is no
directed TR -> LS path. Letter sounds' parents are cognitive ability, hearing,
speech and the intervention - not vocabulary - which is why the letter-sound gain
model ``gf-004`` deliberately carries no skill adjusters. Any association we
estimate here is therefore entirely confounding/selection, dominated by the latent
general-ability (GA) common cause we cannot fully close. Report it as a
latent-ability-confounded adjusted association, an edge-check / robustness read on
the committed DAG, not a hypothesised mechanism.

Revised-DAG adjustment set for the TR -> LS backdoor (re-derived per #405; it
differs from the TR -> WR set of ``mech-088`` because the outcome is now LS, not
WR):

- A (age): linear ``gamma_A`` term; G (group) is always controlled via ``beta_G``;
  L_pre is the autoregressive baseline (the child's own letter-sound level).
- HS (hearing, ``hs`` / ``hs_missing``): common cause ``TR <- HS -> LS`` - adjust.
- IS (intervention sessions, ``attend``): common cause ``TR <- IS -> LS`` - the
  intervention teaches both vocabulary and letter sounds, so session dose is a
  genuine confounder. Adjusted, with the same collider-safety argument as
  ``mech-088``: the ``IG -> IS <- GA`` path is closed at the always-conditioned arm
  ``G``.
- SP (speech, ``deapp_c`` / ``deapp_c_missing``): **not** a parent of TR, so not a
  confounder of this slope (every ``TR <- {A,GA,HS} -> SP -> LS`` backdoor is
  already blocked at A / HS / the GA proxy). It is a strong, stable, pre-treatment
  parent of LS, so it is kept as a **precision adjuster** - and it keeps the TR and
  TE (``mech-104``) adjustment sets identical, where SP *is* a required confounder.
- **RW (phonological memory, ``erbto``) is deliberately NOT adjusted** (unlike
  ``mech-088``): RW -> {TE, EV, TR, RV, PA, NW, PS} never reaches LS, so it is not a
  common cause of TR and LS. Dropping it is the key re-derivation for the LS
  outcome.
- GA (general ability) is latent and unadjustable; the child random intercept
  proxies its time-invariant part, so the slope stays an adjusted association.

Linear mechanism per the vocabulary-exposure precedent (LRP56/57, ``mech-088``):
the HSGP curve shows sampler-geometry pathology on vocabulary predictors, and TR is
a taught-vocabulary measure; the estimand is the LINEAR TR -> L adjusted
association (a single slope, not a shape).

The lagged / predictive sibling that tests growth (TR at period-start -> L at
period-end given L_pre) is ``mech-103``.
"""

from language_reading_predictors.statistical_models.context import ModelSpec
from language_reading_predictors.statistical_models.pipeline import fit_mechanism

SPEC = ModelSpec(
    model_id="lrp-rli-mech-102",
    kind="mechanism",
    title=(
        "Mechanism model: taught receptive vocabulary (TR) -> letter sounds (L), "
        "concurrent"
    ),
    outcome_symbol="L",
    mechanism_symbol="TR",
    adjustment=["G", "A", "L_pre"],
    # Age enters as a linear gamma_A term; the subject random intercept handles the
    # non-independent rows (up to 3 phases x child) and proxies the time-invariant
    # part of latent ability GA.
    extra={
        # Load the exposure (TR) and outcome (L); TR has no measure confounder for
        # the LS backdoor, so the complete-case mask is L + TR + the raw adjusters.
        "outcomes": ("L", "TR"),
        "adjust_baseline_symbol": "L",
        # HS (hearing), IS (attend), SP (speech) - see the module docstring for the
        # per-symbol backdoor derivation. erbto (RW) is intentionally omitted: it
        # does not reach LS.
        "adjust_for": ("hs", "hs_missing", "attend", "deapp_c", "deapp_c_missing"),
        "use_age_gp": False,
        "phase_specific_mechanism": False,
        "use_subject_random_intercept": True,
        # LINEAR mechanism, not the HSGP curve (vocabulary-predictor precedent
        # LRP56/57): the nonparametric curve does not converge on vocabulary
        # exposures at reporting tier. The estimand is the LINEAR TR -> L association.
        "linear_mechanism": True,
        # Concurrent readout: mechanism at post (same wave as the outcome).
        "mechanism_at_pre": False,
    },
)


def fit(config: str = "dev"):
    return fit_mechanism(SPEC, config=config)
