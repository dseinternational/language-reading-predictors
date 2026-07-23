# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""LRP104 - Mechanism model: taught expressive vocabulary (TE) -> letter sounds (L),
concurrent readout.

#405 (companion to #404): the expressive-vocabulary arm of "does taught vocabulary
predict letter-sound knowledge?". Same-wave TE_post -> L_post adjusted association,
conditioning on the child's own letter-sound baseline L_pre. Descriptively TE is a
slightly stronger concurrent correlate of letter sounds than TR (pooled r ~ 0.65);
whether TE predicts letter-sound *growth* is the lagged sibling ``mech-105``, where
the #405 baseline-L-adjusted descriptive signal is positive (r = +0.27, 89% interval
+0.05 to +0.46) and is not presupposed null.

**Exploratory edge-check, never causal.** The revised DAG
(``dag/dag-language-reading.dagitty``) posits **no vocabulary -> letter-sound edge**:
TE -> {EV, EG, EI, PA, WR} and none reaches LS, so there is no directed TE -> LS
path. Any association is latent-general-ability-confounded, an edge-check on the
committed DAG, not a mechanism. Report accordingly.

Revised-DAG adjustment set for the TE -> LS backdoor (re-derived per #405; it
differs from the TE -> WR set of ``mech-089`` because the outcome is now LS):

- A (age): ``gamma_A``; G (group): ``beta_G``; L_pre: autoregressive baseline.
- HS (hearing, ``hs`` / ``hs_missing``): common cause ``TE <- HS -> LS`` - adjust.
- IS (intervention sessions, ``attend``): common cause ``TE <- IS -> LS`` (the
  intervention teaches both) - adjust, with the ``IG -> IS <- GA`` collider closed
  at the always-conditioned arm ``G`` (as in ``mech-088``).
- SP (speech, ``deapp_c`` / ``deapp_c_missing``): **a required confounder here** -
  SP -> TE *and* SP -> LS, so ``TE <- SP -> LS`` is an open backdoor unless SP is
  adjusted. (This is the one adjustment-set difference in intent from the TR model
  ``mech-102``, where SP is only a precision adjuster; the fitted set is identical.)
- **TR and RW are NOT adjusted**: both are parents of TE (``TR -> TE``,
  ``RW -> TE``) but neither reaches LS, so neither is a common cause of TE and LS.
  This differs from ``mech-089`` (TE -> WR), which adjusts TR because TR -> WR.
- GA (general ability) is latent; the child random intercept proxies its
  time-invariant part, so the slope stays an adjusted association.

Linear mechanism per the vocabulary-exposure precedent (LRP56/57, ``mech-089``).
The lagged / predictive sibling (TE_pre -> L_post | L_pre) is ``mech-105``.
"""

from language_reading_predictors.statistical_models.context import ModelSpec
from language_reading_predictors.statistical_models.pipeline import fit_mechanism

SPEC = ModelSpec(
    model_id="lrp-rli-mech-104",
    kind="mechanism",
    title=(
        "Mechanism model: taught expressive vocabulary (TE) -> letter sounds (L), "
        "concurrent"
    ),
    outcome_symbol="L",
    mechanism_symbol="TE",
    adjustment=["G", "A", "L_pre"],
    extra={
        "outcomes": ("L", "TE"),
        "adjust_baseline_symbol": "L",
        # HS, IS (attend), SP (speech, a required confounder here). erbto (RW) and TR
        # omitted: parents of TE that do not reach LS. See the module docstring.
        "adjust_for": ("hs", "hs_missing", "attend", "deapp_c", "deapp_c_missing"),
        "use_age_gp": False,
        "phase_specific_mechanism": False,
        "use_subject_random_intercept": True,
        "linear_mechanism": True,
        "mechanism_at_pre": False,
    },
)


def fit(config: str = "dev"):
    return fit_mechanism(SPEC, config=config)
