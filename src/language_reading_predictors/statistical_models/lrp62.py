# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""LRP62 - how much of the word-reading gain is *phonics-route-compatible*?

The robust replacement for LRP59's single-mediator estimate. Rather than asking
how much of the intervention's word-reading effect flows specifically through
letter-sound knowledge, LRP62 asks how much flows through the **phonics route as
a whole** (decoding-compatible skills) versus a residual **lexical /
whole-word-compatible** path — the team's stated next step
(`notes/2026-05-12-project-review.md`, sec. 124/167).

Design (see `factories._build_route_composite_model` + `mediation.decompose`):

- **Phase 0 only** (`phase_mode="itt"`, t1 -> t2): the single randomised contrast
  (group 1 on intervention, group 2 wait-list control). One row per child, n ~ 53.
- **Treatment** G (randomised). **Mediator** = an equal-weight standardised-logit
  **phonics-route composite** of letter-sound (L) and blending (B), conditioned on
  the baseline composite. Phonetic spelling (P) is **excluded**: the
  measurement-sensitivity audit (`scripts/measurement_audit.py`,
  `notes/202606171000-measurement-sensitivity-audit.md`) found P floored (78% at
  zero at t1, 64% at t2) with few movers, so it carries little usable signal in
  this window. **Outcome** Y = W_t2 (word reading), conditioned on W_t1.
- The composite is continuous, so the mediator leg is **Normal** (not
  Beta-Binomial); the outcome leg and the counterfactual NDE/NIE decomposition
  are identical to LRP59. The "phonics-route share" is NIE / Total.

**Adjustment set {G, A, E, R, W_pre}** (+ the baseline composite, handled
internally). As in LRP59, randomisation handles the G->Y and G->M confounding; the
binding unverifiable assumption is no unmeasured mediator-outcome confounding, so
we adjust for the measured route->reading confounders the mechanism work
identified — age (A), expressive (E) and receptive (R) vocabulary — at **baseline
(t1)** to respect the cross-world assumption.

Read this as **triangulation, not a precise causal split**: with n ~ 53,
contemporaneous mediator/outcome measurement, and the no-unmeasured-confounding
assumption, expect wide intervals. "Most of the word-reading gain is
phonics-route-compatible, with a residual lexical share that cannot be ruled out"
is the honest result; the Total should reconcile in sign and rough magnitude with
LRP52's tau_W.
"""

from language_reading_predictors.statistical_models.context import ModelSpec
from language_reading_predictors.statistical_models.pipeline import fit_mediation

SPEC = ModelSpec(
    model_id="lrp62",
    kind="mediation",
    title=(
        "Reading-route decomposition: how much of the word-reading (W) gain is "
        "phonics-route-compatible (letter-sound + blending) vs lexical?"
    ),
    outcome_symbol="W",
    mechanism_symbol=None,  # the mediator is a composite, not a single measure
    adjustment=["G", "A", "E", "R", "W_pre"],
    extra={
        "mediator_kind": "gaussian_composite",
        # Phonics-route composite. P dropped per the measurement-sensitivity audit.
        "route_symbols": ("L", "B"),
    },
)


def fit(config: str = "dev"):
    return fit_mediation(SPEC, config=config)
