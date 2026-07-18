# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""LRP96 - the decoding channel: letter-sound knowledge (L) -> nonword reading (N).

Tier-1 decoding-specificity mini-suite, design 1A (see
``notes/202607172330-tier1-decoding-specificity-spec.md``). The clean
**total-association** letter-sound -> nonword-decoding slope, i.e. LRP72 *without*
the phoneme-blending moderator. Nonword reading isolates alphabetic decoding (a
string that cannot be sight-read), so a strong ``L -> N`` slope is the signature
that letter-sound knowledge is being *used for decoding* rather than merely
travelling alongside reading through other routes.

Paired with the matched linear ``L -> W`` slope (LRP101) it forms the
**convergent-discriminant contrast** Delta = beta(L->N) - beta(L->W): letter sounds
should feed the pure-decoding channel (N) at least as strongly as the mixed word
channel (W, which can be sight-read). A pure general-ability-confounding account
gives no reason for L to predict N *more* than W, so Delta >= 0 is the decoding-use
signature (Campbell & Fiske 1959 convergent/discriminant logic; not an
identification claim).

Design (mirrors LRP72, moderator removed):

- **Linear mechanism** (``linear_mechanism=True``): decoding is a low, heavily-floored
  0-6 count, so an HSGP dose-response on logit(L) is not identifiable; the mechanism
  enters as a single linear slope ``beta_mech . z(logit L)`` - comparable, per SD of
  the exposure, to the linear ``L -> W`` and negative-control slopes.
- **Graded Beta-Binomial N** (not the off-floor rule). The suite floor rule protects
  *treatment marginals*; for an *exposure-association slope* the graded count is the
  right leg (the LRP72 precedent). The floor still costs power - read a small/uncertain
  slope as floor-limited, not as "letters do not feed decoding".
- **Matched adjustment {G, A, HS, IS, SP} + own baseline** (revised-DAG parents of L,
  #245): group (beta_G), age, hearing (``hs``/``hs_missing``), sessions (``attend``),
  speech (``deapp_c``/``deapp_c_missing``), plus baseline nonword ``N_pre``. Identical
  confounders to LRP58/LRP101 so the cross-outcome contrast is like-for-like. Word
  reading W is **excluded** (sibling/descendant of decoding - over-control / collider
  risk).

GA (latent general ability) is unblockable and the child random intercept does not
stand in for it, so ``beta_mech`` is an **adjusted association**, never a causal
decoding effect. The randomised warrant lives in the ITT suite.
"""

from language_reading_predictors.statistical_models.context import ModelSpec
from language_reading_predictors.statistical_models.pipeline import fit_mechanism

SPEC = ModelSpec(
    model_id="lrp-rli-mech-096",
    kind="mechanism",
    title="Decoding channel: letter-sound knowledge (L) -> nonword reading (N)",
    outcome_symbol="N",
    mechanism_symbol="L",
    adjustment=["G", "A", "N_pre"],
    extra={
        "adjust_baseline_symbol": "N",
        "outcomes": ("L", "N"),
        "adjust_for": ("hs", "hs_missing", "attend", "deapp_c", "deapp_c_missing"),
        "linear_mechanism": True,
        "use_age_gp": False,
        "phase_specific_mechanism": False,
        "use_subject_random_intercept": True,
    },
)


def fit(config: str = "dev"):
    return fit_mechanism(SPEC, config=config)
