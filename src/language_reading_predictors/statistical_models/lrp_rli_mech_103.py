# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""LRP103 - Mechanism model: taught receptive vocabulary (TR) -> letter sounds (L),
lagged / predictive readout.

#405 (companion to #404): the growth-prediction half of "does taught vocabulary
predict letter-sound knowledge?". This is the sibling of the concurrent ``mech-102``
that actually tests *prediction of growth*: taught receptive vocabulary at
**period-start** (TR_pre) predicting letter sounds at **period-end** (L_post),
conditioning on the child's period-start letter-sound level (L_pre). Because the
predictor precedes the outcome increment and the outcome's own baseline is held,
the slope reads as "does where a child's taught vocabulary started predict how much
their letter sounds then changed" - the design behind the descriptive null (baseline
TR ~ -0.1 with the subsequent L gain).

The single build-level difference from ``mech-102`` is ``mechanism_at_pre=True``:
the mechanism regressor is TR's period-start (pre) logit rather than its post logit.
The outcome stays at post and the autoregressive baseline stays at L_pre, so this is
the lagged form ``TR_pre -> L_post | L_pre``. A positive concurrent slope
(``mech-102``) alongside a null lagged slope here is the expected "tracks, doesn't
drive" signature and should be reported as such.

**Exploratory edge-check, never causal.** As for ``mech-102``, the revised DAG
(``dag/dag-language-reading.dagitty``) posits **no vocabulary -> letter-sound edge**,
so any association is latent-general-ability-confounded, not a mechanism.

Adjustment set (re-derived for the TR -> LS backdoor; identical to ``mech-102`` -
see that module's docstring for the per-symbol derivation): A (``gamma_A``), G
(``beta_G``), HS (``hs``), IS (``attend``), SP (``deapp_c``) as a precision adjuster,
plus the L_pre autoregressive baseline; **erbto (RW) omitted** because it does not
reach LS; GA proxied by the child random intercept.

Linear mechanism per the vocabulary-exposure precedent (LRP56/57, ``mech-088``).
The concurrent sibling is ``mech-102``.
"""

from language_reading_predictors.statistical_models.context import ModelSpec
from language_reading_predictors.statistical_models.pipeline import fit_mechanism

SPEC = ModelSpec(
    model_id="lrp-rli-mech-103",
    kind="mechanism",
    title=(
        "Mechanism model: taught receptive vocabulary (TR) -> letter sounds (L), "
        "lagged"
    ),
    outcome_symbol="L",
    mechanism_symbol="TR",
    adjustment=["G", "A", "L_pre"],
    extra={
        "outcomes": ("L", "TR"),
        "adjust_baseline_symbol": "L",
        "adjust_for": ("hs", "hs_missing", "attend", "deapp_c", "deapp_c_missing"),
        "use_age_gp": False,
        "phase_specific_mechanism": False,
        "use_subject_random_intercept": True,
        "linear_mechanism": True,
        # Lagged / predictive readout: mechanism at period-start (pre). This is the
        # sole difference from the concurrent mech-102; it gives TR_pre -> L_post,
        # which with the L_pre baseline tests prediction of letter-sound growth.
        "mechanism_at_pre": True,
    },
)


def fit(config: str = "dev"):
    return fit_mechanism(SPEC, config=config)
