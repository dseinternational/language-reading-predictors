# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""LRP105 - Mechanism model: taught expressive vocabulary (TE) -> letter sounds (L),
lagged / predictive readout.

#405 (companion to #404): the growth-prediction arm for taught expressive
vocabulary. Sibling of the concurrent ``mech-104`` that tests *prediction of
growth*: TE at **period-start** (TE_pre) predicting letter sounds at **period-end**
(L_post), conditioning on the period-start letter-sound level (L_pre). The slope
reads as "does where a child's taught expressive vocabulary started predict how much
their letter sounds then changed" - the design behind the descriptive null.

The single build-level difference from ``mech-104`` is ``mechanism_at_pre=True``
(mechanism regressor at TE's period-start logit). Outcome stays at post, baseline
stays at L_pre, giving the lagged form ``TE_pre -> L_post | L_pre``. A positive
concurrent slope (``mech-104``) with a null lagged slope here is the expected
"tracks, doesn't drive" signature.

**Exploratory edge-check, never causal.** As for ``mech-104``, the revised DAG
posits **no vocabulary -> letter-sound edge**, so any association is
latent-general-ability-confounded, not a mechanism.

Adjustment set (re-derived for the TE -> LS backdoor; identical to ``mech-104`` -
see that module's docstring): A (``gamma_A``), G (``beta_G``), HS (``hs``), IS
(``attend``), SP (``deapp_c``, a required confounder for TE because SP -> TE and
SP -> LS), plus the L_pre autoregressive baseline; **TR and erbto (RW) omitted**
(parents of TE that do not reach LS); GA proxied by the child random intercept.

Linear mechanism per the vocabulary-exposure precedent (LRP56/57, ``mech-089``).
The concurrent sibling is ``mech-104``.
"""

from language_reading_predictors.statistical_models.context import ModelSpec
from language_reading_predictors.statistical_models.pipeline import fit_mechanism

SPEC = ModelSpec(
    model_id="lrp-rli-mech-105",
    kind="mechanism",
    title=(
        "Mechanism model: taught expressive vocabulary (TE) -> letter sounds (L), "
        "lagged"
    ),
    outcome_symbol="L",
    mechanism_symbol="TE",
    adjustment=["G", "A", "L_pre"],
    extra={
        "outcomes": ("L", "TE"),
        "adjust_baseline_symbol": "L",
        "adjust_for": ("hs", "hs_missing", "attend", "deapp_c", "deapp_c_missing"),
        "use_age_gp": False,
        "phase_specific_mechanism": False,
        "use_subject_random_intercept": True,
        "linear_mechanism": True,
        # Lagged / predictive readout: mechanism at period-start (pre); the sole
        # difference from mech-104. Gives TE_pre -> L_post, testing prediction of
        # letter-sound growth given the L_pre baseline.
        "mechanism_at_pre": True,
    },
)


def fit(config: str = "dev"):
    return fit_mechanism(SPEC, config=config)
