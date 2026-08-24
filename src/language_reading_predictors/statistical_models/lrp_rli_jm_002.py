# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""LRPJM02 - phase-stacked joint {word reading, nonword decoding} ANCOVA companion.

#421 Tier 3 (1), second half. ``jm-001`` makes the review's per-wave conditional slope
ratio an identified quantity; this model does the same for the **Tier-1
decoding-specificity Delta** on the parameterisation that quantity was originally
computed on.

Why a second model rather than reading Delta off ``jm-001``. The Tier-1 Delta in
``notes/202607172358-findings-decoding-specificity.md`` is defined on the *mechanism*
(transition / ANCOVA) parameterisation: ``mech-096`` and ``mech-101`` regress each
outcome's post-score on the standardised letter-sound logit **given its own baseline**,
over the three stacked transitions, adjusting for {G, A, HS, IS, SP}. A levels
Delta and an ANCOVA Delta are different estimands - one is "how much higher is the
level per SD of letter sounds", the other "how does the post-level differ per SD of
letter sounds among children who started at the same place" - so re-reporting the
note's Delta from a levels fit would swap the quantity while claiming to identify it.
This model keeps the parameterisation fixed and changes the one thing the review asked
to change: the two outcomes are fitted **together**, so their difference carries the
true cross-outcome covariance instead of the paired-draws convolution.

**An ANCOVA association, not a within-child change effect.** The model regresses each
post-outcome level on the same-period post-letter-sound level, its own pre-outcome
logit, phase, covariates and a child random intercept. It models no letter-sound gain,
no outcome change score and no child-centred letter-sound deviation, so the common
slope blends between-child and within-child information, and the child intercept does
not remove stable general-ability confounding. Do not read a slope as "how much more a
child's score moves" (2026-08-23 follow-up review, #591).

Design. ``phase_mode="all"`` (three stacked transitions), each outcome keeping its own
autoregressive baseline (``W_pre`` / ``N_pre``), its own phase intercepts and its own
Beta-Binomial denominator - 79 items for word reading, 6 for nonword decoding, never
pooled. Adjustment {G, A, HS(``hs``), IS(``attend``), SP(``deapp_c``)} exactly as
``mech-096`` / ``mech-101`` carry it. Note that the *parameterisation* matching is not
sample matching: this model requires both outcome baselines on every retained
transition and standardises the exposure once over that joint union (153 rows, SD
1.41), while each marginal filters to its own outcome's rows and re-standardises there
(152 rows at 1.39, 156 rows at 1.43). One SD is a different raw increment in each, so
the gap between the joint contrast and the paired-marginal sensitivity is not
attributable to cross-outcome covariance alone. The dependence block is a **bivariate child
random intercept** with an LKJ correlation, so ``rho_outcome`` is a between-child
covariance ("children who run high on word reading also run high on decoding"). The
Beta-Binomial ``kappa`` is retained for within-child overdispersion: the two sit at
different levels and are separately identified.

Because the covariance here is between children rather than within a wave, this model
reports ``rho_outcome`` and ``delta_ls_decoding`` and **no** conditional slope ratio -
the "holding decoding fixed at this wave" partialling is ``jm-001``'s quantity.

**Estimand and its limits.** Every slope is an *adjusted association*: latent general
ability is unobserved and the child intercept does not stand in for it. Delta is a
Campbell-Fiske convergent/discriminant argument, not identification of a causal
decoding effect - and that argument assumes a cross-instrument measurement invariance
this model does not impose: with unequal loadings on one general ability, the two
latent-scale slopes differ even with no causal letter-sound route, and the two
instruments differ in item count, floor compression and link discrimination. Nonword
decoding is heavily floored, which costs power on its slope - read a small or uncertain
``beta_mech[N]`` as floor-limited, not as "letters do not feed decoding". The main
PSIS-LOO is leave-one-child-out; the per-outcome LOO-PIT plots are conditional
leave-one-cell-out and are labelled as such. Outcome missingness is assumed ignorable
given the fitted terms, with no MNAR sensitivity registered.
"""

from language_reading_predictors.statistical_models.context import ModelSpec
from language_reading_predictors.statistical_models.joint_mechanism import (
    JointMechanismModelSettings,
)
from language_reading_predictors.statistical_models.pipelines.joint_mechanism import fit_joint_mechanism

SPEC = ModelSpec(
    model_id="lrp-rli-jm-002",
    kind="joint_mechanism",
    title=(
        "Phase-stacked joint {word reading (W), nonword decoding (N)} ANCOVA: "
        "identified Tier-1 decoding-specificity contrast"
    ),
    mechanism_symbol="L",
    adjustment=["G", "A", "W_pre", "N_pre"],
    family="joint_mechanism",
    design="phase-stacked bivariate ANCOVA, LKJ child random intercept",
    estimand_type="association",
    causal_status="none",
    model_settings=JointMechanismModelSettings(
        design="transition",
        outcome_symbols=("W", "N"),
        contrast=("N", "W"),
        # Matched term-for-term to mech-096 / mech-101. The *rows* are not matched:
        # this joint likelihood needs both outcomes' baselines, so it drops rows the
        # single-outcome fits keep and standardises the exposure over a different
        # population. The comparison is approximate; ``comparator_population`` in
        # config.json records the fitted-row identity (2026-08-23 joint audit,
        # finding 7).
        adjust_for=("hs", "hs_missing", "attend", "deapp_c", "deapp_c_missing"),
        confounder_symbols=("G", "A"),
        include_group=True,
    ),
)


def fit(config: str = "dev"):
    return fit_joint_mechanism(SPEC, config=config)
