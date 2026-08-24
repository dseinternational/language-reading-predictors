# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""LRPJM01 - per-wave joint {word reading, nonword decoding} levels model.

#421 Tier 3 (1), the build the letter-sound -> word-reading review note (#424) and the
decoding-specificity note (202607172358) both wait on. Two quantities the suite
currently reports are **product-of-marginals sensitivities**, assembled by pairing
draws from separate fits under a working-independence assumption:

1. the **share retained** - what fraction of the letter-sound -> word-reading slope
   survives holding nonword decoding fixed (0.97 / 0.74 / 0.80 / 0.66 at t1-t4),
   currently the ratio of ``ca-011``'s slope to ``ca-010``'s; and
2. the **decoding-specificity contrast** ``Delta = beta(LS->N) - beta(LS->W)``,
   currently paired from ``mech-096`` and ``mech-101``.

The paired fits share children, so the true joint posterior has a cross-outcome
covariance the pairing sets to zero. This model fits both outcomes together at each
wave with an **LKJ residual correlation** between them, which turns both quantities
into within-model deterministics.

Design. One cross-sectional fit per timepoint (``phase_mode="levels"``, one row per
child), each regressing *both* word-reading and nonword-decoding levels on the same
standardised same-wave letter-sound logit, with a per-outcome slope. The adjustment
set is matched term-for-term to ``ca-010`` / ``ca-011`` - age, hearing (``hs`` plus
its ``hs_missing`` indicator), non-verbal ability (``blocks``) and the flagged group
nuisance (the same wide ``Normal(0, 1)``) - and the slope prior is the same
regularising ``Normal(0, 0.3)``, so the identified ``share_retained`` replaces their
paired-draws ratio like for like. Both outcomes' residuals are drawn
from one bivariate normal with an LKJ correlation; the likelihood is **Binomial**
rather than Beta-Binomial because that residual already carries the extra-binomial
variance (two overdispersion mechanisms on one row is how the ITT joint's LKJ block
went prior-dominated in 2026-04).

The conditional slope follows from the covariance block:

    beta(LS->W | N) = beta_W - rho (sigma_W / sigma_N) beta_N
    share_retained  = beta(LS->W | N) / beta_W

Note this conditions on the **latent** nonword logit where ``ca-011`` conditions on
the *observed* nonword count. Partialling the latent skill is the cleaner reading of
"holding decoding fixed"; it also partials measurement error, so it will generally
retain *less* than the observed-score version. They are **not** two bounds on one
underlying answer: this fit and the ``ca-010`` / ``ca-011`` sensitivities differ in
likelihood, in how missing predictors are treated, in what is conditioned on, in
fitted rows and in estimand, so reading them as bracketing a single quantity is not
supported (2026-08-23 joint audit, finding 4). Read them as different questions whose
answers happen to be comparable in scale. ``share_retained`` is a ratio of posterior
quantities and only interpretable while ``beta_W`` stays clear of zero *and* the
held-fixed outcome's residual scale ``sigma_N`` stays clear of zero, since that scale
divides the conditional slope; the pipeline applies that rule (0.05 logit, 95%
support - ``_jm_ratio_stability``) and withholds the ratio when it fails, publishing
the denominator-free ``abs_slope_reduction`` beside it. Report its median and
interval, never a mean.

**Estimand and its limits.** Every slope is an *adjusted association*, never a causal
effect: latent general ability is unobserved and the residual covariance does not
stand in for it. The contrast is a Campbell-Fiske convergent/discriminant argument (a
pure-general-ability account gives no reason for letter sounds to predict pure
decoding *more* than sight-readable word reading), not identification of a causal
decoding mechanism. Nonword decoding is a 6-item count floored for 72 / 64 / 52 / 40 %
of children at t1-t4, so its residual scale - and through it ``share_retained`` - is
the least well determined quantity in the model. The corresponding randomised-arm
result is an available-case modified ITT estimate in the ITT suite.

Companion: ``jm-002`` re-reports the Tier-1 Delta on the phase-stacked ANCOVA
parameterisation that ``mech-096`` / ``mech-101`` use.
"""

from language_reading_predictors.statistical_models.context import ModelSpec
from language_reading_predictors.statistical_models.joint_mechanism import (
    JointMechanismModelSettings,
)
from language_reading_predictors.statistical_models.pipelines.joint_mechanism import fit_joint_mechanism

SPEC = ModelSpec(
    model_id="lrp-rli-jm-001",
    kind="joint_mechanism",
    title=(
        "Per-wave joint {word reading (W), nonword decoding (N)} levels model: "
        "identified decoding-specificity contrast and share retained"
    ),
    mechanism_symbol="L",
    adjustment=["G", "A"],
    family="joint_mechanism",
    design="per-wave cross-sectional bivariate levels, LKJ residual correlation",
    estimand_type="association",
    causal_status="none",
    model_settings=JointMechanismModelSettings(
        design="levels",
        outcome_symbols=("W", "N"),
        # contrast[0] - contrast[1] is the reported Delta; contrast[1] is also the
        # focal outcome whose slope share_retained partials.
        contrast=("N", "W"),
        # Matched to ca-010 / ca-011: non-verbal ability and hearing — with the
        # hs_missing indicator that the missing-indicator policy pairs with the
        # filled hs (9/54 children have unknown hearing at t1; without it they
        # would be silently coded hearing-clear) — as t1 baselines broadcast
        # across the waves; age via the confounder set; group as a flagged
        # non-interpretable nuisance.
        covariates=("blocks", "hs", "hs_missing"),
        confounder_symbols=("G", "A"),
        include_group=True,
        predictor_slope_sigma=0.3,
    ),
)


def fit(config: str = "dev"):
    return fit_joint_mechanism(SPEC, config=config)
