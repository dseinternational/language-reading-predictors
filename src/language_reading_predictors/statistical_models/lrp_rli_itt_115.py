# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""LRPITT15b - available-case modified ITT generalisation estimate (receptive).

Receptive companion to LRPITT15: the two Block 1 receptive outcomes — directly
taught (``TR`` = b1retau) and not-taught (``UR`` = b1rent) — fitted as factorised
outcome marginals over the randomised window. The headline contrast is
``AME[TR] - AME[UR]`` on the proportion-correct risk-difference scale; the
conditional logit-coefficient difference is secondary. A positive difference shows
a larger taught effect; whether generalisation itself is small must be read from
the marginal not-taught effect against a substantively defined negligible-effect
threshold.
"""

from language_reading_predictors.statistical_models.context import ModelSpec
from language_reading_predictors.statistical_models.joint import (
    JointContrastSettings,
    JointModelSettings,
)
from language_reading_predictors.statistical_models.pipelines.joint import fit_joint

SPEC = ModelSpec(
    model_id="lrp-rli-itt-115",
    kind="joint",
    title=(
        "Available-case modified ITT estimate: receptive taught-versus-not-taught "
        "vocabulary contrast, block 1"
    ),
    # LKJ residual correlation is OFF (same spec as LRPITT15) because an earlier
    # dependence sensitivity mixed poorly. The factorised fit is more stable, but
    # it does not estimate within-child covariance and is not automatically
    # conservative: the omitted covariance can widen or narrow the contrast interval.
    # The registered dependence-aware companion is lrp-rli-itt-315 (#551): the
    # same fit with the block on, derived from this SPEC.
    model_settings=JointModelSettings(
        outcomes=("TR", "UR"),
        # DAG-faithful spec, mirroring the single-outcome suite (own baseline +
        # linear age, no cross-baselines).
        use_cross_baselines=False,
        use_age_linear=True,
        use_residual_correlation=False,
        joint_structure="factorised_outcome_marginals",
        loo_unit="child",
        contrast=JointContrastSettings(
            left="TR",
            right="UR",
            contrast_kind="generalisation",
            contrast_label="Receptive taught versus not-taught vocabulary",
            positive_interpretation=(
                "A positive contrast means the intervention increased the "
                "proportion correct more for taught receptive words than for "
                "not-taught receptive words. It does not by itself show that "
                "generalisation was limited; that requires reading the marginal "
                "not-taught effect against a substantively defined negligible-effect "
                "threshold."
            ),
            negative_interpretation=(
                "A negative contrast means the intervention increased the "
                "proportion correct more for not-taught than taught receptive words."
            ),
            transfer_outcome="UR",
            transfer_interpretation=(
                "Assess whether receptive generalisation is small from the marginal "
                "UR average marginal effect against a substantively defined "
                "negligible-effect threshold."
            ),
            dependence_note=(
                "The fitted model factorises by outcome and does not estimate "
                "within-child residual covariance, so this interval omits that "
                "covariance and is not automatically conservative. The registered "
                "dependence-model sensitivity is lrp-rli-itt-315 (#551): the "
                "same contrast with the per-child LKJ residual-correlation block "
                "on, published as a posterior difference. Read its interval and "
                "P(> 0) beside this one — the point estimate should agree — and "
                "treat this contrast as dependence-checked only once that "
                "companion has passed the house gate."
            ),
            # Machine-readable half of the note above: the release decision
            # verifies this companion is release-ready beside the fit and
            # attaches a dependence-unchecked qualifier when it is not
            # (2026-08-21 joint review, finding 3).
            dependence_companion="lrp-rli-itt-315",
        ),
    ),
)


def fit(config: str = "dev"):
    return fit_joint(SPEC, config=config)
