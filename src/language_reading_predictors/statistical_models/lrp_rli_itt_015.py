# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""LRPITT15 - generalisation contrast (expressive): taught vs not-taught (migrates LRP76).

A within-data test of whether gains are larger in directly-taught than not-taught
skills, one component of the trial's broader generalisation claim. The two Block 1
expressive outcomes — the directly-taught target words (``TE`` = b1extau) and the
not-taught comparison words (``UE`` = b1exnt) — are fitted as factorised outcome
marginals over the randomised window. The headline quantity compares their
probability-scale intervention benefits; the conditional logit-coefficient
difference is secondary. Whether generalisation itself is small must be read from
the marginal not-taught effect against a substantively defined negligible-effect
threshold, not from the between-outcome contrast alone.

Sign convention: ``tau`` is the coefficient on the intervention indicator
(positive => the intervention raised that outcome). The reported contrast is
``AME[TE] - AME[UE]`` (risk difference on taught minus risk difference on
not-taught): a positive value supports a larger intervention effect on directly
taught words. The receptive companion is LRPITT15b.
"""

from language_reading_predictors.statistical_models.context import ModelSpec
from language_reading_predictors.statistical_models.pipeline import fit_joint

SPEC = ModelSpec(
    model_id="lrp-rli-itt-015",
    kind="joint",
    title="Generalisation contrast (expressive): taught vs not-taught vocabulary, block 1",
    # Two-outcome joint Beta-Binomial over the randomised window.
    # ``difference=("TE","UE")`` asks the pipeline to summarise the headline
    # probability-scale AME[TE] - AME[UE], with tau[TE] - tau[UE] retained as a
    # secondary conditional-logit contrast.
    # LKJ residual correlation is OFF because an earlier dependence sensitivity
    # mixed poorly. The factorised fit is more stable, but it does not estimate
    # within-child outcome covariance and is not automatically conservative: the
    # omitted covariance can widen or narrow the contrast interval. The receptive
    # companion LRPITT15b uses the same exploratory sensitivity specification.
    extra={
        "outcomes": ("TE", "UE"),
        # DAG-faithful spec, mirroring the single-outcome suite (own baseline +
        # linear age, no cross-baselines).
        "use_cross_baselines": False,
        "use_age_linear": True,
        "use_residual_correlation": False,
        "joint_structure": "factorised_outcome_marginals",
        "loo_unit": "child",
        "difference": ("TE", "UE"),
        "difference_metadata": {
            "contrast_kind": "generalisation",
            "contrast_label": "Expressive taught versus not-taught vocabulary",
            "positive_interpretation": (
                "A positive contrast means the intervention increased the "
                "proportion correct more for taught expressive words than for "
                "not-taught expressive words. It does not by itself show that "
                "generalisation was limited; that requires reading the marginal "
                "not-taught effect against a substantively defined negligible-effect "
                "threshold."
            ),
            "negative_interpretation": (
                "A negative contrast means the intervention increased the "
                "proportion correct more for not-taught than taught expressive words."
            ),
            "transfer_outcome": "UE",
            "transfer_interpretation": (
                "Assess whether expressive generalisation is small from the marginal "
                "UE average marginal effect against a substantively defined "
                "negligible-effect threshold."
            ),
            "dependence_note": (
                "The fitted model factorises by outcome and does not estimate "
                "within-child residual covariance, so the current interval omits "
                "that covariance. Confirmation requires a paired child-level "
                "randomisation-inference/permutation analysis, bootstrap, sandwich, "
                "or dependence-model sensitivity."
            ),
        },
    },
)


def fit(config: str = "dev"):
    return fit_joint(SPEC, config=config)
