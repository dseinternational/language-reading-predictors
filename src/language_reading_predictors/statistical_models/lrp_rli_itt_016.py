# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""LRPITT16 - modality contrast: taught expressive (TE) vs taught receptive (TR).

The generalisation contrasts LRPITT15 (expressive: TE vs UE) and LRPITT15b/115
(receptive: TR vs UR) each compare taught vs not-taught *within* a modality. Neither
asks whether the intervention moved the directly-taught words more in the
**expressive** modality than in the **receptive** modality. LRPITT16 supplies that
missing contrast: the two Block 1 *taught* outcomes — expressive taught target words
(``TE`` = b1extau) and receptive taught target words (``TR`` = b1rectau) — are
fitted as factorised outcome marginals over the randomised window. The headline
quantity compares their probability-scale intervention benefits.

Sign convention: ``tau`` is the coefficient on the intervention indicator
(positive => the intervention raised that outcome). The reported contrast is
``AME[TE] - AME[TR]`` (proportion-correct risk difference for taught expressive
minus taught receptive): a positive value supports a larger effect in production
than comprehension. The conditional logit-coefficient difference is secondary.

Both component effects use the randomised window. Because residual correlation is
disabled, however, the product model does not identify their within-child posterior
covariance; the paired difference needs an explicit dependence sensitivity.

LKJ residual correlation is OFF, mirroring the convergence-stable factorised fit
used by the LRPITT15 companions. This avoids the poor mixing seen in an earlier
dependence sensitivity, but it does not estimate within-child covariance and is not
automatically conservative: omitted covariance can widen or narrow the contrast
interval.
"""

from language_reading_predictors.statistical_models.context import ModelSpec
from language_reading_predictors.statistical_models.pipeline import fit_joint

SPEC = ModelSpec(
    model_id="lrp-rli-itt-016",
    kind="joint",
    title="Modality contrast: taught expressive (TE) vs taught receptive (TR) vocabulary, block 1",
    extra={
        "outcomes": ("TE", "TR"),
        # DAG-faithful spec, mirroring the single-outcome suite (own baseline +
        # linear age, no cross-baselines).
        "use_cross_baselines": False,
        "use_age_linear": True,
        "use_residual_correlation": False,
        "joint_structure": "factorised_outcome_marginals",
        "loo_unit": "child",
        "difference": ("TE", "TR"),
        "difference_metadata": {
            "contrast_kind": "modality",
            "contrast_label": "Taught expressive versus taught receptive vocabulary",
            "positive_interpretation": (
                "A positive contrast means the intervention increased the "
                "proportion correct more for taught expressive words than for "
                "taught receptive words."
            ),
            "negative_interpretation": (
                "A negative contrast means the intervention increased the "
                "proportion correct more for taught receptive words than for "
                "taught expressive words."
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
