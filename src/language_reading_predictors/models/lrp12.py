# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
LRP12: Predictors of receptive-grammar (TROG-2) level.

``LRP12`` is the exploratory model for receptive-grammar level (``trog``).
The ``trog`` score is the items-correct total from the Test for Reception
of Grammar 2 (TROG-2; Bishop 2003), covering eight grammatical constructs.
The target is near-Gaussian (min 3, max 27, median 14, mean 14.31, std
4.83, skewness 0.29, n ≈ 215) — cleaner than most LRP level targets.

Uniform feature selection (2026-06-21) with a **corr-filter-only exception**
(2026-06-22): ``trog`` has a flat importance distribution, so the uniform
0.005 noise-floor cut prunes it to 3 predictors at a real CV cost (pooled
R² ≈ 0.46 → 0.30). This model therefore keeps the redundancy-filtered set
(no distance-correlation ≥ 0.70 pairs) but **skips the noise-floor step** —
26 predictors — and re-tunes. It is the one model where the uniform
noise-floor is deliberately not applied. See the SelectionStep and
notes/202606211200-uniform-gb-fs.md.
"""

from language_reading_predictors.data_variables import Variables as V
from language_reading_predictors.models.base_model import LevelModel
from language_reading_predictors.models.common import SelectionStep, ShapScatterSpec
from language_reading_predictors.models.lgbm_pipeline import LGBMPipeline


_SELECTION_STEPS: list[SelectionStep] = [
    SelectionStep(
        removed=[
            V.B1RETO, V.SPPHON, V.EOWPVT, V.DEAPPFI, V.ERBNW, V.APTINFO
        ],
        notes=(
            "Uniform feature selection (2026-06-21) with a corr-filter-only exception (2026-06-22): from the full 32-predictor set, a distance-correlation redundancy filter (dcor >= 0.70, keep the highest out-of-fold permutation-importance representative) removes 6 redundant predictors. The 0.005 importance noise-floor cut is deliberately SKIPPED for this model: trog has a flat importance distribution, so applying it prunes to 3 predictors at a real CV cost (pooled R2 ~0.46 -> 0.30). Keeps 26 predictors with no dcor >= 0.70 pairs remaining; re-tuned (Optuna 150-trial MAE, 10-fold GroupKFold, seed 47), tuner-inner CV MAE 2.96 -> 2.78. The standardised-instrument swap (b1exto -> eowpvt) was reverted because it would reintroduce the eowpvt <-> rowpvt redundancy. See notes/202606211200-uniform-gb-fs.md."
        ),
        date="2026-06-22",
        metrics_before={"cv_mae_mean": 2.9616},
        metrics_after={"cv_mae_mean": 2.7835},
    ),
]


# MAE-tuned on the 26-predictor corr-filter-only set (Optuna 150 trials,
# 10-split GroupKFold, seed 47, scoring=mae, lgbm_objective=mae).
# Tuner-inner CV MAE 2.7835.
_LGBM_MAE_PARAMS: dict[str, float | int | str] = {
    "objective": "mae",
    "n_estimators": 124,
    "learning_rate": 0.0362414989436608,
    "num_leaves": 39,
    "max_depth": 5,
    "min_child_samples": 26,
    "subsample": 0.698337301930694,
    "subsample_freq": 1,
    "colsample_bytree": 0.8852598662130469,
    "reg_alpha": 0.0019791895982314697,
    "reg_lambda": 3.1759013241066048,
    "n_jobs": -1,
    "verbosity": -1,
}


class LRP12(LevelModel):
    """TROG-2 receptive-grammar level predictors — exploratory (MAE-tuned, all data).

    Uniform-selected subset of :attr:`Predictors.DEFAULT_LEVEL` (minus the
    target ``trog``) with the noise-floor step skipped (corr-filter-only
    exception), MAE-tuned, no outlier exclusion. See the SelectionStep and
    the module docstring.
    """

    model_id = "lrp12"
    target_var = V.TROG
    description = (
        "LightGBM — TROG-2 (receptive grammar) level predictors "
        "(26 predictors, MAE-tuned, no outlier exclusion)"
    )
    pipeline_cls = LGBMPipeline
    params = _LGBM_MAE_PARAMS
    cv_splits = 51
    outlier_threshold = None
    selection_steps = _SELECTION_STEPS
    shap_scatter_specs = [
        ShapScatterSpec(description="All predictors, SHAP auto-colouring"),
    ]
    notes = (
        "Exploratory model for receptive-grammar level (trog). Uniform "
        "feature selection (2026-06-21) with a corr-filter-only exception "
        "(2026-06-22): the redundancy filter removes 6 predictors but the "
        "0.005 noise-floor cut is skipped (it would prune this flat-importance "
        "target to 3 predictors at a pooled-R2 cost of ~0.46 -> 0.30). 26 "
        "predictors, no dcor >= 0.70 pairs, re-tuned (tuner-inner CV MAE 2.96 "
        "-> 2.78). Treat the reduced ranking as exploratory. See "
        "notes/202606211200-uniform-gb-fs.md."
    )
