# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
LRP12: Predictors of receptive-grammar (TROG-2) level.

``LRP12`` is the baseline exploratory model for receptive-grammar
level (``trog``). The ``trog`` score is the items-correct total
from the Test for Reception of Grammar 2 (TROG-2; Bishop 2003),
covering eight grammatical constructs in blocks of four items
(32 items total; observed max 27 in this sample).

The target is near-Gaussian (``trog`` min 3, max 27, median 14,
mean 14.31, std 4.83, skewness 0.29, n ≈ 215) — cleaner
distribution than most LRP level targets. No floor or ceiling
pathology visible at this sample range.

Feature selection applied 2026-06-20 (replication): reduced from the full 32-predictor set to 26 predictors via a distance-correlation redundancy filter (dcor >= 0.70, keep the highest-importance representative) plus an importance noise-floor cut, then re-tuned on the reduced set. See the SelectionStep below and notes/202606201500-gb-replication-findings.md.

No construct-reduced variant: receptive grammar (``trog``) has no same-skill
sibling among the predictors — expressive grammar (``aptgram``) is a different
modality and concept knowledge (``celf``) a different skill, both kept visible
deliberately. See notes/202606210930-lrp-same-skill-variants.md.
"""

from language_reading_predictors.data_variables import Variables as V
from language_reading_predictors.models.base_model import LevelModel
from language_reading_predictors.models.common import SelectionStep, ShapScatterSpec
from language_reading_predictors.models.lgbm_pipeline import LGBMPipeline


# ── predictor selection steps (shared by all variants) ───────────────────
#
# Feature selection (2026-06-20 replication): distance-correlation
# redundancy filter + importance noise-floor cut; see the SelectionStep.

_SELECTION_STEPS: list[SelectionStep] = [
    SelectionStep(
        removed=[
            V.APTINFO, V.B1RETO, V.EOWPVT, V.ERBNW, V.SPPHON, V.DEAPPFI
        ],
        notes=(
            "Feature selection (replication, 2026-06-20): from the full 32-predictor set, a distance-correlation filter (dcor >= 0.70, keep the highest out-of-fold permutation-importance representative per cluster) plus removal of features at/below the 0.005 importance floor. Reduces to 26 predictors with no dcor >= 0.70 pairs remaining; pooled refit-CV held under matched hyperparameters, then the set was re-tuned. See notes/202606201500-gb-replication-findings.md."
        ),
        date="2026-06-20",
        metrics_before={"cv_mae_mean": 2.7606},
        metrics_after={"cv_mae_mean": 2.7918},
    ),
]


# ── hyperparameter sets ─────────────────────────────────────────────────

# MAE-tuned on the 26-predictor replication-selected set, no outlier
# exclusion (Optuna 150 trials, 10-split GroupKFold, seed 47, scoring=mae,
# lgbm_objective=mae). Tuner-inner CV MAE 2.7918. Supersedes the full-set tune.
_LGBM_MAE_PARAMS: dict[str, float | int | str] = {
    "objective": "mae",
    "n_estimators": 120,
    "learning_rate": 0.04287148759795345,
    "num_leaves": 18,
    "max_depth": 3,
    "min_child_samples": 23,
    "subsample": 0.6893331420047686,
    "subsample_freq": 1,
    "colsample_bytree": 0.9630779729356636,
    "reg_alpha": 0.0434106600675856,
    "reg_lambda": 6.647936941764703,
    "n_jobs": -1,
    "verbosity": -1,
}


# ── primary model (baseline, MAE-tuned) ─────────────────────────────────


class LRP12(LevelModel):
    """TROG-2 receptive-grammar level predictors — baseline (all data, MAE-tuned).

    Uses a feature-selected subset of :attr:`Predictors.DEFAULT_LEVEL`
    (minus the target ``trog``) with MAE-tuned hyperparameters and
    no outlier exclusion. Feature selection was applied (2026-06-20 replication); see the SelectionStep and the module docstring.
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
        "Exploratory model for trog (level). Feature-selected (2026-06-20 replication) from the full 32-predictor default set to 26 predictors via a distance-correlation redundancy filter (no dcor >= 0.70 pairs remain) plus an importance noise-floor cut, then re-tuned on the reduced set (tuner-inner CV MAE 2.761 -> 2.792). Only the dominant predictor is robustly above the importance noise floor; treat the reduced ranking as exploratory. See the SelectionStep and notes/202606201500-gb-replication-findings.md."
    )
