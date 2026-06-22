# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
LRP02: Predictors of word-reading level.

``LRP02`` is the exploratory model for word-reading level (``ewrswr``) —
MAE-tuned with no outlier exclusion. The target is heavily right-skewed
(min 0, median 6.5, max 64) with a hard floor at 0.

Uniform feature selection (2026-06-21): reduced from the full
32-predictor :attr:`Predictors.DEFAULT_LEVEL` set (minus the target) to
6 predictors via a distance-correlation redundancy filter (dcor >= 0.70)
plus an importance noise-floor cut, preferring the standardised
instrument over its bespoke taught sibling where it did not reintroduce
redundancy, then re-tuned. See the SelectionStep and
notes/202606211200-uniform-gb-fs.md.
"""

from language_reading_predictors.data_variables import Variables as V
from language_reading_predictors.models.base_model import LevelModel
from language_reading_predictors.models.common import SelectionStep, ShapScatterSpec
from language_reading_predictors.models.lgbm_pipeline import LGBMPipeline


_SELECTION_STEPS: list[SelectionStep] = [
    SelectionStep(
        removed=[
            V.B1EXTO, V.YARCSI, V.APTINFO, V.AGE, V.NONWORD, V.AGESPEAK, V.B1RETO,
            V.ERBNW, V.DEAPPVO, V.AGEBOOKS, V.GENDER, V.DEAPPIN, V.MUMEDUPOST16,
            V.DADEDUPOST16, V.AREA, V.EARINF, V.VISION, V.HEARING, V.ROWPVT,
            V.GROUP, V.BEHAV, V.TIME, V.NUMCHIL, V.TROG, V.APTGRAM, V.DEAPPFI
        ],
        notes=(
            "Uniform feature selection (2026-06-21): from the full 32-predictor set, a distance-correlation redundancy filter (dcor >= 0.70, keep the highest out-of-fold permutation-importance representative) plus an importance noise-floor cut (<= 0.005). The standardised instrument was preferred over its bespoke taught sibling where it did not reintroduce redundancy. Reduces to 6 predictors with no dcor >= 0.70 pairs remaining; re-tuned on the reduced set (Optuna 150-trial MAE, 10-fold GroupKFold, seed 47). Applied uniformly across all GB models; see notes/202606211200-uniform-gb-fs.md."
        ),
        date="2026-06-21",
        metrics_before={"cv_mae_mean": 6.4941},
        metrics_after={"cv_mae_mean": 6.0936},
    ),
]


# MAE-tuned on the 6-predictor uniform-selected set (Optuna 150
# trials, 10-split GroupKFold, seed 47, scoring=mae, lgbm_objective=mae).
# Tuner-inner CV MAE 6.0936.
_LGBM_MAE_PARAMS: dict[str, float | int | str] = {
    "objective": "mae",
    "n_estimators": 61,
    "learning_rate": 0.05087448729350299,
    "num_leaves": 56,
    "max_depth": 10,
    "min_child_samples": 30,
    "subsample": 0.9231435244978656,
    "subsample_freq": 1,
    "colsample_bytree": 0.6478837495498934,
    "reg_alpha": 0.003266918043241777,
    "reg_lambda": 2.7414726117714094,
    "n_jobs": -1,
    "verbosity": -1,
}


class LRP02(LevelModel):
    """Word-reading level predictors — exploratory model (MAE-tuned, all data).

    Uniform-selected subset of :attr:`Predictors.DEFAULT_LEVEL` (minus the
    target ``ewrswr``) with MAE-tuned hyperparameters and no outlier
    exclusion. See the SelectionStep and the module docstring.
    """

    model_id = "lrp02"
    target_var = V.EWRSWR
    description = (
        "LightGBM — word-reading level predictors "
        "(6 predictors, MAE-tuned, no outlier exclusion)"
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
        "Exploratory model for word-reading level (ewrswr). Uniform feature "
        "selection (2026-06-21) from the full 32-predictor DEFAULT_LEVEL set "
        "to 6 predictors (distance-correlation redundancy filter + importance "
        "noise-floor cut; standardised-instrument swap; no dcor >= 0.70 pairs "
        "remain), re-tuned on the reduced set (tuner-inner CV MAE 6.494 -> "
        "6.094). Treat the reduced ranking as exploratory. See "
        "notes/202606211200-uniform-gb-fs.md."
    )
