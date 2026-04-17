# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
LRP03: Predictors of expressive-vocabulary gains.

``LRP03`` is the exploratory model for expressive-vocabulary gains
(``eowpvt_gain``). It is MAE-tuned on the full 34-predictor
:attr:`Predictors.DEFAULT_GAIN` set (plus the auto-included base
variable ``eowpvt``), with no outlier exclusion, designed to
identify the most important influences on expressive-vocabulary
gains.

The target is signed (``eowpvt_gain`` min ≈ −13, max ≈ 28, median 3,
skewness 0.32, with ~25% negative observations and n ≈ 161). That's
much milder skew than LRP01's ``ewrswr_gain`` and nearly symmetric —
a log / signed-log transform may or may not help and is a question
for future investigation.

The predictor set will be reduced by iterative importance-based
feature selection under the MAE-tuned params (see
``notes/202604171127-lpr03-feature-selection.md``). This is the
initial tuned baseline; no feature-selection steps yet.
"""

from language_reading_predictors.data_variables import Variables as V
from language_reading_predictors.models.base_model import GainModel
from language_reading_predictors.models.common import SelectionStep, ShapScatterSpec
from language_reading_predictors.models.lgbm_pipeline import LGBMPipeline


# ── predictor selection steps (shared by all variants) ───────────────────
#
# LRP03 has not yet been through iterative feature selection. When
# selection variants are introduced, record their rationale here as
# ``SelectionStep`` entries and chain from ``LRP03`` the same way
# ``lrp01.py`` does.

_SELECTION_STEPS: list[SelectionStep] = []


# ── hyperparameter sets ─────────────────────────────────────────────────

# MAE-tuned on the full 34-predictor set (DEFAULT_GAIN + eowpvt), no
# outlier exclusion (Optuna 150 trials, 10-split GroupKFold, seed 47,
# scoring=mae, lgbm_objective=mae). Tuner-inner CV MAE 5.0256 ± 0.7959.
# n=161.
_LGBM_MAE_PARAMS: dict[str, float | int | str] = {
    "objective": "mae",
    "n_estimators": 16,
    "learning_rate": 0.17211058322522463,
    "num_leaves": 7,
    "max_depth": 9,
    "min_child_samples": 17,
    "subsample": 0.8502717837231728,
    "subsample_freq": 1,
    "colsample_bytree": 0.9492076870134684,
    "reg_alpha": 0.3843221349693716,
    "reg_lambda": 0.0011002806788511271,
    "n_jobs": -1,
    "verbosity": -1,
}


# ── primary model (exploratory, MAE-tuned) ──────────────────────────────


class LRP03(GainModel):
    """Expressive-vocabulary gain predictors — exploratory (MAE-tuned, all data).

    Uses the full :attr:`Predictors.DEFAULT_GAIN` predictor set plus
    the base variable ``eowpvt`` (auto-included via :class:`GainModel`)
    with MAE-tuned hyperparameters and no outlier exclusion. The
    starting point for feature selection on the expressive-vocabulary
    gain-prediction task.
    """

    model_id = "lrp03"
    target_var = V.EOWPVT_GAIN
    description = (
        "LightGBM — expressive-vocabulary gain predictors "
        "(34 predictors, MAE-tuned, no outlier exclusion)"
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
        "Exploratory model for identifying important predictors of "
        "expressive-vocabulary gains (eowpvt_gain). MAE-tuned on the "
        "full 34-predictor set (DEFAULT_GAIN + eowpvt) without outlier "
        "exclusion so importance rankings reflect the full range of "
        "outcomes. Feature-selection variants to follow. See "
        "notes/202604171127-lpr03-feature-selection.md."
    )
