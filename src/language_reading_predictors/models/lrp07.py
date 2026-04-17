# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
LRP07: Predictors of receptive vocabulary gains.

``LRP07`` is the exploratory model for receptive vocabulary gains
(``rowpvt_gain``). It is MAE-tuned on the full 34-predictor
:attr:`Predictors.DEFAULT_GAIN` set (plus the auto-included base
variable ``rowpvt``), with no outlier exclusion, designed to
identify the most important influences on receptive vocabulary
gains.

The target is **essentially symmetric** (``rowpvt_gain`` min ≈ −20,
max ≈ 34, median 5, mean 3.84, skewness 0.04, with ~29% negative
and ~3% zero observations, n ≈ 161). Cleaner distribution than any
previous gain target — no skew and no pile-up at zero.

The predictor set will be reduced by iterative importance-based
feature selection under the MAE-tuned params (see
``notes/202604171715-lrp07-feature-selection.md``). This is the
initial tuned baseline; no feature-selection steps yet.
"""

from language_reading_predictors.data_variables import Variables as V  # noqa: F401
from language_reading_predictors.models.base_model import GainModel
from language_reading_predictors.models.common import SelectionStep, ShapScatterSpec
from language_reading_predictors.models.lgbm_pipeline import LGBMPipeline


# ── predictor selection steps (shared by all variants) ───────────────────
#
# LRP07 has not yet been through iterative feature selection. When
# selection variants are introduced, record their rationale here as
# ``SelectionStep`` entries and chain from ``LRP07`` the same way
# ``lrp01.py`` / ``lrp03.py`` / ``lrp05.py`` does.

_SELECTION_STEPS: list[SelectionStep] = []


# ── hyperparameter sets ─────────────────────────────────────────────────

# MAE-tuned on the full 34-predictor set (DEFAULT_GAIN + rowpvt), no
# outlier exclusion (Optuna 150 trials, 10-split GroupKFold, seed 47,
# scoring=mae, lgbm_objective=mae). Tuner-inner CV MAE 6.9827 ±
# 1.3801. n=161.
_LGBM_MAE_PARAMS: dict[str, float | int | str] = {
    "objective": "mae",
    "n_estimators": 42,
    "learning_rate": 0.17957361404938676,
    "num_leaves": 48,
    "max_depth": 3,
    "min_child_samples": 29,
    "subsample": 0.6846157677537632,
    "subsample_freq": 1,
    "colsample_bytree": 0.9683930472827083,
    "reg_alpha": 0.15273064644512227,
    "reg_lambda": 0.19926099863230187,
    "n_jobs": -1,
    "verbosity": -1,
}


# ── primary model (exploratory, MAE-tuned) ──────────────────────────────


class LRP07(GainModel):
    """Receptive vocabulary gain predictors — exploratory (MAE-tuned, all data).

    Uses the full :attr:`Predictors.DEFAULT_GAIN` predictor set plus
    the base variable ``rowpvt`` (auto-included via :class:`GainModel`)
    with MAE-tuned hyperparameters and no outlier exclusion. The
    starting point for feature selection on the receptive vocabulary
    gain-prediction task.
    """

    model_id = "lrp07"
    target_var = V.ROWPVT_GAIN
    description = (
        "LightGBM — receptive vocabulary gain predictors "
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
        "receptive vocabulary gains (rowpvt_gain). MAE-tuned on the "
        "full 34-predictor set (DEFAULT_GAIN + rowpvt) without "
        "outlier exclusion so importance rankings reflect the full "
        "range of outcomes. Target is essentially symmetric (skew "
        "0.04) — cleanest gain target so far. Feature-selection "
        "variants to follow. See "
        "notes/202604171715-lrp07-feature-selection.md."
    )
