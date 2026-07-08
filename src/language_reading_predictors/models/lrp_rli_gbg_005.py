# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
LRPGBG05: Predictors of receptive vocabulary gains.

``LRPGBG05`` is the exploratory model for receptive vocabulary gains
(``rowpvt_gain``). It is MAE-tuned on the full
:attr:`Predictors.DEFAULT_GAIN` set (with the ``rowpvt`` baseline
auto-included) and no outlier exclusion, designed to identify the most
important influences on receptive vocabulary gains.

The target is **essentially symmetric** (``rowpvt_gain`` min ≈ −20,
max ≈ 34, median 5, mean 3.84, skewness 0.04, with ~29% negative
and ~3% zero observations, n ≈ 161). Cleaner distribution than any
previous gain target — no skew and no pile-up at zero.

Fits the full ``Predictors.DEFAULT_GAIN`` set; hyperparameters are
retained from the earlier pruned-set tune (retune-pending, #116 Phase D).
"""

from language_reading_predictors.data_variables import Variables as V
from language_reading_predictors.models.base_model import GainModel
from language_reading_predictors.models.common import DEFAULT_SHAP_SCATTER_SPECS
from language_reading_predictors.models.lgbm_pipeline import LGBMPipeline


# ── hyperparameter sets ─────────────────────────────────────────────────

# MAE-tuned (Optuna 150-trial, seed 47) on the earlier pruned selected set;
# retained as the full-set baseline (retune-pending).
_LGBM_MAE_PARAMS: dict[str, float | int | str] = {
    "objective": "mae",
    "n_estimators": 26,
    "learning_rate": 0.17328403199352835,
    "num_leaves": 23,
    "max_depth": 10,
    "min_child_samples": 37,
    "subsample": 0.6393895136146711,
    "subsample_freq": 1,
    "colsample_bytree": 0.7127905387176482,
    "reg_alpha": 0.5995768124191261,
    "reg_lambda": 2.3376116482336817,
    "n_jobs": -1,
    "verbosity": -1,
}


# ── primary model (exploratory, MAE-tuned) ──────────────────────────────


class LRPGBG05(GainModel):
    """Receptive vocabulary gain predictors — exploratory (MAE-tuned, all data).

    Full ``Predictors.DEFAULT_GAIN`` set, MAE-tuned (params
    retune-pending). Uses the full predictor set plus the base variable
    ``rowpvt`` (auto-included via :class:`GainModel`) with no outlier
    exclusion.
    """

    model_id = "lrp-rli-gbg-005"
    target_var = V.ROWPVT_GAIN
    description = (
        "LightGBM — receptive vocabulary gain predictors "
        "(full predictor set, MAE-tuned, no outlier exclusion)"
    )
    pipeline_cls = LGBMPipeline
    params = _LGBM_MAE_PARAMS
    shap_scatter_specs = DEFAULT_SHAP_SCATTER_SPECS
    notes = (
        "Exploratory model for rowpvt_gain (gain). Fits the full DEFAULT_GAIN predictor set (#116 Phase D retired hard feature selection in favour of full-set ranking); hyperparameters are retained from the earlier pruned-set Optuna tune (retune-pending). Gain models are near-noise (baseline-driven regression to the mean) - treat the ranking as exploratory."
    )
