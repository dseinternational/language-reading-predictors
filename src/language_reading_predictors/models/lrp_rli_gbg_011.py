# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
LRPGBG11: Predictors of phonetic-spelling gains.

``LRPGBG11`` is the exploratory model for phonetic-spelling gains (``spphon_gain``
— change in the phonetic-spelling score), the spelling member of the reading/
spelling outcomes added in #116 Phase B.

Predictor set: :attr:`Predictors.DEFAULT_GAIN` plus the auto-included baseline
``spphon`` (via :class:`GainModel`). No leakage sibling to drop — phonetic
spelling has no taught/total decomposition — so no extra ``exclude``.

Caveat: ``spphon`` is heavily floored (~78% at zero at baseline; see #119/#144),
so the gain target is dominated by zeros and the predictive signal is expected to
be weak. The ranking will surface that honestly (low out-of-fold R²); treat it as
a "how predictable is spelling change at all" read rather than a strong model.

Status: Huber-tuned by Optuna on the full predictor set (150 trials, seed 47;
#169), superseding the earlier parameters borrowed from the phonics-adjacent
letter-sounds analogue LRPGBG09.
"""

from language_reading_predictors.data_variables import Variables as V
from language_reading_predictors.models.base_model import GainModel
from language_reading_predictors.models.common import DEFAULT_SHAP_SCATTER_SPECS
from language_reading_predictors.models.lgbm_pipeline import LGBMPipeline


# Huber-tuned (Optuna 150-trial, seed 47, GroupKFold cv=51, RMSE scoring)
# on the full default predictor set; best mean cross-validated RMSE 11.10.
# Huber threshold alpha = 1.345 x mean |y - median| (MAD is zero)
# of the tuned target
# (2026-09-22 Huber retune, superseding the #169 MAE tune).
_LGBM_HUBER_PARAMS: dict[str, float | int | str] = {
    "objective": "huber",
    "alpha": 12.367232704402516,
    "n_estimators": 201,
    "learning_rate": 0.033178725466685134,
    "num_leaves": 8,
    "max_depth": 6,
    "min_child_samples": 6,
    "subsample": 0.6598931670653165,
    "subsample_freq": 1,
    "colsample_bytree": 0.7842800179852566,
    "reg_alpha": 0.24003499412719698,
    "reg_lambda": 0.0028957499844139848,
    "n_jobs": -1,
    "verbosity": -1,
}


class LRPGBG11(GainModel):
    """Phonetic-spelling gain predictors — exploratory (Huber, all data)."""

    model_id = "lrp-rli-gbg-011"
    target_var = V.SPPHON_GAIN
    description = (
        "LightGBM — phonetic-spelling gain predictors "
        "(DEFAULT_GAIN, Huber, no outlier exclusion)"
    )
    pipeline_cls = LGBMPipeline
    params = _LGBM_HUBER_PARAMS
    shap_scatter_specs = DEFAULT_SHAP_SCATTER_SPECS
    notes = (
        "Exploratory model for predictors of phonetic-spelling gains (spphon_gain). "
        "spphon is heavily floored (~78% at zero), so the gain signal is expected to "
        "be weak and the out-of-fold R² low — a deliberately honest 'how predictable "
        "is spelling change' read. Hyperparameters Huber-tuned by Optuna on the full "
        "set (150 trials, seed 47; Huber retune of 2026-09-22, superseding #169)."
    )
