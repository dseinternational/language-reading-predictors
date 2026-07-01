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

Status: initial exploratory baseline; hyperparameters borrowed from the
phonics-adjacent letter-sounds analogue LRPGBG09 pending a target-specific tune
(``scripts/tune_model.py lrpgbg11``).
"""

from language_reading_predictors.data_variables import Variables as V
from language_reading_predictors.models.base_model import GainModel
from language_reading_predictors.models.common import DEFAULT_SHAP_SCATTER_SPECS
from language_reading_predictors.models.lgbm_pipeline import LGBMPipeline


# Borrowed from LRPGBG09 (letter-sounds gain) pending a target tune.
_LGBM_MAE_PARAMS: dict[str, float | int | str] = {
    "objective": "mae",
    "n_estimators": 41,
    "learning_rate": 0.0853489590463849,
    "num_leaves": 42,
    "max_depth": 12,
    "min_child_samples": 21,
    "subsample": 0.688114728810087,
    "subsample_freq": 1,
    "colsample_bytree": 0.9723492415919919,
    "reg_alpha": 0.001990819330098672,
    "reg_lambda": 3.298314330689827,
    "n_jobs": -1,
    "verbosity": -1,
}


class LRPGBG11(GainModel):
    """Phonetic-spelling gain predictors — exploratory (MAE, all data)."""

    model_id = "lrpgbg11"
    target_var = V.SPPHON_GAIN
    description = (
        "LightGBM — phonetic-spelling gain predictors "
        "(DEFAULT_GAIN, MAE, no outlier exclusion)"
    )
    pipeline_cls = LGBMPipeline
    params = _LGBM_MAE_PARAMS
    shap_scatter_specs = DEFAULT_SHAP_SCATTER_SPECS
    notes = (
        "Exploratory model for predictors of phonetic-spelling gains (spphon_gain). "
        "spphon is heavily floored (~78% at zero), so the gain signal is expected to "
        "be weak and the out-of-fold R² low — a deliberately honest 'how predictable "
        "is spelling change' read. Hyperparameters borrowed from lrpgbg09 (letter "
        "sounds) pending a target-specific tune."
    )
