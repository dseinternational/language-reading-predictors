# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
LRPGBL02: Predictors of taught-vocabulary achievement level.

``LRPGBL02`` is the exploratory model for *taught* expressive-vocabulary level
(``b1extau`` — the Block 1 directly-taught expressive vocabulary score). It is
the taught-vocabulary analogue of :mod:`lrpgbl06` (standardised expressive-vocabulary
level, ``eowpvt``), completing the gain + achievement-level inventory for taught
vocabulary.

The target is mildly right-skewed (``b1extau`` min 0, max 19, median 8, mean
8.16, skewness 0.35, n ≈ 215) — comparable to ``eowpvt`` (skew 0.63), so the
plain MAE LightGBM pipeline with no outlier exclusion used by LRPGBL06 carries over.

Predictor set: :attr:`Predictors.DEFAULT_LEVEL` (target ``b1extau`` is already
excluded — it is in ``DEFAULT_EXCLUDED``) **minus** ``b1exto``. The Block 1
expressive *total* ``b1exto`` equals taught + not-taught (``b1extau + b1exnt``),
so it contains the target directly and must be removed to avoid leakage — the
only deviation from the LRPGBL06 predictor set. (LRPGBL06 additionally dropped
``b1reto`` as redundant with ``rowpvt``; here ``b1reto`` is retained, since for a
*taught*-vocabulary target it carries non-tautological receptive-vocabulary
signal.)

Status: initial exploratory baseline. Fits the full ``Predictors.DEFAULT_LEVEL``
set (minus the leakage exclusions above). Hyperparameters are borrowed from the
LRPGBL06 standardised analogue; a target-specific Optuna tune
(``scripts/tune_model.py lrpgbl02``) is a follow-up.
"""

from language_reading_predictors.data_variables import Variables as V
from language_reading_predictors.models.base_model import LevelModel
from language_reading_predictors.models.common import DEFAULT_SHAP_SCATTER_SPECS
from language_reading_predictors.models.lgbm_pipeline import LGBMPipeline


# ── hyperparameter set ───────────────────────────────────────────────────
#
# Borrowed from LRPGBL06 (MAE-tuned on eowpvt level) as a starting baseline for the
# closely-matched taught-vocabulary level target. Retune for this target before
# any quantitative reporting (scripts/tune_model.py lrpgbl02).
_LGBM_MAE_PARAMS: dict[str, float | int | str] = {
    "objective": "mae",
    "n_estimators": 45,
    "learning_rate": 0.07573022964806482,
    "num_leaves": 30,
    "max_depth": 6,
    "min_child_samples": 10,
    "subsample": 0.8737230089192473,
    "subsample_freq": 1,
    "colsample_bytree": 0.7169131631393786,
    "reg_alpha": 0.0022764472298362187,
    "reg_lambda": 0.003357533830874894,
    "n_jobs": -1,
    "verbosity": -1,
}


class LRPGBL02(LevelModel):
    """Taught expressive-vocabulary level predictors — exploratory (MAE, all data).

    Uses :attr:`Predictors.DEFAULT_LEVEL` (minus the target ``b1extau`` and the
    tautological total ``b1exto``) with no outlier exclusion.
    """

    model_id = "lrpgbl02"
    target_var = V.B1EXTAU
    description = (
        "LightGBM — taught expressive-vocabulary level predictors "
        "(DEFAULT_LEVEL minus b1exto, MAE, no outlier exclusion)"
    )
    pipeline_cls = LGBMPipeline
    params = _LGBM_MAE_PARAMS
    exclude = [V.B1EXTO]
    shap_scatter_specs = DEFAULT_SHAP_SCATTER_SPECS
    notes = (
        "Exploratory model for predictors of taught expressive-vocabulary level "
        "(b1extau), the taught-vocabulary analogue of lrpgbl06. Fits the full "
        "DEFAULT_LEVEL predictor set (b1exto, the Block 1 expressive total = "
        "taught + not-taught, is excluded to avoid target leakage). "
        "Hyperparameters borrowed from lrpgbl06 pending a target-specific tune."
    )
