# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
LRPGBL02: Predictors of taught-vocabulary achievement level.

``LRPGBL02`` is the exploratory model for *taught* expressive-vocabulary level
(``b1extau`` — the Block 1 directly-taught expressive vocabulary score). It is
the taught-vocabulary analogue of :mod:`lrp_rli_gbl_006` (standardised expressive-vocabulary
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

Status: fits the full ``Predictors.DEFAULT_LEVEL`` set (minus the leakage
exclusions above). Hyperparameters MAE-tuned by Optuna on the full predictor set
(150 trials, seed 47; #169), superseding the earlier parameters seeded from the
LRPGBL06 standardised analogue.
"""

from language_reading_predictors.data_variables import Variables as V
from language_reading_predictors.models.base_model import LevelModel
from language_reading_predictors.models.common import DEFAULT_SHAP_SCATTER_SPECS
from language_reading_predictors.models.lgbm_pipeline import LGBMPipeline


# ── hyperparameter set ───────────────────────────────────────────────────
# MAE-tuned by Optuna on the full predictor set (150 trials, seed 47; #169),
# superseding the earlier parameters seeded from LRPGBL06.
_LGBM_MAE_PARAMS: dict[str, float | int | str] = {
    "objective": "mae",
    "n_estimators": 173,
    "learning_rate": 0.03762847988331536,
    "num_leaves": 42,
    "max_depth": 4,
    "min_child_samples": 7,
    "subsample": 0.7429786111553779,
    "subsample_freq": 1,
    "colsample_bytree": 0.6023641005550106,
    "reg_alpha": 0.013509655302654636,
    "reg_lambda": 0.029922670006675835,
    "n_jobs": -1,
    "verbosity": -1,
}


class LRPGBL02(LevelModel):
    """Taught expressive-vocabulary level predictors — exploratory (MAE, all data).

    Uses :attr:`Predictors.DEFAULT_LEVEL` (minus the target ``b1extau`` and the
    tautological total ``b1exto``) with no outlier exclusion.
    """

    model_id = "lrp-rli-gbl-002"
    target_var = V.B1EXTAU
    description = (
        "LightGBM — taught expressive-vocabulary level predictors "
        "(DEFAULT_LEVEL minus b1exto, MAE, no outlier exclusion)"
    )
    pipeline_cls = LGBMPipeline
    params = _LGBM_MAE_PARAMS
    exclude = (V.B1EXTO,)
    shap_scatter_specs = DEFAULT_SHAP_SCATTER_SPECS
    notes = (
        "Exploratory model for predictors of taught expressive-vocabulary level "
        "(b1extau), the taught-vocabulary analogue of lrpgbl06. Fits the full "
        "DEFAULT_LEVEL predictor set (b1exto, the Block 1 expressive total = "
        "taught + not-taught, is excluded to avoid target leakage). "
        "Hyperparameters MAE-tuned by Optuna on the full set (150 trials, seed 47; "
        "#169)."
    )
