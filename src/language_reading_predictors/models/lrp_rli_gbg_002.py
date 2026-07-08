# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
LRPGBG02: Predictors of taught-vocabulary gains.

``LRPGBG02`` is the exploratory model for *taught* expressive-vocabulary gains
(``b1extau_gain`` — change in the Block 1 directly-taught expressive vocabulary
score). It is the taught-vocabulary analogue of :mod:`lrp_rli_gbg_006` (standardised
expressive-vocabulary gains, ``eowpvt_gain``), added so the predictor analysis
covers taught vocabulary too, not only the standardised tests.

The target is signed and mildly skewed (``b1extau_gain`` min ≈ −6, max ≈ 12,
median 1, mean 1.83, skewness 0.71, ~20% negative, n ≈ 161) — comparable to
``eowpvt_gain``, so the plain MAE LightGBM pipeline (no target transform, no
outlier exclusion) used by LRPGBG06 carries over.

Predictor set: :attr:`Predictors.DEFAULT_GAIN` plus the auto-included baseline
``b1extau`` (via :class:`GainModel`), **minus** ``b1exto``. The Block 1 expressive
*total* ``b1exto`` equals taught + not-taught (``b1extau + b1exnt``) so it contains
the target/baseline construct directly; keeping it would make the model a
between-tests calibration of the same instrument rather than an identification of
substantive predictors. This is the only deviation from the LRPGBG06 predictor set.

Status: MAE-tuned by Optuna on the full predictor set (150 trials, seed 47;
#169), superseding the earlier parameters seeded from the LRPGBG06 standardised
analogue. Importance rankings — the purpose of this exploratory model — are
robust to reasonable parameters.
"""

from language_reading_predictors.data_variables import Variables as V
from language_reading_predictors.models.base_model import GainModel
from language_reading_predictors.models.common import DEFAULT_SHAP_SCATTER_SPECS
from language_reading_predictors.models.lgbm_pipeline import LGBMPipeline


# ── hyperparameter set ───────────────────────────────────────────────────
# MAE-tuned by Optuna on the full predictor set (150 trials, seed 47; #169),
# superseding the earlier parameters seeded from LRPGBG06.
_LGBM_MAE_PARAMS: dict[str, float | int | str] = {
    "objective": "mae",
    "n_estimators": 130,
    "learning_rate": 0.055509040295441216,
    "num_leaves": 9,
    "max_depth": 5,
    "min_child_samples": 5,
    "subsample": 0.6160866260410247,
    "subsample_freq": 1,
    "colsample_bytree": 0.8688013411820535,
    "reg_alpha": 0.002155888319490245,
    "reg_lambda": 0.16855387389968438,
    "n_jobs": -1,
    "verbosity": -1,
}


class LRPGBG02(GainModel):
    """Taught expressive-vocabulary gain predictors — exploratory (MAE, all data).

    Uses :attr:`Predictors.DEFAULT_GAIN` plus the auto-included baseline
    ``b1extau`` and minus the tautological total ``b1exto`` (see module
    docstring), with no outlier exclusion.
    """

    model_id = "lrp-rli-gbg-002"
    target_var = V.B1EXTAU_GAIN
    description = (
        "LightGBM — taught expressive-vocabulary gain predictors "
        "(DEFAULT_GAIN minus b1exto, MAE, no outlier exclusion)"
    )
    pipeline_cls = LGBMPipeline
    params = _LGBM_MAE_PARAMS
    exclude = (V.B1EXTO,)
    shap_scatter_specs = DEFAULT_SHAP_SCATTER_SPECS
    notes = (
        "Exploratory model for predictors of taught expressive-vocabulary gains "
        "(b1extau_gain), the taught-vocabulary analogue of lrpgbg06. b1exto (Block 1 "
        "expressive total = taught + not-taught) is excluded to avoid target "
        "leakage. Hyperparameters MAE-tuned by Optuna on the full set (150 trials, "
        "seed 47; #169)."
    )
