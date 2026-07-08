# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
LRPGBL13: Predictors of non-word reading level.

``LRPGBL13`` is the baseline exploratory model for non-word reading
level (``nonword``). ``nonword`` is an items-correct score from
a non-word decoding task (observed range 0–6 in this sample).

The target is **heavily floor-loaded** (``nonword`` min 0, max 6,
median 0, mean 1.24, std 1.81, skewness 1.38, with **57% at
zero**, n ≈ 215). Non-word reading / phonological decoding is a
late-emerging skill — most children in this sample have not yet
started to decode non-words reliably, so the data are genuinely
half zeros. This differs from the LRPGBL12 / LRPGBL09 right-skewed
reading targets where zeros are a minority.

Log or quantile transforms may be more appropriate than a plain
regression here; plan for a ``lrpgbl13_log`` variant in follow-up PRs.
"""

from language_reading_predictors.data_variables import Variables as V
from language_reading_predictors.models.base_model import LevelModel
from language_reading_predictors.models.common import DEFAULT_SHAP_SCATTER_SPECS
from language_reading_predictors.models.lgbm_pipeline import LGBMPipeline


# MAE-tuned by Optuna on the full predictor set (150 trials, seed 47;
# #169 retune, superseding the earlier pruned-set tune).
_LGBM_MAE_PARAMS: dict[str, float | int | str] = {
    "objective": "mae",
    "n_estimators": 118,
    "learning_rate": 0.04811059604375191,
    "num_leaves": 17,
    "max_depth": 3,
    "min_child_samples": 11,
    "subsample": 0.9764536087799128,
    "subsample_freq": 1,
    "colsample_bytree": 0.6690024806000164,
    "reg_alpha": 0.007967682872704603,
    "reg_lambda": 0.016771763411827834,
    "n_jobs": -1,
    "verbosity": -1,
}


class LRPGBL13(LevelModel):
    """Non-word reading level predictors — baseline (all data, MAE-tuned).

    Full ``Predictors.DEFAULT_LEVEL`` set, MAE-tuned on the full set (#169).
    """

    model_id = "lrp-rli-gbl-013"
    target_var = V.NONWORD
    description = (
        "LightGBM — non-word reading level predictors "
        "(full predictor set, MAE-tuned, no outlier exclusion)"
    )
    pipeline_cls = LGBMPipeline
    params = _LGBM_MAE_PARAMS
    shap_scatter_specs = DEFAULT_SHAP_SCATTER_SPECS
    notes = (
        "Exploratory model for nonword (level). Fits the full DEFAULT_LEVEL "
        "predictor set (#116 Phase D retired hard feature selection in favour "
        "of full-set ranking); hyperparameters are re-tuned by Optuna on the full set "
        "(150 trials, seed 47; #169). Treat the ranking as "
        "exploratory."
    )
