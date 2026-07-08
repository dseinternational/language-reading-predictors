# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
LRPGBL07: Predictors of expressive-information (APT) level.

``LRPGBL07`` is the baseline exploratory model for expressive-
information level (``aptinfo``). ``aptinfo`` is the information
raw score from the Action Picture Test (Renfrew, 1997): the child
is shown pictures and asked to describe them, with scoring of the
information content of the response (as distinct from its
grammatical structure, which is scored separately as ``aptgram``
— LRPGBG08/18).

The target is **essentially symmetric** (``aptinfo`` min 0,
max 37.5, median 16.5, mean 16.97, std 7.93, skewness 0.24,
with ~1% at zero, n ≈ 214) — one of the cleanest distributions
in the suite, comparable to LRPGBL15 (`trog`, skew 0.29) and
LRPGBL10 (`blending`, skew 0.01) and much cleaner than the paired
LRPGBL08 (`aptgram`, skew 1.23).
"""

from language_reading_predictors.data_variables import Variables as V
from language_reading_predictors.models.base_model import LevelModel
from language_reading_predictors.models.common import DEFAULT_SHAP_SCATTER_SPECS
from language_reading_predictors.models.lgbm_pipeline import LGBMPipeline


# ── hyperparameter sets ─────────────────────────────────────────────────

# MAE-tuned (Optuna 150-trial, seed 47) on the earlier pruned selected set;
# retained as the full-set baseline (retune-pending).
_LGBM_MAE_PARAMS: dict[str, float | int | str] = {
    "objective": "mae",
    "n_estimators": 276,
    "learning_rate": 0.06362816681666994,
    "num_leaves": 47,
    "max_depth": 6,
    "min_child_samples": 7,
    "subsample": 0.9174036085656114,
    "subsample_freq": 1,
    "colsample_bytree": 0.6656142066855005,
    "reg_alpha": 0.06215178250629114,
    "reg_lambda": 0.04469345927561462,
    "n_jobs": -1,
    "verbosity": -1,
}


# ── primary model (baseline, MAE-tuned) ─────────────────────────────────


class LRPGBL07(LevelModel):
    """APT expressive-information level predictors — baseline (all data, MAE-tuned).

    Full ``Predictors.DEFAULT_LEVEL`` set, MAE-tuned (params retune-pending).
    """

    model_id = "lrp-rli-gbl-007"
    target_var = V.APTINFO
    description = (
        "LightGBM — APT expressive-information level predictors "
        "(full predictor set, MAE-tuned, no outlier exclusion)"
    )
    pipeline_cls = LGBMPipeline
    params = _LGBM_MAE_PARAMS
    shap_scatter_specs = DEFAULT_SHAP_SCATTER_SPECS
    notes = (
        "Exploratory model for aptinfo (level). Fits the full DEFAULT_LEVEL "
        "predictor set (#116 Phase D retired hard feature selection in favour "
        "of full-set ranking); hyperparameters are retained from the earlier "
        "pruned-set Optuna tune (retune-pending). Treat the ranking as "
        "exploratory."
    )
