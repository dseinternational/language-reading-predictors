# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
LRPGBL16: Predictors of DEAP fine-articulation level.

``LRPGBL16`` is the baseline exploratory model for DEAP fine-
articulation level (``deappfi``). ``deappfi`` is a
percentage-scale articulation measure from the Diagnostic
Evaluation of Articulation and Phonology (Dodd et al., 2006) —
the proportion of sounds correctly produced when the child is
asked to name pictures. ``deappfi`` specifically scores the
*final* consonant of each word (distinct from ``deappin``
initial and ``deappvo`` voicing).

The target is **left-skewed with ceiling** (``deappfi`` min
5.4, max 95.2, median 66.6, mean 60.3, std 20.9, skewness
**−0.87**, n ≈ 207). Ceiling effects are possible at the top
end — several children score in the 90s. No zeros in the
sample (unlike the reading targets where floor effects
dominate).

DEAP measures have been used as predictors across every other
model in the suite but never as targets until LRPGBL16/22. First
articulation-domain target.
"""

from language_reading_predictors.data_variables import Variables as V
from language_reading_predictors.models.base_model import LevelModel
from language_reading_predictors.models.common import DEFAULT_SHAP_SCATTER_SPECS
from language_reading_predictors.models.lgbm_pipeline import LGBMPipeline


# ── hyperparameter sets ─────────────────────────────────────────────────

# MAE-tuned by Optuna on the full predictor set (150 trials, seed 47;
# #169 retune, superseding the earlier pruned-set tune).
_LGBM_MAE_PARAMS: dict[str, float | int | str] = {
    "objective": "mae",
    "n_estimators": 137,
    "learning_rate": 0.04651737709141136,
    "num_leaves": 18,
    "max_depth": 11,
    "min_child_samples": 18,
    "subsample": 0.9748319860377574,
    "subsample_freq": 1,
    "colsample_bytree": 0.8739024256791915,
    "reg_alpha": 0.003204453724037499,
    "reg_lambda": 6.290944046440354,
    "n_jobs": -1,
    "verbosity": -1,
}


# ── primary model (baseline, MAE-tuned) ─────────────────────────────────


class LRPGBL16(LevelModel):
    """DEAP fine-articulation level predictors — baseline (all data, MAE-tuned).

    Full ``Predictors.DEFAULT_LEVEL`` set, MAE-tuned on the full set (#169).
    """

    model_id = "lrp-rli-gbl-016"
    target_var = V.DEAPPFI
    description = (
        "LightGBM — DEAP fine-articulation level predictors "
        "(full predictor set, MAE-tuned, no outlier exclusion)"
    )
    pipeline_cls = LGBMPipeline
    params = _LGBM_MAE_PARAMS
    shap_scatter_specs = DEFAULT_SHAP_SCATTER_SPECS
    notes = (
        "Exploratory model for deappfi (level). Fits the full DEFAULT_LEVEL predictor set (#116 Phase D retired hard feature selection in favour of full-set ranking); hyperparameters were re-tuned by Optuna on the full set (150 trials, seed 47; #169). Treat the ranking as exploratory."
    )
