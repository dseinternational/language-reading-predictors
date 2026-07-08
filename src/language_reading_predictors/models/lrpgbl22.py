# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
LRPGBL22: Predictors of DEAP average articulation level (``deappav``).

``deappav`` is the DEAP picture-naming average from the Diagnostic
Evaluation of Articulation and Phonology (Dodd et al., 2006) — the
proportion of sounds correctly produced in a picture-naming task.
It is a composite — ``deappin``, ``deappvo``, ``deappfi`` are its
components and remain in the candidate pool, so a high naive R² is
mechanical (see the same-skill-excluded ranking view, ``ranking_excluding_same_skill.csv``).

The target spans min 47.1, max 94.9, median 77.70, mean 75.27, std
11.30, skew -0.70 (n = 207).

This is an exploratory gradient-boosting discovery model on the
same footing as LRPGBG12–22: it asks how predictable average
articulation is and from what, to inform whether the shared DAG
needs a speech-sound accuracy node. It is not a causal or
intention-to-treat estimate.
"""

from language_reading_predictors.data_variables import Variables as V
from language_reading_predictors.models.base_model import LevelModel
from language_reading_predictors.models.common import ShapScatterSpec
from language_reading_predictors.models.lgbm_pipeline import LGBMPipeline



# ── hyperparameters (MAE-tuned) ──────────────────────────────────────────

_LGBM_MAE_PARAMS: dict[str, float | int | str] = {
    "objective": "mae",
    "learning_rate": 0.039654383749487036,
    "num_leaves": 41,
    "max_depth": 12,
    "min_child_samples": 4,
    "subsample": 0.7337465509959112,
    "colsample_bytree": 0.9969182249863433,
    "reg_alpha": 0.0014782261853234005,
    "reg_lambda": 7.04330979395619,
    "subsample_freq": 1,
    "n_jobs": -1,
    "verbosity": -1,
    "random_state": 47,
    "n_estimators": 197,
}


class LRPGBL22(LevelModel):
    """DEAP average articulation level predictors — baseline (MAE-tuned)."""

    model_id = "lrpgbl22"
    target_var = V.DEAPPAV
    description = (
        "LightGBM — DEAP average articulation level predictors (full predictor set, MAE-tuned, no outlier exclusion)"
    )
    pipeline_cls = LGBMPipeline
    params = _LGBM_MAE_PARAMS
    cv_splits = 51
    outlier_threshold = None
    shap_scatter_specs = (
        ShapScatterSpec(description="All predictors, SHAP auto-colouring"),
    )
    notes = (
        "Exploratory model for deappav (level). Fits the full DEFAULT_LEVEL predictor set (#116 Phase D retired hard feature selection in favour of full-set ranking); hyperparameters are retained from the earlier pruned-set Optuna tune (retune-pending). Treat the ranking as exploratory."
    )

