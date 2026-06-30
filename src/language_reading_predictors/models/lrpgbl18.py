# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
LRPGBL18: Predictors of Early Repetition Battery word repetition level (``erbword``).

``erbword`` is the number of real words correctly repeated from
the Early Repetition Battery (a repetition task indexing verbal /
phonological short-term memory).

The target spans min 0.0, max 28.0, median 12.00, mean 11.35, std
5.22, skew -0.25 (n = 203).

This is an exploratory gradient-boosting discovery model on the
same footing as LRPGBG12–22: it asks how predictable word repetition
is and from what, to inform whether the shared DAG needs a verbal
/ phonological short-term memory node. It is not a causal or
intention-to-treat estimate.

Uniform feature selection (2026-06-23): reduced from the full
32-predictor DEFAULT_LEVEL set to 12 predictors via a distance-
correlation redundancy filter (dcor >= 0.70) plus an importance
noise-floor cut, then re-tuned. See the SelectionStep below.
"""

from language_reading_predictors.data_variables import Variables as V
from language_reading_predictors.models.base_model import LevelModel
from language_reading_predictors.models.common import SelectionStep, ShapScatterSpec
from language_reading_predictors.models.lgbm_pipeline import LGBMPipeline


# ── predictor selection steps ────────────────────────────────────────────

_SELECTION_STEPS: list[SelectionStep] = [
    SelectionStep(
        removed=[
            V.TIME, V.GROUP, V.AREA, V.GENDER, V.AGE, V.APTGRAM,
            V.B1EXTO, V.B1RETO, V.CELF, V.EOWPVT, V.SPPHON, V.YARCLET,
            V.DEAPPFI, V.BEHAV, V.AGESPEAK, V.VISION, V.HEARING,
            V.NUMCHIL, V.AGEBOOKS, V.DADEDUPOST16
        ],
        notes=(
            "Uniform feature selection (2026-06-23): from the full 32-predictor DEFAULT_LEVEL set, a distance-correlation redundancy filter (dcor >= 0.70, keep the highest out-of-fold permutation-importance representative) plus an importance noise-floor cut (<= 0.005). Reduces to 12 predictors with no dcor >= 0.70 pairs remaining; re-tuned on the reduced set (Optuna 150-trial MAE, 10-fold GroupKFold, seed 47). Same method as the LRPGBG12–22 suite; see scripts/rank_predictors.py (the full-set ranking that supersedes the retired hard-selection pass)."
        ),
        date="2026-06-23",
        metrics_before={"cv_mae_mean": 1.9933},
        metrics_after={"cv_mae_mean": 1.9015},
    ),
]


# ── hyperparameters (MAE-tuned on the reduced set) ───────────────────────

_LGBM_MAE_PARAMS: dict[str, float | int | str] = {
    "objective": "mae",
    "learning_rate": 0.12648578524796544,
    "num_leaves": 10,
    "max_depth": 4,
    "min_child_samples": 6,
    "subsample": 0.6442114045867877,
    "colsample_bytree": 0.6424565278401814,
    "reg_alpha": 0.1708379704840523,
    "reg_lambda": 0.018815743974597524,
    "subsample_freq": 1,
    "n_jobs": -1,
    "verbosity": -1,
    "random_state": 47,
    "n_estimators": 133,
}


class LRPGBL18(LevelModel):
    """Early Repetition Battery word repetition level predictors — baseline (MAE-tuned)."""

    model_id = "lrpgbl18"
    target_var = V.ERBWORD
    description = (
        "LightGBM — Early Repetition Battery word repetition level predictors (12 predictors, MAE-tuned, no outlier exclusion)"
    )
    pipeline_cls = LGBMPipeline
    params = _LGBM_MAE_PARAMS
    cv_splits = 51
    outlier_threshold = None
    selection_steps = _SELECTION_STEPS
    shap_scatter_specs = [
        ShapScatterSpec(description="All predictors, SHAP auto-colouring"),
    ]
    notes = (
        "Exploratory model for erbword (level). Uniform feature selection (2026-06-23) from the full 32-predictor DEFAULT_LEVEL set to 12 predictors (distance-correlation redundancy filter + importance noise-floor cut; no dcor >= 0.70 pairs remain), re-tuned on the reduced set (tuner-inner CV MAE 1.961). Treat the reduced ranking as exploratory."
    )

