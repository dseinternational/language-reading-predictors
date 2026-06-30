# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
LRPGBL19: Predictors of Early Repetition Battery total repetition level (``erbto``).

``erbto`` is the ERB total score from the Early Repetition Battery
(a repetition task indexing verbal / phonological short-term
memory). It is a composite — ``erbword``, ``erbnw`` are its
components and remain in the candidate pool, so a high naive R² is
mechanical (see the same-skill-excluded ranking view, ``ranking_excluding_same_skill.csv``).

The target spans min 1.0, max 36.0, median 21.00, mean 20.17, std
9.47, skew -0.21 (n = 202).

This is an exploratory gradient-boosting discovery model on the
same footing as LRPGBG12–22: it asks how predictable total repetition
is and from what, to inform whether the shared DAG needs a verbal
/ phonological short-term memory node. It is not a causal or
intention-to-treat estimate.

Uniform feature selection (2026-06-23): reduced from the full
33-predictor DEFAULT_LEVEL set to 13 predictors via a distance-
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
            V.TIME, V.GROUP, V.AREA, V.APTINFO, V.B1EXTO, V.ERBNW,
            V.NONWORD, V.ROWPVT, V.SPPHON, V.DEAPPIN, V.DEAPPFI,
            V.EWRSWR, V.BEHAV, V.AGESPEAK, V.VISION, V.HEARING,
            V.EARINF, V.NUMCHIL, V.AGEBOOKS, V.MUMEDUPOST16
        ],
        notes=(
            "Uniform feature selection (2026-06-23): from the full 33-predictor DEFAULT_LEVEL set, a distance-correlation redundancy filter (dcor >= 0.70, keep the highest out-of-fold permutation-importance representative) plus an importance noise-floor cut (<= 0.005). Reduces to 13 predictors with no dcor >= 0.70 pairs remaining; re-tuned on the reduced set (Optuna 150-trial MAE, 10-fold GroupKFold, seed 47). Same method as the LRPGBG12–22 suite; see scripts/rank_predictors.py (the full-set ranking that supersedes the retired hard-selection pass)."
        ),
        date="2026-06-23",
        metrics_before={"cv_mae_mean": 0.8413},
        metrics_after={"cv_mae_mean": 2.3414},
    ),
]


# ── hyperparameters (MAE-tuned on the reduced set) ───────────────────────

_LGBM_MAE_PARAMS: dict[str, float | int | str] = {
    "objective": "mae",
    "learning_rate": 0.05596669476177882,
    "num_leaves": 30,
    "max_depth": 3,
    "min_child_samples": 9,
    "subsample": 0.7093898229449795,
    "colsample_bytree": 0.93700430573943,
    "reg_alpha": 1.650810077584773,
    "reg_lambda": 0.03392347265840604,
    "subsample_freq": 1,
    "n_jobs": -1,
    "verbosity": -1,
    "random_state": 47,
    "n_estimators": 150,
}


class LRPGBL19(LevelModel):
    """Early Repetition Battery total repetition level predictors — baseline (MAE-tuned)."""

    model_id = "lrpgbl19"
    target_var = V.ERBTO
    description = (
        "LightGBM — Early Repetition Battery total repetition level predictors (13 predictors, MAE-tuned, no outlier exclusion)"
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
        "Exploratory model for erbto (level). Uniform feature selection (2026-06-23) from the full 33-predictor DEFAULT_LEVEL set to 13 predictors (distance-correlation redundancy filter + importance noise-floor cut; no dcor >= 0.70 pairs remain), re-tuned on the reduced set (tuner-inner CV MAE 2.321). Treat the reduced ranking as exploratory."
    )

