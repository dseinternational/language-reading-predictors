# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
LRP40: Predictors of language sample intelligibility level (``lsamint``).

``lsamint`` is the percentage of intelligible words from a coded
sample of the child's spontaneous connected speech.

The target spans min 18.4, max 100.0, median 82.47, mean 77.95,
std 19.02, skew -1.07 (n = 106).

This is an exploratory gradient-boosting discovery model on the
same footing as LRP01–22: it asks how predictable intelligibility
is and from what, to inform whether the shared DAG needs a
spontaneous connected speech node. It is not a causal or
intention-to-treat estimate. The language-sample measures are
recorded at t1–t2 only, so this level model is doubly exploratory
(≈106 rows, two waves) and no gain model is fitted. The other
language-sample measures are absent from the default predictor
pool (recorded at t1–t2 only), so this model cannot be carried by
same-instrument siblings.

Uniform feature selection (2026-06-23): reduced from the full
33-predictor DEFAULT_LEVEL set to 13 predictors via a distance-
correlation redundancy filter (dcor >= 0.70) plus an importance
noise-floor cut, then re-tuned. See the SelectionStep below and
notes/202606230900-predictability-speech-memory-language.md.
"""

from language_reading_predictors.data_variables import Variables as V
from language_reading_predictors.models.base_model import LevelModel
from language_reading_predictors.models.common import SelectionStep, ShapScatterSpec
from language_reading_predictors.models.lgbm_pipeline import LGBMPipeline


# ── predictor selection steps ────────────────────────────────────────────

_SELECTION_STEPS: list[SelectionStep] = [
    SelectionStep(
        removed=[
            V.GROUP, V.GENDER, V.APTGRAM, V.B1EXTO, V.B1RETO, V.EOWPVT,
            V.ERBNW, V.ERBWORD, V.NONWORD, V.BLENDING, V.SPPHON,
            V.YARCLET, V.DEAPPFI, V.EWRSWR, V.BEHAV, V.VISION, V.EARINF,
            V.NUMCHIL, V.MUMEDUPOST16, V.DADEDUPOST16
        ],
        notes=(
            "Uniform feature selection (2026-06-23): from the full 33-predictor DEFAULT_LEVEL set, a distance-correlation redundancy filter (dcor >= 0.70, keep the highest out-of-fold permutation-importance representative) plus an importance noise-floor cut (<= 0.005). Reduces to 13 predictors with no dcor >= 0.70 pairs remaining; re-tuned on the reduced set (Optuna 150-trial MAE, 10-fold GroupKFold, seed 47). Same method as the LRP01–22 suite; see scripts/uniform_feature_selection.py and notes/202606230900-predictability-speech-memory-language.md."
        ),
        date="2026-06-23",
        metrics_before={"cv_mae_mean": 11.6537},
        metrics_after={"cv_mae_mean": 10.4001},
    ),
]


# ── hyperparameters (MAE-tuned on the reduced set) ───────────────────────

_LGBM_MAE_PARAMS: dict[str, float | int | str] = {
    "objective": "mae",
    "learning_rate": 0.10001913995922147,
    "num_leaves": 45,
    "max_depth": 9,
    "min_child_samples": 11,
    "subsample": 0.6583352046429758,
    "colsample_bytree": 0.7591115445321488,
    "reg_alpha": 0.20438419834894778,
    "reg_lambda": 0.01450279715697431,
    "subsample_freq": 1,
    "n_jobs": -1,
    "verbosity": -1,
    "random_state": 47,
    "n_estimators": 92,
}


class LRP40(LevelModel):
    """language sample intelligibility level predictors — baseline (MAE-tuned)."""

    model_id = "lrp40"
    target_var = V.LSAMINT
    description = (
        "LightGBM — language sample intelligibility level predictors (13 predictors, MAE-tuned, no outlier exclusion)"
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
        "Exploratory model for lsamint (level). Uniform feature selection (2026-06-23) from the full 33-predictor DEFAULT_LEVEL set to 13 predictors (distance-correlation redundancy filter + importance noise-floor cut; no dcor >= 0.70 pairs remain), re-tuned on the reduced set (tuner-inner CV MAE 9.726). Treat the reduced ranking as exploratory. See notes/202606230900-predictability-speech-memory-language.md."
    )
