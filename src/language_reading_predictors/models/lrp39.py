# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
LRP39: Predictors of language sample maximum utterance length level (``lsammax``).

``lsammax`` is the maximum utterance length from a coded sample of
the child's spontaneous connected speech.

The target spans min 1.0, max 13.0, median 5.00, mean 5.22, std
2.18, skew 0.67 (n = 106).

This is an exploratory gradient-boosting discovery model on the
same footing as LRP01–22: it asks how predictable maximum
utterance length is and from what, to inform whether the shared
DAG needs a spontaneous connected speech node. It is not a causal
or intention-to-treat estimate. The language-sample measures are
recorded at t1–t2 only, so this level model is doubly exploratory
(≈106 rows, two waves) and no gain model is fitted. The other
language-sample measures are absent from the default predictor
pool (recorded at t1–t2 only), so this model cannot be carried by
same-instrument siblings.

Uniform feature selection (2026-06-23): reduced from the full
33-predictor DEFAULT_LEVEL set to 5 predictors via a distance-
correlation redundancy filter (dcor >= 0.70) plus an importance
noise-floor cut, then re-tuned. See the SelectionStep below and
"""

from language_reading_predictors.data_variables import Variables as V
from language_reading_predictors.models.base_model import LevelModel
from language_reading_predictors.models.common import SelectionStep, ShapScatterSpec
from language_reading_predictors.models.lgbm_pipeline import LGBMPipeline


# ── predictor selection steps ────────────────────────────────────────────

_SELECTION_STEPS: list[SelectionStep] = [
    SelectionStep(
        removed=[
            V.TIME, V.GROUP, V.AREA, V.GENDER, V.AGE, V.APTINFO,
            V.B1EXTO, V.CELF, V.ERBWORD, V.NONWORD, V.BLENDING,
            V.ROWPVT, V.SPPHON, V.TROG, V.YARCLET, V.YARCSI, V.DEAPPVO,
            V.DEAPPFI, V.EWRSWR, V.BEHAV, V.AGESPEAK, V.VISION,
            V.HEARING, V.EARINF, V.NUMCHIL, V.AGEBOOKS, V.MUMEDUPOST16,
            V.DADEDUPOST16
        ],
        notes=(
            "Uniform feature selection (2026-06-23): from the full 33-predictor DEFAULT_LEVEL set, a distance-correlation redundancy filter (dcor >= 0.70, keep the highest out-of-fold permutation-importance representative) plus an importance noise-floor cut (<= 0.005). Reduces to 5 predictors with no dcor >= 0.70 pairs remaining; re-tuned on the reduced set (Optuna 150-trial MAE, 10-fold GroupKFold, seed 47). Same method as the LRP01–22 suite; scripts/rank_predictors.py (the full-set ranking that supersedes the retired hard-selection pass)."
        ),
        date="2026-06-23",
        metrics_before={"cv_mae_mean": 1.3839},
        metrics_after={"cv_mae_mean": 1.3509},
    ),
]


# ── hyperparameters (MAE-tuned on the reduced set) ───────────────────────

_LGBM_MAE_PARAMS: dict[str, float | int | str] = {
    "objective": "mae",
    "learning_rate": 0.1329073999399487,
    "num_leaves": 53,
    "max_depth": 6,
    "min_child_samples": 12,
    "subsample": 0.7884118562815289,
    "colsample_bytree": 0.8063199668371345,
    "reg_alpha": 0.002165193941670985,
    "reg_lambda": 0.6271374829795932,
    "subsample_freq": 1,
    "n_jobs": -1,
    "verbosity": -1,
    "random_state": 47,
    "n_estimators": 31,
}


class LRP39(LevelModel):
    """language sample maximum utterance length level predictors — baseline (MAE-tuned)."""

    model_id = "lrp39"
    target_var = V.LSAMMAX
    description = (
        "LightGBM — language sample maximum utterance length level predictors (5 predictors, MAE-tuned, no outlier exclusion)"
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
        "Exploratory model for lsammax (level). Uniform feature selection (2026-06-23) from the full 33-predictor DEFAULT_LEVEL set to 5 predictors (distance-correlation redundancy filter + importance noise-floor cut; no dcor >= 0.70 pairs remain), re-tuned on the reduced set (tuner-inner CV MAE 1.381). Treat the reduced ranking as exploratory."
    )
