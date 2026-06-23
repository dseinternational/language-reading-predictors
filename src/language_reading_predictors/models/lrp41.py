# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
LRP41: Predictors of language sample unique words level (``lsamun``).

``lsamun`` is the number of unique words (lexical diversity) from
a coded sample of the child's spontaneous connected speech.

The target spans min 1.0, max 86.0, median 36.00, mean 36.13, std
20.73, skew 0.32 (n = 106).

This is an exploratory gradient-boosting discovery model on the
same footing as LRP01–22: it asks how predictable unique words is
and from what, to inform whether the shared DAG needs a
spontaneous connected speech node. It is not a causal or
intention-to-treat estimate. The language-sample measures are
recorded at t1–t2 only, so this level model is doubly exploratory
(≈106 rows, two waves) and no gain model is fitted. The other
language-sample measures are absent from the default predictor
pool (recorded at t1–t2 only), so this model cannot be carried by
same-instrument siblings.

Uniform feature selection (2026-06-23): reduced from the full
33-predictor DEFAULT_LEVEL set to 10 predictors via a distance-
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
            V.GROUP, V.AREA, V.GENDER, V.AGE, V.APTGRAM, V.B1EXTO,
            V.B1RETO, V.CELF, V.EOWPVT, V.ERBWORD, V.BLENDING, V.SPPHON,
            V.YARCLET, V.YARCSI, V.DEAPPFI, V.BEHAV, V.AGESPEAK,
            V.VISION, V.HEARING, V.NUMCHIL, V.AGEBOOKS, V.MUMEDUPOST16,
            V.DADEDUPOST16
        ],
        notes=(
            "Uniform feature selection (2026-06-23): from the full 33-predictor DEFAULT_LEVEL set, a distance-correlation redundancy filter (dcor >= 0.70, keep the highest out-of-fold permutation-importance representative) plus an importance noise-floor cut (<= 0.005). Reduces to 10 predictors with no dcor >= 0.70 pairs remaining; re-tuned on the reduced set (Optuna 150-trial MAE, 10-fold GroupKFold, seed 47). Same method as the LRP01–22 suite; see scripts/uniform_feature_selection.py and notes/202606230900-predictability-speech-memory-language.md."
        ),
        date="2026-06-23",
        metrics_before={"cv_mae_mean": 11.6995},
        metrics_after={"cv_mae_mean": 11.0723},
    ),
]


# ── hyperparameters (MAE-tuned on the reduced set) ───────────────────────

_LGBM_MAE_PARAMS: dict[str, float | int | str] = {
    "objective": "mae",
    "learning_rate": 0.035596364899835926,
    "num_leaves": 46,
    "max_depth": 4,
    "min_child_samples": 8,
    "subsample": 0.6662442289300283,
    "colsample_bytree": 0.6969455673091792,
    "reg_alpha": 0.001914627094005278,
    "reg_lambda": 0.05464962803233622,
    "subsample_freq": 1,
    "n_jobs": -1,
    "verbosity": -1,
    "random_state": 47,
    "n_estimators": 186,
}


class LRP41(LevelModel):
    """language sample unique words level predictors — baseline (MAE-tuned)."""

    model_id = "lrp41"
    target_var = V.LSAMUN
    description = (
        "LightGBM — language sample unique words level predictors (10 predictors, MAE-tuned, no outlier exclusion)"
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
        "Exploratory model for lsamun (level). Uniform feature selection (2026-06-23) from the full 33-predictor DEFAULT_LEVEL set to 10 predictors (distance-correlation redundancy filter + importance noise-floor cut; no dcor >= 0.70 pairs remain), re-tuned on the reduced set (tuner-inner CV MAE 10.839). Treat the reduced ranking as exploratory. See notes/202606230900-predictability-speech-memory-language.md."
    )
