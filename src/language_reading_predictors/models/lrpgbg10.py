# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
LRPGBG10: Predictors of phoneme-blending gains.

``LRPGBG10`` is the baseline exploratory model for phoneme-blending
gains (``blending_gain``). ``blending`` is a phoneme-blending
(phonological awareness) score on a 0–10 scale: the child selects
which of three pictures depicts the word formed by segmented
phonemes spoken by the examiner.

The target is **mildly right-skewed** (``blending_gain`` min −5,
max 7, median 0, mean 0.48, std 2.15, skewness 0.51, with ~35%
negative and ~20% zero observations, n ≈ 161). Similar in shape
to LRPGBG05 (``rowpvt_gain``, skew 0.04) but with heavier zero
pile-up.

Uniform feature selection (2026-06-21): reduced from the full 34-predictor set to 2 predictors via a distance-correlation redundancy filter plus an importance noise-floor cut, then re-tuned. See the SelectionStep below.
"""

from language_reading_predictors.data_variables import Variables as V
from language_reading_predictors.models.base_model import GainModel
from language_reading_predictors.models.common import DEFAULT_SHAP_SCATTER_SPECS, SelectionStep
from language_reading_predictors.models.lgbm_pipeline import LGBMPipeline


# ── predictor selection steps (shared by all variants) ───────────────────
#
# Feature selection (2026-06-21 uniform): distance-correlation
# redundancy filter + importance noise-floor cut; see the SelectionStep.

_SELECTION_STEPS: list[SelectionStep] = [
    SelectionStep(
        removed=[
            V.ATTEND, V.DEAPPIN, V.ERBWORD, V.APTINFO, V.MUMEDUPOST16,
            V.DADEDUPOST16, V.SPPHON, V.AREA, V.VISION, V.AGESPEAK, V.EARINF,
            V.GENDER, V.AGEBOOKS, V.NUMCHIL, V.HEARING, V.BEHAV, V.GROUP, V.YARCSI,
            V.NONWORD, V.CELF, V.APTGRAM, V.TIME, V.ERBNW, V.EWRSWR, V.AGE,
            V.YARCLET, V.B1RETO, V.TROG, V.DEAPPFI, V.DEAPPVO, V.B1EXTO, V.ROWPVT
        ],
        notes=(
            "Uniform feature selection (2026-06-21): from the full 34-predictor set, a distance-correlation redundancy filter (dcor >= 0.70, keep the highest out-of-fold permutation-importance representative) plus an importance noise-floor cut (<= 0.005). The baseline measure was force-kept (regression-to-the-mean anchor). Reduces to 2 predictors with no dcor >= 0.70 pairs remaining; re-tuned on the reduced set (Optuna 150-trial MAE, 10-fold GroupKFold, seed 47). Applied uniformly across all GB models."
        ),
        date="2026-06-21",
        metrics_before={"cv_mae_mean": 1.5515},
        metrics_after={"cv_mae_mean": 1.4284},
    ),
]


# ── hyperparameter sets ─────────────────────────────────────────────────

# MAE-tuned on the 2-predictor uniform-selected set (Optuna 150
# trials, 10-split GroupKFold, seed 47, scoring=mae, lgbm_objective=mae).
# Tuner-inner CV MAE 1.4284.
_LGBM_MAE_PARAMS: dict[str, float | int | str] = {
    "objective": "mae",
    "n_estimators": 19,
    "learning_rate": 0.1222943988580933,
    "num_leaves": 49,
    "max_depth": 10,
    "min_child_samples": 17,
    "subsample": 0.7939584542266895,
    "subsample_freq": 1,
    "colsample_bytree": 0.8932840140004359,
    "reg_alpha": 0.3614435822731186,
    "reg_lambda": 0.6146738201364021,
    "n_jobs": -1,
    "verbosity": -1,
}


# ── primary model (baseline, MAE-tuned) ─────────────────────────────────


class LRPGBG10(GainModel):
    """Phoneme-blending gain predictors — baseline (all data, MAE-tuned).

    Uses a feature-selected subset of :attr:`Predictors.DEFAULT_GAIN`
    (``blending`` is already a member, so the GainModel auto-include
    is a no-op) with MAE-tuned hyperparameters and no outlier
    exclusion. Feature selection was applied (2026-06-21 uniform); see the SelectionStep and the module docstring.
    """

    model_id = "lrpgbg10"
    target_var = V.BLENDING_GAIN
    description = (
        "LightGBM — phoneme-blending gain predictors "
        "(2 predictors, MAE-tuned, no outlier exclusion)"
    )
    pipeline_cls = LGBMPipeline
    params = _LGBM_MAE_PARAMS
    selection_steps = _SELECTION_STEPS
    shap_scatter_specs = DEFAULT_SHAP_SCATTER_SPECS
    notes = (
        "Exploratory model for blending_gain (gain). Uniform feature selection (2026-06-21) from the full 34-predictor DEFAULT_GAIN set to 2 predictors (distance-correlation redundancy filter + importance noise-floor cut; baseline force-kept; no dcor >= 0.70 pairs remain), re-tuned on the reduced set (tuner-inner CV MAE 1.551 -> 1.428). Gain models are near-noise (baseline-driven regression to the mean) - treat the reduced ranking as exploratory."
    )
