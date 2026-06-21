# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
LRP19: Predictors of expressive-information (APT) gains.

``LRP19`` is the baseline exploratory model for expressive-
information gains (``aptinfo_gain``). ``aptinfo`` is the
information raw score from the Action Picture Test (Renfrew,
1997): the child is shown pictures and asked to describe them,
with scoring of the information content of the response (as
distinct from its grammatical structure, which is scored
separately as ``aptgram`` — LRP17/18).

The target is mildly right-skewed (``aptinfo_gain`` min −7,
max 16, median 2.5, mean 2.61, std 4.44, skewness 0.25, with
~29% negative and ~4% zero observations, n ≈ 160). The low
zero-mass is unusual — most children show measurable change
from timepoint to timepoint (cf LRP11 `trog_gain` ~8% zero,
LRP17 `aptgram_gain` ~11% zero, LRP13 `nonword_gain` ~48%
zero).

Feature selection applied 2026-06-20 (replication): reduced from the full 34-predictor set to 5 predictors via a distance-correlation redundancy filter (dcor >= 0.70, keep the highest-importance representative) plus an importance noise-floor cut, then re-tuned on the reduced set. See the SelectionStep below and notes/202606201500-gb-replication-findings.md.
"""

from language_reading_predictors.data_variables import Variables as V
from language_reading_predictors.models.base_model import GainModel
from language_reading_predictors.models.common import SelectionStep, ShapScatterSpec
from language_reading_predictors.models.lgbm_pipeline import LGBMPipeline


# ── predictor selection steps (shared by all variants) ───────────────────
#
# Feature selection (2026-06-20 replication): distance-correlation
# redundancy filter + importance noise-floor cut; see the SelectionStep.

_SELECTION_STEPS: list[SelectionStep] = [
    SelectionStep(
        removed=[
            V.B1RETO, V.APTGRAM, V.CELF, V.EWRSWR, V.AGEBOOKS, V.YARCSI, V.EARINF,
            V.DADEDUPOST16, V.NUMCHIL, V.GENDER, V.VISION, V.GROUP, V.AREA,
            V.AGESPEAK, V.MUMEDUPOST16, V.BEHAV, V.HEARING, V.DEAPPFI, V.TIME,
            V.EOWPVT, V.DEAPPIN, V.ROWPVT, V.NONWORD, V.YARCLET, V.BLENDING,
            V.ERBNW, V.AGE, V.B1EXTO, V.ATTEND
        ],
        notes=(
            "Uniform feature selection (2026-06-21): from the full 34-predictor set, a distance-correlation redundancy filter (dcor >= 0.70, keep the highest out-of-fold permutation-importance representative) plus an importance noise-floor cut (<= 0.005). The baseline measure was force-kept (regression-to-the-mean anchor). Reduces to 5 predictors with no dcor >= 0.70 pairs remaining; re-tuned on the reduced set (Optuna 150-trial MAE, 10-fold GroupKFold, seed 47). Applied uniformly across all GB models; see notes/202606211200-uniform-gb-fs.md."
        ),
        date="2026-06-21",
        metrics_before={"cv_mae_mean": 3.4050},
        metrics_after={"cv_mae_mean": 3.1504},
    ),
]


# ── hyperparameter sets ─────────────────────────────────────────────────

# MAE-tuned on the 5-predictor uniform-selected set (Optuna 150
# trials, 10-split GroupKFold, seed 47, scoring=mae, lgbm_objective=mae).
# Tuner-inner CV MAE 3.1504.
_LGBM_MAE_PARAMS: dict[str, float | int | str] = {
    "objective": "mae",
    "n_estimators": 275,
    "learning_rate": 0.03817935546050526,
    "num_leaves": 29,
    "max_depth": 12,
    "min_child_samples": 18,
    "subsample": 0.6220122627331144,
    "subsample_freq": 1,
    "colsample_bytree": 0.6383469112272381,
    "reg_alpha": 0.3358114008334534,
    "reg_lambda": 0.06761501795300534,
    "n_jobs": -1,
    "verbosity": -1,
}


# ── primary model (baseline, MAE-tuned) ─────────────────────────────────


class LRP19(GainModel):
    """APT expressive-information gain predictors — baseline (all data, MAE-tuned).

    Uses a feature-selected subset of :attr:`Predictors.DEFAULT_GAIN`
    (``aptinfo`` is already a member, so the GainModel auto-include
    is a no-op) with MAE-tuned hyperparameters and no outlier
    exclusion. Feature selection was applied (2026-06-20 replication); see the SelectionStep and the module docstring.
    """

    model_id = "lrp19"
    target_var = V.APTINFO_GAIN
    description = (
        "LightGBM — APT expressive-information gain predictors "
        "(5 predictors, MAE-tuned, no outlier exclusion)"
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
        "Exploratory model for aptinfo_gain (gain). Feature-selected (2026-06-20 replication) from the full 34-predictor default set to 5 predictors via a distance-correlation redundancy filter (no dcor >= 0.70 pairs remain) plus an importance noise-floor cut, then re-tuned on the reduced set (tuner-inner CV MAE 3.299 -> 3.086). Only the dominant predictor is robustly above the importance noise floor; treat the reduced ranking as exploratory. See the SelectionStep and notes/202606201500-gb-replication-findings.md."
    )
