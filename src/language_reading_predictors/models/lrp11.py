# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
LRP11: Predictors of receptive-grammar (TROG-2) gains.

``LRP11`` is the baseline exploratory model for receptive-grammar
gains (``trog_gain``). The ``trog`` score is the items-correct
total from the Test for Reception of Grammar 2 (TROG-2; Bishop
2003), covering eight grammatical constructs in blocks of four
items (32 items total; observed max 27 in this sample).

The target is mildly left-skewed (``trog_gain`` min ≈ −10,
max ≈ 12, median 2, mean 1.22, std 4.19, skewness −0.17, with
~34% negative and ~8% zero observations, n ≈ 161). Closer in
shape to LRP07 (``rowpvt_gain``, skew 0.04) and LRP05
(``yarclet_gain``, skew 0.45) than to the heavier-skewed gain
targets.

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
            V.TIME, V.GROUP, V.AREA, V.GENDER, V.AGE, V.APTGRAM, V.APTINFO,
            V.B1EXTO, V.B1RETO, V.ERBNW, V.ERBWORD, V.NONWORD, V.BLENDING, V.ROWPVT,
            V.SPPHON, V.YARCLET, V.YARCSI, V.DEAPPIN, V.EWRSWR, V.BEHAV, V.ATTEND,
            V.AGESPEAK, V.VISION, V.HEARING, V.EARINF, V.NUMCHIL, V.AGEBOOKS,
            V.MUMEDUPOST16, V.DADEDUPOST16
        ],
        notes=(
            "Feature selection (replication, 2026-06-20): from the full 34-predictor set, a distance-correlation filter (dcor >= 0.70, keep the highest out-of-fold permutation-importance representative per cluster) plus removal of features at/below the 0.005 importance floor. Reduces to 5 predictors with no dcor >= 0.70 pairs remaining; pooled refit-CV held under matched hyperparameters, then the set was re-tuned. See notes/202606201500-gb-replication-findings.md."
        ),
        date="2026-06-20",
        metrics_before={"cv_mae_mean": 3.1184},
        metrics_after={"cv_mae_mean": 2.9412},
    ),
]


# ── hyperparameter sets ─────────────────────────────────────────────────

# MAE-tuned on the 5-predictor replication-selected set, no outlier
# exclusion (Optuna 150 trials, 10-split GroupKFold, seed 47, scoring=mae,
# lgbm_objective=mae). Tuner-inner CV MAE 2.9412. Supersedes the full-set tune.
_LGBM_MAE_PARAMS: dict[str, float | int | str] = {
    "objective": "mae",
    "n_estimators": 30,
    "learning_rate": 0.18307068825254422,
    "num_leaves": 12,
    "max_depth": 11,
    "min_child_samples": 14,
    "subsample": 0.6266784471053202,
    "subsample_freq": 1,
    "colsample_bytree": 0.6287939774287393,
    "reg_alpha": 0.018270329387307327,
    "reg_lambda": 2.1050936189462623,
    "n_jobs": -1,
    "verbosity": -1,
}


# ── primary model (baseline, MAE-tuned) ─────────────────────────────────


class LRP11(GainModel):
    """TROG-2 receptive-grammar gain predictors — baseline (all data, MAE-tuned).

    Uses a feature-selected subset of :attr:`Predictors.DEFAULT_GAIN`
    (``trog`` is already a member, so the GainModel auto-include
    is a no-op) with MAE-tuned hyperparameters and no outlier
    exclusion. Feature selection was applied (2026-06-20 replication); see the SelectionStep and the module docstring.
    """

    model_id = "lrp11"
    target_var = V.TROG_GAIN
    description = (
        "LightGBM — TROG-2 (receptive grammar) gain predictors "
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
        "Exploratory model for trog_gain (gain). Feature-selected (2026-06-20 replication) from the full 34-predictor default set to 5 predictors via a distance-correlation redundancy filter (no dcor >= 0.70 pairs remain) plus an importance noise-floor cut, then re-tuned on the reduced set (tuner-inner CV MAE 3.118 -> 2.941). Only the dominant predictor is robustly above the importance noise floor; treat the reduced ranking as exploratory. See the SelectionStep and notes/202606201500-gb-replication-findings.md."
    )
