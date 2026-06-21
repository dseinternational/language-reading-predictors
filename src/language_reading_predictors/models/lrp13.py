# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
LRP13: Predictors of non-word reading gains.

``LRP13`` is the baseline exploratory model for non-word reading
gains (``nonword_gain``). ``nonword`` is an items-correct score
from a non-word decoding task (observed range 0–6).

The target is **heavily zero-loaded** (``nonword_gain`` min −6,
max 6, median 0, mean 0.41, std 1.66, skewness 0.40, with ~19%
negative and **~48% zero** observations, n ≈ 153). Half the
children show no change between timepoints — consistent with the
floor-heavy nonword level distribution (57% zero at any given
timepoint). Tree models will struggle to predict non-zero gains
reliably.

Feature selection applied 2026-06-20 (replication): reduced from the full 34-predictor set to 2 predictors via a distance-correlation redundancy filter (dcor >= 0.70, keep the highest-importance representative) plus an importance noise-floor cut, then re-tuned on the reduced set. See the SelectionStep below and notes/202606201500-gb-replication-findings.md.
"""

from language_reading_predictors.data_variables import Variables as V
from language_reading_predictors.models.base_model import GainModel
from language_reading_predictors.models.common import SelectionStep, ShapScatterSpec
from language_reading_predictors.models.lgbm_pipeline import LGBMPipeline


_SELECTION_STEPS: list[SelectionStep] = [
    SelectionStep(
        removed=[
            V.TIME, V.GROUP, V.AREA, V.GENDER, V.AGE, V.APTGRAM, V.APTINFO,
            V.B1EXTO, V.B1RETO, V.CELF, V.EOWPVT, V.ERBWORD, V.BLENDING, V.ROWPVT,
            V.SPPHON, V.TROG, V.YARCLET, V.YARCSI, V.DEAPPIN, V.DEAPPVO, V.DEAPPFI,
            V.EWRSWR, V.BEHAV, V.ATTEND, V.AGESPEAK, V.VISION, V.HEARING, V.EARINF,
            V.NUMCHIL, V.AGEBOOKS, V.MUMEDUPOST16, V.DADEDUPOST16
        ],
        notes=(
            "Feature selection (replication, 2026-06-20): from the full 34-predictor set, a distance-correlation filter (dcor >= 0.70, keep the highest out-of-fold permutation-importance representative per cluster) plus removal of features at/below the 0.005 importance floor. Reduces to 2 predictors with no dcor >= 0.70 pairs remaining; pooled refit-CV held under matched hyperparameters, then the set was re-tuned. See notes/202606201500-gb-replication-findings.md."
        ),
        date="2026-06-20",
        metrics_before={"cv_mae_mean": 0.9726},
        metrics_after={"cv_mae_mean": 0.9273},
    ),
]


# MAE-tuned on the 2-predictor replication-selected set, no outlier
# exclusion (Optuna 150 trials, 10-split GroupKFold, seed 47, scoring=mae,
# lgbm_objective=mae). Tuner-inner CV MAE 0.9273. Supersedes the full-set tune.
_LGBM_MAE_PARAMS: dict[str, float | int | str] = {
    "objective": "mae",
    "n_estimators": 497,
    "learning_rate": 0.10508140765367975,
    "num_leaves": 24,
    "max_depth": 12,
    "min_child_samples": 36,
    "subsample": 0.9087575271581118,
    "subsample_freq": 1,
    "colsample_bytree": 0.6803273216004087,
    "reg_alpha": 5.941599922639021,
    "reg_lambda": 0.021860911996772987,
    "n_jobs": -1,
    "verbosity": -1,
}


class LRP13(GainModel):
    """Non-word reading gain predictors — baseline (all data, MAE-tuned).

    Uses a feature-selected subset of :attr:`Predictors.DEFAULT_GAIN`
    (``nonword`` is already a member) with MAE-tuned hyperparameters
    and no outlier exclusion. Feature selection was applied (2026-06-20 replication); see the SelectionStep and the module docstring.
    """

    model_id = "lrp13"
    target_var = V.NONWORD_GAIN
    description = (
        "LightGBM — non-word reading gain predictors "
        "(2 predictors, MAE-tuned, no outlier exclusion)"
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
        "Exploratory model for nonword_gain (gain). Feature-selected (2026-06-20 replication) from the full 34-predictor default set to 2 predictors via a distance-correlation redundancy filter (no dcor >= 0.70 pairs remain) plus an importance noise-floor cut, then re-tuned on the reduced set (tuner-inner CV MAE 0.973 -> 0.927). Only the dominant predictor is robustly above the importance noise floor; treat the reduced ranking as exploratory. See the SelectionStep and notes/202606201500-gb-replication-findings.md."
    )
