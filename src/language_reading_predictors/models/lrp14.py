# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
LRP14: Predictors of non-word reading level.

``LRP14`` is the baseline exploratory model for non-word reading
level (``nonword``). ``nonword`` is an items-correct score from
a non-word decoding task (observed range 0–6 in this sample).

The target is **heavily floor-loaded** (``nonword`` min 0, max 6,
median 0, mean 1.24, std 1.81, skewness 1.38, with **57% at
zero**, n ≈ 215). Non-word reading / phonological decoding is a
late-emerging skill — most children in this sample have not yet
started to decode non-words reliably, so the data are genuinely
half zeros. This differs from the LRP02 / LRP06 right-skewed
reading targets where zeros are a minority.

Log or quantile transforms may be more appropriate than a plain
regression here; plan for a ``lrp14_log`` variant in follow-up PRs.

Feature selection applied 2026-06-20 (replication): reduced from the full 32-predictor set to 3 predictors via a distance-correlation redundancy filter (dcor >= 0.70, keep the highest-importance representative) plus an importance noise-floor cut, then re-tuned on the reduced set. See the SelectionStep below and notes/202606201500-gb-replication-findings.md.
"""

from language_reading_predictors.data_variables import Variables as V
from language_reading_predictors.models.base_model import LevelModel
from language_reading_predictors.models.common import SelectionStep, ShapScatterSpec
from language_reading_predictors.models.lgbm_pipeline import LGBMPipeline


_SELECTION_STEPS: list[SelectionStep] = [
    SelectionStep(
        removed=[
            V.TIME, V.GROUP, V.AREA, V.GENDER, V.AGE, V.APTGRAM, V.B1EXTO, V.B1RETO,
            V.CELF, V.EOWPVT, V.ERBNW, V.ERBWORD, V.BLENDING, V.ROWPVT, V.SPPHON,
            V.TROG, V.YARCSI, V.DEAPPIN, V.DEAPPVO, V.DEAPPFI, V.BEHAV, V.AGESPEAK,
            V.VISION, V.HEARING, V.EARINF, V.NUMCHIL, V.AGEBOOKS, V.MUMEDUPOST16,
            V.DADEDUPOST16
        ],
        notes=(
            "Feature selection (replication, 2026-06-20): from the full 32-predictor set, a distance-correlation filter (dcor >= 0.70, keep the highest out-of-fold permutation-importance representative per cluster) plus removal of features at/below the 0.005 importance floor. Reduces to 3 predictors with no dcor >= 0.70 pairs remaining; pooled refit-CV held under matched hyperparameters, then the set was re-tuned. See notes/202606201500-gb-replication-findings.md."
        ),
        date="2026-06-20",
        metrics_before={"cv_mae_mean": 0.8916},
        metrics_after={"cv_mae_mean": 0.7724},
    ),
]


# MAE-tuned on the 3-predictor replication-selected set, no outlier
# exclusion (Optuna 150 trials, 10-split GroupKFold, seed 47, scoring=mae,
# lgbm_objective=mae). Tuner-inner CV MAE 0.7724. Supersedes the full-set tune.
_LGBM_MAE_PARAMS: dict[str, float | int | str] = {
    "objective": "mae",
    "n_estimators": 68,
    "learning_rate": 0.08742464937643295,
    "num_leaves": 39,
    "max_depth": 3,
    "min_child_samples": 6,
    "subsample": 0.9814514819214128,
    "subsample_freq": 1,
    "colsample_bytree": 0.6514974124208813,
    "reg_alpha": 0.3492206947118945,
    "reg_lambda": 0.017433408875219325,
    "n_jobs": -1,
    "verbosity": -1,
}


class LRP14(LevelModel):
    """Non-word reading level predictors — baseline (all data, MAE-tuned).

    Uses a feature-selected subset of :attr:`Predictors.DEFAULT_LEVEL`
    (minus the target ``nonword``) with MAE-tuned hyperparameters and
    no outlier exclusion. Feature selection was applied (2026-06-20 replication); see the SelectionStep and the module docstring.
    """

    model_id = "lrp14"
    target_var = V.NONWORD
    description = (
        "LightGBM — non-word reading level predictors "
        "(3 predictors, MAE-tuned, no outlier exclusion)"
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
        "Exploratory model for nonword (level). Feature-selected (2026-06-20 replication) from the full 32-predictor default set to 3 predictors via a distance-correlation redundancy filter (no dcor >= 0.70 pairs remain) plus an importance noise-floor cut, then re-tuned on the reduced set (tuner-inner CV MAE 0.892 -> 0.772). Only the dominant predictor is robustly above the importance noise floor; treat the reduced ranking as exploratory. See the SelectionStep and notes/202606201500-gb-replication-findings.md."
    )


# Construct-reduced variant: MAE-tuned on the 2-predictor set after
# additionally dropping same-construct (reading_decoding) predictors
# (yarclet). Tuner-inner CV MAE 0.7891.
_LGBM_MAE_PARAMS_NOCONSTRUCT: dict[str, float | int | str] = {
    "objective": "mae",
    "n_estimators": 176,
    "learning_rate": 0.03526461559138239,
    "num_leaves": 34,
    "max_depth": 3,
    "min_child_samples": 8,
    "subsample": 0.9494031276497659,
    "subsample_freq": 1,
    "colsample_bytree": 0.6533258433537965,
    "reg_alpha": 0.02826884180155375,
    "reg_lambda": 0.00139661138721428,
    "n_jobs": -1,
    "verbosity": -1,
}


class LRP14NoConstruct(LRP14):
    """nonword — construct-reduced (reading_decoding dropped)."""

    model_id = "lrp14_noconstruct"
    variant_of = "lrp14"
    description = (
        "LightGBM — nonword predictors "
        "(2 predictors, construct-reduced)"
    )
    params = _LGBM_MAE_PARAMS_NOCONSTRUCT
    selection_steps = [
        SelectionStep(
            removed=[V.YARCLET],
            notes=(
                "Construct-reduced variant of lrp14: drops the same-construct (reading_decoding) predictors (yarclet) from the primary set to ask what predicts nonword beyond its sibling measures. Pooled CV falls accordingly; re-tuned on the reduced set. See notes/202606201500-gb-replication-findings.md."
            ),
            date="2026-06-20",
            metrics_after={"cv_mae_mean": 0.7891},
        ),
    ]
    notes = (
        "Construct-reduced variant of lrp14: drops the same-construct (reading_decoding) predictors (yarclet) from the primary set to ask what predicts nonword beyond its sibling measures. Pooled CV falls accordingly; re-tuned on the reduced set. See notes/202606201500-gb-replication-findings.md."
    )
