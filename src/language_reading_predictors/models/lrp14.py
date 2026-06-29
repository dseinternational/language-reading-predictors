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

Uniform feature selection (2026-06-21): reduced from the full 32-predictor set to 3 predictors via a distance-correlation redundancy filter plus an importance noise-floor cut, then re-tuned. See the SelectionStep below and notes/202606211200-uniform-gb-fs.md.

No construct-reduced variant: ``nonword``'s remaining predictors are
*different* reading skills (letter-sound knowledge, phonological awareness,
spelling), which we keep visible rather than treat as concurrent
same-skill restatements. See notes/202606210930-lrp-same-skill-variants.md.
"""

from language_reading_predictors.data_variables import Variables as V
from language_reading_predictors.models.base_model import LevelModel
from language_reading_predictors.models.common import DEFAULT_SHAP_SCATTER_SPECS, SelectionStep
from language_reading_predictors.models.lgbm_pipeline import LGBMPipeline


_SELECTION_STEPS: list[SelectionStep] = [
    SelectionStep(
        removed=[
            V.SPPHON, V.APTGRAM, V.DEAPPVO, V.B1RETO, V.GENDER, V.HEARING,
            V.AGEBOOKS, V.VISION, V.AGESPEAK, V.EARINF, V.NUMCHIL, V.AREA, V.TIME,
            V.BEHAV, V.DADEDUPOST16, V.B1EXTO, V.MUMEDUPOST16, V.GROUP, V.CELF,
            V.ROWPVT, V.DEAPPFI, V.AGE, V.DEAPPIN, V.YARCSI, V.EOWPVT, V.TROG,
            V.ERBNW, V.ERBWORD, V.BLENDING
        ],
        notes=(
            "Uniform feature selection (2026-06-21): from the full 32-predictor set, a distance-correlation redundancy filter (dcor >= 0.70, keep the highest out-of-fold permutation-importance representative) plus an importance noise-floor cut (<= 0.005). Reduces to 3 predictors with no dcor >= 0.70 pairs remaining; re-tuned on the reduced set (Optuna 150-trial MAE, 10-fold GroupKFold, seed 47). Applied uniformly across all GB models; see notes/202606211200-uniform-gb-fs.md."
        ),
        date="2026-06-21",
        metrics_before={"cv_mae_mean": 0.9136},
        metrics_after={"cv_mae_mean": 0.7618},
    ),
]


# MAE-tuned on the 3-predictor uniform-selected set (Optuna 150
# trials, 10-split GroupKFold, seed 47, scoring=mae, lgbm_objective=mae).
# Tuner-inner CV MAE 0.7618.
_LGBM_MAE_PARAMS: dict[str, float | int | str] = {
    "objective": "mae",
    "n_estimators": 112,
    "learning_rate": 0.04472739204117363,
    "num_leaves": 32,
    "max_depth": 3,
    "min_child_samples": 5,
    "subsample": 0.71049510667404,
    "subsample_freq": 1,
    "colsample_bytree": 0.6226024174727751,
    "reg_alpha": 3.6677170651291187,
    "reg_lambda": 0.007696110642370069,
    "n_jobs": -1,
    "verbosity": -1,
}


class LRP14(LevelModel):
    """Non-word reading level predictors — baseline (all data, MAE-tuned).

    Uses a feature-selected subset of :attr:`Predictors.DEFAULT_LEVEL`
    (minus the target ``nonword``) with MAE-tuned hyperparameters and
    no outlier exclusion. Feature selection was applied (2026-06-21 uniform); see the SelectionStep and the module docstring.
    """

    model_id = "lrp14"
    target_var = V.NONWORD
    description = (
        "LightGBM — non-word reading level predictors "
        "(3 predictors, MAE-tuned, no outlier exclusion)"
    )
    pipeline_cls = LGBMPipeline
    params = _LGBM_MAE_PARAMS
    selection_steps = _SELECTION_STEPS
    shap_scatter_specs = DEFAULT_SHAP_SCATTER_SPECS
    notes = (
        "Exploratory model for nonword (level). Uniform feature selection (2026-06-21) from the full 32-predictor DEFAULT_LEVEL set to 3 predictors (distance-correlation redundancy filter + importance noise-floor cut; no dcor >= 0.70 pairs remain), re-tuned on the reduced set (tuner-inner CV MAE 0.914 -> 0.762). Treat the reduced ranking as exploratory. See notes/202606211200-uniform-gb-fs.md."
    )
