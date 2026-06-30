# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
LRPGBL14: Predictors of basic concept knowledge level (CELF).

``LRPGBL14`` is the exploratory model for basic concept knowledge level
(``celf``). The ``celf`` score is drawn from the Clinical Evaluation of
Language Fundamentals Preschool 2nd Ed (Wiig, Secord & Semel 2006); in
this study only the basic-concept-knowledge subtest (18 linguistic
concepts) was administered — so ``celf`` is a lexical/semantic concept
measure, NOT a grammar measure (grammar is covered by ``trog`` for
receptive and ``aptgram`` for expressive grammar).

The target is **mildly left-skewed** (``celf`` min 0, max 18, median 11,
mean 10.88, std 4.24, skewness −0.37, n ≈ 214). The max of 18 is the
instrument maximum but the 95th percentile is below it, so there is no
strong ceiling pathology.

Uniform feature selection (2026-06-21): reduced from the full
32-predictor :attr:`Predictors.DEFAULT_LEVEL` set (minus the target) to
3 predictors via a distance-correlation redundancy filter (dcor >= 0.70)
plus an importance noise-floor cut, then re-tuned. This supersedes the
earlier Select01/Select02 construct-driven hand selection. See the
SelectionStep and notes/202606211200-uniform-gb-fs.md.
"""

from language_reading_predictors.data_variables import Variables as V
from language_reading_predictors.models.base_model import LevelModel
from language_reading_predictors.models.common import DEFAULT_SHAP_SCATTER_SPECS, SelectionStep
from language_reading_predictors.models.lgbm_pipeline import LGBMPipeline


_SELECTION_STEPS: list[SelectionStep] = [
    SelectionStep(
        removed=[
            V.B1RETO, V.ERBNW, V.B1EXTO, V.BEHAV, V.NUMCHIL, V.HEARING, V.VISION,
            V.APTINFO, V.GENDER, V.EARINF, V.GROUP, V.AREA, V.BLENDING,
            V.DADEDUPOST16, V.ERBWORD, V.YARCSI, V.DEAPPFI, V.MUMEDUPOST16,
            V.AGEBOOKS, V.NONWORD, V.SPPHON, V.YARCLET, V.APTGRAM, V.TIME,
            V.AGESPEAK, V.EWRSWR, V.DEAPPVO, V.ROWPVT, V.TROG
        ],
        notes=(
            "Uniform feature selection (2026-06-21): from the full 32-predictor set, a distance-correlation redundancy filter (dcor >= 0.70, keep the highest out-of-fold permutation-importance representative) plus an importance noise-floor cut (<= 0.005). Reduces to 3 predictors with no dcor >= 0.70 pairs remaining; re-tuned on the reduced set (Optuna 150-trial MAE, 10-fold GroupKFold, seed 47). Applied uniformly across all GB models; see notes/202606211200-uniform-gb-fs.md."
        ),
        date="2026-06-21",
        metrics_before={"cv_mae_mean": 2.4964},
        metrics_after={"cv_mae_mean": 2.5670},
    ),
]


# MAE-tuned on the 3-predictor uniform-selected set (Optuna 150
# trials, 10-split GroupKFold, seed 47, scoring=mae, lgbm_objective=mae).
# Tuner-inner CV MAE 2.5670.
_LGBM_MAE_PARAMS: dict[str, float | int | str] = {
    "objective": "mae",
    "n_estimators": 65,
    "learning_rate": 0.043738091375742166,
    "num_leaves": 44,
    "max_depth": 4,
    "min_child_samples": 37,
    "subsample": 0.9853192989415035,
    "subsample_freq": 1,
    "colsample_bytree": 0.9140199440486373,
    "reg_alpha": 0.06457861635292404,
    "reg_lambda": 0.013949746526542768,
    "n_jobs": -1,
    "verbosity": -1,
}


class LRPGBL14(LevelModel):
    """CELF basic concept knowledge level predictors — exploratory (MAE-tuned, all data).

    Uniform-selected subset of :attr:`Predictors.DEFAULT_LEVEL` (minus the
    target ``celf``) with MAE-tuned hyperparameters and no outlier
    exclusion. See the SelectionStep and the module docstring.
    """

    model_id = "lrpgbl14"
    target_var = V.CELF
    description = (
        "LightGBM — CELF (basic concept knowledge) level predictors "
        "(3 predictors, MAE-tuned, no outlier exclusion)"
    )
    pipeline_cls = LGBMPipeline
    params = _LGBM_MAE_PARAMS
    selection_steps = _SELECTION_STEPS
    shap_scatter_specs = DEFAULT_SHAP_SCATTER_SPECS
    notes = (
        "Exploratory model for basic concept knowledge level (celf). Uniform "
        "feature selection (2026-06-21) from the full 32-predictor "
        "DEFAULT_LEVEL set to 3 predictors (distance-correlation redundancy "
        "filter + importance noise-floor cut; no dcor >= 0.70 pairs remain), "
        "re-tuned on the reduced set (tuner-inner CV MAE 2.496 -> 2.567). CELF "
        "here is a lexical/semantic concept measure, not grammar. Treat the "
        "reduced ranking as exploratory. See notes/202606211200-uniform-gb-fs.md."
    )
