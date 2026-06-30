# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
LRPGBL06: Predictors of expressive-vocabulary level.

``LRPGBL06`` is the exploratory model for expressive-vocabulary level
(``eowpvt``). The target is mildly right-skewed (``eowpvt`` min 8, max
77, median 33, skewness 0.63, n ≈ 215) with no hard floor at 0.

Uniform feature selection (2026-06-21): reduced from the full
32-predictor :attr:`Predictors.DEFAULT_LEVEL` set (minus the target) to
7 predictors via a distance-correlation redundancy filter (dcor >= 0.70)
plus an importance noise-floor cut, then re-tuned. See the SelectionStep
and notes/202606211200-uniform-gb-fs.md.

``LRPGBL06NoConstruct`` is the same-skill variant: it additionally drops
``b1exto`` (a bespoke expressive-vocabulary test — the same skill as the
standardised target ``eowpvt``) to ask what predicts expressive
vocabulary beyond a concurrent same-skill measure. See
notes/202606210930-lrp-same-skill-variants.md.
"""

from language_reading_predictors.data_variables import Variables as V
from language_reading_predictors.models.base_model import LevelModel
from language_reading_predictors.models.common import DEFAULT_SHAP_SCATTER_SPECS, SelectionStep
from language_reading_predictors.models.lgbm_pipeline import LGBMPipeline


_SELECTION_STEPS: list[SelectionStep] = [
    SelectionStep(
        removed=[
            V.APTINFO, V.B1RETO, V.NUMCHIL, V.DEAPPIN, V.ERBWORD, V.VISION, V.GROUP,
            V.GENDER, V.EARINF, V.HEARING, V.AGESPEAK, V.MUMEDUPOST16,
            V.DADEDUPOST16, V.AGEBOOKS, V.AREA, V.DEAPPVO, V.YARCSI, V.BEHAV,
            V.SPPHON, V.AGE, V.TROG, V.NONWORD, V.APTGRAM, V.ERBNW, V.BLENDING
        ],
        notes=(
            "Uniform feature selection (2026-06-21): from the full 32-predictor set, a distance-correlation redundancy filter (dcor >= 0.70, keep the highest out-of-fold permutation-importance representative) plus an importance noise-floor cut (<= 0.005). Reduces to 7 predictors with no dcor >= 0.70 pairs remaining; re-tuned on the reduced set (Optuna 150-trial MAE, 10-fold GroupKFold, seed 47). Applied uniformly across all GB models; see notes/202606211200-uniform-gb-fs.md."
        ),
        date="2026-06-21",
        metrics_before={"cv_mae_mean": 6.0503},
        metrics_after={"cv_mae_mean": 6.1592},
    ),
]


# MAE-tuned on the 7-predictor uniform-selected set (Optuna 150
# trials, 10-split GroupKFold, seed 47, scoring=mae, lgbm_objective=mae).
# Tuner-inner CV MAE 6.1592.
_LGBM_MAE_PARAMS: dict[str, float | int | str] = {
    "objective": "mae",
    "n_estimators": 68,
    "learning_rate": 0.08572327409762025,
    "num_leaves": 18,
    "max_depth": 3,
    "min_child_samples": 4,
    "subsample": 0.871675111170649,
    "subsample_freq": 1,
    "colsample_bytree": 0.9388695366733577,
    "reg_alpha": 0.001037151277752068,
    "reg_lambda": 0.019926160950819845,
    "n_jobs": -1,
    "verbosity": -1,
}


class LRPGBL06(LevelModel):
    """Expressive-vocabulary level predictors — exploratory (MAE-tuned, all data).

    Uniform-selected subset of :attr:`Predictors.DEFAULT_LEVEL` (minus the
    target ``eowpvt``) with MAE-tuned hyperparameters and no outlier
    exclusion. See the SelectionStep and the module docstring.
    """

    model_id = "lrpgbl06"
    target_var = V.EOWPVT
    description = (
        "LightGBM — expressive-vocabulary level predictors "
        "(7 predictors, MAE-tuned, no outlier exclusion)"
    )
    pipeline_cls = LGBMPipeline
    params = _LGBM_MAE_PARAMS
    selection_steps = _SELECTION_STEPS
    shap_scatter_specs = DEFAULT_SHAP_SCATTER_SPECS
    notes = (
        "Exploratory model for expressive-vocabulary level (eowpvt). Uniform "
        "feature selection (2026-06-21) from the full 32-predictor "
        "DEFAULT_LEVEL set to 7 predictors (distance-correlation redundancy "
        "filter + importance noise-floor cut; no dcor >= 0.70 pairs remain), "
        "re-tuned on the reduced set (tuner-inner CV MAE 6.050 -> 6.159). "
        "Treat the reduced ranking as exploratory. See "
        "notes/202606211200-uniform-gb-fs.md."
    )


# Same-skill variant: MAE-tuned on the 6-predictor set after dropping
# b1exto — a bespoke expressive-vocabulary test, the same skill as the
# standardised target eowpvt. Tuner-inner CV MAE 7.0809 (vs primary 6.159).
_LGBM_MAE_PARAMS_NOCONSTRUCT: dict[str, float | int | str] = {
    "objective": "mae",
    "n_estimators": 130,
    "learning_rate": 0.017675250240582925,
    "num_leaves": 44,
    "max_depth": 7,
    "min_child_samples": 8,
    "subsample": 0.9143924972566979,
    "subsample_freq": 1,
    "colsample_bytree": 0.6197667976554718,
    "reg_alpha": 0.0018637589742316375,
    "reg_lambda": 0.004928294665389766,
    "n_jobs": -1,
    "verbosity": -1,
}


class LRPGBL06NoConstruct(LRPGBL06):
    """eowpvt — same-skill reduced (bespoke expressive-vocab sibling b1exto dropped)."""

    model_id = "lrpgbl06_noconstruct"
    variant_of = "lrpgbl06"
    description = (
        "LightGBM — eowpvt predictors "
        "(6 predictors, same-skill reduced: expressive-vocab sibling dropped)"
    )
    params = _LGBM_MAE_PARAMS_NOCONSTRUCT
    selection_steps = [
        SelectionStep(
            removed=[V.B1EXTO],
            notes=(
                "Same-skill variant of lrpgbl06: drops b1exto, a bespoke taught expressive-vocabulary test — the same skill as the target eowpvt (standardised expressive vocabulary), measured by a different instrument — to ask what predicts expressive vocabulary beyond a concurrent same-skill measure. Other constructs (receptive vocabulary, reading, articulation) are kept deliberately, to be seen independently. Re-tuned on the reduced set. See notes/202606210930-lrp-same-skill-variants.md."
            ),
            date="2026-06-21",
            metrics_before={"cv_mae_mean": 6.1592},
            metrics_after={"cv_mae_mean": 7.0809},
        ),
    ]
    notes = (
        "Same-skill variant of lrpgbl06: drops b1exto (bespoke expressive vocabulary, the same skill as the standardised target eowpvt) to ask what predicts expressive vocabulary beyond a concurrent same-skill measure. Other constructs kept visible. Re-tuned on the reduced set. See notes/202606210930-lrp-same-skill-variants.md."
    )
