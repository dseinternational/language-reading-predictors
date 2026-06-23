# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
LRP26: Predictors of Early Repetition Battery nonword repetition level (``erbnw``).

``erbnw`` is the number of nonwords correctly repeated from the
Early Repetition Battery (a repetition task indexing verbal /
phonological short-term memory).

The target spans min 0.0, max 18.0, median 9.00, mean 8.98, std
4.74, skew 0.01 (n = 202).

This is an exploratory gradient-boosting discovery model on the
same footing as LRP01–22: it asks how predictable nonword
repetition is and from what, to inform whether the shared DAG
needs a verbal / phonological short-term memory node. It is not a
causal or intention-to-treat estimate.

Uniform feature selection (2026-06-23): reduced from the full
32-predictor DEFAULT_LEVEL set to 11 predictors via a distance-
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
            V.B1RETO, V.CELF, V.EOWPVT, V.NONWORD, V.BLENDING, V.ROWPVT,
            V.SPPHON, V.YARCLET, V.DEAPPIN, V.DEAPPFI, V.EWRSWR,
            V.BEHAV, V.AGESPEAK, V.VISION, V.AGEBOOKS
        ],
        notes=(
            "Uniform feature selection (2026-06-23): from the full 32-predictor DEFAULT_LEVEL set, a distance-correlation redundancy filter (dcor >= 0.70, keep the highest out-of-fold permutation-importance representative) plus an importance noise-floor cut (<= 0.005). Reduces to 11 predictors with no dcor >= 0.70 pairs remaining; re-tuned on the reduced set (Optuna 150-trial MAE, 10-fold GroupKFold, seed 47). Same method as the LRP01–22 suite; see scripts/uniform_feature_selection.py and notes/202606230900-predictability-speech-memory-language.md."
        ),
        date="2026-06-23",
        metrics_before={"cv_mae_mean": 1.9401},
        metrics_after={"cv_mae_mean": 1.8722},
    ),
]


# ── hyperparameters (MAE-tuned on the reduced set) ───────────────────────

_LGBM_MAE_PARAMS: dict[str, float | int | str] = {
    "objective": "mae",
    "learning_rate": 0.016448486269199713,
    "num_leaves": 10,
    "max_depth": 8,
    "min_child_samples": 23,
    "subsample": 0.8128550741391448,
    "colsample_bytree": 0.9858283410142396,
    "reg_alpha": 0.016258934164446813,
    "reg_lambda": 0.014253178652321749,
    "subsample_freq": 1,
    "n_jobs": -1,
    "verbosity": -1,
    "random_state": 47,
    "n_estimators": 384,
}


class LRP26(LevelModel):
    """Early Repetition Battery nonword repetition level predictors — baseline (MAE-tuned)."""

    model_id = "lrp26"
    target_var = V.ERBNW
    description = (
        "LightGBM — Early Repetition Battery nonword repetition level predictors (11 predictors, MAE-tuned, no outlier exclusion)"
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
        "Exploratory model for erbnw (level). Uniform feature selection (2026-06-23) from the full 32-predictor DEFAULT_LEVEL set to 11 predictors (distance-correlation redundancy filter + importance noise-floor cut; no dcor >= 0.70 pairs remain), re-tuned on the reduced set (tuner-inner CV MAE 1.846). Treat the reduced ranking as exploratory. See notes/202606230900-predictability-speech-memory-language.md."
    )


# ── same-skill (_noconstruct) variant ────────────────────────────────────
#
# Drops the same-instrument Early Repetition Battery sibling(s) to separate same-skill
# correlation from cross-domain signal. Pooled OOF R² 0.76 -> 0.47.


class LRP26NoConstruct(LRP26):
    """erbnw — same-skill variant: Early Repetition Battery sibling(s) dropped (R2 0.76 -> 0.47)."""

    model_id = "lrp26_noconstruct"
    variant_of = "lrp26"
    description = (
        "LightGBM — erbnw predictors "
        "(same-skill reduced: Early Repetition Battery sibling(s) dropped)"
    )
    selection_steps = [
        SelectionStep(
            removed=[V.ERBWORD],
            notes=(
                "Same-skill (concurrent) variant of lrp26: drops the same-instrument Early Repetition Battery sibling(s) ``erbword`` (scored from the same instrument as the target). Pooled out-of-fold R² falls from 0.76 to 0.47 -> retains substantial cross-domain signal — not purely measurement-bound. Reuses the primary's MAE-tuned params on the reduced set (the sibling-dropped OOF R² is the headline; re-tuning is a refinement). See notes/202606230900-predictability-speech-memory-language.md."
            ),
            date="2026-06-23",
            metrics_before={"pooled_oof_r2": 0.7619},
            metrics_after={"pooled_oof_r2": 0.4723},
        ),
    ]
    notes = (
        "Same-skill (concurrent) variant of lrp26: drops the same-instrument Early Repetition Battery sibling(s) ``erbword`` (scored from the same instrument as the target). Pooled out-of-fold R² falls from 0.76 to 0.47 -> retains substantial cross-domain signal — not purely measurement-bound. Reuses the primary's MAE-tuned params on the reduced set (the sibling-dropped OOF R² is the headline; re-tuning is a refinement). See notes/202606230900-predictability-speech-memory-language.md."
    )
