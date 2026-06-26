# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
LRP28: Predictors of Early Repetition Battery word repetition level (``erbword``).

``erbword`` is the number of real words correctly repeated from
the Early Repetition Battery (a repetition task indexing verbal /
phonological short-term memory).

The target spans min 0.0, max 28.0, median 12.00, mean 11.35, std
5.22, skew -0.25 (n = 203).

This is an exploratory gradient-boosting discovery model on the
same footing as LRP01–22: it asks how predictable word repetition
is and from what, to inform whether the shared DAG needs a verbal
/ phonological short-term memory node. It is not a causal or
intention-to-treat estimate.

Uniform feature selection (2026-06-23): reduced from the full
32-predictor DEFAULT_LEVEL set to 12 predictors via a distance-
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
            V.TIME, V.GROUP, V.AREA, V.GENDER, V.AGE, V.APTGRAM,
            V.B1EXTO, V.B1RETO, V.CELF, V.EOWPVT, V.SPPHON, V.YARCLET,
            V.DEAPPFI, V.BEHAV, V.AGESPEAK, V.VISION, V.HEARING,
            V.NUMCHIL, V.AGEBOOKS, V.DADEDUPOST16
        ],
        notes=(
            "Uniform feature selection (2026-06-23): from the full 32-predictor DEFAULT_LEVEL set, a distance-correlation redundancy filter (dcor >= 0.70, keep the highest out-of-fold permutation-importance representative) plus an importance noise-floor cut (<= 0.005). Reduces to 12 predictors with no dcor >= 0.70 pairs remaining; re-tuned on the reduced set (Optuna 150-trial MAE, 10-fold GroupKFold, seed 47). Same method as the LRP01–22 suite; see scripts/uniform_feature_selection.py and notes/202606230900-predictability-speech-memory-language.md."
        ),
        date="2026-06-23",
        metrics_before={"cv_mae_mean": 1.9933},
        metrics_after={"cv_mae_mean": 1.9015},
    ),
]


# ── hyperparameters (MAE-tuned on the reduced set) ───────────────────────

_LGBM_MAE_PARAMS: dict[str, float | int | str] = {
    "objective": "mae",
    "learning_rate": 0.12648578524796544,
    "num_leaves": 10,
    "max_depth": 4,
    "min_child_samples": 6,
    "subsample": 0.6442114045867877,
    "colsample_bytree": 0.6424565278401814,
    "reg_alpha": 0.1708379704840523,
    "reg_lambda": 0.018815743974597524,
    "subsample_freq": 1,
    "n_jobs": -1,
    "verbosity": -1,
    "random_state": 47,
    "n_estimators": 133,
}


class LRP28(LevelModel):
    """Early Repetition Battery word repetition level predictors — baseline (MAE-tuned)."""

    model_id = "lrp28"
    target_var = V.ERBWORD
    description = (
        "LightGBM — Early Repetition Battery word repetition level predictors (12 predictors, MAE-tuned, no outlier exclusion)"
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
        "Exploratory model for erbword (level). Uniform feature selection (2026-06-23) from the full 32-predictor DEFAULT_LEVEL set to 12 predictors (distance-correlation redundancy filter + importance noise-floor cut; no dcor >= 0.70 pairs remain), re-tuned on the reduced set (tuner-inner CV MAE 1.961). Treat the reduced ranking as exploratory. See notes/202606230900-predictability-speech-memory-language.md."
    )


# ── same-skill (_noconstruct) variant ────────────────────────────────────
#
# Drops the same-instrument Early Repetition Battery sibling(s) to separate same-skill
# correlation from cross-domain signal. Pooled OOF R² 0.76 -> 0.62.


class LRP28NoConstruct(LRP28):
    """erbword — same-skill variant: Early Repetition Battery sibling(s) dropped (R2 0.76 -> 0.62)."""

    model_id = "lrp28_noconstruct"
    variant_of = "lrp28"
    description = (
        "LightGBM — erbword predictors "
        "(same-skill reduced: Early Repetition Battery sibling(s) dropped)"
    )
    selection_steps = [
        SelectionStep(
            removed=[V.ERBNW],
            notes=(
                "Same-skill (concurrent) variant of lrp28: drops the same-instrument Early Repetition Battery sibling(s) ``erbnw`` (scored from the same instrument as the target). Pooled out-of-fold R² falls from 0.76 to 0.62 -> retains substantial cross-domain signal — not purely measurement-bound. Reuses the primary's MAE-tuned params on the reduced set (the sibling-dropped OOF R² is the headline; re-tuning is a refinement). See notes/202606230900-predictability-speech-memory-language.md."
            ),
            date="2026-06-23",
            metrics_before={"pooled_oof_r2": 0.7648},
            metrics_after={"pooled_oof_r2": 0.6245},
        ),
    ]
    notes = (
        "Same-skill (concurrent) variant of lrp28: drops the same-instrument Early Repetition Battery sibling(s) ``erbnw`` (scored from the same instrument as the target). Pooled out-of-fold R² falls from 0.76 to 0.62 -> retains substantial cross-domain signal — not purely measurement-bound. Reuses the primary's MAE-tuned params on the reduced set (the sibling-dropped OOF R² is the headline; re-tuning is a refinement). See notes/202606230900-predictability-speech-memory-language.md."
    )
