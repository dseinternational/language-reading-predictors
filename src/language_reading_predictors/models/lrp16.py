# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
LRP16: Predictors of phoneme-blending level.

``LRP16`` is the baseline exploratory model for phoneme-blending
level (``blending``). A phoneme-blending (phonological awareness)
score on a 0–10 scale: the child selects which of three pictures
depicts the word formed by segmented phonemes spoken by the
examiner.

The target is **essentially symmetric** (``blending`` min 0, max
10, median 6, mean 5.76, std 2.55, skewness 0.01, n ≈ 215) — one
of the cleanest distributions in the suite. The coarse 0–10
scale may cap achievable R² (similar to CELF's 0–18 scale in
LRP10).

Feature selection applied 2026-06-20 (replication): reduced from the full 32-predictor set to 2 predictors via a distance-correlation redundancy filter (dcor >= 0.70, keep the highest-importance representative) plus an importance noise-floor cut, then re-tuned on the reduced set. See the SelectionStep below and notes/202606201500-gb-replication-findings.md.
"""

from language_reading_predictors.data_variables import Variables as V
from language_reading_predictors.models.base_model import LevelModel
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
            V.B1EXTO, V.B1RETO, V.CELF, V.EOWPVT, V.ERBNW, V.ERBWORD, V.NONWORD,
            V.SPPHON, V.TROG, V.YARCLET, V.YARCSI, V.DEAPPIN, V.DEAPPVO, V.DEAPPFI,
            V.BEHAV, V.AGESPEAK, V.VISION, V.HEARING, V.EARINF, V.NUMCHIL,
            V.AGEBOOKS, V.MUMEDUPOST16, V.DADEDUPOST16
        ],
        notes=(
            "Feature selection (replication, 2026-06-20): from the full 32-predictor set, a distance-correlation filter (dcor >= 0.70, keep the highest out-of-fold permutation-importance representative per cluster) plus removal of features at/below the 0.005 importance floor. The standardised instrument was preferred over its intervention-taught sibling (eowpvt<-b1exto / rowpvt<-b1reto) where it did not cost CV. Reduces to 2 predictors with no dcor >= 0.70 pairs remaining; pooled refit-CV held under matched hyperparameters, then the set was re-tuned. See notes/202606201500-gb-replication-findings.md."
        ),
        date="2026-06-20",
        metrics_before={"cv_mae_mean": 1.7438},
        metrics_after={"cv_mae_mean": 1.7634},
    ),
]


# ── hyperparameter sets ─────────────────────────────────────────────────

# MAE-tuned on the 2-predictor replication-selected set, no outlier
# exclusion (Optuna 150 trials, 10-split GroupKFold, seed 47, scoring=mae,
# lgbm_objective=mae). Tuner-inner CV MAE 1.7634. Supersedes the full-set tune.
_LGBM_MAE_PARAMS: dict[str, float | int | str] = {
    "objective": "mae",
    "n_estimators": 22,
    "learning_rate": 0.04124036041929963,
    "num_leaves": 24,
    "max_depth": 10,
    "min_child_samples": 30,
    "subsample": 0.6144702261058946,
    "subsample_freq": 1,
    "colsample_bytree": 0.6370488804588129,
    "reg_alpha": 0.1825346247915465,
    "reg_lambda": 0.0010070378924338585,
    "n_jobs": -1,
    "verbosity": -1,
}


# ── primary model (baseline, MAE-tuned) ─────────────────────────────────


class LRP16(LevelModel):
    """Phoneme-blending level predictors — baseline (all data, MAE-tuned).

    Uses a feature-selected subset of :attr:`Predictors.DEFAULT_LEVEL`
    (minus the target ``blending``) with MAE-tuned hyperparameters
    and no outlier exclusion. Feature selection was applied (2026-06-20 replication); see the SelectionStep and the module docstring.
    """

    model_id = "lrp16"
    target_var = V.BLENDING
    description = (
        "LightGBM — phoneme-blending level predictors "
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
        "Exploratory model for blending (level). Feature-selected (2026-06-20 replication) from the full 32-predictor default set to 2 predictors via a distance-correlation redundancy filter (no dcor >= 0.70 pairs remain) plus an importance noise-floor cut, then re-tuned on the reduced set (tuner-inner CV MAE 1.744 -> 1.763). Only the dominant predictor is robustly above the importance noise floor; treat the reduced ranking as exploratory. See the SelectionStep and notes/202606201500-gb-replication-findings.md."
    )
