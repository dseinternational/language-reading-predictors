# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
LRP06: Predictors of letter-sound knowledge level.

``LRP06`` is the baseline exploratory model for letter-sound
knowledge level (``yarclet``). It uses the full
:attr:`Predictors.DEFAULT_LEVEL` set (with ``yarclet`` excluded as
the target) with no outlier exclusion so the starting picture is
unfiltered.

The target is **left-skewed with a ceiling at 32** (``yarclet`` min
0, max 32, median 21, skewness −0.60, n ≈ 214). The ceiling effect
(95th percentile = 31, 99th = 32) means many children score at or
near the instrument maximum — a different pathology from the
right-skewed / floor-at-0 targets of LRP02 and LRP04. Log / log1p
transforms are inappropriate here because the skew is in the wrong
direction; a reflection-log or quantile objective might be
considered later.

No tuning has been run for LRP06 yet — it runs on a reasonable
``_LGBM_BASELINE_PARAMS`` dict so later feature-selection variants
have a documented starting point.
"""

from language_reading_predictors.data_variables import Variables as V
from language_reading_predictors.models.base_model import LevelModel
from language_reading_predictors.models.common import SelectionStep, ShapScatterSpec
from language_reading_predictors.models.lgbm_pipeline import LGBMPipeline


# ── predictor selection steps (shared by all variants) ───────────────────
#
# LRP06 has not yet been through iterative feature selection. When
# selection variants are introduced, record their rationale here as
# ``SelectionStep`` entries and chain from ``LRP06`` the same way
# ``lrp02.py`` / ``lrp04.py`` does.

_SELECTION_STEPS: list[SelectionStep] = []


# ── hyperparameter sets ─────────────────────────────────────────────────

# Baseline — no tuning has been run for LRP06 yet. Reasonable defaults
# give the feature-selection work a reproducible starting point. Use
# ``python scripts/tune_model.py lrp06`` to produce a tuned set and
# replace this dict.
_LGBM_BASELINE_PARAMS: dict[str, float | int | str] = {
    "n_estimators": 500,
    "learning_rate": 0.05,
    "num_leaves": 15,
    "max_depth": 6,
    "min_child_samples": 16,
    "subsample": 0.8,
    "subsample_freq": 1,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.1,
    "reg_lambda": 0.1,
    "n_jobs": -1,
    "verbosity": -1,
}


# ── primary model (baseline, untuned) ───────────────────────────────────


class LRP06(LevelModel):
    """Letter-sound knowledge level predictors — baseline (all data, untuned).

    Uses the full :attr:`Predictors.DEFAULT_LEVEL` predictor set
    (minus the target ``yarclet``) and a reasonable
    ``_LGBM_BASELINE_PARAMS`` set. Serves as the starting point for
    feature-selection and tuning work on the letter-sound knowledge
    level-prediction task.
    """

    model_id = "lrp06"
    target_var = V.YARCLET
    description = (
        "LightGBM — letter-sound knowledge level predictors "
        "(baseline, DEFAULT_LEVEL set, untuned)"
    )
    pipeline_cls = LGBMPipeline
    params = _LGBM_BASELINE_PARAMS
    cv_splits = 51
    outlier_threshold = None
    selection_steps = _SELECTION_STEPS
    shap_scatter_specs = [
        ShapScatterSpec(description="All predictors, SHAP auto-colouring"),
    ]
    notes = (
        "Baseline exploratory model for letter-sound knowledge level "
        "(yarclet). Uses the full default level predictor set (minus "
        "the target) without outlier exclusion, and a reasonable "
        "_LGBM_BASELINE_PARAMS starting point — no feature selection "
        "or hyperparameter tuning has been applied yet. Note the "
        "target has a ceiling effect at 32 — many observations cluster "
        "at the instrument maximum, producing a left-skewed "
        "distribution."
    )
