# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
LRP04: Predictors of expressive-vocabulary level.

``LRP04`` is the baseline exploratory model for expressive-vocabulary
level (``eowpvt``). It uses the full :attr:`Predictors.DEFAULT_LEVEL`
set (with ``eowpvt`` excluded as the target) with no outlier exclusion
so the starting picture is unfiltered.

The target is mildly right-skewed (``eowpvt`` min 8, max 77, median 33,
skewness 0.63, n ≈ 215). No hard floor at 0 (unlike ``ewrswr`` in
LRP02), so the motivation for a ``log1p`` transform is less compelling
— but a question for future investigation rather than the baseline.

No tuning has been run for LRP04 yet — it runs on a reasonable
``_LGBM_BASELINE_PARAMS`` dict so later feature-selection variants
have a documented starting point.
"""

from language_reading_predictors.data_variables import Variables as V
from language_reading_predictors.models.base_model import LevelModel
from language_reading_predictors.models.common import SelectionStep, ShapScatterSpec
from language_reading_predictors.models.lgbm_pipeline import LGBMPipeline


# ── predictor selection steps (shared by all variants) ───────────────────
#
# LRP04 has not yet been through iterative feature selection. When
# selection variants are introduced, record their rationale here as
# ``SelectionStep`` entries and chain from ``LRP04`` the same way
# ``lrp02.py`` does.

_SELECTION_STEPS: list[SelectionStep] = []


# ── hyperparameter sets ─────────────────────────────────────────────────

# Baseline — no tuning has been run for LRP04 yet. Reasonable defaults
# give the feature-selection work a reproducible starting point. Use
# ``python scripts/tune_model.py lrp04`` to produce a tuned set and
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


class LRP04(LevelModel):
    """Expressive-vocabulary level predictors — baseline (all data, untuned).

    Uses the full :attr:`Predictors.DEFAULT_LEVEL` predictor set
    (minus the target ``eowpvt``) and a reasonable
    ``_LGBM_BASELINE_PARAMS`` set. Serves as the starting point for
    feature-selection and tuning work on the expressive-vocabulary
    level-prediction task.
    """

    model_id = "lrp04"
    target_var = V.EOWPVT
    description = (
        "LightGBM — expressive-vocabulary level predictors "
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
        "Baseline exploratory model for expressive-vocabulary level "
        "(eowpvt). Uses the full default level predictor set (minus "
        "the target) without outlier exclusion, and a reasonable "
        "_LGBM_BASELINE_PARAMS starting point — no feature selection "
        "or hyperparameter tuning has been applied yet."
    )
