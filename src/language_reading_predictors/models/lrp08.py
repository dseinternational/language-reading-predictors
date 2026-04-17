# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
LRP08: Predictors of receptive vocabulary level.

``LRP08`` is the baseline exploratory model for receptive
vocabulary level (``rowpvt``). It uses the full
:attr:`Predictors.DEFAULT_LEVEL` set (with ``rowpvt`` excluded as
the target) with no outlier exclusion so the starting picture is
unfiltered.

The target is **essentially symmetric and near-Gaussian** (``rowpvt``
min 11, max 82, median 42, mean 41.1, std 14.1, skewness 0.04,
n ≈ 215). No floor, no ceiling, no heavy tail — the cleanest
target distribution of any LRP model to date. Transforms are
unnecessary; standard MAE and RMSE objectives should behave well.

No tuning has been run for LRP08 yet — it runs on a reasonable
``_LGBM_BASELINE_PARAMS`` dict so later feature-selection variants
have a documented starting point.
"""

from language_reading_predictors.data_variables import Variables as V
from language_reading_predictors.models.base_model import LevelModel
from language_reading_predictors.models.common import SelectionStep, ShapScatterSpec
from language_reading_predictors.models.lgbm_pipeline import LGBMPipeline


# ── predictor selection steps (shared by all variants) ───────────────────
#
# LRP08 has not yet been through iterative feature selection. When
# selection variants are introduced, record their rationale here as
# ``SelectionStep`` entries and chain from ``LRP08`` the same way
# ``lrp02.py`` / ``lrp04.py`` / ``lrp06.py`` does.

_SELECTION_STEPS: list[SelectionStep] = []


# ── hyperparameter sets ─────────────────────────────────────────────────

# Baseline — no tuning has been run for LRP08 yet. Reasonable defaults
# give the feature-selection work a reproducible starting point. Use
# ``python scripts/tune_model.py lrp08`` to produce a tuned set and
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


class LRP08(LevelModel):
    """Receptive vocabulary level predictors — baseline (all data, untuned).

    Uses the full :attr:`Predictors.DEFAULT_LEVEL` predictor set
    (minus the target ``rowpvt``) and a reasonable
    ``_LGBM_BASELINE_PARAMS`` set. Serves as the starting point for
    feature-selection and tuning work on the receptive vocabulary
    level-prediction task.
    """

    model_id = "lrp08"
    target_var = V.ROWPVT
    description = (
        "LightGBM — receptive vocabulary level predictors "
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
        "Baseline exploratory model for receptive vocabulary level "
        "(rowpvt). Uses the full default level predictor set (minus "
        "the target) without outlier exclusion, and a reasonable "
        "_LGBM_BASELINE_PARAMS starting point — no feature selection "
        "or hyperparameter tuning has been applied yet. Target is "
        "essentially symmetric / near-Gaussian (skew 0.04, no floor "
        "or ceiling) — cleanest target distribution of any LRP model "
        "to date."
    )
