# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Model registry — all model configurations in one place.

Each model is a ``ModelConfig`` registered via ``_register()``.  Shared
hyperparameters and predictor sets are defined once; individual models
override only what differs. ``pipeline_cls`` on each ``ModelConfig``
selects which estimator pipeline class runs the model (e.g. ``RFPipeline``
or ``LGBMPipeline``).
"""

from typing import Any

from language_reading_predictors.data_variables import Predictors
from language_reading_predictors.models.common import ModelConfig
from language_reading_predictors.models.rf_pipeline import RFPipeline

# ── shared defaults ──────────────────────────────────────────────────────

DEFAULT_RF_PARAMS: dict[str, Any] = dict(
    n_estimators=1200,
    max_depth=8,
    min_samples_leaf=16,
    min_samples_split=4,
    max_features=0.5,
    bootstrap=False,
    criterion="squared_error",
    n_jobs=16,
)

DEFAULT_LGBM_PARAMS: dict[str, Any] = dict(
    n_estimators=1200,
    learning_rate=0.05,
    num_leaves=15,
    max_depth=6,
    min_child_samples=16,
    subsample=0.8,
    subsample_freq=1,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=0.1,
    n_jobs=16,
    verbosity=-1,
)

# ── registry ─────────────────────────────────────────────────────────────

MODELS: dict[str, ModelConfig] = {}


def _register(config: ModelConfig) -> ModelConfig:
    """Add a model config to the global registry."""
    MODELS[config.model_id] = config
    return config


# ── helpers ──────────────────────────────────────────────────────────────


def _predictors(
    target_var: str,
    base: list[str],
    *,
    include: list[str] | None = None,
    exclude: list[str] | None = None,
) -> list[str]:
    """Build a predictor list from a base list with adjustments.

    - The target variable is always removed from predictors.
    - ``include`` vars are prepended (if not already present).
    - ``exclude`` vars are removed.
    """
    exclude_set = {target_var}
    if exclude:
        exclude_set.update(exclude)

    predictors = [p for p in base if p not in exclude_set]

    if include:
        extra = [v for v in include if v not in predictors and v != target_var]
        predictors = extra + predictors

    return predictors


def _estimator_label(pipeline_cls: type) -> str:
    """Human-readable estimator name used in default descriptions."""
    return {
        "RFPipeline": "Random Forest",
        "LGBMPipeline": "LightGBM",
    }.get(pipeline_cls.__name__, pipeline_cls.__name__)


def _gain_model(
    model_id: str,
    target_gain_var: str,
    *,
    description: str | None = None,
    include: list[str] | None = None,
    exclude: list[str] | None = None,
    cv_splits: int = 51,
    outlier_threshold: float | None = None,
    pdp_features: list[str] | None = None,
    pipeline_cls: type = RFPipeline,
    params: dict[str, Any] | None = None,
    **kwargs: Any,
) -> ModelConfig:
    """Create and register a standard gain-prediction model.

    The base variable (e.g. ``ewrswr`` for ``ewrswr_gain``) is automatically
    included as a predictor unless explicitly excluded.
    """
    base_var = target_gain_var.removesuffix("_gain")
    inc = list(include or [])
    if base_var not in inc:
        inc.insert(0, base_var)

    if description is None:
        description = f"{_estimator_label(pipeline_cls)} — {base_var} gain predictors"

    model_params = params if params is not None else DEFAULT_RF_PARAMS

    config = ModelConfig(
        model_id=model_id,
        description=description,
        target_var=target_gain_var,
        predictor_vars=_predictors(
            target_gain_var,
            Predictors.DEFAULT_GAIN,
            include=inc,
            exclude=exclude,
        ),
        model_params=model_params,
        pipeline_cls=pipeline_cls,
        cv_splits=cv_splits,
        outlier_threshold=outlier_threshold,
        pdp_features=pdp_features,
        **kwargs,
    )
    return _register(config)


def _level_model(
    model_id: str,
    target_var: str,
    *,
    description: str | None = None,
    include: list[str] | None = None,
    exclude: list[str] | None = None,
    cv_splits: int = 51,
    pdp_features: list[str] | None = None,
    pipeline_cls: type = RFPipeline,
    params: dict[str, Any] | None = None,
    **kwargs: Any,
) -> ModelConfig:
    """Create and register a standard level-prediction model."""
    if description is None:
        description = f"{_estimator_label(pipeline_cls)} — {target_var} level predictors"

    model_params = params if params is not None else DEFAULT_RF_PARAMS

    config = ModelConfig(
        model_id=model_id,
        description=description,
        target_var=target_var,
        predictor_vars=_predictors(
            target_var,
            Predictors.DEFAULT_LEVEL,
            include=include,
            exclude=exclude,
        ),
        model_params=model_params,
        pipeline_cls=pipeline_cls,
        cv_splits=cv_splits,
        pdp_features=pdp_features,
        **kwargs,
    )
    return _register(config)


# ── model definitions ────────────────────────────────────────────────────
#
# Inline definitions live in per-problem modules (``models/lrp01.py``,
# ``models/lrp02.py``), which are imported by ``models/__init__.py`` so that
# the helpers above run at import time and populate ``MODELS``.
