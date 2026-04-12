# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Model registry — shared defaults and the global ``MODELS`` dict.

``MODELS`` is populated automatically when model-definition classes
(subclasses of ``ModelDefinition``) are imported. This module re-exports
the dict from ``base_model`` so downstream code can still do::

    from language_reading_predictors.models.registry import MODELS
"""

from typing import Any

from language_reading_predictors.models.base_model import MODELS

# ── shared defaults ──────────────────────────────────────────────────────

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

__all__ = ["MODELS", "DEFAULT_LGBM_PARAMS"]
