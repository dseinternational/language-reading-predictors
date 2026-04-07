# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Shared dataclasses for the language-reading-predictors model family.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class RunConfig:
    """Runtime configuration that overrides model defaults for speed control."""

    name: str
    """Preset name, e.g. 'dev', 'test', 'reporting'."""

    n_estimators: int | None = None
    """Override for RandomForestRegressor n_estimators. None = use model default."""

    cv_splits: int | None = None
    """Override for GroupKFold n_splits. None = use model default."""

    perm_importance_repeats: int | None = None
    """Override for permutation importance n_repeats. None = use model default."""

    skip_shap: bool = False
    """If True, skip SHAP analysis."""

    skip_pdp: bool = False
    """If True, skip partial dependence plots."""

    @classmethod
    def from_name(cls, name: str) -> "RunConfig":
        """Create a RunConfig from a preset name."""
        presets = {
            "dev": cls(
                name="dev",
                n_estimators=100,
                cv_splits=5,
                perm_importance_repeats=5,
                skip_shap=True,
                skip_pdp=True,
            ),
            "test": cls(
                name="test",
                n_estimators=500,
                cv_splits=10,
                perm_importance_repeats=10,
            ),
            "reporting": cls(name="reporting"),
        }

        key = name.lower()
        aliases = {
            "development": "dev",
            "testing": "test",
            "rep": "reporting",
            "report": "reporting",
        }
        key = aliases.get(key, key)

        if key not in presets:
            valid = ", ".join(presets.keys())
            msg = f"Unknown run config: {name!r}. Valid options: {valid}"
            raise ValueError(msg)

        return presets[key]


@dataclass
class ModelConfig:
    """All per-model configuration — the only thing that differs between models."""

    model_id: str
    """Short model identifier, e.g. 'lrp01'."""

    description: str
    """Human-readable description shown in console output and reports."""

    target_var: str
    """Column name of the target variable."""

    predictor_vars: list[str]
    """Column names of predictor variables."""

    rf_params: dict[str, Any]
    """Keyword arguments passed to ``RandomForestRegressor``."""

    cv_splits: int
    """Number of GroupKFold splits."""

    outlier_threshold: float | None = None
    """If set, exclude rows where target >= this value."""

    perm_importance_repeats: int = 50
    """Number of repeats for permutation importance."""

    pdp_features: list[str] | None = None
    """Features to include in partial dependence plots (defaults to predictor_vars)."""

    random_seed: int = 47


@dataclass
class ModelFitContext:
    """Shared state passed through every pipeline step."""

    config: ModelConfig
    """Model configuration."""

    run_config: RunConfig
    """Runtime configuration (dev/test/reporting)."""

    output_dir: Path
    """Directory where all artifacts for this run are saved."""

    # Populated by pipeline steps
    df: pd.DataFrame | None = None
    X: pd.DataFrame | None = None
    y: pd.Series | None = None
    groups: pd.Series | None = None

    pipeline: Any = None
    cv_scores: np.ndarray | None = None

    eval_df: pd.DataFrame | None = None
    perm_importance_df: pd.DataFrame | None = None

    shap_values: np.ndarray | None = None
    shap_explainer: Any = None

    plots: dict[str, Any] = field(default_factory=dict)
    dataframes: dict[str, pd.DataFrame] = field(default_factory=dict)
