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
class SelectionStep:
    """Record of a single feature-selection decision.

    Used to document why features were added or removed between model
    variants so the full selection history is persisted in ``config.json``.
    """

    removed: list[str] = field(default_factory=list)
    """Features removed in this selection step."""

    added: list[str] = field(default_factory=list)
    """Features added (e.g. composites) in this selection step."""

    notes: str = ""
    """Free-text rationale for this selection decision."""

    date: str = ""
    """ISO date when this step was taken (e.g. ``"2026-04-16"``)."""

    metrics_before: dict[str, float] = field(default_factory=dict)
    """CV metrics snapshot before this step (e.g.
    ``{"cv_rmse_mean": 3.45, "cv_rmse_std": 0.52}``)."""

    metrics_after: dict[str, float] = field(default_factory=dict)
    """CV metrics snapshot after this step (e.g.
    ``{"cv_rmse_mean": 3.31, "cv_rmse_std": 0.54}``)."""


@dataclass
class RunConfig:
    """Runtime configuration that overrides model defaults for speed control."""

    name: str
    """Preset name, e.g. 'dev', 'test', 'reporting'."""

    n_estimators: int | None = None
    """Override for the estimator's n_estimators. None = use model default."""

    cv_splits: int | None = None
    """Override for GroupKFold n_splits. None = use model default."""

    perm_importance_repeats: int | None = None
    """Override for permutation importance n_repeats. None = use model default."""

    skip_shap: bool = False
    """If True, skip SHAP analysis."""

    skip_pdp: bool = False
    """If True, skip partial dependence plots."""

    skip_correlation: bool = False
    """If True, skip distance-correlation analysis."""

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
                skip_correlation=True,
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

    model_params: dict[str, Any]
    """Keyword arguments passed to the estimator constructor."""

    pipeline_cls: Any = None
    """Pipeline class used to fit this model (currently ``LGBMPipeline``).
    Typed as ``Any`` to keep ``common`` free of pipeline imports; registry
    helpers set this explicitly."""

    cv_splits: int = 51
    """Number of GroupKFold splits."""

    outlier_threshold: float | None = None
    """If set, exclude rows where target >= this value."""

    perm_importance_repeats: int = 50
    """Number of repeats for permutation importance."""

    pdp_features: list[str] | None = None
    """Features to include in partial dependence plots. If None, auto-selected
    from the top ``pdp_top_n`` features by permutation importance."""

    pdp_top_n: int = 15
    """Number of top features (by permutation importance) to include in partial
    dependence plots when ``pdp_features`` is not set explicitly."""

    random_seed: int = 47

    variant_of: str | None = None
    """If set, this model is a selection variant of another model (e.g.
    'lrp01'). Variants are excluded from ``fit_model.py all`` by default."""

    notes: str = ""
    """Free-text rationale stored in ``config.json`` and surfaced in reports —
    used for selection-decision provenance."""

    selection_history: list[SelectionStep] = field(default_factory=list)
    """Ordered list of feature-selection decisions leading to this model's
    predictor set. Persisted in ``config.json`` for provenance."""


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
    cv_results: dict[str, np.ndarray] | None = None

    eval_df: pd.DataFrame | None = None
    perm_importance_df: pd.DataFrame | None = None

    shap_values: np.ndarray | None = None
    shap_explainer: Any = None

    plots: dict[str, Any] = field(default_factory=dict)
    dataframes: dict[str, pd.DataFrame] = field(default_factory=dict)
