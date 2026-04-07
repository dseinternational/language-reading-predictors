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
class ModelFitContext:
    """Shared state passed through every pipeline step."""

    model_id: str
    """Short model identifier, e.g. 'lrp01'."""

    output_dir: Path
    """Directory where all artifacts for this run are saved."""

    random_seed: int = 47

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

    predictor_vars: list[str] = field(default_factory=list)
    target_var: str = ""

    plots: dict[str, Any] = field(default_factory=dict)
    dataframes: dict[str, pd.DataFrame] = field(default_factory=dict)
