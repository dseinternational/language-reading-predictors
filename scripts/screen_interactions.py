# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Reproducible SHAP-interaction screen for the reading models.

A discovery utility (the first step of the project's two-step philosophy): it
ranks candidate pairwise interactions so the moderator choice for the Bayesian
suite is *auditable*, not hand-picked. It is not inferential — see the caveats
in  (pooled rows,
SHAP attributing link-curvature to "interactions"; confirm any signal in a
Bayesian model with subject random effects).

Method (matches the note's spec):

- Fit ``LGBMRegressor(objective="mae", n_estimators=300, learning_rate=0.03)``
  on a target with a fixed predictor list. SHAP is computed on the FULL fit
  (standard for discovery); GroupKFold (by ``subject_id``) is used only for a
  reported cross-validated MAE.
- ``shap.TreeExplainer(model).shap_interaction_values(X)`` returns an
  ``(n_obs, n_feat, n_feat)`` array. Per-feature main effect is the mean
  absolute *diagonal*; pairwise interaction strength is the mean absolute
  *off-diagonal* ``mean_s |inter[s, i, j]|`` for ``i < j``, ranked descending.

Two targets are screened with the canonical registry predictor sets:

- ``ewrswr`` (reading level)      -> ``Predictors.DEFAULT_LEVEL``
- ``ewrswr_gain`` (reading gain)  -> ``Predictors.DEFAULT_GAIN``

(``DEFAULT_LEVEL`` excludes period-related dose variables such as ``attend``;
``DEFAULT_GAIN`` keeps them.)

Outputs, under ``output/interaction_screen/<level|gain>/``:

- ``main_effects.csv``        — feature, mean |diagonal SHAP|
- ``interactions_ranked.csv`` — feature_a, feature_b, strength (mean |off-diag|)
- ``cv_mae.txt``              — GroupKFold mean MAE (+/- sd) and the full spec

Usage::

    python scripts/screen_interactions.py
    python scripts/screen_interactions.py --target level
    python scripts/screen_interactions.py --n-estimators 300 --learning-rate 0.03
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd
import shap
from lightgbm import LGBMRegressor
from rich import print as rprint
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import GroupKFold

from language_reading_predictors import data_utils
from language_reading_predictors.data_utils import DEFAULT_GROUPKFOLD_SPLITS
from language_reading_predictors.data_variables import Predictors, Variables
from language_reading_predictors.models._reporting import (
    print_table,
    ranked_dataframe_table,
    section_header,
)
from language_reading_predictors.statistical_models.environment import OUTPUT_DIR

RANDOM_SEED = 47
_SCREEN_DIR = os.path.join(OUTPUT_DIR, "interaction_screen")

# target key -> (target column, predictor list, human label)
TARGETS: dict[str, tuple[str, list[str], str]] = {
    "level": (Variables.EWRSWR, Predictors.DEFAULT_LEVEL, "reading level (ewrswr)"),
    "gain": (Variables.EWRSWR_GAIN, Predictors.DEFAULT_GAIN, "reading gain (ewrswr_gain)"),
}


def _prepare_X(X: pd.DataFrame) -> pd.DataFrame:
    """Float64, NaN-filled with column means (mirrors notebooks/0003).

    LightGBM handles NaN natively, but ``shap_interaction_values`` is computed
    on a fully-observed matrix so the explanation aligns with a single fitted
    model. The same matrix is used for the fit and the SHAP computation.
    """
    X = X.replace({pd.NA: np.nan}).astype("float64")
    return X.fillna(X.mean())


def _build_model(n_estimators: int, learning_rate: float) -> LGBMRegressor:
    return LGBMRegressor(
        objective="mae",
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        n_jobs=-1,
        verbosity=-1,
        random_state=RANDOM_SEED,
    )


def _cv_mae(
    X: pd.DataFrame,
    y: pd.Series,
    groups: pd.Series,
    n_estimators: int,
    learning_rate: float,
) -> tuple[float, float, int]:
    """GroupKFold (by subject) mean +/- sd of fold MAE. Reported metric only."""
    n_splits = min(DEFAULT_GROUPKFOLD_SPLITS, groups.nunique())
    cv = GroupKFold(n_splits=n_splits)
    maes: list[float] = []
    for train_idx, test_idx in cv.split(X, y, groups=groups):
        model = _build_model(n_estimators, learning_rate)
        model.fit(X.iloc[train_idx], y.iloc[train_idx])
        pred = model.predict(X.iloc[test_idx])
        maes.append(float(mean_absolute_error(y.iloc[test_idx], pred)))
    return float(np.mean(maes)), float(np.std(maes)), n_splits


def _rank_interactions(
    inter: np.ndarray, feature_names: list[str]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (main_effects, interactions_ranked) from SHAP interaction values.

    ``inter`` has shape ``(n_obs, n_feat, n_feat)``. Main effect of feature i is
    ``mean_s |inter[s, i, i]|``; pairwise strength of (i, j) is
    ``mean_s |inter[s, i, j]|`` for ``i < j`` (the off-diagonal half).
    """
    abs_mean = np.abs(inter).mean(axis=0)  # (n_feat, n_feat)

    main = pd.DataFrame(
        {"feature": feature_names, "mean_abs_main": np.diag(abs_mean)}
    ).sort_values("mean_abs_main", ascending=False, ignore_index=True)

    rows: list[dict] = []
    n = len(feature_names)
    for i in range(n):
        for j in range(i + 1, n):
            rows.append(
                {
                    "feature_a": feature_names[i],
                    "feature_b": feature_names[j],
                    "strength": float(abs_mean[i, j]),
                }
            )
    interactions = pd.DataFrame(rows).sort_values(
        "strength", ascending=False, ignore_index=True
    )
    return main, interactions


def screen_target(
    key: str, n_estimators: int, learning_rate: float, top_n: int = 12
) -> None:
    target, predictors, label = TARGETS[key]
    section_header(f"Interaction screen — {label}")

    # Guard against target leakage: the registry sets include the level column
    # itself (``ewrswr`` is in DEFAULT_LEVEL), which would let the model predict
    # the target from itself. Drop it. For gain the target is ``ewrswr_gain``;
    # the ``ewrswr`` baseline legitimately stays in DEFAULT_GAIN.
    predictors = [p for p in predictors if p != target]

    _df, X, y, groups = data_utils.load_and_filter(target, predictors)
    X = _prepare_X(X)
    feature_names = list(X.columns)
    rprint(
        f"  n_obs={len(X)}  n_children={groups.nunique()}  "
        f"n_features={len(feature_names)}"
    )

    cv_mean, cv_sd, n_splits = _cv_mae(X, y, groups, n_estimators, learning_rate)
    rprint(f"  GroupKFold({n_splits}) MAE: {cv_mean:.3f} +/- {cv_sd:.3f}")

    model = _build_model(n_estimators, learning_rate)
    model.fit(X, y)

    explainer = shap.TreeExplainer(model)
    inter = explainer.shap_interaction_values(X)
    main, interactions = _rank_interactions(inter, feature_names)

    out_dir = os.path.join(_SCREEN_DIR, key)
    os.makedirs(out_dir, exist_ok=True)
    main.to_csv(os.path.join(out_dir, "main_effects.csv"), index=False)
    interactions.to_csv(
        os.path.join(out_dir, "interactions_ranked.csv"), index=False
    )
    with open(os.path.join(out_dir, "cv_mae.txt"), "w") as f:
        f.write(
            f"target={target}\n"
            f"predictors={predictors}\n"
            f"lightgbm: objective=mae, n_estimators={n_estimators}, "
            f"learning_rate={learning_rate}, random_state={RANDOM_SEED}\n"
            f"n_obs={len(X)}, n_children={groups.nunique()}\n"
            f"GroupKFold({n_splits}) MAE = {cv_mean:.4f} +/- {cv_sd:.4f}\n"
        )

    print_table(
        ranked_dataframe_table(
            main, title="Main effects (mean |diagonal SHAP|)", max_rows=top_n
        )
    )
    print_table(
        ranked_dataframe_table(
            interactions,
            title="Top interactions (mean |off-diagonal SHAP|)",
            max_rows=top_n,
        )
    )
    rprint(f"  Wrote {out_dir}/{{main_effects,interactions_ranked}}.csv, cv_mae.txt")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--target",
        choices=["level", "gain", "all"],
        default="all",
        help="Which target(s) to screen. Default: all.",
    )
    parser.add_argument("--n-estimators", type=int, default=300)
    parser.add_argument("--learning-rate", type=float, default=0.03)
    parser.add_argument(
        "--top-n", type=int, default=12, help="Rows printed per ranked table."
    )
    args = parser.parse_args()

    keys = ["level", "gain"] if args.target == "all" else [args.target]
    for key in keys:
        screen_target(key, args.n_estimators, args.learning_rate, top_n=args.top_n)


if __name__ == "__main__":
    main()
