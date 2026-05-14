# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Sensitivity check: LRP02 under inverse-frequency subject weighting.

Motivation
----------

``agebooks`` and ``agespeak`` are measured once at baseline (parent
report) and repeated across every timepoint for each child in the long
format. Under the standard fit, each child contributes 3–4 identical
rows of these time-invariant predictors, which inflates their effective
training weight even though no new information is added.

``GroupKFold`` prevents *test-set* leakage (a child's ``agebooks`` value
never appears in both train and test), but during training the model
still sees each child's time-invariant values repeated. This biases
tree splits and permutation importance toward time-invariant features
relative to what would be seen with one row per child.

What this script does
---------------------

1. Loads LRP02's exact data, predictors, and MAE-tuned hyperparameters
   from the model registry.
2. Computes ``sample_weight = 1 / n_timepoints_per_child`` so each
   child contributes a total weight of 1.0.
3. Runs two manual ``GroupKFold`` cross-validations with identical
   splits — one unweighted, one weighted — and records out-of-fold
   predictions.
4. Fits both models on the full dataset and computes permutation
   importance (the weighted model scores permutation importance with
   ``sample_weight`` so the scoring is consistent with the fit).
5. Saves a side-by-side comparison of pooled OOF metrics and
   permutation importance rankings under
   ``output/sensitivity/lrp02_weight/``.

Usage
-----

::

    python scripts/lrp02_weight_sensitivity.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from rich import print
from sklearn.inspection import permutation_importance
from sklearn.model_selection import GroupKFold

import language_reading_predictors.data_utils as data_utils
from language_reading_predictors.models.registry import MODELS

_ROOT_DIR = Path(__file__).resolve().parent.parent
_OUTPUT_DIR = _ROOT_DIR / "output" / "sensitivity" / "lrp02_weight"


def _pooled_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    residuals = y_true - y_pred
    abs_res = np.abs(residuals)
    ss_res = float(np.sum(residuals**2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    return {
        "mae": float(np.mean(abs_res)),
        "rmse": float(np.sqrt(np.mean(residuals**2))),
        "medae": float(np.median(abs_res)),
        "r2": float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan"),
    }


def _oof_predict(
    X: pd.DataFrame,
    y: pd.Series,
    groups: pd.Series,
    params: dict,
    seed: int,
    cv_splits: int,
    weights: np.ndarray | None,
) -> np.ndarray:
    """Manual OOF prediction with optional sample weights.

    Using a manual loop (rather than ``cross_val_predict``) so we can pass
    ``sample_weight`` to ``fit()`` directly without configuring sklearn
    metadata routing. Splits are identical to the unweighted run because
    ``GroupKFold`` is deterministic.
    """
    cv = GroupKFold(n_splits=cv_splits)
    oof = np.empty(len(y), dtype=float)
    # Merge so a ``random_state`` baked into tuned ``params`` does not
    # clash with the explicit kwarg.
    fit_params = {**params, "random_state": seed}
    for train_idx, test_idx in cv.split(X, y, groups=groups):
        est = LGBMRegressor(**fit_params)
        fit_kwargs = {}
        if weights is not None:
            fit_kwargs["sample_weight"] = weights[train_idx]
        est.fit(X.iloc[train_idx], y.iloc[train_idx], **fit_kwargs)
        oof[test_idx] = est.predict(X.iloc[test_idx])
    return oof


def _permutation_importance(
    X: pd.DataFrame,
    y: pd.Series,
    params: dict,
    seed: int,
    weights: np.ndarray | None,
    n_repeats: int,
) -> pd.DataFrame:
    est = LGBMRegressor(**{**params, "random_state": seed})
    fit_kwargs = {"sample_weight": weights} if weights is not None else {}
    est.fit(X, y, **fit_kwargs)
    result = permutation_importance(
        est,
        X,
        y,
        n_repeats=n_repeats,
        random_state=seed,
        sample_weight=weights,
    )
    return pd.DataFrame(
        {
            "feature": X.columns,
            "importance_mean": result.importances_mean,
            "importance_std": result.importances_std,
        }
    )


def main() -> None:
    _OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    cfg = MODELS["lrp02"]
    print(f"Model: {cfg.model_id}")
    print(f"Target: {cfg.target_var}")
    print(f"Predictors ({len(cfg.predictor_vars)}): {cfg.predictor_vars}")

    df, X, y, groups = data_utils.load_and_filter(
        cfg.target_var, cfg.predictor_vars, cfg.outlier_threshold
    )
    print(f"Observations: {len(df)}  |  Subjects: {groups.nunique()}")

    # sample_weight = 1 / n_timepoints_per_child
    # each child contributes total weight 1.0 regardless of timepoint count
    group_sizes = groups.value_counts()
    weights = (1.0 / groups.map(group_sizes)).to_numpy(dtype=float)
    print(
        f"Timepoints per child — min: {group_sizes.min()}, "
        f"max: {group_sizes.max()}, mean: {group_sizes.mean():.2f}"
    )

    # Cap n_estimators if the config has one set — use the registry's pinned value
    # directly (no RunConfig override applied here; sensitivity check mirrors
    # the reporting-config fit).
    params = dict(cfg.model_params)
    seed = cfg.random_seed
    cv_splits = cfg.cv_splits
    n_repeats = cfg.perm_importance_repeats

    # ── OOF predictions ──────────────────────────────────────────────────
    print("\nRunning manual GroupKFold (unweighted) ...")
    oof_unweighted = _oof_predict(
        X, y, groups, params, seed, cv_splits, weights=None
    )
    print("Running manual GroupKFold (weighted) ...")
    oof_weighted = _oof_predict(
        X, y, groups, params, seed, cv_splits, weights=weights
    )

    # Drop-both baseline: refit on the 11-predictor set with agebooks and
    # agespeak removed. Upper-bound estimate of "how much do these features
    # buy" — hyperparameters are not retuned, so if the 11-predictor fit
    # matches or beats the 13-predictor fit the two features carry no
    # unique signal beyond what the remaining 11 predictors cover.
    drop_features = ["agebooks", "agespeak"]
    X_dropped = X.drop(columns=drop_features)
    print(
        f"Running manual GroupKFold (11 predictors, "
        f"{drop_features} dropped) ..."
    )
    oof_dropped = _oof_predict(
        X_dropped, y, groups, params, seed, cv_splits, weights=None
    )

    # ── Pooled metrics (unweighted scoring so both are comparable) ──────
    y_arr = y.to_numpy()
    metrics_unweighted = _pooled_metrics(y_arr, oof_unweighted)
    metrics_weighted = _pooled_metrics(y_arr, oof_weighted)
    metrics_dropped = _pooled_metrics(y_arr, oof_dropped)

    # Also report weighted-scoring metrics for the weighted model — each
    # child contributes equally regardless of how many timepoints they have.
    def _weighted_pooled(y_true, y_pred, w):
        w = w / w.sum()
        residuals = y_true - y_pred
        abs_res = np.abs(residuals)
        mae = float(np.sum(w * abs_res))
        rmse = float(np.sqrt(np.sum(w * residuals**2)))
        y_mean = float(np.sum(w * y_true))
        ss_res = float(np.sum(w * residuals**2))
        ss_tot = float(np.sum(w * (y_true - y_mean) ** 2))
        r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan")
        return {"mae": mae, "rmse": rmse, "r2": r2}

    metrics_weighted_weighted_scoring = _weighted_pooled(
        y_arr, oof_weighted, weights
    )

    # ── Permutation importance ──────────────────────────────────────────
    print("\nComputing permutation importance (unweighted) ...")
    perm_unweighted = _permutation_importance(
        X, y, params, seed, weights=None, n_repeats=n_repeats
    )
    print("Computing permutation importance (weighted) ...")
    perm_weighted = _permutation_importance(
        X, y, params, seed, weights=weights, n_repeats=n_repeats
    )

    # ── Build comparison table ──────────────────────────────────────────
    perm_unweighted = perm_unweighted.rename(
        columns={
            "importance_mean": "imp_unweighted",
            "importance_std": "std_unweighted",
        }
    )
    perm_weighted = perm_weighted.rename(
        columns={
            "importance_mean": "imp_weighted",
            "importance_std": "std_weighted",
        }
    )
    comparison = perm_unweighted.merge(perm_weighted, on="feature")
    comparison["rank_unweighted"] = (
        comparison["imp_unweighted"].rank(ascending=False).astype(int)
    )
    comparison["rank_weighted"] = (
        comparison["imp_weighted"].rank(ascending=False).astype(int)
    )
    comparison["rank_delta"] = (
        comparison["rank_weighted"] - comparison["rank_unweighted"]
    )
    comparison = comparison.sort_values("imp_unweighted", ascending=False)

    comparison_path = _OUTPUT_DIR / "importance_comparison.csv"
    comparison.to_csv(comparison_path, index=False)
    print(f"\nSaved: {comparison_path}")
    print(comparison.to_string(index=False))

    # ── Metrics JSON ────────────────────────────────────────────────────
    metrics = {
        "n_observations": int(len(X)),
        "n_subjects": int(groups.nunique()),
        "cv_splits": int(cv_splits),
        "seed": int(seed),
        "perm_importance_repeats": int(n_repeats),
        "pooled_oof_unweighted_model": metrics_unweighted,
        "pooled_oof_weighted_model_unweighted_scoring": metrics_weighted,
        "pooled_oof_weighted_model_weighted_scoring": (
            metrics_weighted_weighted_scoring
        ),
        "pooled_oof_dropped_agebooks_agespeak": metrics_dropped,
    }
    (_OUTPUT_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"Saved: {_OUTPUT_DIR / 'metrics.json'}")
    print("\nPooled OOF metrics:")
    print(f"  Unweighted model (13 predictors):    {metrics_unweighted}")
    print(f"  Weighted model (unweighted scoring): {metrics_weighted}")
    print(
        f"  Weighted model (weighted scoring):   "
        f"{metrics_weighted_weighted_scoring}"
    )
    print(f"  Dropped agebooks+agespeak (11 pred): {metrics_dropped}")

    # ── Comparison plot ─────────────────────────────────────────────────
    ordered = comparison["feature"].tolist()
    x = np.arange(len(ordered))
    width = 0.4
    fig, ax = plt.subplots(figsize=(9, max(4.0, 0.38 * len(ordered) + 1.5)))
    ax.barh(
        x - width / 2,
        comparison["imp_unweighted"],
        height=width,
        label="Unweighted",
        color="C0",
    )
    ax.barh(
        x + width / 2,
        comparison["imp_weighted"],
        height=width,
        label="Weighted (1/n_timepoints)",
        color="C1",
    )
    ax.set_yticks(x)
    ax.set_yticklabels(ordered)
    ax.invert_yaxis()
    ax.axvline(0.0, color="black", linestyle="--", linewidth=1)
    ax.set_xlabel("Permutation importance (mean decrease in R²)")
    ax.set_ylabel("Predictor")
    ax.set_title("LRP02 permutation importance — unweighted vs subject-weighted")
    ax.legend(loc="lower right")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    plot_path = _OUTPUT_DIR / "importance_comparison.png"
    fig.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {plot_path}")


if __name__ == "__main__":
    main()
