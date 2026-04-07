# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Model LRP01: Random Forest regression predicting word-reading gains,
excluding outliers (ewrswr_gain >= 15).

Pipeline steps can be called individually from a notebook or orchestrated
via the ``fit()`` entry-point (used by ``scripts/fit_model.py``).
"""

import math
import shutil
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from rich import print
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import partial_dependence, permutation_importance
from sklearn.model_selection import GroupKFold, cross_val_score
from sklearn.pipeline import Pipeline

import language_reading_predictors.data_utils as data_utils
from language_reading_predictors.data_variables import Variables as vars
from language_reading_predictors.models.common import ModelFitContext

# ── constants ────────────────────────────────────────────────────────────────

MODEL_ID = "lrp01"

TARGET_VAR = "ewrswr_gain"
OUTLIER_THRESHOLD = 15.0

PREDICTOR_VARS = [
    "time",
    "age",
    "attend",
    "hearing",
    "vision",
    "behav",
    "ewrswr",
    "rowpvt",
    "eowpvt",
    "yarclet",
    "spphon",
    "blending",
    "nonword",
    "trog",
    "aptgram",
    "aptinfo",
    "b1exto",
    "b1reto",
    "b2exto",
    "b2reto",
    "celf",
    "deappin",
    "deappvo",
    "deappfi",
    "deappav",
]

RF_PARAMS = dict(
    n_estimators=1200,
    max_depth=8,
    min_samples_leaf=16,
    min_samples_split=4,
    max_features=0.5,
    bootstrap=False,
    criterion="squared_error",
    n_jobs=16,
)

CV_SPLITS = 53
CV_SCORING = "neg_root_mean_squared_error"
PERM_IMPORTANCE_REPEATS = 50

PLOT_FEATURES = [
    "age",
    "attend",
    "yarclet",
    "celf",
    "trog",
    "eowpvt",
    "b1exto",
    "blending",
    "time",
    "nonword",
    "b1reto",
    "b2reto",
    "aptinfo",
    "deappin",
    "aptgram",
    "rowpvt",
    "behav",
    "b2exto",
    "vision",
]


# ── pipeline steps ───────────────────────────────────────────────────────────


def prepare_data(context: ModelFitContext) -> None:
    """Load data, exclude outliers, and set X / y / groups on the context."""
    print(
        "\n[green]------------------------------------------------------------[/green]"
    )
    print("[bold green]Prepare data[/bold green]")
    print("[green]------------------------------------------------------------[/green]")

    df = data_utils.load_data()

    # Keep only rows with a valid target value
    df = df[df[TARGET_VAR].notna()].copy()

    # Exclude outliers
    df = df[df[TARGET_VAR] < OUTLIER_THRESHOLD]

    context.df = df
    context.X = df[PREDICTOR_VARS]
    context.y = df[TARGET_VAR]
    context.groups = df[vars.SUBJECT_ID]
    context.predictor_vars = PREDICTOR_VARS
    context.target_var = TARGET_VAR

    print(f"  Observations: {len(df)}")
    print(f"  Target: {TARGET_VAR} (outlier threshold < {OUTLIER_THRESHOLD})")
    print(f"  Predictors ({len(PREDICTOR_VARS)}): {PREDICTOR_VARS}")

    desc = context.y.describe()
    context.dataframes["target_describe"] = pd.DataFrame(desc)
    desc.to_csv(context.output_dir / "target_describe.csv")


def configure_model(context: ModelFitContext) -> None:
    """Build the sklearn pipeline and GroupKFold CV splitter."""
    print(
        "\n[green]------------------------------------------------------------[/green]"
    )
    print("[bold green]Configure model[/bold green]")
    print("[green]------------------------------------------------------------[/green]")

    rf = RandomForestRegressor(
        **RF_PARAMS,
        random_state=context.random_seed,
    )
    context.pipeline = Pipeline([("rf", rf)])

    print(f"  RandomForestRegressor: {RF_PARAMS}")
    print(f"  CV: GroupKFold(n_splits={CV_SPLITS})")


def cross_validate(context: ModelFitContext) -> None:
    """Run group-aware cross-validation and save scores."""
    print(
        "\n[green]------------------------------------------------------------[/green]"
    )
    print("[bold green]Cross-validation[/bold green]")
    print("[green]------------------------------------------------------------[/green]")

    cv = GroupKFold(n_splits=CV_SPLITS)
    scores = cross_val_score(
        context.pipeline,
        context.X,
        context.y,
        cv=cv,
        groups=context.groups,
        scoring=CV_SCORING,
    )

    context.cv_scores = scores

    rmse_scores = -scores
    print(f"  Group-aware CV RMSE: {rmse_scores}")
    print(f"  Mean RMSE: {rmse_scores.mean():.4f}")
    print(f"  Std  RMSE: {rmse_scores.std():.4f}")

    scores_df = pd.DataFrame(
        {"fold": range(1, len(rmse_scores) + 1), "rmse": rmse_scores}
    )
    scores_df.to_csv(context.output_dir / "cv_scores.csv", index=False)
    context.dataframes["cv_scores"] = scores_df


def fit_model(context: ModelFitContext) -> None:
    """Fit the pipeline on the full dataset."""
    print(
        "\n[green]------------------------------------------------------------[/green]"
    )
    print("[bold green]Fit model[/bold green]")
    print("[green]------------------------------------------------------------[/green]")

    context.pipeline.fit(context.X, context.y)
    print("  Model fitted on full dataset.")


def evaluate(context: ModelFitContext) -> None:
    """Generate predictions, compute residuals, and save evaluation DataFrame."""
    print(
        "\n[green]------------------------------------------------------------[/green]"
    )
    print("[bold green]Evaluate[/bold green]")
    print("[green]------------------------------------------------------------[/green]")

    y_pred = context.pipeline.predict(context.X)

    eval_df = pd.DataFrame(
        {
            "group": context.groups,
            "y_true": context.y,
            "y_pred": y_pred,
        }
    )
    eval_df["residual"] = eval_df["y_true"] - eval_df["y_pred"]
    eval_df["abs_residual"] = eval_df["residual"].abs()
    eval_df["sq_error"] = (eval_df["y_true"] - eval_df["y_pred"]) ** 2

    context.eval_df = eval_df
    context.dataframes["evaluation"] = eval_df

    eval_df.to_csv(context.output_dir / "evaluation.csv", index=False)

    mae = eval_df["abs_residual"].mean()
    rmse = np.sqrt(eval_df["sq_error"].mean())
    print(f"  In-sample MAE:  {mae:.4f}")
    print(f"  In-sample RMSE: {rmse:.4f}")


def permutation_importance_analysis(context: ModelFitContext) -> None:
    """Compute permutation importance and save results."""
    print(
        "\n[green]------------------------------------------------------------[/green]"
    )
    print("[bold green]Permutation importance[/bold green]")
    print("[green]------------------------------------------------------------[/green]")

    result = permutation_importance(
        context.pipeline,
        context.X,
        context.y,
        n_repeats=PERM_IMPORTANCE_REPEATS,
        random_state=context.random_seed,
    )

    perm_df = pd.DataFrame(
        {
            "feature": context.X.columns,
            "importance_mean": result.importances_mean,
            "importance_std": result.importances_std,
        }
    ).sort_values("importance_mean", ascending=False)

    context.perm_importance_df = perm_df
    context.dataframes["permutation_importance"] = perm_df

    perm_df.to_csv(context.output_dir / "permutation_importance.csv", index=False)

    print(perm_df.to_string(index=False))


def shap_analysis(context: ModelFitContext) -> None:
    """Compute SHAP values and save bar, summary, and waterfall plots."""
    print(
        "\n[green]------------------------------------------------------------[/green]"
    )
    print("[bold green]SHAP analysis[/bold green]")
    print("[green]------------------------------------------------------------[/green]")

    # Constrained layout causes issues with SHAP plots
    mpl.rcParams["figure.constrained_layout.use"] = False

    X_shap = context.X.copy()
    X_shap = X_shap.replace({pd.NA: np.nan})
    X_shap = X_shap.astype("float64")
    X_shap = X_shap.fillna(X_shap.mean())

    rf_fitted = context.pipeline.named_steps["rf"]
    explainer = shap.TreeExplainer(rf_fitted)
    shap_vals = explainer.shap_values(X_shap)

    context.shap_values = shap_vals
    context.shap_explainer = explainer

    explanation = explainer(X_shap)

    # Bar plot
    fig_bar = plt.figure()
    shap.plots.bar(explanation, max_display=12, show=False)
    fig_bar.savefig(
        context.output_dir / "shap_bar.png", dpi=300, bbox_inches="tight"
    )
    context.plots["shap_bar"] = fig_bar
    plt.close(fig_bar)

    # Summary (beeswarm) plot
    plt.figure()
    shap.summary_plot(shap_vals, X_shap, plot_size=0.25, show=False)
    plt.savefig(
        context.output_dir / "shap_summary.png", dpi=300, bbox_inches="tight"
    )
    context.plots["shap_summary"] = plt.gcf()
    plt.close("all")

    # Waterfall plot for most representative observation
    eval_df = context.eval_df
    median_vec = X_shap.median(axis=0)
    dist_from_median = (X_shap - median_vec).abs().sum(axis=1)
    eval_df = eval_df.copy()
    eval_df["dist_from_median"] = dist_from_median.values
    eval_df["representative_score"] = (
        eval_df["abs_residual"] / eval_df["abs_residual"].max()
        + eval_df["dist_from_median"] / eval_df["dist_from_median"].max()
    )
    best_idx = eval_df["representative_score"].idxmin()
    pos_idx = eval_df.index.get_loc(best_idx)

    plt.figure()
    shap.plots.waterfall(explanation[pos_idx], max_display=12, show=False)
    plt.savefig(
        context.output_dir / "shap_waterfall.png", dpi=300, bbox_inches="tight"
    )
    context.plots["shap_waterfall"] = plt.gcf()
    plt.close("all")

    # Clustered bar plot
    clustering = shap.utils.hclust(X_shap, context.y)
    plt.figure()
    shap.plots.bar(
        explanation,
        clustering=clustering,
        clustering_cutoff=0.4,
        max_display=15,
        show=False,
    )
    plt.savefig(
        context.output_dir / "shap_bar_clustered.png", dpi=300, bbox_inches="tight"
    )
    context.plots["shap_bar_clustered"] = plt.gcf()
    plt.close("all")

    print("  SHAP plots saved.")


def partial_dependence_plots(context: ModelFitContext) -> None:
    """Generate partial dependence plots for key features."""
    print(
        "\n[green]------------------------------------------------------------[/green]"
    )
    print("[bold green]Partial dependence plots[/bold green]")
    print("[green]------------------------------------------------------------[/green]")

    X_fp = context.X.astype("float32").copy()

    imp_mean_dict = dict(
        zip(
            context.perm_importance_df["feature"],
            context.perm_importance_df["importance_mean"],
        )
    )
    features_ordered = sorted(
        PLOT_FEATURES, key=lambda f: imp_mean_dict.get(f, 0.0), reverse=True
    )

    n_features = len(features_ordered)
    n_cols = 3
    n_rows = math.ceil(n_features / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4 * n_rows))
    axes = axes.reshape(n_rows, n_cols)

    for idx, feature in enumerate(features_ordered):
        row = idx // n_cols
        col = idx % n_cols
        ax = axes[row, col]

        pdp = partial_dependence(context.pipeline, X_fp, [feature])
        grid = pdp["grid_values"][0]
        avg = pdp["average"][0]

        ax.plot(grid, avg)
        ax.set_title(f"Partial dependence of {feature}")
        ax.set_xlabel(feature)
        ax.set_ylabel("Predicted reading gain")
        ax.set_ylim(1.0, 5.0)

    # Hide empty subplots
    for j in range(n_features, n_rows * n_cols):
        row = j // n_cols
        col = j % n_cols
        axes[row, col].axis("off")

    fig.tight_layout()
    fig.savefig(
        context.output_dir / "partial_dependence.png", dpi=300, bbox_inches="tight"
    )
    context.plots["partial_dependence"] = fig
    plt.close(fig)

    print("  Partial dependence plots saved.")


# ── orchestrator ─────────────────────────────────────────────────────────────


def fit() -> ModelFitContext:
    """Run the full LRP01 pipeline: data → fit → evaluate → explain."""
    print(
        "\n[green]============================================================[/green]"
    )
    print(
        "[bold green]Model LRP01: Random Forest — word-reading gain predictors[/bold green]"
    )
    print("[green]============================================================[/green]")

    output_dir = _resolve_output_dir()

    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    context = ModelFitContext(
        model_id=MODEL_ID,
        output_dir=output_dir,
    )

    prepare_data(context)
    configure_model(context)
    cross_validate(context)
    fit_model(context)
    evaluate(context)
    permutation_importance_analysis(context)
    shap_analysis(context)
    partial_dependence_plots(context)

    print(
        "\n[green]============================================================[/green]"
    )
    print(f"[bold green]Done. Artifacts saved to: {output_dir}[/bold green]")
    print("[green]============================================================[/green]")

    return context


def _resolve_output_dir() -> Path:
    return Path(__file__).resolve().parent.parent.parent.parent / "output" / "models" / MODEL_ID
