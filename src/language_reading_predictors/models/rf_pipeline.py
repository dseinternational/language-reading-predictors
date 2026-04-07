# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Generic Random Forest regression pipeline.

All pipeline steps read configuration from ``context.config`` (a ``ModelConfig``),
so the same code works for any target / predictor / hyperparameter combination.
Individual steps can be called from a notebook for debugging.
"""

import json
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
from language_reading_predictors.models.common import (
    ModelConfig,
    ModelFitContext,
    RunConfig,
)

_ROOT_DIR = Path(__file__).resolve().parent.parent.parent.parent
_DOCS_DIR = _ROOT_DIR / "docs"
_OUTPUT_DIR = _ROOT_DIR / "output" / "models"
_REPORT_TEMPLATE = _DOCS_DIR / "models" / "index.qmd"


# ── pipeline steps ───────────────────────────────────────────────────────────


def prepare_data(context: ModelFitContext) -> None:
    """Load data, optionally exclude outliers, and set X / y / groups."""
    _section("Prepare data")

    cfg = context.config
    df = data_utils.load_data()

    # Keep only rows with a valid target value
    df = df[df[cfg.target_var].notna()].copy()

    # Exclude outliers
    if cfg.outlier_threshold is not None:
        df = df[df[cfg.target_var] < cfg.outlier_threshold]

    context.df = df
    context.X = df[cfg.predictor_vars]
    context.y = df[cfg.target_var]
    context.groups = df[vars.SUBJECT_ID]

    print(f"  Observations: {len(df)}")
    if cfg.outlier_threshold is not None:
        print(f"  Target: {cfg.target_var} (outlier threshold < {cfg.outlier_threshold})")
    else:
        print(f"  Target: {cfg.target_var}")
    print(f"  Predictors ({len(cfg.predictor_vars)}): {cfg.predictor_vars}")

    desc = context.y.describe()
    context.dataframes["target_describe"] = pd.DataFrame(desc)
    desc.to_csv(context.output_dir / "target_describe.csv")


def configure_model(context: ModelFitContext) -> None:
    """Build the sklearn pipeline."""
    _section("Configure model")

    cfg = context.config
    run = context.run_config

    rf_params = dict(cfg.rf_params)
    if run.n_estimators is not None:
        rf_params["n_estimators"] = run.n_estimators

    rf = RandomForestRegressor(
        **rf_params,
        random_state=cfg.random_seed,
    )
    context.pipeline = Pipeline([("rf", rf)])

    print(f"  RandomForestRegressor: {rf_params}")

    cv_splits = run.cv_splits if run.cv_splits is not None else cfg.cv_splits
    print(f"  CV: GroupKFold(n_splits={cv_splits})")


def cross_validate(context: ModelFitContext) -> None:
    """Run group-aware cross-validation and save scores."""
    _section("Cross-validation")

    cfg = context.config
    run = context.run_config
    cv_splits = run.cv_splits if run.cv_splits is not None else cfg.cv_splits
    cv = GroupKFold(n_splits=cv_splits)
    scores = cross_val_score(
        context.pipeline,
        context.X,
        context.y,
        cv=cv,
        groups=context.groups,
        scoring="neg_root_mean_squared_error",
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
    _section("Fit model")

    context.pipeline.fit(context.X, context.y)
    print("  Model fitted on full dataset.")


def evaluate(context: ModelFitContext) -> None:
    """Generate predictions, compute residuals, and save evaluation DataFrame."""
    _section("Evaluate")

    y_pred = context.pipeline.predict(context.X)

    eval_df = pd.DataFrame(
        {
            vars.SUBJECT_ID: context.groups,
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
    _section("Permutation importance")

    cfg = context.config
    run = context.run_config
    n_repeats = (
        run.perm_importance_repeats
        if run.perm_importance_repeats is not None
        else cfg.perm_importance_repeats
    )
    result = permutation_importance(
        context.pipeline,
        context.X,
        context.y,
        n_repeats=n_repeats,
        random_state=cfg.random_seed,
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
    _section("SHAP analysis")

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
    _section("Partial dependence plots")

    cfg = context.config
    pdp_features = cfg.pdp_features if cfg.pdp_features else cfg.predictor_vars

    X_fp = context.X.astype("float64").copy()

    imp_mean_dict = dict(
        zip(
            context.perm_importance_df["feature"],
            context.perm_importance_df["importance_mean"],
        )
    )
    features_ordered = sorted(
        pdp_features, key=lambda f: imp_mean_dict.get(f, 0.0), reverse=True
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
        ax.set_ylabel(f"Predicted {cfg.target_var}")

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


def save_config(context: ModelFitContext) -> None:
    """Save model configuration as JSON for the report template."""
    cfg = context.config
    run = context.run_config
    config_dict = {
        "model_id": cfg.model_id,
        "description": cfg.description,
        "target_var": cfg.target_var,
        "predictor_vars": cfg.predictor_vars,
        "rf_params": {k: _json_safe(v) for k, v in cfg.rf_params.items()},
        "cv_splits": cfg.cv_splits,
        "outlier_threshold": cfg.outlier_threshold,
        "perm_importance_repeats": cfg.perm_importance_repeats,
        "random_seed": cfg.random_seed,
        "run_config": run.name,
    }

    # Note effective overrides applied by the run config
    overrides = {}
    if run.n_estimators is not None:
        overrides["n_estimators"] = run.n_estimators
    if run.cv_splits is not None:
        overrides["cv_splits"] = run.cv_splits
    if run.perm_importance_repeats is not None:
        overrides["perm_importance_repeats"] = run.perm_importance_repeats
    if run.skip_shap:
        overrides["skip_shap"] = True
    if run.skip_pdp:
        overrides["skip_pdp"] = True
    if overrides:
        config_dict["run_config_overrides"] = overrides

    config_path = context.output_dir / "config.json"
    config_path.write_text(json.dumps(config_dict, indent=2))


def report(context: ModelFitContext) -> None:
    """Copy the shared Quarto report template into the output directory."""
    _section("Report")

    if not _REPORT_TEMPLATE.exists():
        print(f"  [bold red]Report template not found: {_REPORT_TEMPLATE}[/bold red]")
        return

    qmd_dest = context.output_dir / "index.qmd"
    shutil.copy(_REPORT_TEMPLATE, qmd_dest)

    print(f"  Report template copied to: {qmd_dest}")
    print(
        f"\n  [bold yellow]To render:[/bold yellow] [blue]quarto render {qmd_dest}[/blue]"
    )
    print(
        f"  [bold yellow]To preview:[/bold yellow] [blue]quarto preview {qmd_dest}[/blue]"
    )


# ── orchestrator ─────────────────────────────────────────────────────────────


def fit(config: ModelConfig, run_config: str = "reporting") -> ModelFitContext:
    """Run the full RF pipeline: data → fit → evaluate → explain → report."""
    run = RunConfig.from_name(run_config)

    print(
        "\n[green]============================================================[/green]"
    )
    print(f"[bold green]Model {config.model_id.upper()}: {config.description}[/bold green]")
    print(f"[bold green]Run config: {run.name}[/bold green]")
    print("[green]============================================================[/green]")

    output_dir = _OUTPUT_DIR / config.model_id

    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    context = ModelFitContext(config=config, run_config=run, output_dir=output_dir)

    prepare_data(context)
    configure_model(context)
    cross_validate(context)
    fit_model(context)
    evaluate(context)
    permutation_importance_analysis(context)

    if not run.skip_shap:
        shap_analysis(context)
    else:
        print(f"\n  [yellow]SHAP analysis skipped (run config: {run.name})[/yellow]")

    if not run.skip_pdp:
        partial_dependence_plots(context)
    else:
        print(f"  [yellow]PDP skipped (run config: {run.name})[/yellow]")

    save_config(context)
    report(context)

    print(
        "\n[green]============================================================[/green]"
    )
    print(f"[bold green]Done. Artifacts saved to: {output_dir}[/bold green]")
    print("[green]============================================================[/green]")

    return context


# ── helpers ──────────────────────────────────────────────────────────────────


def _section(title: str) -> None:
    print(
        "\n[green]------------------------------------------------------------[/green]"
    )
    print(f"[bold green]{title}[/bold green]")
    print("[green]------------------------------------------------------------[/green]")


def _json_safe(v):
    """Convert numpy/non-serialisable values for JSON."""
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return float(v)
    if isinstance(v, np.ndarray):
        return v.tolist()
    return v
