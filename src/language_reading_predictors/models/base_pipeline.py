# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Generic estimator pipeline base class.

All steps read configuration from ``self.context.config`` (a ``ModelConfig``)
so the same code works for any target / predictor / hyperparameter combination.
Individual methods can be called from a notebook for debugging:

    pipeline = LGBMPipeline(config, run_config)
    pipeline.prepare_data()
    pipeline.configure_model()
    pipeline.cross_validate()
    ...

Subclasses override only :meth:`configure_model` to plug in a different
estimator. The fitted estimator is always wrapped in ``Pipeline([(ESTIMATOR_STEP, est)])``
so downstream steps (SHAP, permutation importance, PDP) can look it up by a
consistent name.
"""

import json
import math
import shutil
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shap
from rich import print
from scipy.cluster import hierarchy
from scipy.spatial.distance import squareform
from sklearn.inspection import partial_dependence, permutation_importance
from sklearn.model_selection import GroupKFold, cross_val_score

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

ESTIMATOR_STEP = "est"
"""Name of the estimator step inside the sklearn ``Pipeline``."""


class EstimatorPipeline:
    """Base class for tree-model pipelines.

    Subclasses implement :meth:`configure_model` to construct the estimator
    and wrap it in a sklearn ``Pipeline`` keyed by :data:`ESTIMATOR_STEP`.
    All other steps are estimator-agnostic. ``LGBMPipeline`` is the only
    subclass currently registered.
    """

    def __init__(self, config: ModelConfig, run_config: RunConfig) -> None:
        self.context = ModelFitContext(
            config=config,
            run_config=run_config,
            output_dir=_OUTPUT_DIR / config.model_id,
        )

    # ── pipeline steps ───────────────────────────────────────────────────

    def prepare_data(self) -> None:
        """Load data, optionally exclude outliers, and set X / y / groups.

        Missing values in the predictor frame are cast from pandas nullable
        ``pd.NA`` to numpy ``np.nan`` but left unimputed. LightGBM handles
        NaN natively, and imputing here would discard informative
        missingness (e.g. ``agespeak`` is NaN in 80/152 rows and almost
        certainly correlates with developmental trajectory).
        """
        _section("Prepare data")

        context = self.context
        cfg = context.config
        df = data_utils.load_data()

        df = df[df[cfg.target_var].notna()].copy()

        if cfg.outlier_threshold is not None:
            df = df[df[cfg.target_var] < cfg.outlier_threshold]

        context.df = df
        context.X = (
            df[cfg.predictor_vars].replace({pd.NA: np.nan}).astype("float64")
        )
        context.y = df[cfg.target_var].astype("float64")
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

    def configure_model(self) -> None:
        """Build the sklearn pipeline. Subclasses must override."""
        raise NotImplementedError(
            "EstimatorPipeline subclasses must implement configure_model()."
        )

    def cross_validate(self) -> None:
        """Run group-aware cross-validation and save scores."""
        _section("Cross-validation")

        context = self.context
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

    def fit_model(self) -> None:
        """Fit the pipeline on the full dataset."""
        _section("Fit model")

        context = self.context
        context.pipeline.fit(context.X, context.y)
        print("  Model fitted on full dataset.")

    def evaluate(self) -> None:
        """Generate predictions, compute residuals, and save evaluation DataFrame."""
        _section("Evaluate")

        context = self.context
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

    def permutation_importance_analysis(self) -> None:
        """Compute permutation importance and save results."""
        _section("Permutation importance")

        context = self.context
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

    def correlation_analysis(self) -> None:
        """Compute distance-correlation matrix, dendrogram, and heatmap.

        Uses the same distance-correlation metric as
        ``stats_utils.distance_corr_matrix`` and produces:

        - ``distance_corr_matrix.csv`` — full pairwise matrix.
        - ``distance_corr_dendrogram.png`` — Ward-linkage dendrogram.
        - ``distance_corr_heatmap.png`` — heatmap reordered by dendrogram leaves.
        """
        from language_reading_predictors.stats_utils import distance_corr_matrix

        _section("Distance-correlation analysis")

        context = self.context
        cfg = context.config
        X = context.X

        X_corr = X.replace({pd.NA: np.nan}).astype("float64")
        predictors = list(X_corr.columns)

        print(f"  Computing distance correlation for {len(predictors)} predictors...")
        dcor_matrix = distance_corr_matrix(X_corr)

        pd.DataFrame(dcor_matrix, index=predictors, columns=predictors).to_csv(
            context.output_dir / "distance_corr_matrix.csv"
        )

        # Dissimilarity for clustering
        dcor_dissim = 1.0 - dcor_matrix
        np.fill_diagonal(dcor_dissim, 0.0)
        np.clip(dcor_dissim, 0.0, 1.0, out=dcor_dissim)
        condensed = squareform(dcor_dissim, checks=False)
        linkage = hierarchy.ward(condensed)

        # Dendrogram
        n = len(predictors)
        fig_dendro, ax_dendro = plt.subplots(figsize=(8, max(6, 0.3 * n)))
        dendro = hierarchy.dendrogram(
            linkage,
            labels=predictors,
            orientation="right",
            ax=ax_dendro,
        )
        ax_dendro.set_title(
            f"{cfg.model_id.upper()}: Hierarchical clustering of predictors"
        )
        ax_dendro.set_xlabel("Dissimilarity (1 − distance correlation)")
        fig_dendro.tight_layout()
        fig_dendro.savefig(
            context.output_dir / "distance_corr_dendrogram.png",
            dpi=300,
            bbox_inches="tight",
        )
        context.plots["distance_corr_dendrogram"] = fig_dendro
        plt.close(fig_dendro)

        # Heatmap reordered by dendrogram leaves
        leaf_order = dendro["leaves"]
        ordered_matrix = dcor_matrix[leaf_order, :][:, leaf_order]
        ordered_labels = [predictors[i] for i in leaf_order]

        fig_heat, ax_heat = plt.subplots(
            figsize=(max(8, 0.35 * n), max(6, 0.3 * n))
        )
        sns.heatmap(
            ordered_matrix,
            xticklabels=ordered_labels,
            yticklabels=ordered_labels,
            cmap="viridis",
            square=True,
            cbar_kws={"shrink": 0.6},
            ax=ax_heat,
        )
        ax_heat.set_title(
            f"{cfg.model_id.upper()}: Distance correlation heatmap"
        )
        plt.setp(ax_heat.get_xticklabels(), rotation=45, ha="right")
        fig_heat.tight_layout()
        fig_heat.savefig(
            context.output_dir / "distance_corr_heatmap.png",
            dpi=300,
            bbox_inches="tight",
        )
        context.plots["distance_corr_heatmap"] = fig_heat
        plt.close(fig_heat)

        print("  Distance-correlation plots saved.")

    def shap_analysis(self) -> None:
        """Compute SHAP values and save bar, summary, and waterfall plots."""
        _section("SHAP analysis")

        mpl.rcParams["figure.constrained_layout.use"] = False

        context = self.context
        X_shap = context.X

        estimator = context.pipeline.named_steps[ESTIMATOR_STEP]
        explainer = shap.TreeExplainer(estimator)
        shap_vals = explainer.shap_values(X_shap)

        context.shap_values = shap_vals
        context.shap_explainer = explainer

        explanation = explainer(X_shap)

        fig_bar = plt.figure()
        shap.plots.bar(explanation, max_display=12, show=False)
        fig_bar.savefig(
            context.output_dir / "shap_bar.png", dpi=300, bbox_inches="tight"
        )
        context.plots["shap_bar"] = fig_bar
        plt.close(fig_bar)

        plt.figure()
        shap.summary_plot(shap_vals, X_shap, plot_size=0.25, show=False)
        plt.savefig(
            context.output_dir / "shap_summary.png", dpi=300, bbox_inches="tight"
        )
        context.plots["shap_summary"] = plt.gcf()
        plt.close("all")

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

    def partial_dependence_plots(self) -> None:
        """Generate partial dependence plots for key features."""
        _section("Partial dependence plots")

        context = self.context
        cfg = context.config

        X_fp = context.X

        if cfg.pdp_features:
            pdp_features = cfg.pdp_features
        else:
            pdp_features = (
                context.perm_importance_df
                .head(cfg.pdp_top_n)["feature"]
                .tolist()
            )

        x_cols = set(context.X.columns)
        missing = [f for f in pdp_features if f not in x_cols]
        if missing:
            print(
                f"  [yellow]Skipping PDP features not in predictor set: {missing}[/yellow]"
            )
            pdp_features = [f for f in pdp_features if f in x_cols]

        if not pdp_features:
            print("  [yellow]No PDP features to plot.[/yellow]")
            return

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

    def save_config(self) -> None:
        """Save model configuration as JSON for the report template."""
        context = self.context
        cfg = context.config
        run = context.run_config

        effective_model_params = dict(cfg.model_params)
        if run.n_estimators is not None:
            effective_model_params["n_estimators"] = run.n_estimators
        effective_cv_splits = (
            run.cv_splits if run.cv_splits is not None else cfg.cv_splits
        )
        effective_perm_importance_repeats = (
            run.perm_importance_repeats
            if run.perm_importance_repeats is not None
            else cfg.perm_importance_repeats
        )

        config_dict = {
            "model_id": cfg.model_id,
            "description": cfg.description,
            "pipeline_cls": type(self).__name__,
            "variant_of": cfg.variant_of,
            "notes": cfg.notes,
            "target_var": cfg.target_var,
            "predictor_vars": cfg.predictor_vars,
            "model_params": {
                k: _json_safe(v) for k, v in effective_model_params.items()
            },
            "cv_splits": effective_cv_splits,
            "outlier_threshold": cfg.outlier_threshold,
            "perm_importance_repeats": effective_perm_importance_repeats,
            "random_seed": cfg.random_seed,
            "run_config": run.name,
            "selection_history": [
                {
                    "removed": step.removed,
                    "added": step.added,
                    "notes": step.notes,
                    "date": step.date,
                    "metrics_before": step.metrics_before,
                    "metrics_after": step.metrics_after,
                }
                for step in cfg.selection_history
            ],
        }

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
        if run.skip_correlation:
            overrides["skip_correlation"] = True
        if overrides:
            config_dict["run_config_overrides"] = overrides

        config_path = context.output_dir / "config.json"
        config_path.write_text(json.dumps(config_dict, indent=2))

    def save_metrics(self) -> None:
        """Save aggregated diagnostic metrics for cross-variant comparison."""
        context = self.context
        cfg = context.config
        run = context.run_config

        rmse_scores = -context.cv_scores if context.cv_scores is not None else None
        eval_df = context.eval_df
        in_sample_mae = float(eval_df["abs_residual"].mean())
        in_sample_rmse = float(np.sqrt(eval_df["sq_error"].mean()))

        effective_cv_splits = (
            run.cv_splits if run.cv_splits is not None else cfg.cv_splits
        )

        metrics = {
            "model_id": cfg.model_id,
            "pipeline_cls": type(self).__name__,
            "variant_of": cfg.variant_of,
            "target_var": cfg.target_var,
            "n_observations": int(len(context.X)),
            "n_predictors": int(len(cfg.predictor_vars)),
            "cv_splits": effective_cv_splits,
            "cv_rmse_mean": float(rmse_scores.mean()) if rmse_scores is not None else None,
            "cv_rmse_std": float(rmse_scores.std()) if rmse_scores is not None else None,
            "in_sample_mae": in_sample_mae,
            "in_sample_rmse": in_sample_rmse,
        }

        (context.output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))

    def report(self) -> None:
        """Copy the Quarto report template into the output directory.

        Template lookup order:

        1. ``docs/models/{model_id}/index.qmd`` — bespoke template for this
           exact model.
        2. ``docs/models/{variant_of}/index.qmd`` — bespoke template for the
           parent model (used by selection variants).
        3. ``docs/models/index.qmd`` — shared default template.
        """
        _section("Report")

        context = self.context
        cfg = context.config

        candidates = [_DOCS_DIR / "models" / cfg.model_id / "index.qmd"]
        if cfg.variant_of:
            candidates.append(_DOCS_DIR / "models" / cfg.variant_of / "index.qmd")
        candidates.append(_REPORT_TEMPLATE)

        template = next((c for c in candidates if c.exists()), None)

        if template is None:
            print(
                f"  [bold red]Report template not found for {cfg.model_id}. "
                f"Checked: {', '.join(str(c) for c in candidates)}[/bold red]"
            )
            return

        qmd_dest = context.output_dir / "index.qmd"
        shutil.copy(template, qmd_dest)
        print(f"  Template used: {template.relative_to(_ROOT_DIR)}")

        print(f"  Report template copied to: {qmd_dest}")
        print(
            f"\n  [bold yellow]To render:[/bold yellow] [blue]quarto render {qmd_dest}[/blue]"
        )
        print(
            f"  [bold yellow]To preview:[/bold yellow] [blue]quarto preview {qmd_dest}[/blue]"
        )

    # ── orchestrator ─────────────────────────────────────────────────────

    def fit(self) -> ModelFitContext:
        """Run the full pipeline: data → fit → evaluate → explain → report."""
        context = self.context
        config = context.config
        run = context.run_config

        print(
            "\n[green]============================================================[/green]"
        )
        print(f"[bold green]Model {config.model_id.upper()}: {config.description}[/bold green]")
        print(f"[bold green]Pipeline: {type(self).__name__}[/bold green]")
        print(f"[bold green]Run config: {run.name}[/bold green]")
        print("[green]============================================================[/green]")

        output_dir = context.output_dir
        output_dir.mkdir(parents=True, exist_ok=True)
        _clear_directory(output_dir)

        self.prepare_data()
        self.configure_model()
        self.cross_validate()
        self.fit_model()
        self.evaluate()
        self.save_metrics()
        self.permutation_importance_analysis()

        if not run.skip_correlation:
            self.correlation_analysis()
        else:
            print(
                f"\n  [yellow]Correlation analysis skipped (run config: {run.name})[/yellow]"
            )

        if not run.skip_shap:
            self.shap_analysis()
        else:
            print(f"\n  [yellow]SHAP analysis skipped (run config: {run.name})[/yellow]")

        if not run.skip_pdp:
            self.partial_dependence_plots()
        else:
            print(f"  [yellow]PDP skipped (run config: {run.name})[/yellow]")

        self.save_config()
        self.report()

        print(
            "\n[green]============================================================[/green]"
        )
        print(f"[bold green]Done. Artifacts saved to: {output_dir}[/bold green]")
        print("[green]============================================================[/green]")

        return context


# ── helpers ──────────────────────────────────────────────────────────────────


def _clear_directory(path: Path) -> None:
    """Remove all files and subdirectories inside ``path`` without removing
    ``path`` itself. More robust on Windows than ``rmtree`` + ``mkdir`` when
    another process (editor, file explorer, watcher) holds a handle on the
    directory.
    """
    for entry in path.iterdir():
        if entry.is_dir():
            shutil.rmtree(entry, ignore_errors=True)
        else:
            try:
                entry.unlink()
            except PermissionError:
                pass


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
