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
import shap
from rich import print
from scipy.cluster import hierarchy
from scipy.spatial.distance import squareform
from sklearn.inspection import partial_dependence, permutation_importance
from sklearn.model_selection import GroupKFold, cross_val_predict, cross_validate

import language_reading_predictors.data_utils as data_utils
from language_reading_predictors.data_variables import Variables as V
from language_reading_predictors.plot_utils import (
    plot_heatmap,
    save_shap_scatter_plots,
)
from language_reading_predictors.models.common import (
    ModelConfig,
    ModelFitContext,
    RunConfig,
)

_ROOT_DIR = Path(__file__).resolve().parent.parent.parent.parent
_DOCS_DIR = _ROOT_DIR / "docs"
_OUTPUT_DIR = _ROOT_DIR / "output" / "models"

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
        df, X, y, groups = data_utils.load_and_filter(
            cfg.target_var, cfg.predictor_vars, cfg.outlier_threshold
        )

        context.df = df
        context.X = X
        context.y = y
        context.groups = groups

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

        scoring = {
            "mae": "neg_mean_absolute_error",
            "rmse": "neg_root_mean_squared_error",
            "r2": "r2",
            "medae": "neg_median_absolute_error",
        }

        cv_results = cross_validate(
            context.pipeline,
            context.X,
            context.y,
            cv=cv,
            groups=context.groups,
            scoring=scoring,
        )

        scores_df = pd.DataFrame(
            {
                "fold": range(1, cv_splits + 1),
                "mae": -cv_results["test_mae"],
                "rmse": -cv_results["test_rmse"],
                "r2": cv_results["test_r2"],
                "medae": -cv_results["test_medae"],
            }
        )

        context.cv_scores = -cv_results["test_rmse"]
        context.cv_results = cv_results
        context.dataframes["cv_scores"] = scores_df
        scores_df.to_csv(context.output_dir / "cv_scores.csv", index=False)

        print(f"  MAE:   {scores_df['mae'].mean():.4f} ± {scores_df['mae'].std():.4f}")
        print(f"  RMSE:  {scores_df['rmse'].mean():.4f} ± {scores_df['rmse'].std():.4f}")
        print(f"  R²:    {scores_df['r2'].mean():.4f} ± {scores_df['r2'].std():.4f}")
        print(f"  MedAE: {scores_df['medae'].mean():.4f} ± {scores_df['medae'].std():.4f}")

        # Pooled out-of-fold metrics: score the stacked OOF predictions
        # against the global mean of y, rather than averaging per-fold R².
        # Per-fold R² is meaningless when each fold is a single subject
        # (near-constant y → tiny SST → huge negative R² from small errors).
        oof_pred = cross_val_predict(
            context.pipeline,
            context.X,
            context.y,
            cv=cv,
            groups=context.groups,
        )
        y_true = context.y.to_numpy()
        residuals = y_true - oof_pred
        abs_residuals = np.abs(residuals)
        ss_res = float(np.sum(residuals**2))
        ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
        pooled = {
            "pooled_mae": float(np.mean(abs_residuals)),
            "pooled_rmse": float(np.sqrt(np.mean(residuals**2))),
            "pooled_medae": float(np.median(abs_residuals)),
            "pooled_r2": float(1.0 - ss_res / ss_tot) if ss_tot > 0 else None,
        }
        context.oof_predictions = oof_pred
        context.pooled_cv_metrics = pooled

        oof_df = pd.DataFrame(
            {
                V.SUBJECT_ID: context.groups,
                "y_true": y_true,
                "y_pred_oof": oof_pred,
            }
        )
        oof_df["residual"] = oof_df["y_true"] - oof_df["y_pred_oof"]
        context.dataframes["oof_predictions"] = oof_df
        oof_df.to_csv(context.output_dir / "oof_predictions.csv", index=False)

        print("  Pooled OOF (vs. global mean):")
        print(f"    MAE:   {pooled['pooled_mae']:.4f}")
        print(f"    RMSE:  {pooled['pooled_rmse']:.4f}")
        if pooled["pooled_r2"] is not None:
            print(f"    R²:    {pooled['pooled_r2']:.4f}")
        print(f"    MedAE: {pooled['pooled_medae']:.4f}")

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
        y_true = context.y
        y_pred = context.pipeline.predict(context.X)

        eval_df = pd.DataFrame(
            {
                V.SUBJECT_ID: context.groups,
                "y_true": y_true,
                "y_pred": y_pred,
            }
        )
        eval_df["residual"] = eval_df["y_true"] - eval_df["y_pred"]
        eval_df["abs_residual"] = eval_df["residual"].abs()
        eval_df["sq_error"] = (eval_df["y_true"] - eval_df["y_pred"]) ** 2

        context.eval_df = eval_df
        context.dataframes["evaluation"] = eval_df

        eval_df.to_csv(context.output_dir / "evaluation.csv", index=False)

        mae = float(eval_df["abs_residual"].mean())
        rmse = float(np.sqrt(eval_df["sq_error"].mean()))
        medae = float(eval_df["abs_residual"].median())
        ss_res = eval_df["sq_error"].sum()
        ss_tot = ((y_true - y_true.mean()) ** 2).sum()
        r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan")

        print(f"  In-sample MAE:  {mae:.4f}")
        print(f"  In-sample RMSE: {rmse:.4f}")
        print(f"  In-sample R²:   {r2:.4f}")
        print(f"  In-sample MedAE: {medae:.4f}")

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

        repeats_df = pd.DataFrame(
            result.importances,
            index=context.X.columns,
            columns=[f"repeat_{i}" for i in range(result.importances.shape[1])],
        )
        repeats_df.index.name = "feature"
        repeats_df.loc[perm_df["feature"].tolist()].to_csv(
            context.output_dir / "permutation_importance_repeats.csv"
        )

        print(perm_df.to_string(index=False))

        ordered = perm_df["feature"].tolist()
        data = [result.importances[list(context.X.columns).index(f)] for f in ordered]
        fig_h = max(3.0, 0.35 * len(ordered) + 1.5)
        fig, ax = plt.subplots(figsize=(8, fig_h))
        ax.boxplot(
            data,
            vert=False,
            labels=ordered,
            patch_artist=True,
            boxprops={"facecolor": "white", "edgecolor": "C0"},
            medianprops={"color": "C2"},
            whiskerprops={"color": "C0"},
            capprops={"color": "C0"},
            flierprops={"marker": "o", "markerfacecolor": "C0", "markeredgecolor": "C0", "markersize": 3},
        )
        ax.invert_yaxis()
        ax.axvline(0.0, color="black", linestyle="--", linewidth=1)
        ax.set_xlabel("Decrease in R² score")
        ax.set_ylabel("Predictor variable")
        ax.set_title("Permutation importances")
        ax.grid(axis="x", alpha=0.3)
        fig.tight_layout()
        fig.savefig(
            context.output_dir / "permutation_importance.png",
            dpi=300,
            bbox_inches="tight",
        )
        context.plots["permutation_importance"] = fig
        plt.close(fig)

    def feature_selection_diagnostics(self, cluster_cutoff: float = 0.4) -> None:
        """Compute inter-predictor diagnostics for feature selection.

        Produces:

        - ``spearman_matrix.csv`` / ``spearman_heatmap.png``
        - ``distance_corr_matrix.csv`` / ``distance_corr_heatmap.png``
        - ``distance_corr_dendrogram.png`` (Ward linkage on 1 − dcor)
        - ``mutual_info_heatmap.png``
        - ``cluster_table.csv`` — cluster assignments at *cluster_cutoff*
        - ``importance_pairing.csv`` — clusters joined with permutation importance

        Must run *after* :meth:`permutation_importance_analysis` so that
        ``perm_importance_df`` is available for the importance pairing.
        """
        from language_reading_predictors.stats_utils import (
            distance_corr_matrix,
            mutual_info_dissimilarity,
            spearman_distance_matrix,
        )

        _section("Feature-selection diagnostics")

        context = self.context
        cfg = context.config
        X = context.X.replace({pd.NA: np.nan}).astype("float64")
        X_filled = X.fillna(X.mean())
        predictors = list(X.columns)
        n = len(predictors)
        out = context.output_dir

        print(f"  Predictors: {n}")

        # ── Spearman ────────────────────────────────────────────────────
        print("  Spearman correlation matrix...")
        spearman_dist, spearman_corr = spearman_distance_matrix(X_filled)
        pd.DataFrame(spearman_corr, index=predictors, columns=predictors).to_csv(
            out / "spearman_matrix.csv"
        )
        fig_sp, _ = plot_heatmap(spearman_corr, predictors, "Spearman rank correlation")
        fig_sp.savefig(out / "spearman_heatmap.png", dpi=300, bbox_inches="tight")
        context.plots["spearman_heatmap"] = fig_sp
        plt.close(fig_sp)

        # ── Distance correlation + clustering ───────────────────────────
        print("  Distance-correlation matrix...")
        dcor_matrix = distance_corr_matrix(X_filled)
        pd.DataFrame(dcor_matrix, index=predictors, columns=predictors).to_csv(
            out / "distance_corr_matrix.csv"
        )
        fig_dc, _ = plot_heatmap(dcor_matrix, predictors, "Distance correlation")
        fig_dc.savefig(out / "distance_corr_heatmap.png", dpi=300, bbox_inches="tight")
        context.plots["distance_corr_heatmap"] = fig_dc
        plt.close(fig_dc)

        print("  Distance-correlation dendrogram...")
        dcor_dissim = 1.0 - dcor_matrix
        np.fill_diagonal(dcor_dissim, 0.0)
        np.clip(dcor_dissim, 0.0, 1.0, out=dcor_dissim)
        condensed = squareform(dcor_dissim, checks=False)
        linkage = hierarchy.ward(condensed)

        dendro_h = min(max(3, 0.5 * n), 12)
        dendro_w = min(max(5, 0.4 * n), 10)
        fig_dendro, ax_dendro = plt.subplots(figsize=(dendro_w, dendro_h))
        hierarchy.dendrogram(linkage, labels=predictors, orientation="right", ax=ax_dendro)
        ax_dendro.set_title("Distance-correlation dissimilarity (Ward linkage)")
        ax_dendro.set_xlabel("Dissimilarity (1 \u2212 distance correlation)")
        fig_dendro.savefig(out / "distance_corr_dendrogram.png", dpi=300, bbox_inches="tight")
        context.plots["distance_corr_dendrogram"] = fig_dendro
        plt.close(fig_dendro)

        clusters = hierarchy.fcluster(linkage, t=cluster_cutoff, criterion="distance")
        cluster_df = (
            pd.DataFrame({"feature": predictors, "cluster_id": clusters})
            .sort_values(["cluster_id", "feature"])
            .reset_index(drop=True)
        )
        cluster_df.to_csv(out / "cluster_table.csv", index=False)

        # ── Mutual information ──────────────────────────────────────────
        print("  Mutual information heatmap...")
        mi_dissim = mutual_info_dissimilarity(X_filled, random_state=cfg.random_seed)
        fig_mi, _ = plot_heatmap(
            1.0 - mi_dissim, predictors, "Mutual information (1 \u2212 dissimilarity)"
        )
        fig_mi.savefig(out / "mutual_info_heatmap.png", dpi=300, bbox_inches="tight")
        context.plots["mutual_info_heatmap"] = fig_mi
        plt.close(fig_mi)

        # ── Importance pairing ──────────────────────────────────────────
        perm_df = context.perm_importance_df
        if perm_df is not None:
            print("  Joining permutation importance onto clusters...")
            perm_df = perm_df.copy()
            perm_df["importance_rank"] = (
                perm_df["importance_mean"].rank(ascending=False, method="min").astype(int)
            )
            pairing = cluster_df.merge(
                perm_df[["feature", "importance_mean", "importance_std", "importance_rank"]],
                on="feature",
                how="left",
            ).sort_values(["cluster_id", "importance_rank"])
            pairing.to_csv(out / "importance_pairing.csv", index=False)

        print("  Feature-selection diagnostics saved.")

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

        plt.figure()
        shap.plots.bar(explanation, max_display=12, show=False)
        fig_bar = plt.gcf()
        fig_bar.savefig(
            context.output_dir / "shap_bar.png", dpi=300, bbox_inches="tight"
        )
        context.plots["shap_bar"] = fig_bar
        plt.close("all")

        shap.summary_plot(shap_vals, X_shap, plot_size=0.25, show=False)
        fig_summary = plt.gcf()
        fig_summary.savefig(
            context.output_dir / "shap_summary.png", dpi=300, bbox_inches="tight"
        )
        context.plots["shap_summary"] = fig_summary
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

        shap.plots.waterfall(explanation[pos_idx], max_display=12, show=False)
        fig_waterfall = plt.gcf()
        fig_waterfall.savefig(
            context.output_dir / "shap_waterfall.png", dpi=300, bbox_inches="tight"
        )
        context.plots["shap_waterfall"] = fig_waterfall
        plt.close("all")

        clustering = shap.utils.hclust(X_shap, context.y)
        shap.plots.bar(
            explanation,
            clustering=clustering,
            clustering_cutoff=0.4,
            max_display=15,
            show=False,
        )
        fig_clustered = plt.gcf()
        fig_clustered.savefig(
            context.output_dir / "shap_bar_clustered.png", dpi=300, bbox_inches="tight"
        )
        context.plots["shap_bar_clustered"] = fig_clustered
        plt.close("all")

        print("  SHAP plots saved.")

    def shap_scatter_plots(
        self,
        predictors: list[str] | None = None,
        color_by: str | None = None,
        filename_suffix: str | None = None,
    ) -> list:
        """Save one ``shap.plots.scatter`` per predictor (PNG and SVG).

        Call this explicitly (not wired into :meth:`fit`) so multiple sets
        of scatters can be generated with different ``color_by`` choices.

        Parameters
        ----------
        predictors : list[str] | None
            Features to plot. Defaults to all predictors in ``context.X``.
            Must be subset of the fitted predictor set.
        color_by : str | None
            Column name used to colour each point.

            * If it is a predictor in ``context.X``, the corresponding
              Explanation slice is used — this produces a classical SHAP
              dependence plot that surfaces the interaction between the two
              features.
            * If it is a column in ``context.df`` but *not* a predictor
              (e.g. a baseline score not used by the model), the raw values
              are wrapped in a single-feature Explanation for colouring.
            * ``None`` (default) falls back to SHAP's auto-interaction
              colouring.
        filename_suffix : str | None
            Suffix appended to output filenames (before the extension), so
            repeated calls with different ``color_by`` values do not
            overwrite each other. Defaults to ``f"by_{color_by}"`` when
            ``color_by`` is set, otherwise empty.

        Returns
        -------
        list of Path
            PNG paths written.
        """
        _section("SHAP scatter plots")

        context = self.context

        if context.shap_explainer is None:
            print("  [yellow]SHAP explainer not available. Run shap_analysis() first.[/yellow]")
            return []

        explanation = context.shap_explainer(context.X)
        all_predictors = list(context.X.columns)

        if predictors is None:
            predictors = all_predictors
        else:
            missing = [p for p in predictors if p not in all_predictors]
            if missing:
                msg = f"predictors not in fitted model: {missing}"
                raise ValueError(msg)

        color = None
        color_name = None
        suffix = filename_suffix
        if color_by is not None:
            if color_by in all_predictors:
                color = explanation[:, color_by]
            elif context.df is not None and color_by in context.df.columns:
                color = context.df[color_by].to_numpy(dtype=float)
                color_name = color_by
            else:
                msg = (
                    f"color_by '{color_by}' is not a predictor and not a "
                    "column in context.df"
                )
                raise ValueError(msg)
            if suffix is None:
                suffix = f"by_{color_by}"

        written = save_shap_scatter_plots(
            explanation,
            predictors,
            context.output_dir,
            color=color,
            color_name=color_name,
            filename_suffix=suffix,
        )
        label = f" (coloured by {color_by})" if color_by else ""
        print(f"  Saved {len(written)} scatter plot(s){label}.")
        return written

    def run_shap_scatter_specs(self) -> None:
        """Run every ``ShapScatterSpec`` declared on the model config.

        The model's class-level ``shap_scatter_specs`` list drives what
        plots get produced — each spec is passed straight to
        :meth:`shap_scatter_plots`. Empty list → no-op.
        """
        context = self.context
        specs = context.config.shap_scatter_specs
        if not specs:
            return

        _section(f"SHAP scatter specs ({len(specs)})")
        for idx, spec in enumerate(specs, start=1):
            label = spec.description or (
                f"colour={spec.color_by}" if spec.color_by else "auto-colour"
            )
            print(f"  [{idx}/{len(specs)}] {label}")
            self.shap_scatter_plots(
                predictors=spec.predictors,
                color_by=spec.color_by,
                filename_suffix=spec.filename_suffix,
            )

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
        n_cols = 2
        n_rows = math.ceil(n_features / n_cols)
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 4 * n_rows))
        axes = axes.reshape(n_rows, n_cols)

        for idx, feature in enumerate(features_ordered):
            row = idx // n_cols
            col = idx % n_cols
            ax = axes[row, col]

            pdp = partial_dependence(context.pipeline, X_fp, [feature])
            grid = pdp["grid_values"][0]
            avg = pdp["average"][0]

            ax.plot(grid, avg, label="PDP")

            # Add target mean and median as horizontal reference lines
            y_mean = float(context.y.mean())
            y_median = float(context.y.median())
            ax.axhline(y_mean, color="C1", linestyle="--", alpha=0.7, label="Target mean")
            ax.axhline(y_median, color="C2", linestyle=":", alpha=0.7, label="Target median")

            ax.set_title(f"Partial dependence of {feature}")
            ax.set_xlabel(feature)
            ax.set_ylabel(f"Predicted {cfg.target_var}")
            ax.legend(fontsize=8)

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

        effective_model_params = _cap_n_estimators(cfg.model_params, run)
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

        eval_df = context.eval_df
        y_true = context.y

        in_sample_mae = float(eval_df["abs_residual"].mean())
        in_sample_rmse = float(np.sqrt(eval_df["sq_error"].mean()))
        in_sample_medae = float(eval_df["abs_residual"].median())
        ss_res = eval_df["sq_error"].sum()
        ss_tot = ((y_true - y_true.mean()) ** 2).sum()
        in_sample_r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else None

        cv_scores_df = context.dataframes.get("cv_scores")

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
            "cv_mae_mean": float(cv_scores_df["mae"].mean()) if cv_scores_df is not None else None,
            "cv_mae_std": float(cv_scores_df["mae"].std()) if cv_scores_df is not None else None,
            "cv_rmse_mean": float(cv_scores_df["rmse"].mean()) if cv_scores_df is not None else None,
            "cv_rmse_std": float(cv_scores_df["rmse"].std()) if cv_scores_df is not None else None,
            "cv_r2_mean": float(cv_scores_df["r2"].mean()) if cv_scores_df is not None else None,
            "cv_r2_std": float(cv_scores_df["r2"].std()) if cv_scores_df is not None else None,
            "cv_medae_mean": float(cv_scores_df["medae"].mean()) if cv_scores_df is not None else None,
            "cv_medae_std": float(cv_scores_df["medae"].std()) if cv_scores_df is not None else None,
            "cv_pooled_mae": (context.pooled_cv_metrics or {}).get("pooled_mae"),
            "cv_pooled_rmse": (context.pooled_cv_metrics or {}).get("pooled_rmse"),
            "cv_pooled_r2": (context.pooled_cv_metrics or {}).get("pooled_r2"),
            "cv_pooled_medae": (context.pooled_cv_metrics or {}).get("pooled_medae"),
            "in_sample_mae": in_sample_mae,
            "in_sample_rmse": in_sample_rmse,
            "in_sample_r2": in_sample_r2,
            "in_sample_medae": in_sample_medae,
        }

        (context.output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))

    def report(self) -> None:
        """Copy the Quarto report template into the output directory.

        Template lookup order:

        1. ``docs/models/{model_id}/index.qmd`` — bespoke template for this
           exact model.
        2. ``docs/models/{variant_of}/index.qmd`` — bespoke template for the
           parent model (used by selection variants).
        """
        _section("Report")

        context = self.context
        cfg = context.config

        candidates = [_DOCS_DIR / "models" / cfg.model_id / "index.qmd"]
        if cfg.variant_of:
            candidates.append(_DOCS_DIR / "models" / cfg.variant_of / "index.qmd")

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
            self.feature_selection_diagnostics()
        else:
            print(
                f"\n  [yellow]Feature-selection diagnostics skipped (run config: {run.name})[/yellow]"
            )

        if not run.skip_shap:
            self.shap_analysis()
            self.run_shap_scatter_specs()
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
                print(f"[yellow]Warning: could not delete {entry} (PermissionError)[/yellow]")


def _cap_n_estimators(
    model_params: dict, run_config: RunConfig
) -> dict:
    """Build effective LGBM params, capping n_estimators to the run config limit.

    Returns a *copy* of ``model_params`` with n_estimators adjusted if needed.
    The run config value is an upper bound, not a replacement — models with a
    tuned n_estimators below the cap keep their value.
    """
    params = dict(model_params)
    if run_config.n_estimators is not None:
        tuned = params.get("n_estimators", run_config.n_estimators)
        params["n_estimators"] = min(tuned, run_config.n_estimators)
    return params


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
