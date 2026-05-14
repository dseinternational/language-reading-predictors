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
from sklearn.model_selection import GroupKFold, cross_validate

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
from language_reading_predictors.models._reporting import (
    cv_fold_metrics_table,
    in_sample_metrics_table,
    model_header_panel,
    pooled_oof_table,
    print_panel,
    print_table,
    ranked_dataframe_table,
    run_summary_panel,
    section_header,
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
        section_header("Prepare data")

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
            print(
                f"  Target: {cfg.target_var} (outlier threshold < {cfg.outlier_threshold})"
            )
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
        section_header("Cross-validation")

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
            return_estimator=True,
            return_indices=True,
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

        context.cv_results = cv_results
        context.dataframes["cv_scores"] = scores_df
        scores_df.to_csv(context.output_dir / "cv_scores.csv", index=False)

        print_table(cv_fold_metrics_table(scores_df))

        # Pooled out-of-fold metrics. OOF predictions are reused from the
        # per-fold estimators returned by ``cross_validate`` so we do not
        # refit the model a second time. R² is computed against the
        # per-fold *training-mean* baseline (a constant predictor that
        # only sees training data), so the baseline is not contaminated
        # by the held-out values it is being compared against.
        y_true = context.y.to_numpy()
        oof_pred = np.full_like(y_true, np.nan, dtype=float)
        ss_res_total = 0.0
        ss_tot_total = 0.0
        train_idx_iter = cv_results["indices"]["train"]
        test_idx_iter = cv_results["indices"]["test"]
        for est, tr_idx, val_idx in zip(
            cv_results["estimator"], train_idx_iter, test_idx_iter
        ):
            fold_pred = est.predict(context.X.iloc[val_idx])
            oof_pred[val_idx] = fold_pred
            y_train_mean = float(np.mean(y_true[tr_idx]))
            fold_resid = y_true[val_idx] - fold_pred
            ss_res_total += float(np.sum(fold_resid**2))
            ss_tot_total += float(
                np.sum((y_true[val_idx] - y_train_mean) ** 2)
            )

        residuals = y_true - oof_pred
        abs_residuals = np.abs(residuals)
        pooled = {
            "pooled_mae": float(np.mean(abs_residuals)),
            "pooled_rmse": float(np.sqrt(np.mean(residuals**2))),
            "pooled_medae": float(np.median(abs_residuals)),
            "pooled_r2": float(1.0 - ss_res_total / ss_tot_total)
            if ss_tot_total > 0
            else None,
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

        print_table(pooled_oof_table(pooled))

    def fit_model(self) -> None:
        """Fit the pipeline on the full dataset."""
        section_header("Fit model")

        context = self.context
        context.pipeline.fit(context.X, context.y)
        print("  Model fitted on full dataset.")

    def evaluate(self) -> None:
        """Generate predictions, compute residuals, and save evaluation DataFrame."""
        section_header("Evaluate")

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

        print_table(in_sample_metrics_table(mae, rmse, r2, medae))

    def permutation_importance_analysis(self) -> None:
        """Compute group-aware out-of-fold permutation importance.

        For each CV fold, permutation importance is computed on the
        held-out validation rows using the estimator fitted on the
        corresponding training rows (reused from :meth:`cross_validate`).
        Per-fold importance arrays are concatenated, so each feature
        ends up with ``n_folds * n_repeats`` permutation deltas. This
        avoids the in-sample optimism that arises when permutation is
        run against the training set the model has just memorised, and
        respects the subject-level grouping enforced elsewhere in the
        pipeline. Scoring is ``neg_root_mean_squared_error`` so the
        units match the headline CV metric quoted in the report.
        """
        section_header("Permutation importance")

        context = self.context
        cfg = context.config
        run = context.run_config
        n_repeats = (
            run.perm_importance_repeats
            if run.perm_importance_repeats is not None
            else cfg.perm_importance_repeats
        )

        cv_results = context.cv_results
        if cv_results is None or "estimator" not in cv_results:
            msg = (
                "permutation_importance_analysis() requires cross_validate() "
                "to have been called first."
            )
            raise RuntimeError(msg)

        fold_importances = []
        for est, val_idx in zip(
            cv_results["estimator"], cv_results["indices"]["test"]
        ):
            X_val = context.X.iloc[val_idx]
            y_val = context.y.iloc[val_idx]
            result = permutation_importance(
                est,
                X_val,
                y_val,
                n_repeats=n_repeats,
                random_state=cfg.random_seed,
                scoring="neg_root_mean_squared_error",
            )
            fold_importances.append(result.importances)

        # ``permutation_importance.importances`` is signed so that a
        # positive value means score *dropped* when the column was
        # permuted. With ``neg_root_mean_squared_error`` that means a
        # positive value indicates the unpermuted column contributed
        # to lower RMSE — i.e. it was useful.
        all_importances = np.concatenate(fold_importances, axis=1)
        importances_mean = all_importances.mean(axis=1)
        importances_std = all_importances.std(axis=1)

        perm_df = pd.DataFrame(
            {
                "feature": context.X.columns,
                "importance_mean": importances_mean,
                "importance_std": importances_std,
            }
        ).sort_values("importance_mean", ascending=False)

        context.perm_importance_df = perm_df
        context.dataframes["permutation_importance"] = perm_df

        perm_df.to_csv(context.output_dir / "permutation_importance.csv", index=False)

        n_total_repeats = all_importances.shape[1]
        repeats_df = pd.DataFrame(
            all_importances,
            index=context.X.columns,
            columns=[f"repeat_{i}" for i in range(n_total_repeats)],
        )
        repeats_df.index.name = "feature"
        repeats_df.loc[perm_df["feature"].tolist()].to_csv(
            context.output_dir / "permutation_importance_repeats.csv"
        )

        print_table(
            ranked_dataframe_table(
                perm_df,
                title="Permutation importance",
                columns=["feature", "importance_mean", "importance_std"],
            )
        )

        ordered = perm_df["feature"].tolist()
        col_index = {f: i for i, f in enumerate(context.X.columns)}
        data = [all_importances[col_index[f]] for f in ordered]
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
            flierprops={
                "marker": "o",
                "markerfacecolor": "C0",
                "markeredgecolor": "C0",
                "markersize": 3,
            },
        )
        ax.invert_yaxis()
        ax.axvline(0.0, color="black", linestyle="--", linewidth=1)
        ax.set_xlabel("Decrease in held-out RMSE when feature is permuted")
        ax.set_ylabel("Predictor variable")
        ax.set_title("Out-of-fold permutation importance")
        ax.grid(axis="x", alpha=0.3)
        fig.tight_layout()
        fig.savefig(
            context.output_dir / "permutation_importance.png",
            dpi=300,
            bbox_inches="tight",
        )
        context.plots["permutation_importance"] = fig
        plt.close(fig)

    def construct_importance(self) -> None:
        """Aggregate permutation importance by construct family.

        Uses :attr:`Variables.CONSTRUCTS` via :meth:`Variables.construct_of`
        to map each feature to a construct (language composite, expressive
        vocabulary, receptive vocabulary, reading word, reading decoding,
        articulation, phonological memory, speech sampling, health, home
        literacy, demographics family / child, cognition, social,
        intervention, study structure, or other).

        Saves ``construct_importance.csv`` with per-construct sum, mean,
        max, and member count. Must run after
        :meth:`permutation_importance_analysis`.

        Motivation: dominant *constructs* are more stable across variants
        than dominant individual features — within-construct substitution
        (e.g. ``b1exto`` ↔ ``eowpvt`` ↔ ``aptinfo``) is mostly noise. The
        construct-level view surfaces which *domains* matter rather than
        which specific instrument won the permutation-importance race on
        this particular fit.
        """
        from language_reading_predictors.data_variables import Variables as V

        section_header("Construct-level importance")

        context = self.context
        perm_df = context.perm_importance_df.copy()
        perm_df["construct"] = perm_df["feature"].map(V.construct_of)

        grouped = (
            perm_df.groupby("construct")
            .agg(
                total_importance=("importance_mean", "sum"),
                mean_importance=("importance_mean", "mean"),
                max_importance=("importance_mean", "max"),
                n_members=("feature", "count"),
                top_feature=("feature", "first"),
            )
            .sort_values("total_importance", ascending=False)
            .reset_index()
        )

        context.dataframes["construct_importance"] = grouped
        grouped.to_csv(context.output_dir / "construct_importance.csv", index=False)

        print_table(
            ranked_dataframe_table(
                grouped,
                title="Construct importance",
                columns=[
                    "construct",
                    "total_importance",
                    "mean_importance",
                    "max_importance",
                    "n_members",
                    "top_feature",
                ],
            )
        )

    def shap_direction_diagnostics(self) -> None:
        """Per-feature direction and monotonicity diagnostics from SHAP.

        For each feature, compute:

        - ``shap_mean_abs`` — magnitude (basically a SHAP-side mirror of
          permutation importance; agreement with permutation importance
          is itself a sanity check).
        - ``shap_std`` — spread of SHAP values across observations. Large
          spread with small mean-abs indicates a feature with high-
          variance but low-consistency effect.
        - ``feature_shap_spearman`` — Spearman rank correlation between
          the feature value and its SHAP value. Sign gives the direction
          of effect; magnitude gives monotonicity. +1 is clean
          monotonic positive; −1 is clean monotonic negative; near-0 is
          non-monotonic or bimodal.
        - ``shape_flag`` — a categorical verdict:
          ``"monotonic_+"`` / ``"monotonic_-"`` when ``|spearman| > 0.7``,
          ``"noisy_+"`` / ``"noisy_-"`` when ``0.3 < |spearman| ≤ 0.7``,
          and ``"non_monotonic"`` otherwise.

        Reads ``context.shap_values`` so must run after
        :meth:`shap_analysis`. A feature with low permutation importance
        *and* ``non_monotonic`` / high-spread SHAP is a candidate for
        drop on direction grounds, even when importance alone would be
        borderline.
        """
        from scipy import stats as _stats

        section_header("SHAP direction diagnostics")

        context = self.context
        shap_vals = context.shap_values
        if shap_vals is None:
            print("  [yellow]No SHAP values available — skipped.[/yellow]")
            return

        X = context.X
        rows = []
        for i, feat in enumerate(X.columns):
            feat_vals = X.iloc[:, i].to_numpy()
            shap_col = shap_vals[:, i]
            mask = ~np.isnan(feat_vals)
            if mask.sum() < 3:
                spearman = float("nan")
            else:
                res = _stats.spearmanr(feat_vals[mask], shap_col[mask])
                spearman = (
                    float(res.correlation) if not np.isnan(res.correlation) else 0.0
                )
            mean_abs = float(np.mean(np.abs(shap_col)))
            std = float(np.std(shap_col))
            if abs(spearman) > 0.7:
                flag = "monotonic_+" if spearman > 0 else "monotonic_-"
            elif abs(spearman) > 0.3:
                flag = "noisy_+" if spearman > 0 else "noisy_-"
            else:
                flag = "non_monotonic"
            rows.append(
                {
                    "feature": feat,
                    "shap_mean_abs": mean_abs,
                    "shap_std": std,
                    "feature_shap_spearman": spearman,
                    "shape_flag": flag,
                }
            )

        diag = pd.DataFrame(rows).sort_values("shap_mean_abs", ascending=False)
        context.dataframes["shap_direction_diagnostics"] = diag
        diag.to_csv(context.output_dir / "shap_direction_diagnostics.csv", index=False)
        print_table(
            ranked_dataframe_table(
                diag,
                title="SHAP direction diagnostics",
                columns=[
                    "feature",
                    "shap_mean_abs",
                    "shap_std",
                    "feature_shap_spearman",
                    "shape_flag",
                ],
            )
        )

    def stability_selection(
        self,
        n_bootstraps: int = 30,
        subject_fraction: float = 0.8,
        top_k: int = 5,
        n_repeats: int = 10,
    ) -> None:
        """Subject-level bootstrap stability of permutation importance.

        Draws ``n_bootstraps`` subsamples of subjects (``subject_fraction``
        of the unique subjects, sampled with replacement), refits the
        estimator on each subsample, and recomputes permutation
        importance with ``n_repeats`` repeats. Records:

        - ``appearance_rate_top_k`` — fraction of bootstraps where the
          feature placed in the top *top_k* by importance;
        - ``importance_mean`` / ``importance_std`` — bootstrap distribution
          of per-feature importance;
        - ``rank_median`` / ``rank_iqr`` — distribution of the feature's
          rank across bootstraps.

        High appearance rate + low rank IQR → robustly important.
        Compare against the single-point permutation importance to
        identify features whose ranking is fold-luck rather than signal
        — especially useful for the low-R² gain models (LRP01, LRP03)
        where a single fit's ranking is least stable.

        Gated to reporting configs via ``fit()``; costly for large
        predictor sets but not ruinous at the current n.
        """
        from sklearn.base import clone
        from sklearn.utils import resample

        section_header("Stability selection")

        context = self.context
        cfg = context.config
        rng = np.random.default_rng(cfg.random_seed)
        unique_subjects = np.asarray(context.groups.unique())
        n_sub = max(1, int(round(len(unique_subjects) * subject_fraction)))

        rank_records: dict[str, list[int]] = {f: [] for f in context.X.columns}
        imp_records: dict[str, list[float]] = {f: [] for f in context.X.columns}
        appearance_top: dict[str, int] = {f: 0 for f in context.X.columns}

        print(
            f"  Bootstraps: {n_bootstraps}, subjects per draw: {n_sub} "
            f"of {len(unique_subjects)}, perm repeats per draw: {n_repeats}"
        )

        # Map each subject to its row indices once; bootstraps then
        # build a row-index array by *repeating* the indices of each
        # drawn subject. ``isin``+set would silently collapse the
        # with-replacement draw back to unique subjects.
        subject_rows: dict = {
            s: np.flatnonzero(context.groups.to_numpy() == s)
            for s in unique_subjects
        }

        for b in range(n_bootstraps):
            seed = int(rng.integers(0, 2**31 - 1))
            drawn = resample(
                unique_subjects,
                replace=True,
                n_samples=n_sub,
                random_state=seed,
            )
            row_idx = np.concatenate([subject_rows[s] for s in drawn])
            X_b = context.X.iloc[row_idx]
            y_b = context.y.iloc[row_idx]

            est = clone(context.pipeline)
            est.fit(X_b, y_b)
            result = permutation_importance(
                est,
                X_b,
                y_b,
                n_repeats=n_repeats,
                random_state=seed,
            )

            # Rank features in this bootstrap (highest importance = rank 1)
            order = np.argsort(-result.importances_mean)
            ranks = np.empty_like(order)
            ranks[order] = np.arange(1, len(order) + 1)
            for i, feat in enumerate(context.X.columns):
                rank_records[feat].append(int(ranks[i]))
                imp_records[feat].append(float(result.importances_mean[i]))
                if ranks[i] <= top_k:
                    appearance_top[feat] += 1

        rows = []
        for feat in context.X.columns:
            ranks_arr = np.asarray(rank_records[feat])
            imps_arr = np.asarray(imp_records[feat])
            rows.append(
                {
                    "feature": feat,
                    "appearance_rate_top_k": appearance_top[feat] / n_bootstraps,
                    "importance_mean": float(np.mean(imps_arr)),
                    "importance_std": float(np.std(imps_arr)),
                    "rank_median": float(np.median(ranks_arr)),
                    "rank_q25": float(np.quantile(ranks_arr, 0.25)),
                    "rank_q75": float(np.quantile(ranks_arr, 0.75)),
                }
            )

        stab = pd.DataFrame(rows).sort_values("appearance_rate_top_k", ascending=False)
        stab.attrs["n_bootstraps"] = n_bootstraps
        stab.attrs["top_k"] = top_k
        context.dataframes["stability_selection"] = stab
        stab.to_csv(context.output_dir / "stability_selection.csv", index=False)
        print_table(
            ranked_dataframe_table(
                stab,
                title=f"Stability selection (top-{top_k} across {n_bootstraps} bootstraps)",
                columns=[
                    "feature",
                    "appearance_rate_top_k",
                    "importance_mean",
                    "importance_std",
                    "rank_median",
                    "rank_q25",
                    "rank_q75",
                ],
            )
        )

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

        section_header("Feature-selection diagnostics")

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
        hierarchy.dendrogram(
            linkage, labels=predictors, orientation="right", ax=ax_dendro
        )
        ax_dendro.set_title("Distance-correlation dissimilarity (Ward linkage)")
        ax_dendro.set_xlabel("Dissimilarity (1 \u2212 distance correlation)")
        fig_dendro.savefig(
            out / "distance_corr_dendrogram.png", dpi=300, bbox_inches="tight"
        )
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
                perm_df["importance_mean"]
                .rank(ascending=False, method="min")
                .astype(int)
            )
            pairing = cluster_df.merge(
                perm_df[
                    ["feature", "importance_mean", "importance_std", "importance_rank"]
                ],
                on="feature",
                how="left",
            ).sort_values(["cluster_id", "importance_rank"])
            pairing.to_csv(out / "importance_pairing.csv", index=False)

        print("  Feature-selection diagnostics saved.")

    def shap_analysis(self) -> None:
        """Compute SHAP values and save bar, summary, and waterfall plots."""
        section_header("SHAP analysis")

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
        section_header("SHAP scatter plots")

        context = self.context

        if context.shap_explainer is None:
            print(
                "  [yellow]SHAP explainer not available. Run shap_analysis() first.[/yellow]"
            )
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

        section_header(f"SHAP scatter specs ({len(specs)})")
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
        section_header("Partial dependence plots")

        context = self.context
        cfg = context.config

        X_fp = context.X

        if cfg.pdp_features:
            pdp_features = cfg.pdp_features
        else:
            pdp_features = context.perm_importance_df.head(cfg.pdp_top_n)[
                "feature"
            ].tolist()

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
            ax.axhline(
                y_mean, color="C1", linestyle="--", alpha=0.7, label="Target mean"
            )
            ax.axhline(
                y_median, color="C2", linestyle=":", alpha=0.7, label="Target median"
            )

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
            "cv_mae_mean": float(cv_scores_df["mae"].mean())
            if cv_scores_df is not None
            else None,
            "cv_mae_std": float(cv_scores_df["mae"].std())
            if cv_scores_df is not None
            else None,
            "cv_rmse_mean": float(cv_scores_df["rmse"].mean())
            if cv_scores_df is not None
            else None,
            "cv_rmse_std": float(cv_scores_df["rmse"].std())
            if cv_scores_df is not None
            else None,
            "cv_r2_mean": float(cv_scores_df["r2"].mean())
            if cv_scores_df is not None
            else None,
            "cv_r2_std": float(cv_scores_df["r2"].std())
            if cv_scores_df is not None
            else None,
            "cv_medae_mean": float(cv_scores_df["medae"].mean())
            if cv_scores_df is not None
            else None,
            "cv_medae_std": float(cv_scores_df["medae"].std())
            if cv_scores_df is not None
            else None,
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
        section_header("Report")

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

        print()
        print_panel(
            model_header_panel(
                model_id=config.model_id,
                description=config.description,
                pipeline_cls=type(self).__name__,
                run_config=run.name,
                target=config.target_var,
                n_predictors=len(config.predictor_vars),
                variant_of=config.variant_of,
            )
        )

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
        self.construct_importance()

        if not run.skip_correlation:
            self.feature_selection_diagnostics()
        else:
            print(
                f"\n  [yellow]Feature-selection diagnostics skipped (run config: {run.name})[/yellow]"
            )

        if not run.skip_shap:
            self.shap_analysis()
            self.run_shap_scatter_specs()
            self.shap_direction_diagnostics()
        else:
            print(
                f"\n  [yellow]SHAP analysis skipped (run config: {run.name})[/yellow]"
            )

        # Stability selection is expensive; reporting config only.
        if run.name == "reporting":
            self.stability_selection()
        else:
            print(
                f"\n  [yellow]Stability selection skipped (run config: {run.name})[/yellow]"
            )

        if not run.skip_pdp:
            self.partial_dependence_plots()
        else:
            print(f"  [yellow]PDP skipped (run config: {run.name})[/yellow]")

        self.save_config()
        self.report()

        print()
        print_panel(run_summary_panel(output_dir=output_dir))

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
                print(
                    f"[yellow]Warning: could not delete {entry} (PermissionError)[/yellow]"
                )


def _cap_n_estimators(model_params: dict, run_config: RunConfig) -> dict:
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


def _json_safe(v):
    """Convert numpy/non-serialisable values for JSON."""
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return float(v)
    if isinstance(v, np.ndarray):
        return v.tolist()
    return v
