# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Feature-selection diagnostics for a model's predictor set.

Runs Spearman, distance-correlation, and mutual-information analyses
against a registered model's ``predictor_vars`` and writes diagnostics to
``output/feature_selection/{model_id}/``. If the model has already been
fitted, permutation importance is joined onto the cluster assignments so
the user can see which of each correlated pair currently earns its place.

Output directory lives outside ``output/models/`` so ``fit_model.py`` does
not wipe it.

Usage
-----
    python scripts/analyze_predictors.py lrp01
    python scripts/analyze_predictors.py lrp01 --cluster-cutoff 0.4
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from rich import print
from scipy.cluster import hierarchy
from scipy.spatial.distance import squareform

import language_reading_predictors.data_utils as data_utils
from language_reading_predictors.data_variables import Variables as V
from language_reading_predictors.models.registry import MODELS
from language_reading_predictors.stats_utils import (
    distance_corr_matrix,
    mutual_info_dissimilarity,
    spearman_distance_matrix,
)

_ROOT_DIR = Path(__file__).resolve().parent.parent
_OUTPUT_DIR = _ROOT_DIR / "output" / "feature_selection"
_MODELS_DIR = _ROOT_DIR / "output" / "models"


def _load_predictor_frame(cfg) -> pd.DataFrame:
    """Load the data, apply the same outlier filter as the fit pipeline, and
    return just the predictor columns.
    """
    df = data_utils.load_data()
    df = df[df[cfg.target_var].notna()].copy()
    if cfg.outlier_threshold is not None:
        df = df[df[cfg.target_var] < cfg.outlier_threshold]
    return df[cfg.predictor_vars].astype("float64")


def _heatmap(matrix: np.ndarray, labels: list[str], title: str, path: Path) -> None:
    n = len(labels)
    fig, ax = plt.subplots(figsize=(max(8, 0.35 * n), max(6, 0.3 * n)))
    sns.heatmap(
        matrix,
        xticklabels=labels,
        yticklabels=labels,
        cmap="vlag",
        center=0.0 if matrix.min() < 0 else None,
        square=True,
        cbar_kws={"shrink": 0.6},
        ax=ax,
    )
    ax.set_title(title)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _dendrogram(linkage: np.ndarray, labels: list[str], path: Path) -> None:
    n = len(labels)
    fig, ax = plt.subplots(figsize=(8, max(6, 0.3 * n)))
    hierarchy.dendrogram(linkage, labels=labels, orientation="right", ax=ax)
    ax.set_title("Distance-correlation dissimilarity (Ward linkage)")
    ax.set_xlabel("Dissimilarity (1 − distance correlation)")
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _cluster_table(
    linkage: np.ndarray, labels: list[str], cutoff: float
) -> pd.DataFrame:
    clusters = hierarchy.fcluster(linkage, t=cutoff, criterion="distance")
    return (
        pd.DataFrame({"feature": labels, "cluster_id": clusters})
        .sort_values(["cluster_id", "feature"])
        .reset_index(drop=True)
    )


def analyze(model_id: str, cluster_cutoff: float) -> None:
    key = model_id.lower()
    if key not in MODELS:
        print(f"[bold red]Unknown model: {model_id}[/bold red]")
        print(f"Available: {', '.join(MODELS.keys())}")
        raise SystemExit(1)

    cfg = MODELS[key]
    predictors = [p for p in cfg.predictor_vars if p != V.SUBJECT_ID]

    out_dir = _OUTPUT_DIR / key
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[bold green]Feature-selection diagnostics for {key}[/bold green]")
    print(f"  Output: {out_dir}")
    print(f"  Predictors ({len(predictors)}): {predictors}")

    X = _load_predictor_frame(cfg)[predictors]
    X = X.replace({pd.NA: np.nan}).fillna(X.mean())

    # ── Spearman ────────────────────────────────────────────────────────
    print("\n[bold]Spearman correlation matrix[/bold]")
    spearman_dist, spearman_corr = spearman_distance_matrix(X)
    pd.DataFrame(spearman_corr, index=predictors, columns=predictors).to_csv(
        out_dir / "spearman_matrix.csv"
    )
    _heatmap(
        spearman_corr,
        predictors,
        "Spearman rank correlation",
        out_dir / "spearman_heatmap.png",
    )

    # ── distance correlation + hierarchical clustering ─────────────────
    print("[bold]Distance-correlation dendrogram[/bold]")
    dcor_matrix = distance_corr_matrix(X)
    dcor_dissim = 1.0 - dcor_matrix
    np.fill_diagonal(dcor_dissim, 0.0)
    np.clip(dcor_dissim, 0.0, 1.0, out=dcor_dissim)
    condensed = squareform(dcor_dissim, checks=False)
    linkage = hierarchy.ward(condensed)

    _dendrogram(linkage, predictors, out_dir / "distance_corr_dendrogram.png")

    cluster_df = _cluster_table(linkage, predictors, cluster_cutoff)
    cluster_df.to_csv(out_dir / "cluster_table.csv", index=False)

    # ── mutual information ──────────────────────────────────────────────
    print("[bold]Mutual information heatmap[/bold]")
    mi_dissim = mutual_info_dissimilarity(X, random_state=cfg.random_seed)
    mi_similarity = 1.0 - mi_dissim
    _heatmap(
        mi_similarity,
        predictors,
        "Mutual information (1 − dissimilarity)",
        out_dir / "mutual_info_heatmap.png",
    )

    # ── importance pairing ─────────────────────────────────────────────
    perm_path = _MODELS_DIR / key / "permutation_importance.csv"
    if perm_path.exists():
        print("[bold]Joining permutation importance onto clusters[/bold]")
        perm_df = pd.read_csv(perm_path)
        perm_df["importance_rank"] = (
            perm_df["importance_mean"].rank(ascending=False, method="min").astype(int)
        )
        pairing = cluster_df.merge(
            perm_df[["feature", "importance_mean", "importance_std", "importance_rank"]],
            on="feature",
            how="left",
        ).sort_values(["cluster_id", "importance_rank"])
        pairing.to_csv(out_dir / "importance_pairing.csv", index=False)
    else:
        print(
            f"[yellow]No permutation_importance.csv at {perm_path} — "
            "skipping importance pairing. Run fit_model.py first for this.[/yellow]"
        )

    print(f"\n[bold green]Done. Artifacts in {out_dir}[/bold green]")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Feature-selection diagnostics for a model's predictor set."
    )
    parser.add_argument("model", type=str, help="Model id, e.g. lrp01")
    parser.add_argument(
        "--cluster-cutoff",
        type=float,
        default=0.4,
        help="Distance cutoff for forming clusters from the Ward linkage. Default: 0.4",
    )
    args = parser.parse_args()
    analyze(args.model, args.cluster_cutoff)


if __name__ == "__main__":
    main()
