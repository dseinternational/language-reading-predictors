# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Cluster-first predictor-ranking tables (#116).

The library home for the ranking-table assembly shared by two callers:

- :meth:`models.base_pipeline.EstimatorPipeline.cluster_ranking_analysis` — run on
  **every** GB fit, using a light per-cluster *aggregate* of the already-computed
  permutation importance (:func:`aggregate_cluster_importance`). This is what the
  per-model report renders.
- ``scripts/rank_predictors.py`` — the dedicated ranking run, which computes a more
  rigorous **grouped-permutation** cluster importance (re-permuting whole clusters)
  but then assembles the same tables via the functions here.

Both share :func:`assemble_ranking` (per-feature, cluster-first detail) and
:func:`cluster_ranking_table` (one row per cluster, with a representative), so the
two surfaces stay schema-compatible.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
from dse_research_utils.ml.feature_groups import (
    feature_groups_from_linkage,
    linkage_from_dissimilarity,
)

# ── Curated same-skill (predictor↔OUTCOME contamination) map ───────────────────
# Source: curated project mapping. A predictor listed here is
# a *concurrent restatement of the outcome* (same skill, possibly a different
# instrument). Distance-correlation clustering is predictor↔predictor and CANNOT
# catch this, so it survives as curated annotation — NOT a prune. Keyed by outcome.
SAME_SKILL_SIBLINGS: dict[str, list[str]] = {
    "eowpvt": ["b1exto"],    # expressive vocab — EOWPVT vs total b1exto
    "aptgram": ["aptinfo"],  # same APT elicited sample
    "aptinfo": ["aptgram"],  # same APT elicited sample
    "deappfi": ["deappin", "deappvo"],  # same DEAP picture-naming instrument
    "deappin": ["deappfi", "deappvo"],  # same DEAP picture-naming instrument
    "deappvo": ["deappfi", "deappin"],  # DEAP voicing — same DEAP instrument
    "deappav": ["deappin", "deappvo", "deappfi"],  # DEAP average — its deterministic components
    "deapp_c": ["deappin", "deappvo", "deappfi"],  # DEAP composite — its deterministic components
    "rowpvt": ["b1reto"],    # receptive vocab — ROWPVT vs total b1reto
    # Early Repetition Battery (ERB) phonological-memory sub-scores — same instrument.
    "erbnw": ["erbword", "erbto"],
    "erbword": ["erbnw", "erbto"],
    "erbto": ["erbnw", "erbword"],
    # gain outcomes (ewrswr_gain, rowpvt_gain, erb*_gain, ...): none — the baseline
    # level is the regression-to-the-mean anchor, not contamination.
}



# ── dendrogram cut → cluster membership ───────────────────────────────────────
# Both ranking surfaces cut the SAME average-linkage tree over the 1 − distance
# correlation dissimilarity, so the cut lives here once (#662). The library owns
# the tree construction and the cut; this project owns the feature order, the
# cut height, the ``cluster_id`` column dtype and the table's row order.


def average_linkage_tree(dissimilarity: np.ndarray) -> np.ndarray:
    """Average-linkage tree over a precomputed dissimilarity matrix.

    Average linkage, not Ward: Ward is correctly defined only for Euclidean
    distances, and 1 − distance correlation is a precomputed non-Euclidean
    dissimilarity in general (#631 finding 18). The shared constructor validates
    what ``squareform(..., checks=False)`` used to skip — finite, nonnegative,
    exactly symmetric, exactly zero-diagonal — and refuses Euclidean-only
    methods outright, so the choice cannot be undone by a later caller.
    """
    return linkage_from_dissimilarity(dissimilarity, method="average")


def cluster_ids_by_feature(
    names: Sequence[str], linkage: np.ndarray, *, cutoff: float
) -> dict[str, int]:
    """Feature → SciPy cut label at ``cutoff``, in the caller's feature order.

    The numeric labels are SciPy's own, kept (rather than renumbered) because
    ``cluster_table.csv``, ``importance_pairing.csv`` and the cluster-importance
    tables all join on them. They are local to one tree and one cut, not stable
    identities across a changed feature set, matrix or cut height.
    """
    groups = feature_groups_from_linkage(list(names), linkage, threshold=cutoff)
    return {
        member: cluster_id for cluster_id, members in groups.items() for member in members
    }


def cluster_table(
    names: Sequence[str], linkage: np.ndarray, *, cutoff: float
) -> pd.DataFrame:
    """The ``cluster_table.csv`` frame: one row per feature, cluster-major order.

    ``cluster_id`` keeps the ``int32`` dtype SciPy's ``fcluster`` produced, so a
    downstream merge against a table built the same way still matches on dtype;
    building the column from Python ints would silently widen it to ``int64``.
    """
    membership = cluster_ids_by_feature(names, linkage, cutoff=cutoff)
    features = list(names)
    return (
        pd.DataFrame(
            {
                "feature": features,
                "cluster_id": np.asarray(
                    [membership[name] for name in features], dtype=np.int32
                ),
            }
        )
        .sort_values(["cluster_id", "feature"])
        .reset_index(drop=True)
    )


def _standardise_perm(df: pd.DataFrame) -> pd.DataFrame:
    """Rename the permutation-importance columns to the ranking schema and select.

    Shared by both entry points so the two surfaces standardise the (differently
    sourced) permutation-importance frame identically.
    """
    return df.rename(
        columns={"importance_mean": "perm_imp_mean", "importance_std": "perm_imp_sd"}
    )[["feature", "perm_imp_mean", "perm_imp_sd"]]


def aggregate_cluster_importance(
    perm_df: pd.DataFrame, clusters: pd.DataFrame
) -> pd.DataFrame:
    """Per-cluster importance by **aggregating** the per-feature permutation scores.

    A light, single-fit proxy for the dedicated ranking run's grouped-permutation
    cluster importance: the cluster score is the **mean** per-feature out-of-fold
    permutation importance of its members (with the member SDs propagated). Ranks
    clusters by that mean. Returns the same schema the grouped-permutation path
    produces, so :func:`assemble_ranking` / :func:`cluster_ranking_table` are shared.

    ``perm_df`` has ``feature`` / ``importance_mean`` / ``importance_std``;
    ``clusters`` has ``feature`` / ``cluster_id``.
    """
    m = clusters.merge(_standardise_perm(perm_df), on="feature", how="left")
    g = m.groupby("cluster_id", sort=False)
    out = pd.DataFrame(
        {
            "cluster_id": list(g.groups.keys()),
            "cluster_perm_imp_mean": g["perm_imp_mean"].mean().to_numpy(),
            # Root-mean-square of the per-member permutation SDs: a rough
            # *descriptive* spread of member uncertainty, NOT the standard error
            # of the cluster mean (that would divide the quadrature sum by
            # n_members). Display-only — clusters are ranked by the mean above —
            # and not directly comparable to the grouped-permutation path's
            # cluster_perm_imp_sd in scripts/rank_predictors.py (the std of the
            # regrouped delta samples, a different statistic on the same column).
            "cluster_perm_imp_sd": g["perm_imp_sd"]
            .apply(lambda s: float(np.sqrt(np.nanmean(np.square(s.to_numpy())))))
            .to_numpy(),
            "n_members": g["feature"].size().to_numpy(),
            "members": g["feature"].apply(lambda s: ", ".join(map(str, s))).to_numpy(),
        }
    )
    out = out.sort_values("cluster_perm_imp_mean", ascending=False).reset_index(drop=True)
    out["cluster_rank"] = np.arange(1, len(out) + 1)
    return out


def assemble_ranking(pipe, target, siblings, cluster_imp):
    """Per-feature detail table, ordered cluster-first then within-cluster importance."""
    ctx = pipe.context
    od = ctx.output_dir
    perm = _standardise_perm(ctx.perm_importance_df)
    clusters = pd.read_csv(od / "cluster_table.csv")  # feature, cluster_id
    shapd = pd.read_csv(od / "shap_direction_diagnostics.csv")[
        ["feature", "shap_mean_abs", "feature_shap_spearman"]
    ]
    stab_path = od / "stability_selection.csv"
    if stab_path.exists():  # absent in --quick mode (stability skipped)
        stab = pd.read_csv(stab_path)[["feature", "appearance_rate_top_k"]]
    else:
        stab = pd.DataFrame(
            {"feature": clusters["feature"], "appearance_rate_top_k": np.nan}
        )

    df = (
        clusters
        .merge(perm, on="feature", how="left")
        .merge(shapd, on="feature", how="left")
        .merge(stab, on="feature", how="left")
        .merge(
            cluster_imp[["cluster_id", "cluster_rank", "cluster_perm_imp_mean"]],
            on="cluster_id", how="left",
        )
    )
    df["z"] = df["perm_imp_mean"] / df["perm_imp_sd"].replace(0.0, np.nan)
    df["sign"] = (
        np.sign(df["feature_shap_spearman"]).map({1.0: "+", -1.0: "-"}).fillna("0")
    )
    df["same_skill_of_outcome"] = df["feature"].isin(siblings)
    df = df.rename(
        columns={
            "feature": "member",
            "shap_mean_abs": "mean_abs_shap",
            "appearance_rate_top_k": "topk_freq",
        }
    )
    # cluster-first ordering, then per-feature importance within each cluster
    df = df.sort_values(
        ["cluster_rank", "perm_imp_mean"], ascending=[True, False]
    ).reset_index(drop=True)
    df["within_cluster_rank"] = df.groupby("cluster_rank").cumcount() + 1
    cols = [
        "cluster_rank", "cluster_id", "within_cluster_rank", "member",
        "cluster_perm_imp_mean", "perm_imp_mean", "perm_imp_sd", "z",
        "mean_abs_shap", "topk_freq", "sign", "same_skill_of_outcome",
    ]
    return df[cols]


def cluster_ranking_table(cluster_imp, ranking, siblings):
    """Primary artefact: one row per cluster, with the representative member.

    The representative is the highest per-feature importance member; an
    ``representative_excl_same_skill`` column gives the highest non-flagged member, so
    a downstream consumer taking cluster representatives never picks a restatement of
    the outcome.
    """
    rows = []
    for _, c in cluster_imp.iterrows():
        members = ranking[ranking["cluster_id"] == c["cluster_id"]].sort_values(
            "perm_imp_mean", ascending=False
        )
        rep = members.iloc[0]["member"]
        non_skill = members[~members["same_skill_of_outcome"]]
        rep_excl = non_skill.iloc[0]["member"] if len(non_skill) else None
        rows.append({
            "cluster_rank": int(c["cluster_rank"]),
            "cluster_id": int(c["cluster_id"]),
            "cluster_perm_imp_mean": c["cluster_perm_imp_mean"],
            "cluster_perm_imp_sd": c["cluster_perm_imp_sd"],
            "n_members": int(c["n_members"]),
            "representative": rep,
            "representative_excl_same_skill": rep_excl,
            "any_same_skill": bool(members["same_skill_of_outcome"].any()),
            "members": c["members"],
        })
    return pd.DataFrame(rows)
