# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Unit tests for the shared cluster-ranking aggregation (#116).

The fit-side ``cluster_ranking_analysis`` aggregates per-feature permutation
importance over the diagnostic clusters; these lock the aggregator's schema and
ranking and the cluster-table assembly that the per-model report renders.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy.cluster import hierarchy
from scipy.spatial.distance import squareform

from language_reading_predictors.models.cluster_ranking import (
    SAME_SKILL_SIBLINGS,
    aggregate_cluster_importance,
    average_linkage_tree,
    cluster_ids_by_feature,
    cluster_ranking_table,
    cluster_table,
)


def test_deap_composite_targets_have_same_skill_siblings():
    """The DEAP composite level targets (LRP-RLI-GBL-022/LRP-RLI-GBL-023) leave their
    deterministic components in the pool, so they must carry a curated same-skill map
    so the ``ranking_excluding_same_skill.csv`` view their docstrings promise is emitted.
    """
    components = ["deappin", "deappvo", "deappfi"]
    assert SAME_SKILL_SIBLINGS.get("deappav") == components  # LRP-RLI-GBL-022 target
    assert SAME_SKILL_SIBLINGS.get("deapp_c") == components  # LRP-RLI-GBL-023 target


def _perm_and_clusters():
    perm = pd.DataFrame(
        {
            "feature": ["a", "b", "c", "d"],
            "importance_mean": [0.5, 0.3, 0.1, 0.05],
            "importance_std": [0.1, 0.1, 0.05, 0.02],
        }
    )
    clusters = pd.DataFrame(
        {"feature": ["a", "b", "c", "d"], "cluster_id": [1, 1, 2, 2]}
    )
    return perm, clusters


def test_aggregate_cluster_importance_schema_and_ranking():
    perm, clusters = _perm_and_clusters()
    ci = aggregate_cluster_importance(perm, clusters)
    assert {
        "cluster_id", "cluster_rank", "cluster_perm_imp_mean",
        "cluster_perm_imp_sd", "n_members", "members",
    } <= set(ci.columns)
    top = ci.sort_values("cluster_rank").iloc[0]
    # cluster 1 (mean of 0.5, 0.3 = 0.4) outranks cluster 2 (mean 0.075).
    assert top["cluster_id"] == 1
    assert top["cluster_rank"] == 1
    assert top["n_members"] == 2
    assert abs(top["cluster_perm_imp_mean"] - 0.4) < 1e-9


def test_cluster_ranking_table_picks_representative():
    perm, clusters = _perm_and_clusters()
    ci = aggregate_cluster_importance(perm, clusters)
    # minimal ranking frame with the columns cluster_ranking_table consumes.
    ranking = pd.DataFrame(
        {
            "cluster_id": [1, 1, 2, 2],
            "member": ["a", "b", "c", "d"],
            "perm_imp_mean": [0.5, 0.3, 0.1, 0.05],
            "same_skill_of_outcome": [True, False, False, False],
        }
    )
    tbl = cluster_ranking_table(ci, ranking, siblings=["a"])
    row1 = tbl[tbl["cluster_id"] == 1].iloc[0]
    # highest-importance member is the representative; the same-skill 'a' is
    # excluded from representative_excl_same_skill in favour of 'b'.
    assert row1["representative"] == "a"
    assert row1["representative_excl_same_skill"] == "b"
    assert bool(row1["any_same_skill"]) is True


# --- Shared feature grouping (#662) ------------------------------------------


def _symmetric_dissimilarity(rng, n: int) -> np.ndarray:
    """A 1 − distance-correlation-shaped matrix: symmetric, zero diagonal, in [0, 1]."""
    matrix = rng.random((n, n))
    matrix = (matrix + matrix.T) / 2.0
    np.fill_diagonal(matrix, 0.0)
    np.clip(matrix, 0.0, 1.0, out=matrix)
    return matrix


@pytest.mark.parametrize("seed", range(6))
def test_shared_grouping_reproduces_the_local_tree_and_cut(seed):
    """The tree, the labels and the table must be identical to the retired path.

    ``cluster_id`` joins ``cluster_table.csv`` to ``importance_pairing.csv`` and
    to the cluster-importance tables, so the labels are not free to change: they
    are SciPy's own, and the column keeps SciPy's ``int32`` (building it from
    Python ints would widen it to ``int64`` and break a dtype-sensitive merge).
    """
    rng = np.random.default_rng(seed)
    n = int(rng.integers(2, 25))
    names = [f"f{index:02d}" for index in range(n)]
    dissimilarity = _symmetric_dissimilarity(rng, n)
    cutoff = float(rng.uniform(0.05, 0.95))

    retired_tree = hierarchy.average(squareform(dissimilarity, checks=False))
    retired_labels = hierarchy.fcluster(retired_tree, t=cutoff, criterion="distance")
    retired_table = (
        pd.DataFrame({"feature": names, "cluster_id": retired_labels})
        .sort_values(["cluster_id", "feature"])
        .reset_index(drop=True)
    )

    tree = average_linkage_tree(dissimilarity)
    np.testing.assert_array_equal(tree, retired_tree)
    assert cluster_ids_by_feature(names, tree, cutoff=cutoff) == {
        name: int(label) for name, label in zip(names, retired_labels, strict=True)
    }
    table = cluster_table(names, tree, cutoff=cutoff)
    pd.testing.assert_frame_equal(table, retired_table)
    assert table["cluster_id"].dtype == np.int32


def test_the_tree_builder_refuses_a_repaired_or_euclidean_only_input():
    """The checks ``squareform(..., checks=False)`` used to skip are now enforced."""
    asymmetric = np.array([[0.0, 0.2], [0.3, 0.0]])
    with pytest.raises(ValueError, match="symmetric"):
        average_linkage_tree(asymmetric)
    with pytest.raises(ValueError, match="zero diagonal"):
        average_linkage_tree(np.array([[0.1, 0.2], [0.2, 0.1]]))
    with pytest.raises(ValueError, match="nonnegative"):
        average_linkage_tree(np.array([[0.0, -0.2], [-0.2, 0.0]]))


def test_a_singleton_feature_set_still_produces_one_cluster():
    """The GB diagnostics run on whatever predictor set a model declares."""
    tree = average_linkage_tree(np.zeros((1, 1)))
    assert tree.shape == (0, 4)
    assert cluster_ids_by_feature(["only"], tree, cutoff=0.4) == {"only": 1}
    table = cluster_table(["only"], tree, cutoff=0.4)
    assert list(table["feature"]) == ["only"]
    assert table["cluster_id"].dtype == np.int32
