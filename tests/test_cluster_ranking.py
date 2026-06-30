# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Unit tests for the shared cluster-ranking aggregation (#116).

The fit-side ``cluster_ranking_analysis`` aggregates per-feature permutation
importance over the diagnostic clusters; these lock the aggregator's schema and
ranking and the cluster-table assembly that the per-model report renders.
"""

from __future__ import annotations

import pandas as pd

from language_reading_predictors.models.cluster_ranking import (
    aggregate_cluster_importance,
    cluster_ranking_table,
)


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
