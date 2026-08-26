# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Pins for the GB pipeline's clustering linkage and SHAP-interaction artefacts (#631).

Finding 18: Ward linkage is correctly defined only for Euclidean distances, so
the 1 − dcor dissimilarity must be clustered with AVERAGE linkage — both in
``EstimatorPipeline.feature_selection_diagnostics`` and in
``scripts/rank_predictors.py``'s cut-height sensitivity, which replicates it.

Finding 20c: the SHAP interaction CSV and heatmap must share one convention —
the summed symmetric |SHAP interaction| — so a pair's heatmap cell equals its
table value.
"""

from __future__ import annotations

import importlib.util
import inspect
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.cluster import hierarchy
from scipy.spatial.distance import squareform

from language_reading_predictors.models.base_pipeline import (
    EstimatorPipeline,
    summed_symmetric_interactions,
)

_RANK_SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "rank_predictors.py"


@pytest.fixture(scope="module")
def rank_predictors():
    spec = importlib.util.spec_from_file_location("rank_predictors_linkage", _RANK_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ── linkage method (finding 18) ────────────────────────────────────────────────


def test_feature_selection_diagnostics_uses_average_linkage():
    """Pin the linkage method: average, never Ward, on the 1 − dcor dissimilarity."""
    src = inspect.getsource(EstimatorPipeline.feature_selection_diagnostics)
    assert "hierarchy.average(" in src
    assert "hierarchy.ward(" not in src


def test_rank_predictors_average_linkage_matches_scipy_average(rank_predictors):
    """``average_linkage`` must reproduce scipy average linkage on the same
    mean-filled distance-correlation dissimilarity (and differ from Ward)."""
    from language_reading_predictors.stats_utils import distance_corr_matrix

    rng = np.random.default_rng(7)
    n = 60
    a = rng.normal(size=n)
    X = pd.DataFrame(
        {
            "a0": a,
            "a1": a + 0.1 * rng.normal(size=n),
            "b0": rng.normal(size=n),
            "b1": rng.normal(size=n),
            "c0": rng.normal(size=n),
        }
    )

    Z = rank_predictors.average_linkage(X)

    Xf = X.replace({pd.NA: np.nan}).astype("float64")
    Xf = Xf.fillna(Xf.mean())
    dissim = 1.0 - distance_corr_matrix(Xf)
    np.fill_diagonal(dissim, 0.0)
    np.clip(dissim, 0.0, 1.0, out=dissim)
    condensed = squareform(dissim, checks=False)

    assert np.allclose(Z, hierarchy.average(condensed))
    assert not np.allclose(Z, hierarchy.ward(condensed))


# ── SHAP interaction convention (finding 20c) ──────────────────────────────────


def test_summed_symmetric_interactions_table_matches_heatmap():
    """The plotted matrix must equal the saved pair values (#631 finding 20c)."""
    rng = np.random.default_rng(11)
    feats = ["w", "x", "y", "z"]
    mean_abs = rng.uniform(0.0, 1.0, size=(4, 4))  # deliberately asymmetric

    inter_df, heat = summed_symmetric_interactions(mean_abs, feats)

    # Heatmap matrix: summed symmetric with a zero diagonal.
    assert np.allclose(heat, heat.T)
    assert np.allclose(np.diag(heat), 0.0)

    # Every CSV pair row equals the corresponding heatmap cell, and both carry
    # the summed [i, j] + [j, i] convention.
    pos = {f: i for i, f in enumerate(feats)}
    assert len(inter_df) == 6  # 4 choose 2 off-diagonal pairs
    for _, row in inter_df.iterrows():
        i, j = pos[row["feature_a"]], pos[row["feature_b"]]
        assert row["mean_abs_interaction"] == pytest.approx(heat[i, j])
        assert row["mean_abs_interaction"] == pytest.approx(
            mean_abs[i, j] + mean_abs[j, i]
        )

    # Ranked descending.
    vals = inter_df["mean_abs_interaction"].to_numpy()
    assert np.all(np.diff(vals) <= 0)
