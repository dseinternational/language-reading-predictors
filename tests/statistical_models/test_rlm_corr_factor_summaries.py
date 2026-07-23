# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Independent unit tests for the isolated RLM corr-factor summaries (#394 pillar 6).

Exercises the pure ``rlm_corr_factor_summaries`` functions on a hand-built posterior
— no factory, PyMC model, output directory, Quarto template or plotting session —
confirming the mm-001 measurement-headline tables are computable in isolation.
"""

from __future__ import annotations

import numpy as np
import xarray as xr

from language_reading_predictors.statistical_models.rlm_corr_factor_summaries import (
    factor_correlation_matrix,
    factor_correlation_pairs,
    loadings_communalities_table,
)


def _synthetic_posterior():
    """Two chains x three draws over three indicators (two domains)."""
    loading = np.array(
        [
            [[0.8, 0.6, 0.5], [0.9, 0.7, 0.4], [0.7, 0.5, 0.6]],
            [[0.6, 0.8, 0.5], [0.8, 0.6, 0.5], [0.7, 0.7, 0.4]],
        ]
    )  # (chain=2, draw=3, indicator=3)
    communality = loading**2
    # A (chain, draw, domain, domain_b) correlation cube: constant per draw here.
    base = np.array([[1.0, 0.7], [0.7, 1.0]])
    factor_corr = np.broadcast_to(base, (2, 3, 2, 2)).copy()
    post = xr.Dataset(
        {
            "loading": (("chain", "draw", "indicator"), loading),
            "communality": (("chain", "draw", "indicator"), communality),
            "factor_corr": (("chain", "draw", "domain", "domain_b"), factor_corr),
        },
        coords={
            "chain": [0, 1],
            "draw": np.arange(3),
            "indicator": ["basread", "basspel", "bpvs"],
            "domain": ["reading", "language"],
            "domain_b": ["reading", "language"],
        },
    )
    return post


def test_loadings_table_columns_and_values():
    post = _synthetic_posterior()
    domains = {"reading": ("basread", "basspel"), "language": ("bpvs",)}
    df = loadings_communalities_table(post, domains, lo_q=0.055)

    assert list(df["indicator"]) == ["basread", "basspel", "bpvs"]
    # Domain lookup maps each indicator to its declared domain.
    assert list(df["domain"]) == ["reading", "reading", "language"]
    # The three quantity families are all present at the three summary widths.
    for q in ("loading", "correlation", "communality"):
        for stat in ("median", "mean", "lo", "hi", "lo50", "hi50"):
            assert f"{q}_{stat}" in df.columns
    # basread loadings across the six draws -> mean matches by hand.
    basread = np.array([0.8, 0.9, 0.7, 0.6, 0.8, 0.7])
    np.testing.assert_allclose(df.loc[0, "loading_mean"], basread.mean())
    # correlation = sqrt(communality) = sqrt(loading**2) = |loading|.
    np.testing.assert_allclose(
        df.loc[0, "correlation_mean"], np.sqrt(basread**2).mean()
    )
    np.testing.assert_allclose(df.loc[0, "communality_mean"], (basread**2).mean())


def test_factor_correlation_matrix_is_domain_indexed_mean():
    post = _synthetic_posterior()
    corr = factor_correlation_matrix(post)
    assert list(corr.index) == ["reading", "language"]
    assert list(corr.columns) == ["reading", "language"]
    np.testing.assert_allclose(np.diag(corr.to_numpy()), 1.0)
    np.testing.assert_allclose(corr.loc["reading", "language"], 0.7)


def test_factor_correlation_pairs_upper_triangle_only():
    post = _synthetic_posterior()
    pairs = factor_correlation_pairs(post, lo_q=0.055)
    # Two domains -> exactly one upper-triangle pair.
    assert len(pairs) == 1
    row = pairs.iloc[0]
    assert row["domain_i"] == "reading" and row["domain_j"] == "language"
    np.testing.assert_allclose(row["mean"], 0.7)
    assert row["prob_pos"] == 1.0
