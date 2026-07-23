# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Independent unit tests for the isolated correlated-factor summaries (#394 pillar 6).

Exercises the pure ``corr_factor_summaries`` functions on a hand-built posterior — no
factory, PyMC model, output directory, Quarto template or plotting session —
confirming the measurement-headline tables are computable in isolation, and that the
``loading_var`` parameter selects the right posterior variable (``lambda_load`` for
the RLI model, ``loading`` for the Byrne model).
"""

from __future__ import annotations

import numpy as np
import xarray as xr

from language_reading_predictors.statistical_models.corr_factor_summaries import (
    factor_correlation_matrix,
    factor_correlation_pairs,
    loadings_communalities_table,
)


def _synthetic_posterior(loading_var: str = "lambda_load"):
    """Two chains x three draws over three indicators (two domains)."""
    loading = np.array(
        [
            [[0.8, 0.6, 0.5], [0.9, 0.7, 0.4], [0.7, 0.5, 0.6]],
            [[0.6, 0.8, 0.5], [0.8, 0.6, 0.5], [0.7, 0.7, 0.4]],
        ]
    )  # (chain=2, draw=3, indicator=3)
    communality = loading**2
    base = np.array([[1.0, 0.7], [0.7, 1.0]])
    factor_corr = np.broadcast_to(base, (2, 3, 2, 2)).copy()
    post = xr.Dataset(
        {
            loading_var: (("chain", "draw", "indicator"), loading),
            "communality": (("chain", "draw", "indicator"), communality),
            "factor_corr": (("chain", "draw", "domain", "domain_b"), factor_corr),
        },
        coords={
            "chain": [0, 1],
            "draw": np.arange(3),
            "indicator": ["R", "E", "L"],
            "domain": ["vocab", "code"],
            "domain_b": ["vocab", "code"],
        },
    )
    return post


def test_loadings_table_reads_the_named_loading_var():
    domains = {"vocab": ("R", "E"), "code": ("L",)}
    r_loadings = np.array([0.8, 0.9, 0.7, 0.6, 0.8, 0.7])

    # RLI model: loadings under ``lambda_load``.
    df = loadings_communalities_table(
        _synthetic_posterior("lambda_load"), domains, lo_q=0.055,
        loading_var="lambda_load",
    )
    assert list(df["domain"]) == ["vocab", "vocab", "code"]
    np.testing.assert_allclose(df.loc[0, "loading_mean"], r_loadings.mean())
    np.testing.assert_allclose(df.loc[0, "communality_mean"], (r_loadings**2).mean())
    # correlation column = sqrt(communality).
    np.testing.assert_allclose(
        df.loc[0, "correlation_mean"], np.sqrt(r_loadings**2).mean()
    )

    # Byrne model: identical numbers when the loadings live under ``loading``.
    df_byrne = loadings_communalities_table(
        _synthetic_posterior("loading"), domains, lo_q=0.055, loading_var="loading",
    )
    np.testing.assert_allclose(
        df_byrne["loading_mean"].to_numpy(), df["loading_mean"].to_numpy()
    )


def test_factor_correlation_matrix_and_pairs():
    post = _synthetic_posterior()
    corr = factor_correlation_matrix(post)
    assert list(corr.index) == ["vocab", "code"]
    np.testing.assert_allclose(np.diag(corr.to_numpy()), 1.0)
    np.testing.assert_allclose(corr.loc["vocab", "code"], 0.7)

    pairs = factor_correlation_pairs(post, lo_q=0.055)
    assert len(pairs) == 1  # two domains -> one upper-triangle pair
    row = pairs.iloc[0]
    assert row["domain_i"] == "vocab" and row["domain_j"] == "code"
    np.testing.assert_allclose(row["mean"], 0.7)
    assert row["prob_pos"] == 1.0
