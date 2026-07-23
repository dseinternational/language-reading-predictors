# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Pure summary computations for the Byrne correlated-domain-factor model (mm-001).

The measurement-headline tables of :func:`pipeline.fit_rlm_corr_factor` — the
loadings / correlations / communalities table and the domain-factor correlation
matrix (mean matrix + per-pair posterior summary) — computed as pure functions of
the posterior. Extracted from the fit orchestration (#394 pillar 6, "separate
computation from presentation", mirroring :mod:`lcf_summaries`) so the numeric
summaries are testable without an output directory, Quarto template or Matplotlib
session; the fit function keeps the CSV persistence, console table and plots.

Each function takes the posterior group (``trace.posterior``) and returns a
``pandas.DataFrame`` with exactly the columns the fit previously wrote, so the
published ``loadings_summary.csv`` / ``factor_correlation.csv`` /
``factor_correlation_summary.csv`` are byte-for-byte unchanged.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def loadings_communalities_table(
    post: Any, domains: dict[str, tuple[str, ...]], *, lo_q: float
) -> pd.DataFrame:
    """Per-indicator loading, indicator-factor correlation (= sqrt communality) and communality
    posterior summaries (median, mean, ``lo_q``/``1 - lo_q`` and 50% quantiles)."""
    dom_of = {s: d for d, syms in domains.items() for s in syms}
    rows = []
    for j, name in enumerate(str(s) for s in post["indicator"].values):
        lam_d = post["loading"].isel(indicator=j).values.reshape(-1)
        com_d = post["communality"].isel(indicator=j).values.reshape(-1)
        corr_d = np.sqrt(com_d)
        rows.append(
            {
                "indicator": name,
                "domain": dom_of.get(name, "?"),
                "loading_median": float(np.median(lam_d)),
                "loading_mean": float(np.mean(lam_d)),
                "loading_lo": float(np.quantile(lam_d, lo_q)),
                "loading_hi": float(np.quantile(lam_d, 1 - lo_q)),
                "loading_lo50": float(np.quantile(lam_d, 0.25)),
                "loading_hi50": float(np.quantile(lam_d, 0.75)),
                "correlation_median": float(np.median(corr_d)),
                "correlation_mean": float(np.mean(corr_d)),
                "correlation_lo": float(np.quantile(corr_d, lo_q)),
                "correlation_hi": float(np.quantile(corr_d, 1 - lo_q)),
                "correlation_lo50": float(np.quantile(corr_d, 0.25)),
                "correlation_hi50": float(np.quantile(corr_d, 0.75)),
                "communality_median": float(np.median(com_d)),
                "communality_mean": float(np.mean(com_d)),
                "communality_lo": float(np.quantile(com_d, lo_q)),
                "communality_hi": float(np.quantile(com_d, 1 - lo_q)),
                "communality_lo50": float(np.quantile(com_d, 0.25)),
                "communality_hi50": float(np.quantile(com_d, 0.75)),
            }
        )
    return pd.DataFrame(rows)


def factor_correlation_matrix(post: Any) -> pd.DataFrame:
    """Posterior-mean domain-factor correlation matrix (domains x domains)."""
    corr_draws = post["factor_corr"]
    dnames = [str(d) for d in post["domain"].values]
    return pd.DataFrame(
        corr_draws.mean(dim=("chain", "draw")).values, index=dnames, columns=dnames
    )


def factor_correlation_pairs(post: Any, *, lo_q: float) -> pd.DataFrame:
    """Per-upper-triangle-pair domain-factor correlation posterior summary
    (median, mean, ``lo_q``/``1 - lo_q`` and 50% quantiles, and ``P(corr > 0)``)."""
    corr_draws = post["factor_corr"]
    dnames = [str(d) for d in post["domain"].values]
    corr_stacked = corr_draws.stack(sample=("chain", "draw"))
    rows = []
    for i, di in enumerate(dnames):
        for j, dj in enumerate(dnames):
            if j <= i:
                continue
            pair = np.asarray(
                corr_stacked.isel(domain=i, domain_b=j).values
            ).reshape(-1)
            rows.append(
                {
                    "domain_i": di,
                    "domain_j": dj,
                    "median": float(np.median(pair)),
                    "mean": float(np.mean(pair)),
                    "lo": float(np.quantile(pair, lo_q)),
                    "hi": float(np.quantile(pair, 1 - lo_q)),
                    "lo50": float(np.quantile(pair, 0.25)),
                    "hi50": float(np.quantile(pair, 0.75)),
                    "prob_pos": float(np.mean(pair > 0)),
                }
            )
    return pd.DataFrame(rows)
