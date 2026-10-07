# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Shared interval summaries and diagnostics that retain chain identity."""

from __future__ import annotations
from dse_research_utils.statistics.array_intervals import equal_tail_interval

import arviz as az
import numpy as np
import xarray as xr


# Default outer posterior interval; prediction checks choose coverage separately.
# See notes/202607172359-credible-interval-standard.md.
REPORTING_CI_PROB = 0.89


def band50(draws: np.ndarray) -> tuple[float, float]:
    """Return the 25th and 75th percentiles for the inner 50% interval."""
    if np.size(draws) == 0:
        raise ValueError("Cannot summarise an empty posterior sample.")
    lo, hi = equal_tail_interval(draws, prob=0.5, axis=None, nonfinite="propagate")
    return float(lo), float(hi)


def derived_mc_diagnostics(
    draws: np.ndarray,
    *,
    n_chains: int,
    n_draws: int,
    prefix: str = "",
) -> dict[str, float]:
    """Sampling precision of a quantity calculated from posterior draws.

    ``draws`` must contain every draw, ordered by chain with draw varying fastest.
    Effective sample size and Monte Carlo error need the original chain layout.
    If values are missing or undefined, return unavailable diagnostics (NaN).
    Combining the remaining values into one chain can overstate precision.

    These diagnostics measure posterior sampling error. They do not measure
    numerical error in mediator integration or establish causal identification.
    """
    if n_chains < 1 or n_draws < 1:
        raise ValueError("n_chains and n_draws must both be positive")
    arr = np.asarray(draws, dtype=float).ravel()
    if arr.size != n_chains * n_draws or not np.all(np.isfinite(arr)):
        return {f"{prefix}{name}": float("nan") for name in ("ess_bulk", "ess_tail", "mcse_median")}
    da = xr.DataArray(arr.reshape(n_chains, n_draws), dims=("chain", "draw"))
    return {
        f"{prefix}ess_bulk": float(az.ess(da, method="bulk")),
        f"{prefix}ess_tail": float(az.ess(da, method="tail")),
        f"{prefix}mcse_median": float(az.mcse(da, method="median")),
    }


def loo_delta(loo_a: az.ELPDData, loo_b: az.ELPDData) -> dict[str, float]:
    """Delta-ELPD between two models using ArviZ compare.

    arviz 1.x ``az.compare`` reports the ELPD in an ``elpd`` column (the 0.x
    ``elpd_loo`` was renamed); ``dse`` is unchanged.
    """
    df = az.compare({"a": loo_a, "b": loo_b})
    # ``az.compare`` reports ``dse`` relative to the top-ranked (reference) model,
    # whose own ``dse`` is 0; the SE of the ELPD difference sits on the *other*
    # row. Reading ``df.loc["a", "dse"]`` returns 0 whenever "a" ranks first
    # (misleadingly certain). The pairwise difference SE is the single non-zero
    # ``dse`` across the two rows, so take the max (the reference's is exactly 0).
    if "dse" in df.columns:
        d_se = float(max(df.loc["a", "dse"], df.loc["b", "dse"]))
    else:
        d_se = float("nan")
    return {
        "d_elpd": float(df.loc["a", "elpd"] - df.loc["b", "elpd"]),
        "d_se": d_se,
    }


def beta_summary(trace: xr.DataTree, name: str, ci_prob: float) -> dict[str, float]:
    """Summarise ``name`` with median, mean, equal-tailed intervals and P(>0)."""
    draws = trace.posterior[name].stack(sample=("chain", "draw")).values
    lo, hi = equal_tail_interval(draws, prob=ci_prob, axis=None, nonfinite="propagate")
    lo50, hi50 = band50(draws)
    return {
        "median": float(np.median(draws)),
        "mean": float(np.mean(draws)),
        "lo": float(lo),
        "hi": float(hi),
        "lo50": lo50,
        "hi50": hi50,
        "prob_pos": float(np.mean(draws > 0)),
    }


def coef_row(label: str, draws: np.ndarray, hdi_prob: float) -> dict[str, str | float]:
    """Return a labelled coefficient summary with equal-tailed intervals.

    ``hdi_prob`` retains its legacy name but sets equal-tailed coverage.
    """
    d = np.asarray(draws).reshape(-1)
    lo, hi = equal_tail_interval(d, prob=hdi_prob, axis=None, nonfinite="propagate")
    lo50, hi50 = band50(d)
    return {
        "coefficient": label,
        "median": float(np.median(d)),
        "mean": float(np.mean(d)),
        "lo": float(lo),
        "hi": float(hi),
        "lo50": lo50,
        "hi50": hi50,
        "prob_pos": float(np.mean(d > 0)),
    }
