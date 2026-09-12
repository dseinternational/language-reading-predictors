# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Shared interval summaries and diagnostics that retain chain identity."""

from __future__ import annotations
import arviz as az
import numpy as np
import xarray as xr


# Default outer posterior interval; prediction checks choose coverage separately.
# See notes/202607172359-credible-interval-standard.md.
REPORTING_CI_PROB = 0.89


def band50(draws: np.ndarray) -> tuple[float, float]:
    """Inner 50 % equal-tailed band ``(lo25, hi75)`` reported alongside the headline.

    A single inner band so the summary builders that report only a headline
    ``ci_prob`` interval can also carry the inner 50 % equal-tailed interval
    without re-deriving quantiles at each call site. The wider ITT / growth
    summaries use the shared ``eti_bands`` helper; this covers the families that
    emit a single headline interval.
    """
    return float(np.quantile(draws, 0.25)), float(np.quantile(draws, 0.75))


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
        return {
            f"{prefix}{name}": float("nan")
            for name in ("ess_bulk", "ess_tail", "mcse_median")
        }
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
    """Posterior mean, equal-tailed ``ci_prob``-coverage interval, and P(>0) for ``name``.

    The interval is equal-tailed at ``ci_prob`` coverage, not an HDI — the parameter
    was previously named ``hdi``, which misdescribed it (the callers already pass
    ``ctx.reporting.ci_prob``).
    """
    draws = trace.posterior[name].stack(sample=("chain", "draw")).values
    lo_q, hi_q = (1 - ci_prob) / 2, 1 - (1 - ci_prob) / 2
    return {
        "median": float(np.median(draws)),
        "mean": float(np.mean(draws)),
        "lo": float(np.quantile(draws, lo_q)),
        "hi": float(np.quantile(draws, hi_q)),
        "lo50": float(np.quantile(draws, 0.25)),
        "hi50": float(np.quantile(draws, 0.75)),
        "prob_pos": float(np.mean(draws > 0)),
    }


def coef_row(label: str, draws: np.ndarray, hdi_prob: float) -> dict[str, str | float]:
    """Posterior mean, equal-tailed central interval and ``P(coef > 0)``.

    Equal-tailed quantiles at coverage ``hdi_prob`` — the same convention as
    :func:`reporting.tau_summary_itt` (not a highest-density interval).
    """
    d = np.asarray(draws).reshape(-1)
    lo_q = (1 - hdi_prob) / 2
    return {
        "coefficient": label,
        "median": float(np.median(d)),
        "mean": float(np.mean(d)),
        "lo": float(np.quantile(d, lo_q)),
        "hi": float(np.quantile(d, 1 - lo_q)),
        "lo50": float(np.quantile(d, 0.25)),
        "hi50": float(np.quantile(d, 0.75)),
        "prob_pos": float(np.mean(d > 0)),
    }
