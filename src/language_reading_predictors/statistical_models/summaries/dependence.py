# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Dependence calculations and summaries."""

from __future__ import annotations
import numpy as np
import pandas as pd
import xarray as xr


#: Descriptive spread thresholds only, not tests of distributional equality.
DEPENDENCE_PRIOR_DOMINATED_RATIO = 0.95


#: Below this the posterior SD is less than three quarters of the prior SD.
DEPENDENCE_INFORMED_RATIO = 0.75


def _dependence_verdict(ratio: float | None) -> str:
    if ratio is None or not np.isfinite(ratio):
        return "not assessable"
    if ratio >= DEPENDENCE_PRIOR_DOMINATED_RATIO:
        return "little or no contraction"
    if ratio >= DEPENDENCE_INFORMED_RATIO:
        return "moderate contraction"
    return "substantial contraction"


def dependence_identification_summary(trace: xr.DataTree, *, ci_prob: float) -> pd.DataFrame | None:
    """Compare posterior and prior spread, location and sign probabilities.

    An SD ratio measures relative spread only. It cannot establish that the data
    leave the distribution unchanged. Prior draws from this fit are preferred;
    a closed-form LKJ SD is used only when those draws are absent.
    """
    posterior = getattr(trace, "posterior", None)
    if posterior is None or "u_corr_pair" not in posterior:
        return None
    prior = None
    try:
        groups = {str(g).strip("/") for g in getattr(trace, "groups", ())}
        if "prior" in groups:
            prior = trace["prior"]
    except Exception:  # pragma: no cover - defensive
        prior = None

    lo_q = (1 - ci_prob) / 2
    hi_q = 1 - lo_q
    n_outcomes = int(posterior.sizes.get("outcome", 0))
    rows: list[dict] = []

    for name, role, dim in (
        ("u_corr_pair", "residual correlation", "outcome_pair"),
        ("sigma_outcome", "residual SD", "outcome"),
    ):
        if name not in posterior:
            continue
        labels = [str(v) for v in posterior[name].coords[dim].values]
        for index, label in enumerate(labels):
            post = np.asarray(posterior[name].isel({dim: index}).values, dtype=float).ravel()
            prior_sd: float | None = None
            prior_source = "unavailable"
            prior_median = prior_positive = None
            if prior is not None and name in prior:
                draws = np.asarray(prior[name].isel({dim: index}).values, dtype=float).ravel()
                if draws.size > 1:
                    prior_sd = float(draws.std(ddof=1))
                    prior_median = float(np.median(draws))
                    prior_positive = float(np.mean(draws > 0))
                    prior_source = "fitted prior draws"
            if prior_sd is None and name == "u_corr_pair" and n_outcomes >= 2:
                from language_reading_predictors.statistical_models.priors import (
                    residual_correlation_prior_sd,
                )

                prior_sd = residual_correlation_prior_sd(n_outcomes)
                prior_source = "LKJ closed form"
                prior_median, prior_positive = 0.0, 0.5
            post_sd = float(post.std(ddof=1))
            ratio = post_sd / prior_sd if prior_sd is not None and prior_sd > 0 else None
            rows.append(
                {
                    "parameter": f"{name}[{label}]",
                    "role": role,
                    "posterior_median": float(np.median(post)),
                    "lo": float(np.quantile(post, lo_q)),
                    "hi": float(np.quantile(post, hi_q)),
                    "posterior_sd": post_sd,
                    "prior_sd": prior_sd,
                    "prior_source": prior_source,
                    "posterior_prior_sd_ratio": ratio,
                    "sd_contraction": None if ratio is None else 1.0 - ratio,
                    "prior_median": prior_median,
                    "prior_prob_positive": prior_positive,
                    "posterior_prob_positive": float(np.mean(post > 0)),
                    "verdict": _dependence_verdict(ratio),
                    "ci_prob": ci_prob,
                }
            )
    return pd.DataFrame(rows)
