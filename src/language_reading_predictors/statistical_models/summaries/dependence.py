# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Dependence calculations and summaries."""

from __future__ import annotations
import numpy as np
import pandas as pd
import xarray as xr


#: Posterior-to-prior SD ratios at or above this leave the block's correlation
#: indistinguishable from its prior. A convention for reading the table, not a
#: gate threshold: the ratio itself is reported so a reader can judge it.
DEPENDENCE_PRIOR_DOMINATED_RATIO = 0.95


#: Below this the block has genuinely narrowed the parameter.
DEPENDENCE_INFORMED_RATIO = 0.75


def _dependence_verdict(ratio: float | None) -> str:
    if ratio is None or not np.isfinite(ratio):
        return "not assessable"
    if ratio >= DEPENDENCE_PRIOR_DOMINATED_RATIO:
        return "prior-dominated"
    if ratio >= DEPENDENCE_INFORMED_RATIO:
        return "weakly informed"
    return "informed"


def dependence_identification_summary(
    trace: xr.DataTree, *, ci_prob: float
) -> pd.DataFrame | None:
    """How far the LKJ residual-dependence block is informed by the data.

    Returns ``None`` for a fit without the block. One row per free parameter of
    the block — each outcome pair's residual correlation (``u_corr_pair``) and
    each outcome's residual SD (``sigma_outcome``) — comparing the posterior SD
    against the prior SD **measured from this fit's own prior group**.

    Measuring the prior empirically rather than from the LKJ closed form is
    deliberate. The marginal SD of an off-diagonal under ``LKJ(eta)`` is
    ``1 / sqrt(2a + 1)`` with ``a = eta + (d - 2) / 2``
    (:func:`priors.residual_correlation_prior_sd`), and for the registered
    two-outcome companions that closed form and the fitted prior agree to three
    decimals. For ``d > 2`` they do **not**: drawing from this environment's
    ``pm.LKJCorr(n=3, eta=4)`` gives off-diagonal SDs of 0.316, 0.302 and 0.301,
    where a true LKJ is exchangeable and all three must match the closed form's
    0.316. Reading the yardstick off the fit's own ``prior`` group sidesteps that
    entirely and is correct whatever the sampler produces. The closed form is used
    only as a fallback when no prior group was persisted, and the ``prior_source``
    column records which was used.

    Why this exists (2026-08-22 ITT audit, finding 3). The three registered
    companions publish a dependence-corrected interval for a paired contrast, and
    their prose invites the reader to treat that interval as the data's verdict on
    within-child covariance. It is not: at n = 53 the correlation posteriors have
    SD 0.334, 0.337 and 0.334 against a prior SD of 0.334 — ratios of 1.002, 1.008
    and 1.001. The posterior *is* the prior, so the "correction" is the LKJ
    prior's, not the data's. The residual SDs are a different story and the table
    shows it: those posteriors sit well inside their prior, which is why the block
    is reported per parameter rather than with one verdict for the whole thing.
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
            post = np.asarray(
                posterior[name].isel({dim: index}).values, dtype=float
            ).ravel()
            prior_sd: float | None = None
            prior_source = "unavailable"
            if prior is not None and name in prior:
                draws = np.asarray(
                    prior[name].isel({dim: index}).values, dtype=float
                ).ravel()
                if draws.size > 1:
                    prior_sd = float(draws.std(ddof=1))
                    prior_source = "fitted prior draws"
            if prior_sd is None and name == "u_corr_pair" and n_outcomes >= 2:
                from language_reading_predictors.statistical_models.priors import (
                    residual_correlation_prior_sd,
                )

                prior_sd = residual_correlation_prior_sd(n_outcomes)
                prior_source = "LKJ closed form"
            post_sd = float(post.std(ddof=1))
            ratio = (
                post_sd / prior_sd
                if prior_sd is not None and prior_sd > 0
                else None
            )
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
                    "information_gain": None if ratio is None else 1.0 - ratio,
                    "verdict": _dependence_verdict(ratio),
                    "ci_prob": ci_prob,
                }
            )
    return pd.DataFrame(rows)
