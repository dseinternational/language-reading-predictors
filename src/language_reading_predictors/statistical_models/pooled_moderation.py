# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Pooled treatment x baseline moderation across outcomes (#228 item 9).

The gain-factor models each estimate a treatment x own-baseline interaction
(``gamma_int_trt_own``): is the intervention's effect on that outcome smaller for
children who already started higher? It is negative in 7 of the 8 outcomes, but
every single interval spans (or nearly spans) zero. This module pools those eight
estimates with a Bayesian random-effects meta-analysis to give **one** posterior
for the equity question - "does the intervention flatten the baseline gradient,
on average across skills?" - with far more power than eight separate looks.

    theta_k ~ Normal(mu, tau)          # per-outcome true interaction
    d_k     ~ Normal(theta_k, se_k)    # each gain-factor model's estimate

``mu`` is the headline pooled interaction (negative = the intervention narrows the
baseline gap); ``tau`` is how much the interaction genuinely varies across skills.

**Second-stage caveat (load-bearing).** The eight outcomes are measured on the
same ~54 children, so the eight estimates are correlated, not independent studies.
This meta-analysis treats them as independent, which makes ``mu``'s interval
somewhat **too narrow**; read the direction and rough size, not a precise bound. A
single joint gain model with a pooled interaction hyperprior removes this caveat
and is the recommended follow-on (see the companion note). This is exploratory and
associational (the across-period interaction is an adjusted association, not the
randomised effect), per ``METHODS.md``.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd


def build_pooled_moderation_model(
    effects: np.ndarray,
    ses: np.ndarray,
    *,
    mu_sigma: float = 0.5,
    tau_sigma: float = 0.3,
):
    """Non-centred Bayesian random-effects meta-analysis over the interactions.

    ``effects`` / ``ses`` are the per-outcome ``gamma_int_trt_own`` posterior means
    and standard deviations. Priors: ``mu ~ Normal(0, mu_sigma)`` and
    ``tau ~ HalfNormal(tau_sigma)`` - weakly informative on the interaction's
    logit-ish scale (the raw estimates sit within about +/-0.3).
    """
    import pymc as pm

    effects = np.asarray(effects, dtype=float)
    ses = np.asarray(ses, dtype=float)
    if effects.shape != ses.shape or effects.ndim != 1 or effects.size < 2:
        raise ValueError("effects and ses must be 1-D of equal length >= 2")

    coords = {"outcome": [f"k{i}" for i in range(effects.size)]}
    with pm.Model(coords=coords) as model:
        d = pm.Data("effect_obs", effects, dims="outcome")
        s = pm.Data("effect_se", ses, dims="outcome")
        mu = pm.Normal("mu", 0.0, mu_sigma)
        tau = pm.HalfNormal("tau", tau_sigma)
        z = pm.Normal("z", 0.0, 1.0, dims="outcome")
        theta = pm.Deterministic("theta", mu + tau * z, dims="outcome")
        pm.Normal("obs", mu=theta, sigma=s, observed=d, dims="outcome")
    return model


def summarise(
    idata,
    labels: list[str],
    effects: np.ndarray,
    ses: np.ndarray,
    *,
    ci_prob: float = 0.95,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Shape the meta-analysis posterior into a pooled summary and a per-outcome table.

    Returns ``(pooled, by_outcome)``. ``pooled`` has the pooled interaction ``mu``
    (with ``prob_lt_0`` = the posterior probability the intervention flattens the
    gradient) and the heterogeneity ``tau``. ``by_outcome`` pairs each outcome's raw
    estimate with its shrunken posterior, so the pull toward the pooled mean is visible.
    """
    q_lo, q_hi = (1.0 - ci_prob) / 2.0, 1.0 - (1.0 - ci_prob) / 2.0
    post = idata.posterior
    mu = np.asarray(post["mu"].values).ravel()
    tau = np.asarray(post["tau"].values).ravel()

    def _row(term, draws, prob_neg):
        return {
            "term": term,
            "median": float(np.median(draws)),
            "lo": float(np.quantile(draws, q_lo)),
            "hi": float(np.quantile(draws, q_hi)),
            "prob_lt_0": float((draws < 0).mean()) if prob_neg else np.nan,
            "ci_prob": ci_prob,
        }

    pooled = pd.DataFrame(
        [
            _row("pooled interaction (mu)", mu, True),
            _row("heterogeneity (tau)", tau, False),
        ]
    )

    theta = np.asarray(post["theta"].values)  # (chain, draw, outcome)
    theta = theta.reshape(-1, theta.shape[-1])  # (sample, outcome)
    if theta.shape[1] != len(labels):
        raise ValueError("labels length must match the number of outcomes in theta")
    by = []
    for i, lab in enumerate(labels):
        d = theta[:, i]
        by.append(
            {
                "outcome": lab,
                "raw_mean": float(effects[i]),
                "raw_se": float(ses[i]),
                "shrunk_median": float(np.median(d)),
                "shrunk_lo": float(np.quantile(d, q_lo)),
                "shrunk_hi": float(np.quantile(d, q_hi)),
            }
        )
    return pooled, pd.DataFrame(by)


def run_pooled_moderation(
    effects: np.ndarray,
    ses: np.ndarray,
    labels: list[str],
    *,
    draws: int = 1000,
    tune: int = 1000,
    chains: int = 4,
    seed: int = 0,
    ci_prob: float = 0.95,
):
    """Build, sample and summarise. Returns ``(pooled, by_outcome, idata)``."""
    import pymc as pm

    model = build_pooled_moderation_model(effects, ses)
    with model:
        idata = pm.sample(
            draws=draws,
            tune=tune,
            chains=chains,
            random_seed=seed,
            progressbar=False,
            target_accept=0.95,
        )
    pooled, by = summarise(idata, labels, effects, ses, ci_prob=ci_prob)
    return pooled, by, idata


def _fake_idata(mu, tau, theta):
    """Test helper: wrap arrays as an ``idata``-like ``.posterior`` (chain, draw[, outcome])."""
    import xarray as xr

    ds = xr.Dataset(
        {
            "mu": (("chain", "draw"), np.asarray(mu)),
            "tau": (("chain", "draw"), np.asarray(tau)),
            "theta": (("chain", "draw", "outcome"), np.asarray(theta)),
        }
    )
    return SimpleNamespace(posterior=ds)
