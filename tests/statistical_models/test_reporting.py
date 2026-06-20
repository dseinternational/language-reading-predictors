# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Unit tests for :mod:`statistical_models.reporting`."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import xarray as xr
from scipy.special import expit

from language_reading_predictors.statistical_models.reporting import tau_summary_itt


def _trace(eta, tau, tau_i=None):
    """Wrap synthetic posterior arrays as an object exposing ``.posterior``.

    ``eta`` has shape (chain, draw, obs); ``tau`` (chain, draw); optional
    ``tau_i`` (chain, draw, obs). Using a plain ``Dataset`` keeps the test
    independent of the installed ArviZ version's ``from_dict`` behaviour.
    """
    n_chain, n_draw, n_obs = eta.shape
    data = {
        "eta": (("chain", "draw", "obs_id"), eta),
        "tau": (("chain", "draw"), tau),
    }
    if tau_i is not None:
        data["tau_i"] = (("chain", "draw", "obs_id"), tau_i)
    ds = xr.Dataset(
        data,
        coords={
            "chain": np.arange(n_chain),
            "draw": np.arange(n_draw),
            "obs_id": np.arange(n_obs),
        },
    )
    return SimpleNamespace(posterior=ds)


def _ame_by_loop(eta, delta, G):
    """Reference AME: per draw, average over obs of expit(eta0+delta)-expit(eta0)."""
    n_draw, n_obs = eta.shape[1], eta.shape[2]
    per_draw = []
    for d in range(n_draw):
        diffs = []
        for i in range(n_obs):
            d_i = delta[0, d, i] if delta.ndim == 3 else delta[0, d]
            eta0 = eta[0, d, i] - d_i * G[i]
            diffs.append(expit(eta0 + d_i) - expit(eta0))
        per_draw.append(np.mean(diffs))
    return float(np.mean(per_draw))


def test_tau_summary_itt_constant_tau_average_marginal_effect():
    # 1 chain, 2 draws, 3 observations.
    eta = np.array([[[0.0, 1.0, -0.5], [0.2, -1.0, 0.3]]])
    tau = np.array([[0.4, 0.6]])
    G = np.array([1.0, 0.0, 1.0])

    out = tau_summary_itt(_trace(eta, tau), hdi_prob=0.9, G=G)

    assert out["tau_prob_mean"] == pytest.approx(_ame_by_loop(eta, tau, G))
    assert out["tau_logit_mean"] == pytest.approx(float(np.mean(tau)))
    assert out["prob_tau_pos"] == pytest.approx(1.0)  # both tau draws > 0


def test_tau_summary_itt_operating_point_comes_from_full_eta():
    # Two constant etas (stand-ins for the cross-baseline / adjuster / GP terms
    # the old scalar baseline ignored) give different probability-scale effects
    # for the *same* tau, because expit is non-linear — confirming eta drives
    # the operating point rather than a single own-baseline mean.
    tau = np.array([[0.5]])
    G = np.array([1.0, 1.0])
    near_floor = tau_summary_itt(_trace(np.array([[[-2.0, -2.0]]]), tau), hdi_prob=0.9, G=G)
    near_mid = tau_summary_itt(_trace(np.array([[[0.0, 0.0]]]), tau), hdi_prob=0.9, G=G)
    assert near_floor["tau_prob_mean"] != pytest.approx(near_mid["tau_prob_mean"])
    # Logit-scale summary is the operating-point-invariant tau itself.
    assert near_floor["tau_logit_mean"] == pytest.approx(near_mid["tau_logit_mean"])


def test_tau_summary_itt_varying_tau_uses_tau_i():
    eta = np.array([[[0.1, -0.2, 0.4]]])
    tau = np.array([[0.5]])  # headline tau -> logit-scale summary
    tau_i = np.array([[[0.3, 0.7, 0.5]]])  # per-obs effect -> drives the AME
    G = np.array([1.0, 0.0, 1.0])

    out = tau_summary_itt(_trace(eta, tau, tau_i=tau_i), hdi_prob=0.9, G=G)

    assert out["tau_prob_mean"] == pytest.approx(_ame_by_loop(eta, tau_i, G))
    assert out["tau_logit_mean"] == pytest.approx(0.5)


def test_tau_summary_itt_rejects_misaligned_G():
    eta = np.array([[[0.0, 1.0, -0.5]]])  # 3 observations
    tau = np.array([[0.4]])
    with pytest.raises(ValueError, match="aligned with the fitted subset"):
        tau_summary_itt(_trace(eta, tau), hdi_prob=0.9, G=np.array([1.0, 0.0]))
