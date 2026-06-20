# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Unit tests for the g-formula mediation decomposition (``mediation.decompose``).

These build the joint mediator + outcome model on synthetic data to obtain a
``MediationData`` aligned to the model, then run :func:`decompose` against a
small *synthetic* posterior (no MCMC sampling) — fast and deterministic. The
focus is the issue #85 guarantee: the g-formula adjusts for exactly the
confounder set the model was fitted with, sourced from ``MediationData`` rather
than a hardcoded ``{E, R}``.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import xarray as xr

from language_reading_predictors.statistical_models.factories import (
    build_mediation_model,
)
from language_reading_predictors.statistical_models.mediation import decompose
from language_reading_predictors.statistical_models.preprocessing import (
    load_and_prepare,
)

from .test_factories import _write_synthetic

_QUANTITIES = {"total", "NDE", "NIE", "proportion_mediated"}
# Outcome- and mediator-leg coefficients the decomposition reads, minus the
# per-confounder ``b_<s>`` / ``a_<s>`` terms (added per test to match the
# fitted confounder set).
_OUTCOME_DRAWS = ["b0", "b_G", "b_M", "b_GM", "b_W", "b_A"]
_BB_MEDIATOR_DRAWS = ["a0", "a_G", "a_L", "a_A"]
_GAUSSIAN_MEDIATOR_DRAWS = ["a0", "a_G", "a_comp", "a_A"]


def _prepare(tmp_path, n_children: int = 15):
    p = _write_synthetic(tmp_path, n_children=n_children)
    return load_and_prepare(path=p, phase_mode="itt")


def _fake_trace(names, *, positive=(), chains: int = 2, draws: int = 20, seed: int = 0):
    """A minimal posterior stand-in: an xarray ``Dataset`` of (chain, draw)
    coefficient draws exposed as ``.posterior`` — exactly what ``decompose``
    consumes (``trace.posterior[name].stack(...)``)."""
    rng = np.random.default_rng(seed)
    data = {}
    for name in names:
        arr = rng.normal(0.0, 0.2, size=(chains, draws))
        if name in positive:
            arr = np.abs(arr) + 5.0  # kappa / sigma must be positive
        data[name] = (("chain", "draw"), arr)
    return SimpleNamespace(posterior=xr.Dataset(data))


def test_decompose_beta_binomial(tmp_path):
    """LRP59: single Beta-Binomial mediator, confounders {E, R}."""
    prep = _prepare(tmp_path)
    _, med = build_mediation_model(prep, confounder_symbols=("E", "R"))
    names = _OUTCOME_DRAWS + ["b_E", "b_R"] + _BB_MEDIATOR_DRAWS + ["a_E", "a_R", "kappa_M"]
    df = decompose(_fake_trace(names, positive=["kappa_M"]), med, n_replicates=4)
    assert set(df["quantity"]) == _QUANTITIES
    effects = df[df["quantity"].isin(["total", "NDE", "NIE"])]
    assert effects[["prob_mean", "words_mean"]].notna().all().all()


def test_decompose_gaussian_composite(tmp_path):
    """LRP62: Gaussian route-composite mediator, confounders {E, R}."""
    prep = _prepare(tmp_path)
    _, med = build_mediation_model(
        prep,
        mediator_kind="gaussian_composite",
        route_symbols=("L", "B"),
        confounder_symbols=("E", "R"),
    )
    names = (
        _OUTCOME_DRAWS + ["b_E", "b_R"] + _GAUSSIAN_MEDIATOR_DRAWS + ["a_E", "a_R", "sigma_M"]
    )
    df = decompose(_fake_trace(names, positive=["sigma_M"]), med, n_replicates=4)
    assert set(df["quantity"]) == _QUANTITIES


def test_decompose_follows_fitted_confounder_set(tmp_path):
    """The core issue #85 guarantee: ``decompose`` adjusts for exactly the
    confounders carried on ``MediationData``. Fit with a single, *non-default*
    confounder (R only) and supply a posterior that contains only the R
    coefficients — under the old hardcoded ``{E, R}`` this would ``KeyError`` on
    the missing E coefficients / ``conf_logit['E']``."""
    prep = _prepare(tmp_path)
    _, med = build_mediation_model(prep, confounder_symbols=("R",))
    assert med.confounder_symbols == ("R",)
    assert set(med.conf_logit) == {"R"}
    names = _OUTCOME_DRAWS + ["b_R"] + _BB_MEDIATOR_DRAWS + ["a_R", "kappa_M"]
    df = decompose(_fake_trace(names, positive=["kappa_M"]), med, n_replicates=4)
    assert set(df["quantity"]) == _QUANTITIES
